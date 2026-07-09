# SGRPO Mathematical Derivations

## 1. Problem Statement: SSM State Contamination in GRPO

### 1.1 Background

Group Relative Policy Optimization (GRPO) generates G rollouts {o₁, o₂, ..., o_G} for a prompt q and computes advantages relative to the group:

$$A_i = \frac{r_i - \frac{1}{G}\sum_{j=1}^{G} r_j}{\text{std}(\{r_j\}_{j=1}^{G})}$$

The validity of this advantage estimator relies on a critical assumption:

**Assumption (i.i.d.):** Each rollout oᵢ is sampled independently and identically from π_θ(·|q).

### 1.2 The Violation in Mamba/SSM Architectures

For Mamba models, generation depends on recurrent SSM hidden states h_t:

$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$$

where $\bar{A}_t = \exp(\Delta_t A)$ and $\bar{B}_t = (\Delta_t A)^{-1}(\exp(\Delta_t A) - I) \cdot \Delta_t B$.

**Key observation:** When generating rollouts sequentially with a shared cache:

$$o_k \sim \pi_\theta(\cdot | q, h_T^{(k-1)})$$

where $h_T^{(k-1)}$ is the terminal hidden state of rollout k-1, NOT the clean post-prompt state $h_0$.

This means:
- o₁ ~ π_θ(·|q, h₀)
- o₂ ~ π_θ(·|q, h_T^{(1)}) ≠ π_θ(·|q, h₀)
- o₃ ~ π_θ(·|q, h_T^{(2)}) ≠ π_θ(·|q, h₀)

The rollouts are **not identically distributed** because they start from different hidden states.

### 1.3 Consequence: Biased Advantage Estimation

The group mean $\bar{r} = \frac{1}{G}\sum_j r_j$ is an estimator of $\mathbb{E}_{o \sim \pi_\theta(\cdot|q)}[r(o)]$.

When rollouts are not i.i.d., this estimator is biased:

$$\mathbb{E}[\bar{r}] = \frac{1}{G}\sum_{j=1}^{G} \mathbb{E}_{o_j \sim \pi_\theta(\cdot|q, h_T^{(j-1)})}[r(o_j)] \neq \mathbb{E}_{o \sim \pi_\theta(\cdot|q, h_0)}[r(o)]$$

The bias magnitude depends on how much the SSM states drift from h₀.

---

## 2. SGRPO: State-Isolated Group Relative Policy Optimization

### 2.1 The Fix: State Isolation

Before each rollout k, restore the SSM states to h₀:

$$\forall k \in \{1, ..., G\}: \quad h^{(k)}_{\text{init}} \leftarrow h_0$$

This ensures:

$$\forall k: \quad o_k \sim \pi_\theta(\cdot | q, h_0)$$

which restores the i.i.d. assumption.

### 2.2 The SGRPO Objective

With isolation, the advantage estimator becomes unbiased. We denote the isolated advantage as $A_i^{ISO}$:

$$A_i^{ISO} = \frac{r_i - \frac{1}{G}\sum_{j=1}^{G} r_j}{\text{std}(\{r_j\}_{j=1}^{G})}$$

where all $r_j$ are computed from rollouts starting at the same $h_0$.

Combined with DAPO's token-level normalization and FIPO's Future-KL weighting:

$$\mathcal{L}^{SGRPO} = -\frac{1}{\sum_{i=1}^{G}|o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \min\left(r_t \cdot \hat{A}_t^{ISO},\ \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot \hat{A}_t^{ISO}\right)$$

where:

$$r_t = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})}$$

$$\hat{A}_t^{ISO} = A_i^{ISO} \cdot w(\text{FutureKL}_t)$$

### 2.3 State Isolation Cost Analysis

For Mamba-130m with L=24 layers, d_inner=1536, d_state=16:

**Per-layer state size:** d_inner × d_state = 24,576 floats = 98,304 bytes (float32)

**Total snapshot cost:** L × 98,304 = 2,359,296 bytes ≈ 2.3 MB

**Per-rollout restore cost:** Same as snapshot (memcpy)

**Overhead per group:** G × 2.3 MB memcpy ≈ negligible vs. forward pass cost

**As fraction of model parameters:** 589,824 / 129,916,160 ≈ 0.45%

---

## 3. Future-KL Token Weighting

### 3.1 Motivation

Not all tokens in a rollout contribute equally to the policy update. Tokens that anchor significant probability mass shifts should receive higher gradient weight.

### 3.2 Definition

$$\text{FutureKL}_t = \sum_{k=t}^{T} \gamma^{k-t} \cdot M_k \cdot \delta_k$$

where:
- $\delta_k = \log \pi_\theta(y_k) - \log \pi_{\theta_{old}}(y_k)$ is the log-probability shift at position k
- $\gamma = 2^{-1/\tau}$ is the exponential decay factor
- $\tau$ is the decay rate (default: 30)
- $M_k$ is the attention mask (1 for real tokens, 0 for padding)

### 3.3 Influence Weight Function

$$w(\text{FutureKL}_t) = \text{clip}\left(\exp(\text{FutureKL}_t),\ w_{low},\ w_{high}\right)$$

Default: $w_{low} = 1.0$, $w_{high} = 1.2$

### 3.4 Interpretation

- $\text{FutureKL}_t > 0$: token t anchors a trajectory being reinforced → weight > 1
- $\text{FutureKL}_t < 0$: token t anchors a trajectory being suppressed → weight < 1 (clamped to w_low)
- $\text{FutureKL}_t ≈ 0$: token t is approximately neutral → weight ≈ 1

The clipping prevents extreme weights that would destabilize training.

### 3.5 Computational Complexity

The Future-KL computation is a reverse exponential moving average:

$$\text{FutureKL}_T = \delta_T$$
$$\text{FutureKL}_t = \delta_t + \gamma \cdot \text{FutureKL}_{t+1}$$

This is computed in O(G × T) time with a single reverse scan. The implementation uses `torch.flip` + forward scan + `torch.flip` for GPU-friendly execution.

---

## 4. Gradient Analysis

### 4.1 Policy Gradient Under SGRPO

The gradient of the SGRPO objective with respect to θ:

$$\nabla_\theta \mathcal{L}^{SGRPO} = -\frac{1}{\sum_i |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \nabla_\theta f(r_t, \hat{A}_t^{ISO})$$

where f is the clipped surrogate:

$$f(r, A) = \min(rA, \text{clip}(r, 1-\varepsilon, 1+\varepsilon) \cdot A)$$

The gradient of f with respect to θ (through r_t):

$$\nabla_\theta f = \begin{cases}
\hat{A}_t^{ISO} \cdot \nabla_\theta r_t & \text{if } |r_t - 1| < \varepsilon \text{ (unclipped)} \\
0 & \text{if } r_t \text{ is clipped and clip objective is lower}
\end{cases}$$

### 4.2 Why BAPO's Adaptive Clipping is Incompatible with Future-KL

BAPO derives adaptive bounds assuming uniform per-token advantage weighting:

$$\varepsilon_{high} = \varepsilon \cdot \frac{N_{neg}}{N_{pos} + N_{neg}}, \quad \varepsilon_{low} = \varepsilon \cdot \frac{N_{pos}}{N_{pos} + N_{neg}}$$

where $N_{pos}$ and $N_{neg}$ count tokens with positive/negative advantages.

With Future-KL weighting, the effective advantage at each token is $A \cdot w_t$, which changes the positive/negative split continuously. The adaptive bounds would need to be:

$$\varepsilon_{high}^{weighted} = \varepsilon \cdot \frac{\sum_t \mathbb{1}[A \cdot w_t < 0]}{\sum_t 1}$$

This is computable but the theoretical justification for this specific functional form under weighted advantages has not been established. Using it without proof would be mathematically unjustified.

---

## 5. Convergence Intuition

### 5.1 Why State Isolation Improves Convergence

Without isolation, the advantage estimates have bias B and variance V:

$$\text{MSE}(\hat{A}) = B^2 + V$$

State isolation eliminates B (the systematic bias from contamination), reducing MSE:

$$\text{MSE}(\hat{A}^{ISO}) = 0 + V^{ISO}$$

Since $V^{ISO} \leq V$ (identical starting states reduce variance), state isolation strictly reduces the MSE of advantage estimates.

Lower MSE → more accurate policy gradients → faster and more stable convergence.

### 5.2 Why Future-KL Improves Sample Efficiency

Future-KL upweights tokens that anchor significant probability mass shifts. These are the tokens where the policy update is most impactful. By concentrating gradient on these tokens, Future-KL achieves:

1. **Higher signal-to-noise ratio** in the gradient estimate
2. **Reduced variance** in the policy gradient (tokens with near-zero influence contribute less noise)
3. **Better credit assignment** (early tokens that set up successful reasoning chains get more credit)

---

## 6. Numerical Stability Analysis

### 6.1 Log-Ratio Clamping

The importance sampling ratio $r_t = \exp(\log \pi_\theta - \log \pi_{\theta_{old}})$ can overflow if the log-ratio is too large.

We clamp: $\log r_t \in [-20, 20]$, which bounds $r_t \in [2 \times 10^{-9}, 5 \times 10^{8}]$.

After PPO clipping, the effective range is $[1-\varepsilon, 1+\varepsilon]$, so the clamping only affects the gradient computation for extreme ratios that would be clipped anyway.

### 6.2 Advantage Normalization

Division by std(r) with additive epsilon:

$$A_i = \frac{r_i - \bar{r}}{\text{std}(r) + 10^{-8}}$$

The epsilon prevents division by zero when all rewards are identical. In this case, the degenerate group filter catches it before normalization.

### 6.3 Future-KL Stability

The exp() in influence weight computation is guarded by clamping the input to [-10, 10]:

$$w_t = \text{clip}(\exp(\text{clamp}(\text{FutureKL}_t, -10, 10)), w_{low}, w_{high})$$

Since $w_{high} = 1.2$ and $\exp(-10) \approx 5 \times 10^{-5} < w_{low} = 1.0$, the clamping has no effect on the final output — it only prevents intermediate overflow.
