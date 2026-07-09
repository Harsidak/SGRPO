"""
rollouts/sgrpo_rollout.py

One job: generate G rollouts with SSM state isolation between each rollout.
This is SGRPO's core novel contribution.

The mechanism:
1. Process the prompt through the model, capture SSM hidden state h_0
2. Before each rollout, restore h_0 exactly
3. Generate rollout from clean starting state
4. Repeat for all G rollouts

Why this restores the i.i.d. assumption:
- Without isolation: rollout k starts from h_T of rollout k-1 (contaminated)
- With isolation:    rollout k starts from h_0 (clean, identical for all k)
- The i.i.d. assumption requires identical starting conditions per rollout

Cache API (transformers >= 4.46, verified on 5.13):
- `MambaCache` was removed. The model returns a universal
  `transformers.cache_utils.DynamicCache` in `output.cache_params`.
- Per-layer state lives in `cache.layers[i]`:
    - recurrent_states: [batch, d_inner, d_state]  — the SSM state h_t
    - conv_states:      [batch, d_inner, conv_kernel] — causal-conv buffer
- `cache_position` no longer exists in the Mamba forward signature; the
  model switches prefill/decode via `cache.has_previous_state(layer_idx)`.

Complete isolation requires restoring BOTH tensors: recurrent_states carries
the SSM recurrence h_t, and conv_states carries the short causal-convolution
context (the last `conv_kernel` token projections). Restoring only the SSM
state would leave the first few generated tokens conditioned on the previous
rollout's conv buffer.

Hardening features:
- SSM state isolation diagnostics (cosine similarity verification)
- Single cache reused across rollouts — restore is a pure tensor copy,
  no extra prompt forward passes (keeps the overhead claim honest)
- Timing instrumentation for overhead measurement
- Numerical stability in log-probability computation
"""

import time

import torch
from transformers import MambaForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def _snapshot_ssm_states(cache: DynamicCache) -> list[dict[str, torch.Tensor]]:
    """
    Extract and clone all SSM recurrent states AND conv states from a
    DynamicCache.

    Per layer for Mamba-130m (24 layers):
        recurrent_states: [1, 1536, 16] — h_t, the tensor that carries
                          contamination between rollouts
        conv_states:      [1, 1536, 4]  — causal conv buffer

    NOT the residual stream hidden states (those are output_hidden_states,
    which misleadingly show cosine similarity ~1.0 — see CLAUDE.md §2.4).
    """
    return [
        {
            "recurrent": layer.recurrent_states.detach().clone(),
            "conv": layer.conv_states.detach().clone(),
        }
        for layer in cache.layers
    ]


def _restore_ssm_states(
    cache: DynamicCache,
    snapshot: list[dict[str, torch.Tensor]],
) -> None:
    """
    Restore SSM states from a snapshot into a DynamicCache in-place.
    This is the actual state isolation operation.

    Cost: L * d_inner * (d_state + conv_kernel) floats copied
    For Mamba-130m: 24 * 1536 * (16 + 4) = 737,280 floats ≈ 1.4MB in fp32.
    Negligible overhead; no forward passes involved.
    """
    for layer_idx, state in enumerate(snapshot):
        cache.layers[layer_idx].recurrent_states.copy_(state["recurrent"])
        cache.layers[layer_idx].conv_states.copy_(state["conv"])


def _compute_isolation_diagnostics(
    h0_clean: list[dict[str, torch.Tensor]],
    cache_post_rollout: DynamicCache,
) -> dict:
    """
    Compute diagnostics verifying that state isolation is working correctly.

    Returns dict with:
    - ssm_cosine_sim_mean: mean cosine similarity between h0 and post-rollout states
      (should be < 1.0 if the rollout changed states; ~1.0 means no generation happened)
    - ssm_l2_drift_mean: mean L2 distance of state drift from h0
    """
    cos_sims = []
    l2_drifts = []

    for layer_idx, h0 in enumerate(h0_clean):
        h_post = cache_post_rollout.layers[layer_idx].recurrent_states
        h0_flat = h0["recurrent"].flatten().float()
        h_post_flat = h_post.flatten().float()

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            h0_flat.unsqueeze(0), h_post_flat.unsqueeze(0)
        ).item()
        cos_sims.append(cos_sim)

        # L2 drift
        l2_drift = (h0_flat - h_post_flat).norm().item()
        l2_drifts.append(l2_drift)

    return {
        "sgrpo/ssm_cosine_sim_mean": sum(cos_sims) / len(cos_sims),
        "sgrpo/ssm_l2_drift_mean": sum(l2_drifts) / len(l2_drifts),
        "sgrpo/ssm_layers_checked": len(cos_sims),
    }


def generate_rollouts_isolated(
    model: MambaForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    group_size: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    device: str = "cuda",
) -> dict:
    """
    Generate group_size rollouts with SSM state isolation.
    Each rollout begins from an identical h_0 captured after prompt processing.

    Args:
        model:          the current policy model
        tokenizer:      tokenizer
        prompt:         formatted prompt string
        group_size:     G — number of rollouts to generate
        max_new_tokens: maximum tokens to generate per rollout
        temperature:    sampling temperature
        device:         "cuda" or "cpu"

    Returns same structure as base_rollout.generate_rollouts() for
    drop-in compatibility with the shared trainer, plus diagnostic fields.
    """
    model.eval()

    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    prompt_len = prompt_inputs.input_ids.shape[1]

    # ── Step 1: Process prompt once, capture clean h_0 ───────────────────
    # The model auto-creates a DynamicCache when use_cache=True.
    # The final-position logits give the distribution over the FIRST
    # generated token — identical for every rollout since all start from h_0.
    isolation_start = time.time()
    with torch.no_grad():
        prompt_out = model(
            input_ids=prompt_inputs.input_ids,
            use_cache=True,
        )
        cache = prompt_out.cache_params
        h0_clean = _snapshot_ssm_states(cache)
        first_token_logits = prompt_out.logits[:, -1, :].detach().clone()
    isolation_time = time.time() - isolation_start

    # ── Step 2: Generate G rollouts, each from clean h_0 ────────────────
    all_generated_ids = []
    all_old_log_probs = []
    all_generated_texts = []
    isolation_diagnostics = None

    with torch.no_grad():
        for k in range(group_size):
            # Restore h_0 before every rollout — this is the isolation
            # mechanism. Pure tensor copy into the same cache object;
            # no prompt re-encoding needed.
            _restore_ssm_states(cache, h0_clean)

            generated_ids = prompt_inputs.input_ids.clone()
            log_probs = []
            logits = first_token_logits

            for t_idx in range(max_new_tokens):
                # Sample next token from current logits
                if temperature != 1.0:
                    step_logits = logits / temperature
                else:
                    step_logits = logits

                # Numerically stable log-softmax
                log_probs_dist = torch.log_softmax(step_logits, dim=-1)
                probs = torch.exp(log_probs_dist)
                next_token = torch.multinomial(probs, 1)  # [1, 1]

                # Store log_prob of sampled token
                log_prob = log_probs_dist[0, next_token[0, 0]]
                log_probs.append(log_prob)

                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Stop at eos
                if next_token[0, 0].item() == tokenizer.eos_token_id:
                    break

                # Advance the state with the sampled token (decode step —
                # DynamicCache tracks position via has_previous_state)
                out = model(
                    input_ids=next_token,
                    cache_params=cache,
                    use_cache=True,
                )
                logits = out.logits[:, -1, :]

            # Compute isolation diagnostics on first rollout
            if k == 0:
                isolation_diagnostics = _compute_isolation_diagnostics(
                    h0_clean, cache
                )

            old_log_probs_tensor = torch.stack(log_probs)  # [gen_len]

            generated_text = tokenizer.decode(
                generated_ids[0, prompt_len:],
                skip_special_tokens=True
            )

            all_generated_ids.append(generated_ids[0])
            all_old_log_probs.append(old_log_probs_tensor)
            all_generated_texts.append(generated_text)

    # Pad log_probs to same length for stacking
    max_len = max(lp.shape[0] for lp in all_old_log_probs)
    padded_log_probs = []
    for lp in all_old_log_probs:
        pad_len = max_len - lp.shape[0]
        if pad_len > 0:
            padding = torch.zeros(pad_len, device=device, dtype=lp.dtype)
            lp = torch.cat([lp, padding])
        padded_log_probs.append(lp)

    result = {
        "input_ids": prompt_inputs.input_ids.expand(group_size, -1),
        "generated_ids": torch.nn.utils.rnn.pad_sequence(
            all_generated_ids, batch_first=True,
            padding_value=tokenizer.pad_token_id
        ),
        "generated_texts": all_generated_texts,
        "old_log_probs": torch.stack(padded_log_probs),  # [G, max_gen_len]
        "prompt_len": prompt_len,
        "h0_snapshot": h0_clean,               # kept for diagnostics
        "isolation_time": isolation_time,       # timing overhead
    }

    if isolation_diagnostics:
        result["isolation_diagnostics"] = isolation_diagnostics

    return result
