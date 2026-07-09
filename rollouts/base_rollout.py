"""
rollouts/base_rollout.py

One job: generate G rollouts for a given prompt.
Standard rollout — no SSM state isolation.
Used by PPO, GRPO, DAPO, BAPO.

Returns tokens and log_probs needed for the loss function.
The log_probs returned here are the OLD log_probs (pi_theta_old)
used in the importance sampling ratio during optimization.
"""

import torch
from transformers import MambaForCausalLM, AutoTokenizer


def generate_rollouts(
    model: MambaForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    group_size: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    device: str = "cuda",
) -> dict:
    """
    Generate group_size rollouts for a single prompt.

    Args:
        model:          the current policy model (pi_theta_old during rollout)
        tokenizer:      tokenizer
        prompt:         formatted prompt string
        group_size:     G — number of rollouts to generate
        max_new_tokens: maximum tokens to generate per rollout
        temperature:    sampling temperature (1.0 = standard sampling)
        device:         "cuda" or "cpu"

    Returns dict with:
        "input_ids":        [G, prompt_len] — prompt token IDs
        "generated_ids":    [G, prompt_len + gen_len] — full sequence IDs
        "generated_texts":  list of G decoded strings (generated portion only)
        "old_log_probs":    [G, gen_len] — log probs under pi_theta_old
                            CRITICAL: stored BEFORE any gradient update.
                            Used in importance sampling ratio during optimization.
    """
    model.eval()

    # Tokenize prompt
    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    prompt_len = prompt_inputs.input_ids.shape[1]

    all_generated_ids = []
    all_old_log_probs = []
    all_generated_texts = []

    with torch.no_grad():
        for k in range(group_size):
            # Generate one rollout
            # NOTE: for Mamba, generate() re-encodes prompt each time
            # because no cache is passed — this means no state contamination
            # in base_rollout (each generate() call starts fresh).
            # The contamination issue happens when you manually manage
            # the cache across rollouts — which naive GRPO implementations do.
            output = model.generate(
                prompt_inputs.input_ids,
                attention_mask=prompt_inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

            generated_ids = output.sequences  # [1, prompt_len + gen_len]
            scores = output.scores            # tuple of [1, vocab_size] per generated token

            # Compute log_probs for each generated token
            # scores[t] is the logit distribution at generation step t
            gen_len = len(scores)
            log_probs = []
            for t in range(gen_len):
                token_id = generated_ids[0, prompt_len + t]
                log_prob = torch.log_softmax(scores[t][0], dim=-1)[token_id]
                log_probs.append(log_prob)

            old_log_probs = torch.stack(log_probs)  # [gen_len]

            # Decode only the generated portion
            generated_text = tokenizer.decode(
                generated_ids[0, prompt_len:],
                skip_special_tokens=True
            )

            all_generated_ids.append(generated_ids[0])
            all_old_log_probs.append(old_log_probs)
            all_generated_texts.append(generated_text)

    # Rollouts stop at EOS independently, so lengths differ within the
    # group — pad to the longest before stacking. Padded log-prob positions
    # are zeros; the trainer's attention mask excludes them from the loss.
    padded_log_probs = []
    max_lp_len = max(lp.shape[0] for lp in all_old_log_probs)
    for lp in all_old_log_probs:
        pad_len = max_lp_len - lp.shape[0]
        if pad_len > 0:
            lp = torch.cat([lp, torch.zeros(pad_len, device=lp.device,
                                            dtype=lp.dtype)])
        padded_log_probs.append(lp)

    return {
        "input_ids": prompt_inputs.input_ids.expand(group_size, -1),
        "generated_ids": torch.nn.utils.rnn.pad_sequence(
            all_generated_ids, batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ),                                                       # [G, full_len]
        "generated_texts": all_generated_texts,                  # list of G strings
        "old_log_probs": torch.stack(padded_log_probs),          # [G, gen_len]
        "prompt_len": prompt_len,
    }
