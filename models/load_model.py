"""
models/load_model.py

One job: load any HuggingFace causal LM and return (model, tokenizer).

Universal loader: supports pure-SSM models (Mamba-1/2), hybrid SSM+attention
models (Jamba, Zamba), and standard Transformers (Qwen, Phi, Gemma, ...).
The architecture class determines whether SGRPO's state isolation applies:

- "ssm":         all layers carry recurrent state → isolation required
- "hybrid":      SSM layers carry state, attention layers are stateless
                 → isolation of SSM layers required (future work)
- "transformer": stateless between generate() calls → isolation is a no-op;
                 SGRPO degenerates to GRPO (this is the paper's control)

Default remains state-spaces/mamba-130m-hf:
- Fits in 6GB VRAM during training at batch_size=1
- Fast enough for rapid iteration on loss function math
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "state-spaces/mamba-130m-hf"

# Architecture detection by model class name
SSM_ARCHITECTURES = {
    "MambaForCausalLM",
    "Mamba2ForCausalLM",
    "FalconMambaForCausalLM",
}
HYBRID_ARCHITECTURES = {
    "JambaForCausalLM",
    "ZambaForCausalLM",
    "Zamba2ForCausalLM",
    "BambaForCausalLM",
    "NemotronHForCausalLM",
}


def detect_architecture(model) -> str:
    """
    Return 'ssm', 'hybrid', or 'transformer' for a loaded model.
    This determines whether SGRPO's state isolation is needed.
    """
    class_name = model.__class__.__name__
    if class_name in SSM_ARCHITECTURES:
        return "ssm"
    if class_name in HYBRID_ARCHITECTURES:
        return "hybrid"
    return "transformer"


def load_model(device: str = "cuda", dtype: str = "bfloat16",
               model_name: str = MODEL_NAME):
    """
    Load a causal LM and tokenizer.

    Args:
        device:     "cuda" or "cpu"
        dtype:      "bfloat16" (default, saves VRAM) or "float32" (debugging)
        model_name: any HuggingFace model name or local path.
                    Default: state-spaces/mamba-130m-hf

    Returns:
        model:     causal LM moved to device, eval mode by default
        tokenizer: AutoTokenizer with pad_token set

    Use detect_architecture(model) to decide whether state isolation applies.

    VRAM cost for mamba-130m at bfloat16: ~260MB for weights alone.
    Training overhead (optimizer + gradients + activations): ~3-4GB total.
    Fits within RTX 4060 6GB at batch_size=1, grad_accum=8.
    """
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    print(f"Loading {model_name} on {device} in {dtype}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Many tokenizers (Mamba's included) ship without a pad_token — this
    # causes silent errors in batched generation. Set it explicitly.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    ).to(device)

    arch = detect_architecture(model)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,} "
          f"| Architecture: {arch}")
    if torch.cuda.is_available() and device != "cpu":
        print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return model, tokenizer


def get_model_name() -> str:
    return MODEL_NAME
