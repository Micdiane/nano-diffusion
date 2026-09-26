"""A local attention boundary; installing SageAttention is optional."""

import torch
import torch.nn.functional as F


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              backend: str = "sdpa") -> torch.Tensor:
    """Noncausal diffusion attention on [B,H,N,D], without GQA or dropout.

    SageAttention is approximate; selecting it does not imply identical outputs.
    """
    if backend not in ("sdpa", "sage"):
        raise ValueError("attention backend must be 'sdpa' or 'sage'")
    if any(x.ndim != 4 or any(n <= 0 for n in x.shape) for x in (q, k, v)):
        raise ValueError("q/k/v must be nonempty [B,H,N,D] tensors")
    if k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q/k/v must share batch, heads and head_dim; k/v shapes must match")
    if any(x.device != q.device or x.dtype != q.dtype for x in (k, v)):
        raise ValueError("q/k/v must have identical devices and dtypes")
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("attention supports float32, float16 and bfloat16")
    if backend == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    if q.device.type != "cuda":
        raise ValueError("SageAttention requires CUDA; choose backend='sdpa' for CPU")
    if q.dtype not in (torch.float16, torch.bfloat16) or q.shape[-1] not in (64, 128):
        raise ValueError("this Sage adapter requires FP16/BF16 and head_dim 64 or 128")
    try:
        from sageattention import sageattn
    except ImportError as exc:
        raise ImportError("backend='sage' requires a compatible SageAttention installation") from exc
    return sageattn(q.contiguous(), k.contiguous(), v.contiguous(),
                    tensor_layout="HND", is_causal=False)
