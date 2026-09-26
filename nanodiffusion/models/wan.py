"""Wan2.1 T2V components, with sampling owned by nano-diffusion.

The shape, T5 padding and VAE normalization follow Diffusers' pipeline_wan.py.
Only the single-transformer, 16-channel Wan2.1 T2V path is supported. There is
no pipeline.__call__, external scheduler, I2V, quantization or attention patch.
The intended checkpoint is Wan-AI/Wan2.1-T2V-1.3B-Diffusers. Real weights and
4090 memory requirements have not been tested by this educational adapter.
"""

import html
import re
from collections.abc import Mapping
from contextlib import contextmanager

import torch

from nanodiffusion.config import SamplingParams


def _get(config, name, default=None):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def _clean_prompt(prompt: str) -> str:
    # Like Diffusers: ftfy is optional; unescape twice before collapsing space.
    try:
        from ftfy import fix_text
    except ImportError:
        pass
    else:
        prompt = fix_text(prompt)
    return re.sub(r"\s+", " ", html.unescape(html.unescape(prompt))).strip()


class WanAdapter:
    """Wrap already-loaded components; preserve their mixed-precision weights.

    ``dtype`` must match the transformer's compute dtype. In particular, do
    not blanket-cast a loaded Wan transformer: its FP32 exceptions should stay
    FP32. ``from_pretrained`` arranges component dtypes, including an FP32 VAE.
    CPU offload moves whole components sequentially, not individual layers.
    This class is single-request and not thread safe.
    """

    default_shift = 3.0
    num_train_timesteps = 1000

    def __init__(self, tokenizer, text_encoder, transformer, vae, *,
                 device: str | torch.device = "cuda", dtype: torch.dtype = torch.bfloat16,
                 cpu_offload: bool = True, max_sequence_length: int = 512):
        self.device, self.dtype = torch.device(device), dtype
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("WanAdapter supports CPU and CUDA")
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("WanAdapter requires float32, float16 or bfloat16")
        if not isinstance(cpu_offload, bool):
            raise TypeError("cpu_offload must be bool")
        if (isinstance(max_sequence_length, bool) or not isinstance(max_sequence_length, int)
                or not 1 <= max_sequence_length <= 512):
            raise ValueError("max_sequence_length must be an integer in [1, 512]")
        if transformer is None or vae is None or text_encoder is None or tokenizer is None:
            raise ValueError("Wan T2V requires tokenizer, text encoder, transformer and VAE")

        tc, vc = transformer.config, vae.config
        self.latent_channels = _get(vc, "z_dim")
        self.temporal_scale = _get(vc, "scale_factor_temporal")
        self.spatial_scale = _get(vc, "scale_factor_spatial")
        if (self.latent_channels, self.temporal_scale, self.spatial_scale) != (16, 4, 8):
            raise ValueError("only the Wan2.1 VAE (16 channels, temporal/spatial scale 4/8) is supported")
        if (_get(tc, "in_channels") != 16 or _get(tc, "out_channels") != 16
                or tuple(_get(tc, "patch_size", ())) != (1, 2, 2)
                or any(_get(tc, name) is not None for name in
                       ("image_dim", "added_kv_proj_dim", "pos_embed_seq_len"))):
            raise ValueError("only Wan2.1 T2V is supported; I2V/TI2V or other transformer layouts are unsupported")
        mean = torch.as_tensor(_get(vc, "latents_mean"), dtype=torch.float32)
        std = torch.as_tensor(_get(vc, "latents_std"), dtype=torch.float32)
        if (mean.shape != (16,) or std.shape != (16,) or not torch.isfinite(mean).all()
                or not torch.isfinite(std).all() or not (std > 0).all()):
            raise ValueError("VAE latents_mean/std must contain 16 finite values with positive std")
        self._mean, self._std = mean.reshape(1, 16, 1, 1, 1), std.reshape(1, 16, 1, 1, 1)
        self.text_dim = _get(tc, "text_dim")
        self.cpu_offload, self.max_sequence_length = cpu_offload, max_sequence_length
        self.tokenizer, self.text_encoder = tokenizer, text_encoder
        self.transformer, self.vae = transformer, vae
        self._denoising = self._closed = False
        initial_device = torch.device("cpu") if cpu_offload else self.device
        for module in (text_encoder, transformer, vae):
            module.eval().requires_grad_(False).to(device=initial_device)
        self.vae.to(dtype=torch.float32)

    @classmethod
    def from_pretrained(cls, model_path: str, *, device: str | torch.device = "cuda",
                        dtype: torch.dtype = torch.bfloat16, cpu_offload: bool = True,
                        local_files_only: bool = True, max_sequence_length: int = 512):
        """Read local/cache files by default; False explicitly permits downloads.

        A Diffusers pipeline is used only to load components, then discarded.
        Its scheduler is not retained: nano-diffusion uses static FlowEuler.
        """
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be bool")
        try:
            from diffusers import AutoencoderKLWan, WanPipeline
        except ImportError as exc:
            raise ImportError("WanAdapter requires the optional 'wan' dependencies") from exc
        config = WanPipeline.load_config(model_path, local_files_only=local_files_only)
        if (_get(config, "_class_name") != "WanPipeline"
                or _get(config, "boundary_ratio") is not None
                or _get(config, "expand_timesteps", False)
                or _get(config, "transformer_2") not in (None, [None, None], (None, None))):
            raise ValueError("only single-transformer Wan2.1 T2V is supported (no dual transformer or expanded timestep)")
        vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae",
                                              torch_dtype=torch.float32,
                                              local_files_only=local_files_only)
        pipeline = WanPipeline.from_pretrained(model_path, vae=vae, torch_dtype=dtype,
                                                local_files_only=local_files_only)
        if (getattr(pipeline, "transformer_2", None) is not None
                or getattr(pipeline, "image_encoder", None) is not None):
            raise ValueError("dual-transformer and image-conditioned pipelines are unsupported")
        # We deliberately choose a static 480p-style shift, not an arbitrary
        # checkpoint scheduler's dynamic/multistep settings.
        return cls(pipeline.tokenizer, pipeline.text_encoder, pipeline.transformer, pipeline.vae,
                   device=device, dtype=dtype, cpu_offload=cpu_offload,
                   max_sequence_length=max_sequence_length)

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("WanAdapter is closed")

    @contextmanager
    def _use_component(self, module):
        """Keep text/VAE residency and error cleanup in one place."""
        try:
            module.to(device=self.device)
            yield
        finally:
            if self.cpu_offload:
                module.to(device="cpu")

    def latent_shape(self, params: SamplingParams) -> tuple[int, int, int, int, int]:
        self._ensure_open()
        # Reject instead of silently rounding the user's requested output size.
        if params.height % 16 or params.width % 16:
            raise ValueError("Wan height and width must be multiples of 16")
        if (params.num_frames - 1) % self.temporal_scale:
            raise ValueError("Wan num_frames must be 4*n+1 (for example 1, 5, 9)")
        return (1, 16, (params.num_frames - 1) // 4 + 1,
                params.height // 8, params.width // 8)

    @torch.inference_mode()
    def encode(self, prompt: str) -> torch.Tensor:
        self._ensure_open()
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if self._denoising:
            raise RuntimeError("cannot encode while denoising")
        tokens = self.tokenizer([_clean_prompt(prompt)], padding="max_length",
                                max_length=self.max_sequence_length, truncation=True,
                                add_special_tokens=True, return_attention_mask=True,
                                return_tensors="pt")
        mask = tokens.attention_mask
        length = int(mask.gt(0).sum(dim=1)[0])
        with self._use_component(self.text_encoder):
            embeddings = self.text_encoder(tokens.input_ids.to(self.device),
                                           mask.to(self.device)).last_hidden_state
            embeddings = embeddings.to(device=self.device, dtype=self.dtype)
            # T5's padded positions need not be zero even with an attention mask.
            embeddings[:, length:] = 0
            return embeddings

    def begin_denoising(self) -> None:
        self._ensure_open()
        self.transformer.to(device=self.device)
        self._denoising = True

    @torch.inference_mode()
    def predict_velocity(self, latents: torch.Tensor, timestep: torch.Tensor,
                         conditioning: torch.Tensor) -> torch.Tensor:
        self._ensure_open()
        if not self._denoising:
            raise RuntimeError("call begin_denoising before predict_velocity")
        if latents.ndim != 5 or latents.shape[1] != 16 or any(n <= 0 for n in latents.shape):
            raise ValueError("Wan latents must be nonempty [B,16,F,H,W]")
        if timestep.shape != (latents.shape[0],):
            raise ValueError("Wan requires one scalar timestep per batch element")
        if (conditioning.ndim != 3 or conditioning.shape[0] != latents.shape[0]
                or conditioning.shape[1] != self.max_sequence_length
                or (self.text_dim is not None and conditioning.shape[-1] != self.text_dim)):
            raise ValueError("conditioning must be [B,max_sequence_length,text_dim]")
        return self.transformer(hidden_states=latents, timestep=timestep,
                                encoder_hidden_states=conditioning, return_dict=False)[0]

    def end_denoising(self) -> None:
        if self._closed:
            return
        try:
            if self.cpu_offload:
                self.transformer.to(device="cpu")
        finally:
            self._denoising = False

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self._ensure_open()
        if self._denoising:
            raise RuntimeError("call end_denoising before decode")
        if latents.ndim != 5 or latents.shape[1] != 16:
            raise ValueError("Wan latents must have shape [B,16,F,H,W]")
        with self._use_component(self.vae):
            # Diffusers divides by reciprocal std: x * std + mean, NOT x / std.
            decoded_latents = latents.to(device=self.device, dtype=torch.float32)
            decoded_latents = decoded_latents * self._std.to(self.device) + self._mean.to(self.device)
            video = self.vae.decode(decoded_latents, return_dict=False)[0]
            if video.ndim != 5 or video.shape[1] != 3:
                raise ValueError("VAE output must be [B,3,F,H,W]")
            return (video.float() / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 4, 1).contiguous().cpu()

    def close(self) -> None:
        if self._closed:
            return
        for module in (self.text_encoder, self.transformer, self.vae):
            module.to(device="cpu")
        self.tokenizer = self.text_encoder = self.transformer = self.vae = None
        self._denoising, self._closed = False, True
