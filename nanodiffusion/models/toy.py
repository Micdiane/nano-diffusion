"""Small, randomly initialized video DiT for tracing the engine on CPU.

No weights are downloaded or trained. Images are diagnostic patterns, NOT
semantic text-to-image/video generations. The byte encoder and RGB decoder
demonstrate interfaces; neither is a pretrained text encoder or VAE.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from nanodiffusion.attention import attention
from nanodiffusion.config import SamplingParams


class ToyBlock(nn.Module):
    """One pre-norm self-attention/MLP block over all space-time tokens."""

    def __init__(self, width: int, backend: str):
        super().__init__()
        self.backend = backend
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # One attention head: [B,N,3C] -> three [B,1,N,C] tensors.
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        attended = attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
                             backend=self.backend).squeeze(1)
        x = x + self.proj(attended)
        return x + self.mlp(self.norm2(x))


class ToyAdapter(nn.Module):
    """Random 126k-parameter adapter with fourfold spatial latent compression.

    Latents are [B,4,F,H/4,W/4]; frames remain temporally uncompressed.
    Full space-time attention scales quadratically with F*(H/4)*(W/4).
    """

    default_shift = 1.0
    num_train_timesteps = 1000
    latent_channels = 4
    spatial_scale = 4
    width = 64

    def __init__(self, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32,
                 backend: str = "sdpa", seed: int = 0):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("ToyAdapter supports CPU and CUDA")
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("ToyAdapter requires float32, float16 or bfloat16")
        if backend not in ("sdpa", "sage"):
            raise ValueError("backend must be 'sdpa' or 'sage'")
        if backend == "sage" and (self.device.type != "cuda" or dtype == torch.float32):
            raise ValueError("SageAttention requires CUDA FP16/BF16; use SDPA for CPU")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        # Initialize on CPU, restoring its RNG state and never reseeding CUDA.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(seed)
            self.text = nn.Embedding(257, self.width)  # byte IDs plus a BOS token
            self.input_proj = nn.Linear(self.latent_channels, self.width)
            self.position_proj = nn.Linear(6, self.width)
            self.time_mlp = nn.Sequential(nn.Linear(self.width, self.width), nn.SiLU(),
                                          nn.Linear(self.width, self.width))
            self.blocks = nn.ModuleList([ToyBlock(self.width, backend) for _ in range(2)])
            self.output_norm = nn.LayerNorm(self.width)
            self.output_proj = nn.Linear(self.width, self.latent_channels)
            self.rgb = nn.Conv3d(self.latent_channels, 3, kernel_size=1)
        self.to(device=self.device, dtype=self.dtype)
        self.eval()

    def latent_shape(self, params: SamplingParams) -> tuple[int, int, int, int, int]:
        if params.height % self.spatial_scale or params.width % self.spatial_scale:
            raise ValueError("toy height and width must be positive multiples of 4")
        return (1, self.latent_channels, params.num_frames,
                params.height // self.spatial_scale, params.width // self.spatial_scale)

    @torch.inference_mode()
    def encode(self, prompt: str) -> torch.Tensor:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        # UTF-8 bytes are deterministic across processes; cap educational inputs.
        ids = [256, *prompt.encode("utf-8")[:256]]
        tokens = torch.tensor([ids], device=self.device, dtype=torch.long)
        return self.text(tokens)  # [1,text_length,64]

    def begin_denoising(self) -> None:
        pass

    def end_denoising(self) -> None:
        pass

    @torch.inference_mode()
    def predict_velocity(self, latents: torch.Tensor, timestep: torch.Tensor,
                         conditioning: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 5 or latents.shape[1] != self.latent_channels:
            raise ValueError("latents must have shape [B,4,F,H,W]")
        b, _, frames, height, width = latents.shape
        if any(n <= 0 for n in latents.shape):
            raise ValueError("latent dimensions must be positive")
        if timestep.numel() != b:
            raise ValueError("provide one timestep per batch element")
        if (conditioning.ndim != 3 or conditioning.shape[0] not in (1, b)
                or conditioning.shape[1] == 0 or conditioning.shape[-1] != self.width):
            raise ValueError("conditioning must be [1 or B,L,64] with L > 0")
        x = latents.permute(0, 2, 3, 4, 1).reshape(b, -1, self.latent_channels)
        axes = [torch.linspace(0, 1, n, device=self.device) for n in (frames, height, width)]
        coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        position = torch.cat((torch.sin(math.pi * coords), torch.cos(math.pi * coords)), dim=-1)
        frequencies = torch.exp(-math.log(10000) * torch.arange(
            self.width // 2, device=self.device, dtype=torch.float32) / (self.width // 2))
        phase = timestep.reshape(b, 1) * frequencies
        time = torch.cat((phase.sin(), phase.cos()), dim=-1).to(self.dtype)
        text = conditioning.mean(dim=1)
        condition = self.time_mlp(time) + text
        x = self.input_proj(x) + self.position_proj(position.to(self.dtype)) + condition[:, None]
        for block in self.blocks:
            x = block(x)
        velocity = self.output_proj(self.output_norm(x))
        return velocity.reshape(b, frames, height, width, self.latent_channels).permute(0, 4, 1, 2, 3)

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 5 or latents.shape[1] != self.latent_channels:
            raise ValueError("latents must have shape [B,4,F,H,W]")
        rgb = self.rgb(latents.to(device=self.device, dtype=self.dtype)).float()
        b, _, frames, height, width = rgb.shape
        images = rgb.permute(0, 2, 1, 3, 4).reshape(b * frames, 3, height, width)
        images = F.interpolate(images, scale_factor=self.spatial_scale,
                               mode="bilinear", align_corners=False).sigmoid()
        return images.reshape(b, frames, 3, height * 4, width * 4).permute(0, 1, 3, 4, 2).cpu()

    def close(self) -> None:
        pass
