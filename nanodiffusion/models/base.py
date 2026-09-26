from typing import Protocol
import torch

from nanodiffusion.config import SamplingParams


class ModelAdapter(Protocol):
    """Component boundary; the engine owns noise, CFG and numerical integration."""

    device: torch.device
    dtype: torch.dtype
    default_shift: float
    num_train_timesteps: int

    def latent_shape(self, params: SamplingParams) -> tuple[int, int, int, int, int]: ...

    def encode(self, prompt: str) -> torch.Tensor: ...

    def begin_denoising(self) -> None: ...

    def predict_velocity(self, latents: torch.Tensor, timestep: torch.Tensor,
                         conditioning: torch.Tensor) -> torch.Tensor:
        """Runner supplies device-local latents/conditioning in compute dtype, FP32 time."""
        ...

    def end_denoising(self) -> None: ...

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Return CPU float32 frames [B,F,H,W,3] in [0,1]."""
        ...

    def close(self) -> None: ...
