from typing import Protocol, TYPE_CHECKING
import torch

from nanodiffusion.config import SamplingParams
if TYPE_CHECKING:
    from diffusers import DPMSolverMultistepScheduler
    from nanodiffusion.models.sana import Conditioning


class ModelAdapter(Protocol):
    """The engine owns noise, CFG and per-request sampler state."""
    device: torch.device
    dtype: torch.dtype

    def latent_shape(self, params: SamplingParams) -> tuple[int, int, int, int]: ...
    def make_sampler(self, params: SamplingParams) -> 'DPMSolverMultistepScheduler': ...
    def encode(self, prompt: str) -> 'Conditioning': ...
    def predict_velocity(self, latents: torch.Tensor, timestep: torch.Tensor,
                         conditioning: 'Conditioning') -> torch.Tensor: ...
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Return CPU float32 images [B,H,W,3] in [0,1]."""
        ...
    def close(self) -> None: ...
