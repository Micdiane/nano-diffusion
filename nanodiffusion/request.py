from dataclasses import dataclass
from typing import TYPE_CHECKING
import torch

from nanodiffusion.config import SamplingParams
if TYPE_CHECKING:
    from diffusers import DPMSolverMultistepScheduler
    from nanodiffusion.models.sana import Conditioning


@dataclass
class Request:
    """Mutable sampling state; queue membership determines its lifecycle."""
    request_id: int
    prompt: str
    params: SamplingParams
    step_index: int = 0
    latents: torch.Tensor | None = None
    positive: 'Conditioning | None' = None
    negative: 'Conditioning | None' = None
    sampler: 'DPMSolverMultistepScheduler | None' = None
    host_timesteps: list[float] | None = None


@dataclass
class GenerationOutput:
    request_id: int
    prompt: str
    seed: int
    latents: torch.Tensor
    images: torch.Tensor
    num_model_calls: int


@dataclass
class StepOutput:
    request_id: int
    completed_steps: int
    total_steps: int
    timestep: float
    result: GenerationOutput | None = None
