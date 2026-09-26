from dataclasses import dataclass
import torch

from nanodiffusion.config import SamplingParams
from nanodiffusion.sampler import FlowEuler


@dataclass
class Request:
    """Mutable sampling state; queue membership determines its lifecycle."""
    request_id: int
    prompt: str
    params: SamplingParams
    step_index: int = 0
    latents: torch.Tensor | None = None
    positive: torch.Tensor | None = None
    negative: torch.Tensor | None = None
    sampler: FlowEuler | None = None


@dataclass
class GenerationOutput:
    request_id: int
    prompt: str
    seed: int
    latents: torch.Tensor
    frames: torch.Tensor
    num_model_calls: int


@dataclass
class StepOutput:
    request_id: int
    completed_steps: int
    total_steps: int
    timestep: float
    result: GenerationOutput | None = None
