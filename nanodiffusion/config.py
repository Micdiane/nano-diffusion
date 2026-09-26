from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SamplingParams:
    height: int = 32
    width: int = 32
    num_frames: int = 1
    num_steps: int = 8
    guidance_scale: float = 5.0
    seed: int = 0
    negative_prompt: str = ""
    flow_shift: float | None = None

    def __post_init__(self):
        for name in ("height", "width", "num_frames", "num_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 1:
            raise ValueError("guidance_scale must be finite and >= 1; 1 disables CFG")
        if self.flow_shift is not None and (not math.isfinite(self.flow_shift) or self.flow_shift <= 0):
            raise ValueError("flow_shift must be finite and positive")
        if not isinstance(self.negative_prompt, str):
            raise TypeError("negative_prompt must be a string")
