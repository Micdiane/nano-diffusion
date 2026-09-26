from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SamplingParams:
    height: int = 512
    width: int = 512
    num_steps: int = 20
    guidance_scale: float = 4.5
    seed: int = 42
    negative_prompt: str = ''

    def __post_init__(self):
        for name in ('height', 'width', 'num_steps'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63)')
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 1:
            raise ValueError('guidance_scale must be finite and >= 1; 1 disables CFG')
        if not isinstance(self.negative_prompt, str):
            raise TypeError('negative_prompt must be a string')
