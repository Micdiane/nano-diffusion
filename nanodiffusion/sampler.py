import math
import torch


class FlowEuler:
    """Integrate dx/dsigma = v(x,sigma), from noise sigma=1 to data sigma=0."""

    def __init__(self, num_steps: int, shift: float = 1.0, num_train_timesteps: int = 1000):
        if num_steps <= 0 or num_train_timesteps <= 0:
            raise ValueError("step counts must be positive")
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("shift must be finite and positive")
        # Match the usual static flow schedule: training sigma_min, then terminal zero.
        sigma = torch.linspace(1.0, 1.0 / num_train_timesteps, num_steps, dtype=torch.float64)
        # This denominator avoids cancellation at sigma=1 for tiny positive shifts.
        sigma = shift * sigma / ((1 - sigma) + shift * sigma)
        self.sigmas = torch.cat([sigma, sigma.new_zeros(1)])
        self.timesteps = self.sigmas[:-1] * num_train_timesteps

    def step(self, velocity: torch.Tensor, index: int, latents: torch.Tensor) -> torch.Tensor:
        if not 0 <= index < len(self.timesteps):
            raise IndexError("sampling step outside schedule")
        if velocity.shape != latents.shape:
            raise ValueError("velocity and latents must have the same shape")
        delta = float(self.sigmas[index + 1] - self.sigmas[index])
        return latents.float() + delta * velocity.float()


def classifier_free_guidance(positive: torch.Tensor, negative: torch.Tensor | None,
                            scale: float) -> torch.Tensor:
    if scale == 1.0:
        return positive.float()
    if negative is None or negative.shape != positive.shape:
        raise ValueError("CFG requires a negative prediction with matching shape")
    return negative.float() + scale * (positive.float() - negative.float())
