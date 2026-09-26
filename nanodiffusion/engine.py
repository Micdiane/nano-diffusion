from itertools import count
from typing import Callable

from nanodiffusion.config import SamplingParams
from nanodiffusion.models.base import ModelAdapter
from nanodiffusion.request import GenerationOutput, Request, StepOutput
from nanodiffusion.runner import ModelRunner
from nanodiffusion.scheduler import RequestScheduler


class Diffusion:
    """Small offline engine: request admission -> model runner -> numerical sampler."""

    def __init__(self, model: ModelAdapter):
        self.runner = ModelRunner(model)
        self.scheduler = RequestScheduler()
        self._ids = count()
        self._closed = False
        self._cleanup_done = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("engine is closed")

    def _abort_active(self) -> None:
        request = self.scheduler.active
        if request is not None:
            try:
                self.runner.model.end_denoising()
            except Exception:
                pass  # Preserve the original model/callback failure.
            self.runner.release(request)
            self.scheduler.finish()

    def add_request(self, prompt: str, params: SamplingParams | None = None) -> int:
        self._check_open()
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        params = SamplingParams() if params is None else params
        self.runner.model.latent_shape(params)  # Validate before changing the queue.
        request_id = next(self._ids)
        self.scheduler.add(Request(request_id, prompt, params))
        return request_id

    def step(self) -> StepOutput:
        self._check_open()
        request = self.scheduler.schedule()
        try:
            if request.latents is None:
                self.runner.prepare(request)
            t = self.runner.denoise_step(request)
            result = None
            if request.step_index == request.params.num_steps:
                result = self.runner.finish(request)
                self.scheduler.finish()
            return StepOutput(request.request_id, request.step_index, request.params.num_steps, t, result)
        except Exception:
            self._abort_active()
            raise

    def generate(self, prompts: str | list[str], params: SamplingParams | None = None,
                 on_step: Callable[[StepOutput], None] | None = None) -> list[GenerationOutput]:
        self._check_open()
        if not self.scheduler.idle:
            raise RuntimeError("finish pending requests with step() before using generate()")
        prompts = [prompts] if isinstance(prompts, str) else prompts
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise TypeError("prompts must be a string or list of strings")
        for prompt in prompts:
            self.add_request(prompt, params)
        results = []
        while not self.scheduler.idle:
            update = self.step()
            if update.result is not None:
                results.append(update.result)
            if on_step is not None:
                try:
                    on_step(update)
                except Exception:
                    self._abort_active()
                    raise
        return results

    def close(self) -> None:
        if self._cleanup_done:
            return
        self._closed = True
        self._abort_active()
        self.scheduler.waiting.clear()
        self.runner.model.close()
        self._cleanup_done = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc is None:
            self.close()
        else:
            try:
                self.close()
            except Exception as cleanup_error:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"Adapter cleanup also failed: {cleanup_error}")
