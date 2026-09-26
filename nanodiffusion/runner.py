import torch

from nanodiffusion.models.base import ModelAdapter
from nanodiffusion.request import GenerationOutput, Request
from nanodiffusion.sampler import FlowEuler, classifier_free_guidance


class ModelRunner:
    def __init__(self, model: ModelAdapter):
        self.model = model

    @torch.inference_mode()
    def prepare(self, request: Request) -> None:
        params = request.params
        shape = self.model.latent_shape(params)
        shift = self.model.default_shift if params.flow_shift is None else params.flow_shift
        request.sampler = FlowEuler(params.num_steps, shift, self.model.num_train_timesteps)
        request.positive = self.model.encode(request.prompt)
        if params.guidance_scale > 1:
            request.negative = self.model.encode(params.negative_prompt)
        # A private CPU generator makes a request independent of queue order/global RNG.
        generator = torch.Generator(device="cpu").manual_seed(params.seed)
        request.latents = torch.randn(shape, generator=generator, dtype=torch.float32).to(self.model.device)
        self.model.begin_denoising()

    @torch.inference_mode()
    def denoise_step(self, request: Request) -> float:
        if request.sampler is None or request.latents is None or request.positive is None:
            raise RuntimeError("request was not prepared")
        index = request.step_index
        t = float(request.sampler.timesteps[index])
        timestep = torch.full((request.latents.shape[0],), t, device=self.model.device, dtype=torch.float32)
        inputs = request.latents.to(self.model.dtype)
        positive = self.model.predict_velocity(inputs, timestep, request.positive)
        negative = None
        if request.negative is not None:
            negative = self.model.predict_velocity(inputs, timestep, request.negative)
        velocity = classifier_free_guidance(positive, negative, request.params.guidance_scale)
        request.latents = request.sampler.step(velocity, index, request.latents)
        request.step_index += 1
        return t

    @torch.inference_mode()
    def finish(self, request: Request) -> GenerationOutput:
        self.model.end_denoising()
        if request.latents is None:
            raise RuntimeError("request has no latents")
        frames = self.model.decode(request.latents)
        calls_per_step = 2 if request.params.guidance_scale > 1 else 1
        result = GenerationOutput(request.request_id, request.prompt, request.params.seed,
                                  request.latents.cpu(), frames,
                                  request.step_index * calls_per_step)
        self.release(request)
        return result

    def release(self, request: Request) -> None:
        request.latents = request.positive = request.negative = request.sampler = None
