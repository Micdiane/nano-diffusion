import torch

from nanodiffusion.models.base import ModelAdapter
from nanodiffusion.request import GenerationOutput, Request


class ModelRunner:
    def __init__(self, model: ModelAdapter):
        self.model = model

    @torch.inference_mode()
    def prepare(self, request: Request) -> None:
        params = request.params
        request.sampler = self.model.make_sampler(params)
        request.host_timesteps = request.sampler.timesteps.tolist()
        request.positive = self.model.encode(request.prompt)
        if params.guidance_scale > 1:
            request.negative = self.model.encode(params.negative_prompt)
        # CPU float32 noise, independent of queue order and global RNG state.
        generator = torch.Generator(device='cpu').manual_seed(params.seed)
        request.latents = torch.randn(self.model.latent_shape(params), generator=generator,
                                      dtype=torch.float32).to(self.model.device)

    @torch.inference_mode()
    def denoise_step(self, request: Request) -> float:
        if request.sampler is None or request.latents is None or request.positive is None:
            raise RuntimeError('request was not prepared')
        index = request.step_index
        timestep = request.sampler.timesteps[index]
        model_time = timestep.expand(request.latents.shape[0]).float()
        inputs = request.latents.to(self.model.dtype)
        velocity = self.model.predict_velocity(inputs, model_time, request.positive).float()
        if request.negative is not None:
            negative = self.model.predict_velocity(inputs, model_time, request.negative).float()
            velocity = negative + request.params.guidance_scale * (velocity - negative)
        request.latents = request.sampler.step(velocity, timestep, request.latents, return_dict=False)[0]
        request.step_index += 1
        return request.host_timesteps[index]

    @torch.inference_mode()
    def finish(self, request: Request) -> GenerationOutput:
        if request.latents is None:
            raise RuntimeError('request has no latents')
        images = self.model.decode(request.latents)
        calls_per_step = 2 if request.params.guidance_scale > 1 else 1
        result = GenerationOutput(request.request_id, request.prompt, request.params.seed,
                                  request.latents.cpu(), images,
                                  request.step_index * calls_per_step)
        self.release(request)
        return result

    def release(self, request: Request) -> None:
        request.latents = request.positive = request.negative = request.sampler = None
        request.host_timesteps = None
