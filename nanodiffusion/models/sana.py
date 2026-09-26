"""Pretrained SANA components. The runner, not a library pipeline, samples images."""
from dataclasses import dataclass
from pathlib import Path

import torch

from nanodiffusion.config import SamplingParams
from nanodiffusion.models.sana_transformer import SanaTransformer

MODEL_ID = 'Efficient-Large-Model/Sana_600M_512px_diffusers'
REVISION = '83d7a190bfd1fd070570a793d2dab5c7a3231b9d'


@dataclass
class Conditioning:
    hidden_states: torch.Tensor
    mask: torch.Tensor


class SanaModel:
    def __init__(self, checkpoint, *, device='cuda', dtype=torch.bfloat16):
        from diffusers import AutoencoderDC, DPMSolverMultistepScheduler
        from transformers import AutoTokenizer, Gemma2Model

        self.checkpoint = Path(checkpoint)
        self.device, self.dtype = torch.device(device), dtype
        self.transformer = SanaTransformer.from_pretrained(self.checkpoint / 'transformer', device=device, dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint / 'tokenizer', padding_side='right', local_files_only=True)
        self.text_encoder = Gemma2Model.from_pretrained(
            self.checkpoint / 'text_encoder', dtype=torch.bfloat16,
            attn_implementation='eager', local_files_only=True,
        ).to(device).eval().requires_grad_(False)
        self.vae = AutoencoderDC.from_pretrained(
            self.checkpoint / 'vae', torch_dtype=torch.float32, local_files_only=True,
        ).to(device).eval().requires_grad_(False)
        self.scheduler_config = DPMSolverMultistepScheduler.load_config(self.checkpoint / 'scheduler')

    def latent_shape(self, params: SamplingParams):
        if params.height % 32 or params.width % 32:
            raise ValueError('SANA height and width must be divisible by 32')
        return (1, self.transformer.config['in_channels'], params.height // 32, params.width // 32)

    def make_sampler(self, params):
        from diffusers import DPMSolverMultistepScheduler
        # Each request owns history: multistep samplers cannot be shared across requests.
        sampler = DPMSolverMultistepScheduler.from_config(self.scheduler_config)
        sampler.set_timesteps(params.num_steps, device=self.device)
        return sampler

    @torch.inference_mode()
    def encode(self, prompt):
        # Deliberately explicit: no prompt enhancement, no caption rewriting.
        tokens = self.tokenizer(prompt, padding='max_length', max_length=300,
                                truncation=True, return_tensors='pt').to(self.device)
        hidden = self.text_encoder(**tokens, use_cache=False).last_hidden_state
        return Conditioning(hidden.to(self.dtype), tokens.attention_mask)

    def predict_velocity(self, latents, timestep, conditioning):
        scale = self.transformer.config.get('timestep_scale', 1.0)
        return self.transformer(latents, timestep * scale, conditioning.hidden_states, conditioning.mask)

    @torch.inference_mode()
    def decode(self, latents):
        pixels = self.vae.decode(latents.float() / self.vae.config.scaling_factor, return_dict=False)[0]
        return (pixels / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()

    def close(self):
        self.transformer = self.text_encoder = self.vae = None
