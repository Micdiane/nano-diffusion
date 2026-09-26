"""Generate a real SANA image from a local, pinned Diffusers checkpoint."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from nanodiffusion import Diffusion, SamplingParams


def main(argv=None):
    parser = argparse.ArgumentParser(description='nano-diffusion: readable pretrained SANA inference')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Local Sana_600M_512px_diffusers directory')
    parser.add_argument('--prompt', default='A small red panda drinking tea in a bamboo forest, watercolor painting')
    parser.add_argument('--negative-prompt', default='')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--cfg', type=float, default=4.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', choices=['bfloat16', 'float16'], default='bfloat16')
    parser.add_argument('--output', type=Path, default=Path('outputs/sana.png'))
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != '.png':
        parser.error('--output must end in .png')
    if not (args.checkpoint / 'model_index.json').is_file():
        parser.error('--checkpoint must contain a Diffusers model_index.json; see scripts/download_model.py')
    try:
        params = SamplingParams(height=args.height, width=args.width, num_steps=args.steps,
                                guidance_scale=args.cfg, seed=args.seed, negative_prompt=args.negative_prompt)
        if params.height % 32 or params.width % 32:
            raise ValueError('height and width must be divisible by 32')
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    if not torch.cuda.is_available():
        parser.error('CUDA is required for this real-model CLI')

    from PIL import Image
    from nanodiffusion.models.sana import SanaModel
    model = SanaModel(args.checkpoint, dtype=getattr(torch, args.dtype))
    def progress(update):
        if not args.quiet:
            print(f'step={update.completed_steps}/{update.total_steps} t={update.timestep:.3f}')
    with Diffusion(model) as engine:
        result = engine.generate(args.prompt, params, on_step=progress)[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pixels = result.images[0].mul(255).round().byte().numpy()
    Image.fromarray(pixels).save(args.output)
    metadata = {'prompt': args.prompt, 'sampling': asdict(params), 'checkpoint': str(args.checkpoint),
                'dit_dtype': args.dtype, 'text_encoder_dtype': 'bfloat16', 'vae_dtype': 'float32',
                'sampler': 'checkpoint DPMSolverMultistepScheduler', 'num_model_calls': result.num_model_calls}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved {args.output}')
    return 0
