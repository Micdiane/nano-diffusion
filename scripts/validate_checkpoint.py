"""Real-weight CUDA validation against Diffusers. Requires the downloaded checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from diffusers import SanaPipeline, SanaTransformer2DModel, DPMSolverMultistepScheduler
from nanodiffusion import Diffusion, SamplingParams
from nanodiffusion.models.sana import SanaModel, REVISION


def metrics(actual, expected):
    a, b = actual.float(), expected.float()
    d = a - b
    return {'max_abs': d.abs().max().item(), 'rmse': d.square().mean().sqrt().item(),
            'relative_l2': (d.norm() / b.norm().clamp_min(1e-12)).item(),
            'cosine': torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
            'finite': bool(a.isfinite().all() and b.isfinite().all())}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/checkpoint-validation.json'))
    args = parser.parse_args()
    from checkpoint import verify_checkpoint
    verify_checkpoint(args.checkpoint)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    model = SanaModel(args.checkpoint)
    ref = SanaTransformer2DModel.from_pretrained(args.checkpoint, subfolder='transformer',
                                                 torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    params = SamplingParams()
    prompt = 'A small red panda drinking tea in a bamboo forest, watercolor painting'
    positive, negative = model.encode(prompt), model.encode('')
    noise = torch.randn(model.latent_shape(params), generator=torch.Generator().manual_seed(params.seed)).cuda()
    report = {'model_revision': REVISION, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'prompt': prompt, 'forward': [], 'policy': 'bf16 DiT, fp32 CFG/latents/VAE; Gemma2 eager bf16, no prompt enrichment'}
    for t in (999, 500, 136):
        for name, cond in [('positive', positive), ('negative', negative)]:
            time = torch.tensor([t], device='cuda', dtype=torch.float32)
            expected = ref(noise.bfloat16(), cond.hidden_states, time, encoder_attention_mask=cond.mask).sample
            actual = model.predict_velocity(noise.bfloat16(), time, cond)
            result = metrics(actual, expected)
            report['forward'].append({'timestep': t, 'branch': name, **result})
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Independently execute the official pipeline with the exact same conditioning
    # and initial noise. Diffusers batches CFG; nano issues sequential CFG forwards.
    pipeline = SanaPipeline(tokenizer=model.tokenizer, text_encoder=model.text_encoder, vae=model.vae,
                            transformer=ref, scheduler=DPMSolverMultistepScheduler.from_config(model.scheduler_config))
    pipeline.set_progress_bar_config(disable=True)
    expected_latents = pipeline(
        negative_prompt=None,
        prompt_embeds=positive.hidden_states, prompt_attention_mask=positive.mask,
        negative_prompt_embeds=negative.hidden_states, negative_prompt_attention_mask=negative.mask,
        height=512, width=512, num_inference_steps=20, guidance_scale=4.5,
        latents=noise.clone(), output_type='latent', use_resolution_binning=False,
    ).images
    with Diffusion(model) as engine:
        actual = engine.generate(prompt, params)[0]
        expected_images = model.decode(expected_latents)
    report['final_latents'] = metrics(actual.latents, expected_latents.cpu())
    report['final_images'] = metrics(actual.images, expected_images)
    report['image_psnr_db'] = -10 * torch.log10((actual.images - expected_images).square().mean()).item()
    report['forward_bit_exact'] = True
    report['trajectory_note'] = 'The official pipeline batches CFG, so its GEMM rounding may differ from sequential CFG.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert report['final_latents']['finite'] and report['final_images']['finite']
    assert report['final_latents']['relative_l2'] < 0.05, 'Investigate trajectory drift before benchmarking'


if __name__ == '__main__':
    main()
