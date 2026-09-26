"""Single-GPU, batch-one steady-state benchmark. Run the engines sequentially."""
import argparse
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import math
import sys
import subprocess
import time

import torch
from PIL import Image

from nanodiffusion import Diffusion, SamplingParams
from nanodiffusion.models.sana import SanaModel, MODEL_ID, REVISION

PROMPTS = [
    'A small red panda drinking tea in a bamboo forest, watercolor painting',
    'A small wooden boat on a calm alpine lake at sunrise, realistic landscape photography',
    'A ceramic teapot and two cups on a wooden table beside a window, soft morning light',
]


class NanoBackend:
    def __init__(self, checkpoint):
        self.engine = Diffusion(SanaModel(checkpoint))
        self.denoise_events = None
        self.model_calls = 0
        def count_forward(module, inputs, output):
            self.model_calls += 1
        self.engine.runner.model.transformer.register_forward_hook(count_forward)
        forward = self.engine.runner.denoise_step
        def measured_forward(request):
            if request.step_index == 0:
                self.denoise_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                self.denoise_events[0].record()
            result = forward(request)
            if request.step_index == request.params.num_steps:
                self.denoise_events[1].record()
            return result
        self.engine.runner.denoise_step = measured_forward

    def generate(self, prompt, seed):
        self.model_calls = 0
        output = self.engine.generate(prompt, SamplingParams(seed=seed))[0]
        return output.images, output.latents

    def close(self):
        self.engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['nano', 'sglang'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--runs', type=int, default=12)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/rtx4090'))
    args = parser.parse_args()
    if args.warmups < 1 or args.runs < 3:
        parser.error('use at least one warmup and three measured requests')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from checkpoint import verify_checkpoint
    verify_checkpoint(args.checkpoint)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    if args.engine == 'nano':
        backend = NanoBackend(args.checkpoint)
    else:
        from sglang_backend import SGLangBackend
        backend = SGLangBackend(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    samples = Path('outputs') / args.output.name / args.engine
    samples.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(-args.warmups, args.runs):
        case = index % len(PROMPTS)
        seed = 42 + case
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        images, latents = backend.generate(PROMPTS[case], seed)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        denoise_ms = backend.denoise_events[0].elapsed_time(backend.denoise_events[1])
        finite = bool(images.isfinite().all() and latents.isfinite().all())
        if not finite:
            raise RuntimeError('Non-finite real-model output')
        if backend.model_calls != 40:
            raise RuntimeError(f'Expected 20 steps x 2 CFG calls, got {backend.model_calls}')
        record = {'index': index, 'warmup': index < 0, 'prompt': PROMPTS[case], 'seed': seed,
                  'model_calls': backend.model_calls,
                  'seconds': elapsed, 'denoise_gpu_ms': denoise_ms,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'finite': finite}
        records.append(record)
        print(json.dumps(record), flush=True)
        # Save all unique outputs outside the timed region, never model weights.
        if 0 <= index < len(PROMPTS):
            Image.fromarray(images[0].mul(255).round().byte().numpy()).save(args.output / f'{args.engine}-{case}.png')
            torch.save({'images': images, 'latents': latents}, samples / f'{case}.pt')
    timed = [r for r in records if not r['warmup']]
    durations = [r['seconds'] for r in timed]
    versions = {name: importlib.metadata.version(name) for name in
                ['torch', 'triton', 'diffusers', 'transformers', 'safetensors', 'numpy']}
    report = {
        'engine': args.engine, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'gpu': torch.cuda.get_device_name(), 'gpu_capability': torch.cuda.get_device_capability(),
        'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
        'python': platform.python_version(), 'versions': versions,
        'model_id': MODEL_ID, 'model_revision': REVISION,
        'sglang_revision': '0e2aac500bbd7010503bff67bd221bfbd7f04389' if args.engine == 'sglang' else None,
        'sampling': asdict(SamplingParams()), 'threads': 8, 'batch_size': 1,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          sorted([*Path('nanodiffusion').rglob('*.py'), Path(__file__),
                                  Path(__file__).with_name('sglang_backend.py')])},
        'policy': {
            'dit': 'bf16 eager native model', 'text_encoder': 'shared HF Gemma2 bf16 eager (softcapping enabled)',
            'vae': 'shared HF DC-AE fp32', 'latents_noise_cfg': 'fp32', 'noise_generator': 'cpu',
            'prompt_enrichment': False, 'text_length': 300, 'padding': 'max_length/right',
            'cfg_execution': 'sequential positive/negative', 'tf32': False, 'cudnn_sdpa': False,
            'cpu_offload': False, 'cache': False, 'torch_compile': False, 'cuda_graphs': False,
            'progress_bars': False,
            'timing': 'wall clock, synchronized before/after; full in-process pipeline through CPU float images and latents; excludes load, warmup and file writes',
            'denoise_timing': 'CUDA events around the denoising loop; includes GPU idle gaps between launches',
            'memory': 'PyTorch peak allocated/reserved in this process, including resident model weights',
            'sglang_scope': 'native SanaPipeline via loaded_modules; shared HF text encoder/VAE and FP32 CFG/latent config; no upstream source patches',
        },
        'summary': {
            'requests': len(timed), 'total_seconds': sum(durations), 'images_per_second': len(timed) / sum(durations),
            'mean_seconds': statistics.mean(durations), 'median_seconds': statistics.median(durations),
            'stdev_seconds': statistics.stdev(durations), 'p95_seconds': sorted(durations)[max(0, math.ceil(.95 * len(durations)) - 1)],
            'mean_denoise_gpu_ms': statistics.mean(r['denoise_gpu_ms'] for r in timed),
            'peak_allocated_gib': max(r['peak_allocated_bytes'] for r in timed) / 2**30,
            'peak_reserved_gib': max(r['peak_reserved_bytes'] for r in timed) / 2**30,
        }, 'runs': records,
    }
    (args.output / f'{args.engine}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2), flush=True)
    backend.close()


if __name__ == '__main__':
    main()
