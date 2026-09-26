"""Small offline CLI. The default toy mode downloads no model or extra dependency."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from nanodiffusion import Diffusion, SamplingParams


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Readable flow-matching engine; toy mode is UNTRAINED")
    parser.add_argument("--model", choices=("toy", "wan"), default="toy")
    parser.add_argument("--checkpoint", help="Local Diffusers-format Wan2.1 T2V 1.3B directory")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly allow remote Wan component downloads")
    parser.add_argument("--prompt", default="A small boat on a calm lake")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shift", type=float)
    parser.add_argument("--device", help="Default: CPU for toy, CUDA for Wan")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--attention", choices=("sdpa", "sage"), default="sdpa", help="Toy adapter only; Wan uses native SDPA")
    parser.add_argument("--no-offload", action="store_true", help="Keep all Wan components on the device")
    parser.add_argument("--output", type=Path, default=Path("outputs/sample.pt"))
    parser.add_argument("--mp4", type=Path, help="Optional video export; requires imageio[ffmpeg]")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.output.suffix != ".pt":
        parser.error("--output must use .pt (stores frames, latents and sampling metadata)")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.mp4 is not None and args.mp4.suffix.lower() != ".mp4":
        parser.error("--mp4 must end in .mp4")
    if args.model == "wan" and not args.checkpoint:
        parser.error("--model wan requires --checkpoint; downloads are disabled unless --allow-download is set")
    if args.model == "wan" and args.attention != "sdpa":
        parser.error("the Wan adapter currently supports native SDPA only")
    if args.model == "toy" and (args.checkpoint or args.allow_download or args.no_offload):
        parser.error("--checkpoint, --allow-download and --no-offload apply only to Wan")

    is_toy = args.model == "toy"
    device = args.device or ("cpu" if is_toy else "cuda")
    dtype_name = args.dtype or ("float32" if is_toy else "bfloat16")
    dtype = getattr(torch, dtype_name)
    try:
        params = SamplingParams(
            height=args.height if args.height is not None else (32 if is_toy else 256),
            width=args.width if args.width is not None else (32 if is_toy else 256),
            num_frames=args.frames if args.frames is not None else (1 if is_toy else 17),
            num_steps=args.steps if args.steps is not None else (8 if is_toy else 30),
            guidance_scale=args.cfg, seed=args.seed, negative_prompt=args.negative_prompt,
            flow_shift=args.shift,
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    # Check optional video dependency before any expensive real-model loading.
    if args.mp4 is not None:
        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg
            imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as exc:
            parser.error(f"video export requires imageio[ffmpeg]: {exc}")

    if is_toy:
        from nanodiffusion.models.toy import ToyAdapter
        model = ToyAdapter(device=device, dtype=dtype, backend=args.attention)
        print("UNTRAINED toy model: output is a diagnostic pattern, not semantic text-to-video.")
    else:
        from nanodiffusion.models.wan import WanAdapter
        print("Wan2.1 components + first-order Flow Euler; not the original UniPC sampling recipe.")
        model = WanAdapter.from_pretrained(
            args.checkpoint, device=device, dtype=dtype, cpu_offload=not args.no_offload,
            local_files_only=not args.allow_download,
        )

    def progress(update):
        if not args.quiet:
            print(f"request={update.request_id} step={update.completed_steps}/{update.total_steps} t={update.timestep:.3f}")

    with Diffusion(model) as engine:
        result = engine.generate(args.prompt, params, on_step=progress)[0]
    metadata = {
        "model": args.model, "checkpoint": args.checkpoint, "untrained": is_toy,
        "prompt": args.prompt, "sampling": asdict(params),
        "sampler": "first-order FlowEuler", "effective_shift": params.flow_shift or model.default_shift,
        "attention": args.attention, "device": device, "dtype": dtype_name,
        "num_model_calls": result.num_model_calls,
        "frames_shape": list(result.frames.shape), "latents_shape": list(result.latents.shape),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"frames": result.frames, "latents": result.latents, "metadata": metadata}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if args.mp4 is not None:
        args.mp4.parent.mkdir(parents=True, exist_ok=True)
        video = result.frames[0].mul(255).round().clamp(0, 255).to(torch.uint8).numpy()
        imageio.mimwrite(str(args.mp4), video, fps=args.fps, macro_block_size=1)
    print(f"Saved {args.output} and {args.output.with_suffix('.json')}; frames={tuple(result.frames.shape)}")
    return 0
