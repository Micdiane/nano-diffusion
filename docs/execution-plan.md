# Real-model execution target

The project is a small, readable inference engine for a real pretrained model also
supported by SGLang Diffusion. Random DiTs are not a product feature or performance
baseline.

## First target

- GPU: one NVIDIA RTX 4090 (24 GB).
- Checkpoint: `Efficient-Large-Model/Sana_600M_512px_diffusers`.
- Checkpoint revision: `83d7a190bfd1fd070570a793d2dab5c7a3231b9d`.
- Reference SGLang source: `0e2aac500bbd7010503bff67bd221bfbd7f04389`.
- Workload: 512 × 512 text-to-image, 20 steps, CFG 4.5, explicit shared negative
  prompt, fixed seeds. Start with one image per request.
- Sampler: checkpoint's second-order DPM-Solver++ with flow prediction and
  flow-shift 3.0; not the old generic Euler demonstration.

## Acceptance gates

1. Load the real checkpoint, including Gemma2, SANA-600M DiT and DC-AE. Keep the
   request → scheduler → runner → model → sampler path readable. Do not delegate
   the entire engine to a Diffusers pipeline call.
2. Validate native model math against a pretrained reference and check final
   latents/images. Investigate text preprocessing, masks, Gemma2 logit softcapping,
   CFG execution, timestep construction and precision before interpreting speed.
3. Generate and inspect actual images on the 4090.
4. Run native SGLang and nano sequentially with identical checkpoint/configuration.
   Record warmups, individual synchronized timings, images/second, latency and
   memory. Distinguish full pipeline time from denoising time and state whether
   output transfer/postprocessing is included. Exclude download/load/JIT warmup
   from steady-state timing and exclude PNG writing from both engines.
5. Preserve commands, package versions, configuration, raw results, examples and
   any compatibility patches. Report numerical differences and failures honestly.
6. Remove the toy product path, update CLI/tests/README and publish to the existing
   public repository. CPU unit tests supplement real GPU validation.

## Study direction after the baseline

Profile the actual 512px workload, then change one mechanism at a time: batching,
fused projections, CUDA Graphs, or Triton normalization/modulation. Keep an eager
reference and compare quality as well as time. The first deliverable establishes
a reproducible baseline; it does not assume nano is faster than SGLang.

## Completed baseline (2026-09-26)

The real SANA path, 4090 generation, Diffusers validation and controlled native
SGLang comparison have been executed. The final 3-warmup/12-request runs measured
0.699 images/s for nano and 0.612 for SGLang (1.143x), with image PSNR
33.91–43.23 dB across three prompts. See `benchmarks/README.md` for the explicit
shared-component SGLang configuration and the limits of this eager batch-one result.

The next study milestone is profiling this fixed real workload, followed by a
single controlled CUDA Graph experiment. Projection fusion and Triton AdaLN follow
only when measurements justify them. These later optimizations are not implemented
or represented as measured achievements in the current baseline.
