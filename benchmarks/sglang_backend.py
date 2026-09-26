"""Controlled SGLang native pipeline for a same-components comparison.

No upstream files are patched. Use its public loaded_modules/config extension
points to share official Gemma2/DC-AE semantics with nano. The DiT, stage executor,
request validation, latent preparation and DPM scheduler wrapper remain native
SGLang. This is an in-process pipeline comparison, not an HTTP serving benchmark.
"""
import json
from functools import partial
import subprocess
import tempfile
from pathlib import Path

import torch
from diffusers import AutoencoderDC
from transformers import AutoTokenizer, Gemma2Model

from sglang.multimodal_gen.configs.pipeline_configs.sana import SanaPipelineConfig
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.distributed.parallel_state import maybe_init_distributed_environment_and_model_parallel
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines.sana import SanaPipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args


class Float32CFG(CFGPolicy):
    def combine(self, predictions, *args, **kwargs):
        return super().combine([p.float() for p in predictions], *args, **kwargs)


class MatchedSanaConfig(SanaPipelineConfig):
    def get_latent_dtype(self, prompt_dtype):
        return torch.float32


class SGLangBackend:
    def __init__(self, checkpoint):
        import sglang
        source = Path(sglang.__file__).resolve().parents[2]
        revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != '0e2aac500bbd7010503bff67bd221bfbd7f04389':
            raise ValueError(f'Use the pinned SGLang checkout, got {revision}')
        subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', 'HEAD', '--', 'python/sglang'], check=True, stdout=subprocess.DEVNULL)
        checkpoint = Path(checkpoint)
        config = MatchedSanaConfig()
        config.cfg_policy = Float32CFG()
        config.text_encoder_extra_args = [{'padding': 'max_length', 'return_attention_mask': True}]
        config.vae_config.update_model_arch(json.loads((checkpoint / 'vae/config.json').read_text()))
        config.vae_config.post_init()
        config.text_encoder_configs[0].update_model_arch(json.loads((checkpoint / 'text_encoder/config.json').read_text()))
        self.args = ServerArgs.from_kwargs(
            model_path=str(checkpoint), backend='sglang', pipeline_config=config,
            num_gpus=1, attention_backend='torch_sdpa', enable_cfg_parallel=False,
            dit_cpu_offload=False, text_encoder_cpu_offload=False, vae_cpu_offload=False,
            image_encoder_cpu_offload=False, dit_layerwise_offload=False,
            enable_torch_compile=False, enable_breakable_cuda_graph=False,
            warmup_mode='off', log_level='warning',
        )
        set_global_server_args(self.args)
        self.rendezvous = tempfile.TemporaryDirectory(prefix='nano-sana-dist-')
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=1, sp_size=1,
            distributed_init_method='file://' + str(Path(self.rendezvous.name) / 'rendezvous'),
        )
        self.pipeline = SanaPipeline(str(checkpoint), self.args, loaded_modules={
            'tokenizer': AutoTokenizer.from_pretrained(checkpoint / 'tokenizer', padding_side='right', local_files_only=True),
            'text_encoder': Gemma2Model.from_pretrained(checkpoint / 'text_encoder', dtype=torch.bfloat16,
                                                       attn_implementation='eager', local_files_only=True).cuda().eval().requires_grad_(False),
            'vae': AutoencoderDC.from_pretrained(checkpoint / 'vae', torch_dtype=torch.float32,
                                                 local_files_only=True).cuda().eval().requires_grad_(False),
        })
        self.denoise_events = None
        self.final_latents = None
        self.model_calls = 0
        def count_forward(module, inputs, output):
            self.model_calls += 1
        self.pipeline.get_module('transformer').register_forward_hook(count_forward)
        stage = next(s for s in self.pipeline.stages if type(s).__name__ == 'DenoisingStage')
        stage.progress_bar = partial(stage.progress_bar, disable=True)
        forward = stage.forward
        def measured_forward(batch, server_args):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = forward(batch, server_args)
            end.record()
            self.denoise_events = (start, end)
            self.final_latents = result.latents
            return result
        stage.forward = measured_forward

    @torch.inference_mode()
    def generate(self, prompt, seed):
        self.model_calls = 0
        sampling = SamplingParams.from_pretrained(
            self.args.model_path, prompt=prompt, negative_prompt='', height=512, width=512,
            num_inference_steps=20, guidance_scale=4.5, seed=seed, generator_device='cpu',
            save_output=False, return_frames=True, enable_cache_dit=False,
        )
        request = prepare_request(self.args, sampling)
        request.suppress_logs = True
        output = self.pipeline.forward(request, self.args)
        if output.error:
            raise RuntimeError(output.error)
        pixels = output.output
        if not isinstance(pixels, torch.Tensor):
            raise TypeError(f'Expected tensor pixels, got {type(pixels)}')
        if pixels.ndim == 5:
            pixels = pixels.squeeze(2)
        images = pixels.permute(0, 2, 3, 1).float().cpu()
        return images, self.final_latents.cpu()

    def close(self):
        self.pipeline = None
        self.final_latents = None
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        self.rendezvous.cleanup()
