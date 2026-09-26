"""CPU component-contract tests; no Diffusers, weights, network or CUDA needed."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from nanodiffusion import Diffusion
from nanodiffusion.config import SamplingParams
from nanodiffusion.models.wan import WanAdapter


class FakeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.moves = []

    def to(self, *args, **kwargs):
        self.moves.append((args, kwargs))
        return super().to(*args, **kwargs)


class FakeTokenizer:
    def __call__(self, prompts, **kwargs):
        self.prompts, self.kwargs = prompts, kwargs
        ids = torch.arange(kwargs["max_length"]).unsqueeze(0)
        mask = torch.zeros_like(ids)
        mask[:, :3] = 1
        return SimpleNamespace(input_ids=ids, attention_mask=mask)


class FakeTextEncoder(FakeModule):
    def forward(self, input_ids, attention_mask):
        self.last_mask = attention_mask.clone()
        # Deliberately nonzero padded outputs: adapter must zero these itself.
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 4) + 1
        return SimpleNamespace(last_hidden_state=hidden)


class FakeTransformer(FakeModule):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(in_channels=16, out_channels=16,
                                      patch_size=(1, 2, 2), text_dim=4,
                                      image_dim=None, added_kv_proj_dim=None,
                                      pos_embed_seq_len=None)
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return (torch.full_like(kwargs["hidden_states"], 0.25),)


class FakeVAE(FakeModule):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(z_dim=16, scale_factor_temporal=4,
                                      scale_factor_spatial=8,
                                      latents_mean=list(range(16)), latents_std=[2.0] * 16)

    def decode(self, latents, return_dict):
        self.decoded_latents = latents.clone()
        assert return_dict is False
        # Expose output layout and clamping with three different channels.
        rgb = latents[:, :3].clone()
        rgb[:, 0], rgb[:, 1], rgb[:, 2] = -2.0, 0.0, 2.0
        return (rgb,)


def components():
    return FakeTokenizer(), FakeTextEncoder(), FakeTransformer(), FakeVAE()


def make_adapter(**kwargs):
    return WanAdapter(*components(), device="cpu", dtype=torch.float32,
                      max_sequence_length=8, **kwargs)


class WanAdapterTests(unittest.TestCase):
    def test_latent_shape_uses_causal_temporal_compression_and_patch_grid(self):
        model = make_adapter()
        self.assertEqual(model.latent_shape(SamplingParams(height=32, width=48, num_frames=9)),
                         (1, 16, 3, 4, 6))
        self.assertEqual(model.latent_shape(SamplingParams(height=16, width=16, num_frames=1)),
                         (1, 16, 1, 2, 2))
        for params in (SamplingParams(height=24), SamplingParams(width=24),
                       SamplingParams(num_frames=8)):
            with self.subTest(params=params), self.assertRaises(ValueError):
                model.latent_shape(params)

    def test_prompt_cleaning_mask_and_zero_padding(self):
        model = make_adapter()
        model.text_encoder.moves.clear()
        embeddings = model.encode("  A\ncat\t&amp;amp;  dog  ")
        self.assertEqual(model.tokenizer.prompts, ["A cat & dog"])
        self.assertEqual(model.tokenizer.kwargs["padding"], "max_length")
        self.assertTrue(model.tokenizer.kwargs["truncation"])
        self.assertTrue(model.tokenizer.kwargs["return_attention_mask"])
        self.assertEqual(embeddings.shape, (1, 8, 4))
        torch.testing.assert_close(embeddings[0, :3, 0], torch.tensor([1., 2., 3.]))
        self.assertEqual(torch.count_nonzero(embeddings[:, 3:]).item(), 0)
        torch.testing.assert_close(model.text_encoder.last_mask,
                                   torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]]))
        self.assertEqual(len(model.text_encoder.moves), 2)  # load and offload
        self.assertEqual(model.text_encoder.moves[-1][1]["device"], "cpu")
        self.assertFalse(model.text_encoder.training)
        self.assertFalse(model.text_encoder.weight.requires_grad)

    def test_predictions_are_component_calls_without_cfg_or_sampling(self):
        model = make_adapter()
        embeddings = model.encode("cat")
        latent = torch.ones((1, 16, 1, 2, 2))
        timestep = torch.tensor([500.], dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "begin_denoising"):
            model.predict_velocity(latent, timestep, embeddings)
        model.begin_denoising()
        predicted = model.predict_velocity(latent, timestep, embeddings)
        self.assertEqual(len(model.transformer.calls), 1)
        call = model.transformer.calls[0]
        self.assertEqual(set(call), {"hidden_states", "timestep", "encoder_hidden_states", "return_dict"})
        self.assertFalse(call["return_dict"])
        self.assertEqual(call["timestep"].dtype, torch.float32)
        self.assertEqual(predicted.shape, latent.shape)
        torch.testing.assert_close(predicted, torch.full_like(latent, .25))
        torch.testing.assert_close(latent, torch.ones_like(latent))
        with self.assertRaisesRegex(RuntimeError, "cannot encode"):
            model.encode("dog")
        with self.assertRaises(ValueError):
            model.predict_velocity(latent, timestep[:, None], embeddings)
        model.end_denoising()
        self.assertEqual(model.transformer.moves[-1][1]["device"], "cpu")

    def test_embedding_and_prediction_dtype_preserves_mixed_precision_weights(self):
        tokenizer, encoder, transformer, vae = components()
        # Mimic FP32 exceptions in a preloaded low-precision model.
        transformer.main = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        model = WanAdapter(tokenizer, encoder, transformer, vae, device="cpu",
                           dtype=torch.bfloat16, max_sequence_length=8)
        self.assertEqual(transformer.weight.dtype, torch.float32)
        self.assertEqual(transformer.main.dtype, torch.bfloat16)
        with Diffusion(model) as engine:
            result = engine.generate("cat", SamplingParams(
                height=16, width=16, num_steps=1, guidance_scale=1))[0]
        self.assertEqual(result.latents.dtype, torch.float32)
        self.assertEqual(result.frames.dtype, torch.float32)
        call = transformer.calls[0]
        self.assertEqual(call["hidden_states"].dtype, torch.bfloat16)
        self.assertEqual(call["encoder_hidden_states"].dtype, torch.bfloat16)
        self.assertEqual(call["timestep"].dtype, torch.float32)

    def test_decode_denormalizes_in_fp32_and_returns_cpu_channels_last(self):
        model = make_adapter()
        model.vae.moves.clear()
        latent = torch.ones((1, 16, 2, 2, 2), dtype=torch.bfloat16)
        frames = model.decode(latent)
        expected = torch.arange(16).float().reshape(1, 16, 1, 1, 1) + 2
        torch.testing.assert_close(model.vae.decoded_latents, expected.expand(1, 16, 2, 2, 2))
        self.assertEqual(model.vae.decoded_latents.dtype, torch.float32)
        self.assertEqual(frames.shape, (1, 2, 2, 2, 3))
        self.assertEqual(frames.dtype, torch.float32)
        self.assertEqual(frames.device.type, "cpu")
        torch.testing.assert_close(frames[0, 0, 0, 0], torch.tensor([0., .5, 1.]))
        self.assertTrue(frames.is_contiguous())
        self.assertEqual(len(model.vae.moves), 2)

    def test_component_offload_happens_even_after_component_failure(self):
        model = make_adapter()
        with patch.object(model.text_encoder, "forward", side_effect=RuntimeError("bad text")):
            with self.assertRaisesRegex(RuntimeError, "bad text"):
                model.encode("cat")
        self.assertEqual(model.text_encoder.moves[-1][1]["device"], "cpu")
        with patch.object(model.vae, "decode", side_effect=RuntimeError("bad vae")):
            with self.assertRaisesRegex(RuntimeError, "bad vae"):
                model.decode(torch.ones(1, 16, 1, 2, 2))
        self.assertEqual(model.vae.moves[-1][1]["device"], "cpu")

    def test_no_offload_mode_and_close_lifecycle(self):
        model = make_adapter(cpu_offload=False)
        text, transformer, vae = model.text_encoder, model.transformer, model.vae
        text.moves.clear()
        model.encode("cat")
        self.assertEqual(len(text.moves), 1)
        model.begin_denoising()
        calls = len(transformer.moves)
        model.end_denoising()
        self.assertEqual(len(transformer.moves), calls)
        model.close()
        model.close()  # idempotent
        model.end_denoising()
        self.assertIsNone(model.transformer)
        for module in (text, transformer, vae):
            self.assertEqual(module.moves[-1][1]["device"], "cpu")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            model.encode("cat")

    def test_rejects_unsupported_components(self):
        for component, field, value in ((2, "in_channels", 36), (2, "image_dim", 1280),
                                        (2, "patch_size", (2, 2, 2)), (3, "z_dim", 48),
                                        (3, "latents_std", [0.] * 16)):
            parts = components()
            setattr(parts[component].config, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                WanAdapter(*parts, device="cpu", dtype=torch.float32)

    def fake_diffusers(self, config=None):
        tokenizer, text_encoder, transformer, vae = components()
        pipeline = SimpleNamespace(tokenizer=tokenizer, text_encoder=text_encoder,
                                   transformer=transformer, vae=vae, transformer_2=None)
        config = {"_class_name": "WanPipeline"} if config is None else config
        pipeline_cls = SimpleNamespace(load_config=Mock(return_value=config),
                                       from_pretrained=Mock(return_value=pipeline))
        vae_cls = SimpleNamespace(from_pretrained=Mock(return_value=vae))
        return SimpleNamespace(WanPipeline=pipeline_cls, AutoencoderKLWan=vae_cls)

    def test_loader_is_local_by_default_and_drops_pipeline(self):
        fake = self.fake_diffusers()
        with patch.dict(sys.modules, {"diffusers": fake}):
            model = WanAdapter.from_pretrained("/local/model", device="cpu", dtype=torch.float32)
        fake.WanPipeline.load_config.assert_called_once_with("/local/model", local_files_only=True)
        self.assertTrue(fake.WanPipeline.from_pretrained.call_args.kwargs["local_files_only"])
        self.assertTrue(fake.AutoencoderKLWan.from_pretrained.call_args.kwargs["local_files_only"])
        self.assertEqual(fake.AutoencoderKLWan.from_pretrained.call_args.kwargs["torch_dtype"], torch.float32)
        self.assertFalse(hasattr(model, "pipeline"))
        self.assertFalse(hasattr(model, "scheduler"))
        self.assertEqual(model.max_sequence_length, 512)

    def test_download_requires_explicit_false_and_import_is_lazy(self):
        fake = self.fake_diffusers()
        with patch.dict(sys.modules, {"diffusers": fake}):
            WanAdapter.from_pretrained("remote/repo", device="cpu", local_files_only=False)
        for loader in (fake.WanPipeline.load_config, fake.WanPipeline.from_pretrained,
                       fake.AutoencoderKLWan.from_pretrained):
            self.assertIs(loader.call_args.kwargs["local_files_only"], False)
        with patch.dict(sys.modules, {"diffusers": None}):
            with self.assertRaisesRegex(ImportError, "optional 'wan'"):
                WanAdapter.from_pretrained("/local/model")

    def test_loader_rejects_other_pipeline_variants_before_loading_weights(self):
        for override in ({"_class_name": "WanImageToVideoPipeline"}, {"expand_timesteps": True},
                         {"boundary_ratio": .875}, {"transformer_2": ["diffusers", "WanTransformer3DModel"]}):
            fake = self.fake_diffusers({"_class_name": "WanPipeline", **override})
            with self.subTest(override=override), patch.dict(sys.modules, {"diffusers": fake}):
                with self.assertRaisesRegex(ValueError, "single-transformer"):
                    WanAdapter.from_pretrained("/local/model")
            fake.AutoencoderKLWan.from_pretrained.assert_not_called()
            fake.WanPipeline.from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
