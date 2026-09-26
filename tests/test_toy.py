"""CPU behavior tests for the educational model and local attention boundary."""

import math
import unittest

import torch

from nanodiffusion.attention import attention
from nanodiffusion.config import SamplingParams
from nanodiffusion.models.toy import ToyAdapter


def noise(shape, seed=7):
    return torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(seed))


class AttentionTests(unittest.TestCase):
    def test_sdpa_matches_explicit_self_and_cross_attention(self):
        for q_length, kv_length in ((3, 5), (5, 5)):
            with self.subTest(lengths=(q_length, kv_length)):
                q = noise((2, 3, q_length, 8), seed=1)
                k = noise((2, 3, kv_length, 8), seed=2)
                v = noise((2, 3, kv_length, 8), seed=3)
                scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
                expected = scores.softmax(dim=-1) @ v
                actual = attention(q, k, v)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_invalid_shapes_and_dtypes_raise_clear_errors(self):
        q = torch.zeros(1, 1, 3, 8)
        cases = [
            ((q.squeeze(0), q, q), {}, "nonempty"),
            ((q[:, :, :0], q, q), {}, "nonempty"),
            ((q, q.expand(1, 2, 3, 8), q.expand(1, 2, 3, 8)), {}, "share"),
            ((q, q, q[:, :, :2]), {}, "shapes must match"),
            ((q, q[:, :, :, :4], q[:, :, :, :4]), {}, "head_dim"),
            ((q, q.double(), q.double()), {}, "identical devices and dtypes"),
            ((q.long(), q.long(), q.long()), {}, "supports float"),
        ]
        for args, kwargs, message in cases:
            with self.subTest(message=message, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    attention(*args, **kwargs)

    def test_backend_contracts_fail_before_optional_import(self):
        q = torch.zeros(1, 1, 3, 64)
        with self.assertRaisesRegex(ValueError, "backend"):
            attention(q, q, q, backend="unknown")
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            attention(q, q, q, backend="sage")


class ToyAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.addClassCleanup(torch.set_num_threads, cls.previous_threads)
        cls.model = ToyAdapter(seed=19)

    def test_model_seed_is_reproducible_without_changing_global_rng(self):
        before = torch.random.get_rng_state().clone()
        first, same, different = ToyAdapter(seed=5), ToyAdapter(seed=5), ToyAdapter(seed=6)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for key, value in first.state_dict().items():
            with self.subTest(parameter=key):
                self.assertTrue(torch.equal(value, same.state_dict()[key]))
        self.assertTrue(any(not torch.equal(value, different.state_dict()[key])
                            for key, value in first.state_dict().items()))

    def test_video_shape_and_decoded_pixel_contract(self):
        for height, width, frames in ((4, 8, 1), (12, 16, 3)):
            with self.subTest(height=height, width=width, frames=frames):
                params = SamplingParams(height=height, width=width, num_frames=frames)
                shape = self.model.latent_shape(params)
                self.assertEqual(shape, (1, 4, frames, height // 4, width // 4))
                latents = noise((2, *shape[1:]))
                velocity = self.model.predict_velocity(
                    latents, torch.tensor([600., 600.]), self.model.encode("你好 nano"))
                self.assertEqual(velocity.shape, latents.shape)
                self.assertTrue(torch.isfinite(velocity).all())
                pixels = self.model.decode(latents - 0.1 * velocity)
                self.assertEqual(pixels.shape, (2, frames, height, width, 3))
                self.assertEqual(pixels.device.type, "cpu")
                self.assertEqual(pixels.dtype, torch.float32)
                self.assertTrue(torch.isfinite(pixels).all())
                self.assertGreaterEqual(pixels.min().item(), 0)
                self.assertLessEqual(pixels.max().item(), 1)

    def test_text_and_time_condition_outputs_without_using_global_rng(self):
        latents = noise((1, 4, 2, 2, 3))
        before = torch.random.get_rng_state().clone()
        text = self.model.encode("中文 prompt")
        torch.testing.assert_close(text, self.model.encode("中文 prompt"), atol=0, rtol=0)
        prediction = self.model.predict_velocity(latents, torch.tensor([700.]), text)
        repeat = self.model.predict_velocity(latents, torch.tensor([700.]), text)
        other_time = self.model.predict_velocity(latents, torch.tensor([400.]), text)
        other_text = self.model.predict_velocity(latents, torch.tensor([700.]), self.model.encode(""))
        torch.testing.assert_close(prediction, repeat, atol=0, rtol=0)
        self.assertFalse(torch.allclose(prediction, other_time))
        self.assertFalse(torch.allclose(prediction, other_text))
        self.model.decode(latents)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_invalid_resolution_conditioning_and_latents(self):
        for height, width in ((7, 8), (8, 7)):
            with self.subTest(height=height, width=width):
                with self.assertRaisesRegex(ValueError, "multiples of 4"):
                    self.model.latent_shape(SamplingParams(height=height, width=width))
        valid = torch.zeros(1, 4, 1, 2, 2)
        text = self.model.encode("")
        cases = [
            (valid.squeeze(0), torch.tensor([1.]), text, "latents"),
            (valid[:, :3], torch.tensor([1.]), text, "latents"),
            (valid[:, :, :0], torch.tensor([1.]), text, "positive"),
            (valid, torch.tensor([1., 2.]), text, "one timestep"),
            (valid, torch.tensor([1.]), text[:, :0], "conditioning"),
            (valid, torch.tensor([1.]), text[:, :, :32], "conditioning"),
        ]
        for latents, timestep, condition, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.model.predict_velocity(latents, timestep, condition)
        with self.assertRaisesRegex(ValueError, "latents"):
            self.model.decode(valid[:, :3])
        with self.assertRaisesRegex(TypeError, "string"):
            self.model.encode(None)

    def test_invalid_constructor_options(self):
        for kwargs, message in (({"backend": "invalid"}, "backend"),
                                ({"backend": "sage"}, "CUDA FP16/BF16"),
                                ({"dtype": torch.int32}, "requires float"),
                                ({"seed": -1}, "seed")):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    ToyAdapter(**kwargs)


if __name__ == "__main__":
    unittest.main()
