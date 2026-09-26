"""CPU checks of numerical semantics and the request lifecycle; no model weights."""
import unittest

import torch

from nanodiffusion import Diffusion, SamplingParams
from nanodiffusion.sampler import FlowEuler, classifier_free_guidance


class ConstantModel:
    """Known vector field makes the entire sampler trajectory analytically checkable."""

    device = torch.device("cpu")
    dtype = torch.float32
    default_shift = 3.0
    num_train_timesteps = 1000

    def __init__(self):
        self.calls = []
        self.encodings = []
        self.closed = 0
        self.ended = 0
        self.fail_next = False
        self.fail_cleanup = False

    def latent_shape(self, params):
        if params.width % 4 or params.height % 4:
            raise ValueError("dimensions must be multiples of four")
        return (1, 1, params.num_frames, params.height // 4, params.width // 4)

    def encode(self, prompt):
        self.encodings.append(prompt)
        return torch.tensor([2.0 if prompt else 1.0])

    def begin_denoising(self):
        pass

    def predict_velocity(self, latents, timestep, conditioning):
        if self.fail_next:
            self.fail_next = False
            raise ValueError("intentional model failure")
        self.calls.append(float(timestep.item()))
        return torch.ones_like(latents) * conditioning.item()

    def end_denoising(self):
        self.ended += 1
        if self.fail_cleanup:
            self.fail_cleanup = False
            raise RuntimeError("intentional cleanup failure")

    def decode(self, latents):
        return latents.permute(0, 2, 3, 4, 1).expand(-1, -1, -1, -1, 3).sigmoid()

    def close(self):
        self.closed += 1


class SamplerTests(unittest.TestCase):
    def test_constant_velocity_exact_integral_with_shift(self):
        # dx/dsigma=2 over sigma:1->0 must give x_final=x_initial-2.
        for steps in (1, 2, 9):
            for shift in (1.0, 3.0):
                with self.subTest(steps=steps, shift=shift):
                    sampler = FlowEuler(steps, shift)
                    x = torch.tensor([3.0])
                    self.assertEqual(sampler.sigmas[0], 1)
                    self.assertEqual(sampler.sigmas[-1], 0)
                    self.assertTrue(torch.all(sampler.sigmas[:-1] > sampler.sigmas[1:]))
                    for index in range(steps):
                        x = sampler.step(torch.tensor([2.0]), index, x)
                    torch.testing.assert_close(x, torch.tensor([1.0]))

    def test_shifted_interior_and_cfg_fp32(self):
        sampler = FlowEuler(3, shift=3.0)
        raw = (1.0 + 0.001) / 2
        self.assertAlmostEqual(float(sampler.sigmas[1]), 3 * raw / (1 + 2 * raw))
        positive, negative = torch.tensor([3.0]).half(), torch.tensor([1.0]).half()
        out = classifier_free_guidance(positive, negative, 4.0)
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out, torch.tensor([9.0]))
        torch.testing.assert_close(classifier_free_guidance(positive, None, 1.0), positive.float())
        with self.assertRaises(ValueError):
            classifier_free_guidance(positive, None, 4.0)

    def test_extreme_positive_shift_keeps_endpoints_finite(self):
        for shift in (1e-20, 1e20):
            with self.subTest(shift=shift):
                sigmas = FlowEuler(8, shift).sigmas
                self.assertTrue(torch.all(torch.isfinite(sigmas)))
                self.assertEqual(sigmas[0], 1)
                self.assertEqual(sigmas[-1], 0)
                self.assertTrue(torch.all(sigmas[:-1] >= sigmas[1:]))

    def test_invalid_params(self):
        for kwargs in ({"num_steps": 0}, {"seed": -1}, {"guidance_scale": 0.5},
                       {"guidance_scale": float("nan")}, {"flow_shift": 0},
                       {"num_frames": True}, {"width": 2.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SamplingParams(**kwargs)


class EngineTests(unittest.TestCase):
    def params(self, **kwargs):
        return SamplingParams(height=8, width=8, num_steps=3, **kwargs)

    def test_fifo_one_denoising_step_at_a_time(self):
        model = ConstantModel()
        with Diffusion(model) as engine:
            ids = [engine.add_request(p, self.params()) for p in ("first", "second")]
            first, second = list(engine.scheduler.waiting)
            updates = [engine.step() for _ in range(6)]
            self.assertEqual([u.request_id for u in updates], [ids[0]] * 3 + [ids[1]] * 3)
            self.assertEqual([u.completed_steps for u in updates], [1, 2, 3, 1, 2, 3])
            self.assertEqual([u.result is not None for u in updates], [False, False, True] * 2)
            self.assertIsNone(first.latents)
            self.assertIsNone(second.latents)
            self.assertTrue(engine.scheduler.idle)
            self.assertEqual(updates[-1].result.num_model_calls, 6)
            with self.assertRaisesRegex(RuntimeError, "no queued"):
                engine.step()

    def test_cfg_and_private_rng_full_trajectory(self):
        model = ConstantModel()
        params = self.params(seed=123, guidance_scale=4.0)
        global_rng = torch.get_rng_state().clone()
        initial = torch.randn(model.latent_shape(params),
                              generator=torch.Generator().manual_seed(123))
        with Diffusion(model) as engine:
            output = engine.generate("positive", params)[0]
            # v_uncond=1, v_cond=2 -> CFG field=5, exact integral=-5.
            torch.testing.assert_close(output.latents, initial - 5)
            self.assertEqual(model.encodings, ["positive", ""])
            self.assertEqual(len(model.calls), 6)
            self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
            engine.generate("unrelated", self.params(seed=17))
            repeated = engine.generate("positive", params)[0]
            torch.testing.assert_close(output.latents, repeated.latents, rtol=0, atol=0)

    def test_cfg_disabled_and_callback(self):
        model, updates = ConstantModel(), []
        with Diffusion(model) as engine:
            result = engine.generate("prompt", self.params(guidance_scale=1), updates.append)[0]
            self.assertEqual(model.encodings, ["prompt"])
            self.assertEqual(len(model.calls), 3)
            self.assertEqual(result.num_model_calls, 3)
            self.assertEqual(len(updates), 3)
            self.assertIs(updates[-1].result, result)

    def test_failed_request_releases_state_preserves_queue_and_error(self):
        model = ConstantModel()
        with Diffusion(model) as engine:
            engine.add_request("bad", self.params())
            second_id = engine.add_request("next", self.params())
            first = engine.scheduler.waiting[0]
            model.fail_next = model.fail_cleanup = True
            with self.assertRaisesRegex(ValueError, "intentional model"):
                engine.step()
            self.assertIsNone(first.latents)
            self.assertIsNone(first.positive)
            self.assertIsNone(engine.scheduler.active)
            for _ in range(3):
                update = engine.step()
            self.assertEqual(update.result.request_id, second_id)
            self.assertTrue(engine.scheduler.idle)

    def test_admission_close_and_mixed_api_guard(self):
        model = ConstantModel()
        engine = Diffusion(model)
        with self.assertRaises(ValueError):
            engine.add_request("invalid", SamplingParams(width=7))
        self.assertTrue(engine.scheduler.idle)
        with self.assertRaises(TypeError):
            engine.generate(["valid", 1])
        self.assertTrue(engine.scheduler.idle)
        self.assertEqual(engine.generate([]), [])
        engine.add_request("pending", self.params())
        with self.assertRaisesRegex(RuntimeError, "pending"):
            engine.generate("other", self.params())
        engine.step()
        engine.close()
        engine.close()
        self.assertTrue(engine.scheduler.idle)
        self.assertEqual(model.closed, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            engine.add_request("late")

    def test_callback_failure_aborts_active_but_preserves_waiting(self):
        model = ConstantModel()
        with Diffusion(model) as engine:
            def fail_callback(update):
                raise ValueError("callback failed")

            with self.assertRaisesRegex(ValueError, "callback failed"):
                engine.generate(["first", "next"], self.params(), fail_callback)
            self.assertIsNone(engine.scheduler.active)
            self.assertEqual(model.ended, 1)
            self.assertEqual(len(engine.scheduler.waiting), 1)
            for _ in range(3):
                update = engine.step()
            self.assertEqual(update.result.prompt, "next")

    def test_context_preserves_original_error_and_close_can_be_retried(self):
        class CloseFailModel(ConstantModel):
            def close(self):
                self.closed += 1
                if self.closed == 1:
                    raise RuntimeError("cleanup failed")

        model = CloseFailModel()
        engine = Diffusion(model)
        with self.assertRaisesRegex(ValueError, "original error"):
            with engine:
                raise ValueError("original error")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            engine.generate("late")
        engine.close()
        self.assertEqual(model.closed, 2)
        engine.close()
        self.assertEqual(model.closed, 2)


if __name__ == "__main__":
    unittest.main()
