"""Request lifecycle tests. These CPU fixtures are not generation benchmarks."""
import unittest
import torch
from diffusers import DPMSolverMultistepScheduler
from nanodiffusion import Diffusion, SamplingParams


class ConstantField:
    device, dtype = torch.device('cpu'), torch.float32

    def __init__(self):
        self.closed = 0
        self.fail = False
        self.samplers = []

    def latent_shape(self, params):
        if params.width % 32 or params.height % 32:
            raise ValueError('dimensions must be divisible by 32')
        return (1, 1, params.height // 32, params.width // 32)

    def make_sampler(self, params):
        sampler = DPMSolverMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, flow_shift=3.0)
        sampler.set_timesteps(params.num_steps)
        self.samplers.append(sampler)
        return sampler

    def encode(self, prompt):
        return 2.0 if prompt else 1.0

    def predict_velocity(self, latents, timestep, conditioning):
        if self.fail:
            self.fail = False
            raise ValueError('model failure')
        return torch.full_like(latents, conditioning)

    def decode(self, latents):
        return latents.permute(0, 2, 3, 1).expand(-1, -1, -1, 3).sigmoid()

    def close(self):
        self.closed += 1


class EngineTests(unittest.TestCase):
    def params(self, **kwargs):
        return SamplingParams(height=64, width=64, num_steps=3, **kwargs)

    def test_fifo_and_sampler_history_isolation(self):
        model = ConstantField()
        with Diffusion(model) as engine:
            ids = [engine.add_request(p, self.params()) for p in ('first', 'second')]
            requests = list(engine.scheduler.waiting)
            updates = [engine.step() for _ in range(6)]
            self.assertEqual([u.request_id for u in updates], [ids[0]] * 3 + [ids[1]] * 3)
            self.assertEqual([u.completed_steps for u in updates], [1, 2, 3, 1, 2, 3])
            self.assertEqual([u.result is not None for u in updates], [False, False, True] * 2)
            self.assertIsNot(model.samplers[0], model.samplers[1])
            self.assertTrue(all(r.latents is None and r.sampler is None for r in requests))
            self.assertTrue(engine.scheduler.idle)

    def test_constant_flow_cfg_rng_and_repeatability(self):
        model = ConstantField()
        params = self.params(seed=123, guidance_scale=4.0)
        rng = torch.get_rng_state().clone()
        initial = torch.randn(model.latent_shape(params), generator=torch.Generator().manual_seed(123))
        with Diffusion(model) as engine:
            out = engine.generate('positive', params)[0]
            # Constant flow 1 + 4*(2-1) = 5, integrated over sigma_initial -> 0.
            sigma_initial = float(model.samplers[-1].sigmas[0])
            torch.testing.assert_close(out.latents, initial - sigma_initial * 5, atol=1e-5, rtol=1e-5)
            self.assertEqual(out.num_model_calls, 6)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            engine.generate('unrelated', self.params(seed=17))
            repeated = engine.generate('positive', params)[0]
            torch.testing.assert_close(out.latents, repeated.latents, rtol=0, atol=0)

    def test_cfg_disabled_and_callback(self):
        updates = []
        with Diffusion(ConstantField()) as engine:
            out = engine.generate('p', self.params(guidance_scale=1), updates.append)[0]
            self.assertEqual(out.num_model_calls, 3)
            self.assertEqual(len(updates), 3)
            self.assertIs(updates[-1].result, out)

    def test_failed_request_releases_state_and_preserves_queue(self):
        model = ConstantField()
        with Diffusion(model) as engine:
            engine.add_request('bad', self.params())
            second = engine.add_request('next', self.params())
            request = engine.scheduler.waiting[0]
            model.fail = True
            with self.assertRaisesRegex(ValueError, 'model failure'):
                engine.step()
            self.assertIsNone(request.latents)
            self.assertIsNone(engine.scheduler.active)
            for _ in range(3):
                update = engine.step()
            self.assertEqual(update.result.request_id, second)

    def test_callback_failure_and_close(self):
        model = ConstantField()
        engine = Diffusion(model)
        def fail(update):
            raise ValueError('callback')
        with self.assertRaisesRegex(ValueError, 'callback'):
            engine.generate(['first', 'second'], self.params(), fail)
        self.assertIsNone(engine.scheduler.active)
        self.assertEqual(len(engine.scheduler.waiting), 1)
        engine.close()
        engine.close()
        self.assertTrue(engine.scheduler.idle)
        self.assertEqual(model.closed, 1)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            engine.generate('late')

    def test_invalid_admission(self):
        with Diffusion(ConstantField()) as engine:
            with self.assertRaises(ValueError):
                engine.add_request('invalid', SamplingParams(width=7))
            with self.assertRaises(TypeError):
                engine.generate(['valid', 1])
            self.assertTrue(engine.scheduler.idle)
            engine.add_request('pending', self.params())
            with self.assertRaisesRegex(RuntimeError, 'pending'):
                engine.generate('other')
        for kwargs in ({'num_steps': 0}, {'seed': -1}, {'guidance_scale': float('nan')}, {'width': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SamplingParams(**kwargs)
