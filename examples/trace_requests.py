"""python -m examples.trace_requests --checkpoint /path/to/Sana_600M_512px_diffusers"""
import argparse
from nanodiffusion import Diffusion, SamplingParams
from nanodiffusion.models.sana import SanaModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    with Diffusion(SanaModel(args.checkpoint)) as engine:
        engine.add_request('A small boat on a calm lake', SamplingParams(seed=7))
        engine.add_request('A snowy mountain at sunrise', SamplingParams(seed=11))
        while not engine.scheduler.idle:
            update = engine.step()
            print(f'request {update.request_id}: {update.completed_steps}/{update.total_steps}')
            if update.result is not None:
                print('  finished:', tuple(update.result.images.shape))
