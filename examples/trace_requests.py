"""Run from the project directory: python -m examples.trace_requests."""
from nanodiffusion import Diffusion, SamplingParams
from nanodiffusion.models.toy import ToyAdapter


def main():
    with Diffusion(ToyAdapter()) as engine:
        # Different step counts are legal because this version runs one request at a time.
        engine.add_request("boat", SamplingParams(num_steps=2, seed=7))
        engine.add_request("mountain", SamplingParams(num_steps=3, seed=11))
        while not engine.scheduler.idle:
            update = engine.step()
            print(f"request {update.request_id}: {update.completed_steps}/{update.total_steps}")
            if update.result is not None:
                print("  finished:", tuple(update.result.frames.shape), "(untrained diagnostic output)")


if __name__ == "__main__":
    main()
