"""Compare measured runs and their real generated images; no model inference."""
import argparse
import json
from pathlib import Path
import torch


def difference(a, b):
    a, b = a.float(), b.float()
    d = a - b
    return {'max_abs': d.abs().max().item(), 'rmse': d.square().mean().sqrt().item(),
            'relative_l2': (d.norm() / b.norm()).item(),
            'cosine': torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('benchmarks/rtx4090'))
    args = parser.parse_args()
    reports = {e: json.loads((args.results / f'{e}.json').read_text()) for e in ['nano', 'sglang']}
    for key in ['model_id', 'model_revision', 'sampling', 'threads', 'batch_size', 'versions', 'policy']:
        if reports['nano'][key] != reports['sglang'][key]:
            raise ValueError(f'Incomparable settings: {key}')
    for nano, sglang in zip(reports['nano']['runs'], reports['sglang']['runs'], strict=True):
        for key in ['index', 'warmup', 'prompt', 'seed']:
            if nano[key] != sglang[key]:
                raise ValueError(f'Incomparable request: {key}')
    results = {'throughput_ratio_nano_over_sglang': reports['nano']['summary']['images_per_second'] / reports['sglang']['summary']['images_per_second'],
               'cases': []}
    for case in range(3):
        tensors = {e: torch.load(Path('outputs') / args.results.name / e / f'{case}.pt', weights_only=True)
                   for e in ['nano', 'sglang']}
        a, b = tensors['nano'], tensors['sglang']
        result = {'case': case, 'latents': difference(a['latents'], b['latents']),
                  'images': difference(a['images'], b['images']),
                  'image_psnr_db': -10 * torch.log10((a['images'] - b['images']).square().mean()).item()}
        results['cases'].append(result)
    (args.results / 'comparison.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
