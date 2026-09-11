"""Export a compact inference checkpoint with reproducibility metadata."""
import argparse
import json
from pathlib import Path
import torch
from .download import sha256
from .train import atomic_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory for an immutable export')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if saved.get('format') != 'airsign_act_v1':
        raise ValueError('Unsupported checkpoint')
    removed = {'optimizer', 'rng_torch', 'rng_cuda', 'rng_python', 'rng_numpy'}
    inference = {key: value for key, value in saved.items() if key not in removed}
    inference['inference_only'] = True
    inference['training_checkpoint_sha256'] = sha256(args.checkpoint)
    args.output.mkdir(parents=True)
    target = args.output / 'policy.pt'
    atomic_checkpoint(target, inference)
    manifest = {'team': 'AirSign', 'task': saved['task'], 'training_step': saved['step'],
                'dataset_coverage': saved.get('dataset_coverage', 'unspecified; inspect source dataset manifest'),
                'checkpoint_sha256': sha256(target), 'training_checkpoint_sha256': inference['training_checkpoint_sha256'],
                'dataset_sha256': saved['dataset_sha256'], 'validation': saved['validation'],
                'model_config': saved['model_config'], 'schema': saved['schema'],
                'cameras': saved['cameras'], 'physical_validation': None}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(target), 'sha256': manifest['checkpoint_sha256']}))


if __name__ == '__main__':
    main()
