"""Collect completed software checks with exact policy and dataset provenance."""
import argparse
import json
from pathlib import Path
import re
import shutil

from ebim_phase2.download import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.workspace.resolve()
    log = base / 'final-pytest.log'
    summary = next(line for line in reversed(log.read_text().splitlines()) if ' passed' in line)
    if not re.search(r'^\d+ passed', summary) or 'failed' in summary or 'error' in summary:
        raise ValueError('Expected a completed passing test suite')
    probe_path = base/'runs/mujoco_actuator_probe_valid_initial.json'
    probe = json.loads(probe_path.read_text())
    probe.pop('records')
    probe['final_max_abs_tracking_error_rad'] = probe.pop('max_tracking_error_rad')
    probe['full_log_sha256'] = sha256(probe_path)
    checks = {'team': 'AirSign', 'physical_validation': None,
              'pytest': {'summary': summary, 'log_sha256': sha256(log)},
              'wheel': json.loads((base/'runs/wheel-check.json').read_text()),
              'vision_pretrained': json.loads((base/'runs/vision-pretrained-verified/summary.json').read_text()),
              'mujoco_actuator_probe': probe,
              'docker': {'tested_container': False, 'reason': 'Docker Hub base-image pull failed; no alternate registry path established'},
              'tasks': {}}
    for task in (1, 2):
        export = json.loads((base/f'exports/task{task}/manifest.json').read_text())
        results = {}
        for label in ('interface', 'http', 'vision-ablation'):
            result = json.loads((base/f'runs/task{task}-full-{label}.json').read_text())
            if result['checkpoint_sha256'] != export['checkpoint_sha256']:
                raise ValueError(f'Task {task} {label} checked a different policy')
            if label != 'vision-ablation' and result['dataset_sha256'] != export['dataset_sha256']:
                raise ValueError(f'Task {task} {label} used a different dataset')
            results[label] = result
        checks['tasks'][str(task)] = results
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'software-validation.json').write_text(json.dumps(checks, indent=2)+'\n')
    shutil.copyfile(log, args.output/'pytest.log')
    shutil.copyfile(base/'wheel-vision-pytest.log', args.output/'wheel-vision-pytest.log')
    print(json.dumps({'report': str(args.output/'software-validation.json'), 'pytest': summary}))


if __name__ == '__main__':
    main()
