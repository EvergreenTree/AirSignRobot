"""Summarize local download, preparation and experiment progress without loading weights."""
import argparse
import json
from pathlib import Path


def status(workspace):
    result = {'downloads': {}, 'prepared_320': {}, 'experiments': {}, 'runs': {}}
    for root in sorted((workspace/'data/raw').glob('*')):
        info_path = root/'meta/info.json'
        if not info_path.exists():
            continue
        info = json.loads(info_path.read_text())
        result['downloads'][root.name] = {
            'videos': len(list(root.glob('videos/**/*.mp4'))), 'expected_videos': 3*info['total_episodes'],
            'release_verified': (root/'download_manifest.json').exists()}
    for task, expected in ((1, 50), (2, 238)):
        result['prepared_320'][str(task)] = {
            'episodes': len(list((workspace/f'data/prepared-task{task}-320').glob('*/episode.json'))),
            'expected_episodes': expected}
        path = workspace/f'task{task}-experiment.json'
        if path.exists():
            result['experiments'][str(task)] = json.loads(path.read_text())['stage']
    for path in sorted((workspace/'runs').glob('*/metrics.jsonl')):
        records = []
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # A producer may be appending the final line.
        if not records:
            continue
        record = {'step': records[-1]['step']}
        valid = [r for r in records if 'validation' in r]
        if valid:
            best = min(valid, key=lambda r: r['validation']['normalized_mae'])
            record.update({'best_validation_step': best['step'],
                           'best_validation_mae': round(best['validation']['normalized_mae'], 6)})
        result['runs'][path.parent.name] = record
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path('..'))
    args = p.parse_args()
    print(json.dumps(status(args.workspace.resolve()), indent=2))
