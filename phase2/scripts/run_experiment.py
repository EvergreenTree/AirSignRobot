"""Wait for verified data, freeze a full dataset, train, evaluate once, and export."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from snapshot_prepared import snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--task', type=int, choices=[1, 2], required=True)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--backbone-checkpoint', type=Path, required=True)
    p.add_argument('--prepared', type=Path, help='Optional task-specific prepared image cache')
    args = p.parse_args()
    base = args.workspace.resolve()
    prepared = args.prepared.resolve() if args.prepared else base/'data/prepared'
    roots = [base/'data/raw'/r for r in (['task1_part1', 'task1_part2'] if args.task == 1 else ['task2_munich'])]
    dataset, output = base/f'data/task{args.task}-full', base/f'runs/task{args.task}-act'
    state_path = base/f'task{args.task}-experiment.json'

    def status(stage, **details):
        record = {'task': args.task, 'stage': stage, 'updated_unix_s': time.time(), **details}
        temporary = state_path.with_suffix('.partial')
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(state_path)
        print(json.dumps(record), flush=True)

    if not (dataset/'manifest.json').exists():
        status('waiting_for_complete_verified_releases')
        while not all((r/'download_manifest.json').exists() for r in roots):
            time.sleep(15)
        status('waiting_for_episode_preparation')
        expected = [f"{root.name}_{json.loads(line)['episode_index']:06d}" for root in roots
                    for line in (root/'meta/episodes.jsonl').read_text().splitlines()]
        while not all((prepared/key/'episode.json').exists() for key in expected):
            time.sleep(15)
        status('freezing_and_auditing_full_dataset')
        snapshot(roots, prepared, dataset, args.task, require_complete=True)
    status('waiting_for_gpu')
    (base/'runs').mkdir(exist_ok=True)
    with (base/'runs/gpu-training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        common = [sys.executable, '-m', 'ebim_phase2.train', '--task', str(args.task),
                  '--dataset', str(dataset), '--batch-size', str(args.batch_size), '--workers', str(args.workers)]
        command = common + ['--output', str(output), '--steps', str(args.steps),
                            '--backbone-checkpoint', str(args.backbone_checkpoint)]
        if (output/'last.pt').exists():
            command += ['--resume', str(output/'last.pt')]
        status('training', command=command)
        with (base/f'task{args.task}-train.log').open('a') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        evaluation = base/f'runs/task{args.task}-test'
        if not (evaluation/'test_metrics.json').exists():
            status('evaluating_reserved_test_episodes')
            with (base/f'task{args.task}-test.log').open('a') as log:
                subprocess.run(common + ['--output', str(evaluation), '--resume', str(output/'best.pt'),
                                         '--evaluate', 'test', '--eval-samples', '1000000'],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        exported = base/f'exports/task{args.task}'
        if not (exported/'manifest.json').exists():
            status('exporting')
            subprocess.run([sys.executable, '-m', 'ebim_phase2.export', '--checkpoint', str(output/'best.pt'),
                            '--output', str(exported)], check=True)
        status('offline_experiment_complete', export=str(exported), test_metrics=str(evaluation/'test_metrics.json'),
               physical_validation=None)


if __name__ == '__main__':
    main()
