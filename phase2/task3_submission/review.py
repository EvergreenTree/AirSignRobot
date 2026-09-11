#!/usr/bin/env python3
"""Offline Task 3 evidence review, software checks, or JSONL policy interface.

This entrypoint has no robot transport and never imports the on-site scripts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def verify():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for item in manifest['files']:
        path = (ROOT / item['path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Missing or invalid evidence path: {item['path']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item['sha256']:
            raise ValueError(f"Checksum mismatch: {item['path']}")
    summary = json.loads((HERE / 'evidence/summary.json').read_text())
    initial = json.loads((HERE / 'evidence/arm/before.json').read_text())
    final = json.loads((HERE / 'evidence/arm/final.json').read_text())
    delta = [final['O_T_EE'][12+i] - initial['O_T_EE'][12+i] for i in range(3)]
    net = math.sqrt(sum(x*x for x in delta))
    yaw = 0.0
    for step in range(2, 22):
        rows = [json.loads(line) for line in (HERE / f'evidence/base/turn{step:02d}.jsonl').read_text().splitlines()]
        valid = [row for row in rows if row['valid']]
        if len(valid) < 20:
            raise ValueError(f'Insufficient settled scan evidence in turn {step}')
        yaw += statistics.median(row['yaw'] for row in valid[-20:])
    turn = -math.degrees(yaw)
    if not math.isclose(net, summary['right_arm']['net_displacement_m'], abs_tol=1e-10):
        raise ValueError('Arm displacement differs from retained telemetry')
    if not math.isclose(turn, summary['base']['right_turn_degrees_from_step2_reference'], abs_tol=1e-8):
        raise ValueError('Turn estimate differs from retained scan-registration logs')
    if final['has_current_errors'] or final['robot_mode_code'] != 1:
        raise ValueError('Retained final arm sample is not idle and error-free')
    for key in ('plate_contact_verified', 'grasp_verified', 'lift_verified'):
        if summary['right_arm'][key] is not False:
            raise ValueError('Unsupported task-outcome claim')
    return {'status': 'PASS', 'scope': 'offline evidence integrity and derived metrics only',
            'files_verified': len(manifest['files']), 'right_arm_net_displacement_m': net,
            'base_right_turn_estimate_degrees': turn, 'verified_completed_task3_stages': 0,
            'robot_commands_sent': False}


def child_env():
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'phase2')
    return env


def self_test():
    print(json.dumps(verify()), flush=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                    str(ROOT / 'phase2/tests/test_tasks_control.py'),
                    str(ROOT / 'phase2/tests/test_tasks_scoring.py')],
                   env=child_env(), cwd=ROOT, check=True)
    command = [sys.executable, '-m', 'ebim_phase2.tasks', '--task', 'task3',
               '--calibration', str(HERE / 'examples/calibration.synthetic.json'),
               '--assignments', str(HERE / 'examples/seats.synthetic.json'), '--replay-clock']
    data = (HERE / 'examples/missing-observations.synthetic.jsonl').read_text()
    result = subprocess.run(command, input=data, text=True, capture_output=True,
                            env=child_env(), cwd=ROOT, check=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    if len(rows) != 3 or rows[0]['action'] is not None or rows[1]['status'] != 'WAITING' or rows[2]['status'] != 'INVALID_OBSERVATION' or rows[2]['action'] is not None:
        raise ValueError(f'Unexpected missing-observation behavior: {rows}')
    if any(x != 0 for x in rows[1]['action']):
        raise ValueError('Missing observations produced movement in the zero-state fixture')
    print(json.dumps({'status': 'PASS', 'scope': 'synthetic Task 3 JSONL failure cases',
                      'cases': len(rows), 'robot_commands_sent': False}), flush=True)
    return 0


def main():
    args = sys.argv[1:]
    mode = args.pop(0) if args else 'review'
    if mode == 'review' and not args:
        print(json.dumps(verify(), indent=2))
        return 0
    if mode == 'self-test' and not args:
        return self_test()
    if mode in ('describe', 'policy'):
        command = [sys.executable, '-m', 'ebim_phase2.tasks', '--task', 'task3']
        if mode == 'describe':
            command.append('--describe')
        return subprocess.call(command + args, env=child_env(), cwd=ROOT)
    print('usage: review.py [review|self-test|describe|policy <Task 3 CLI arguments>]', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
