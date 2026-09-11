"""JSONL observation→action adapter, without implicit hardware actuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from .control import Controller, Observation, ServoCalibration
from .plans import task2_plan, task3_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("task2", "task3"), required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--assignments", type=Path, help="Task3 JSON map plate/cup/bowl/spoon to assigned seat")
    parser.add_argument("--peel-required", action="store_true", help="Enable only if the actual site presents an attached pad")
    parser.add_argument('--pad-arms', choices=['left', 'right', 'both'], default='right',
                        help='Task2 grasp configuration; bimanual handling is an explicit site choice')
    parser.add_argument("--episode-timeout", type=float, default=1800., help="Local controller timeout; set site limit explicitly")
    parser.add_argument("--describe", action="store_true", help="Print pose/signal requirements for every waypoint")
    parser.add_argument("--replay-clock", action="store_true", help="Offline fixtures only: use recorded clock instead of host wall time")
    args = parser.parse_args(argv)
    if args.task == "task3":
        if args.assignments is None and not args.describe:
            parser.error("task3 requires --assignments from the organizer's assigned seats")
        assignments = (json.loads(args.assignments.read_text()) if args.assignments else
                       {'plate': 'assigned_plate_seat', 'cup': 'assigned_cup_seat',
                        'bowl': 'assigned_head_seat', 'spoon': 'assigned_head_seat'})
        plan = task3_plan(assignments)
    else:
        plan = task2_plan(peel_required=args.peel_required,
                          arms=('left','right') if args.pad_arms == 'both' else (args.pad_arms,))
    if args.describe:
        for item in plan:
            print(json.dumps({"waypoint": item.name, "stage": item.stage,
                              "poses": {arm: target.key for arm, target in item.targets.items()},
                              "navigate": item.navigate, "require": item.require,
                              "require_while": item.require_while, "near_head": item.near_head}))
        return 0
    if args.calibration is None:
        parser.error('execution requires --calibration; use --describe to inspect requirements')
    calibration = ServoCalibration(**json.loads(args.calibration.read_text()))
    controller = Controller(plan, calibration, episode_timeout=args.episode_timeout)
    for line_number, line in enumerate(sys.stdin, start=1):
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError('Observation must be a JSON object')
            if message.get("reset") is True:
                controller.reset()
                print(json.dumps({"status": "RESET"}), flush=True)
                continue
            now = message.get('now', message['timestamp']) if args.replay_clock else time.time()
            decision = controller.step(Observation.from_dict(message), now=now)
            print(json.dumps(decision.to_dict(), allow_nan=False), flush=True)
        except (KeyError, TypeError, ValueError) as error:
            # Invalid robot state has no defensible hold command. Never reuse
            # a stale action: let the transport's deadman watchdog stop it.
            print(json.dumps({"status": "INVALID_OBSERVATION", "action": None,
                              "line": line_number, "reason": str(error)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
