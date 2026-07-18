#!/usr/bin/env python3
"""Validate AirSign's four-stage browser replay and metrics evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tree(value: Any, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite number at {location}")
    if isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, f"{location}[{index}]")
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{location}.{key}")


def require_truth_boundary(document: dict[str, Any], name: str) -> None:
    if document.get("official_stage_completion_claimed") is not False:
        raise ValueError(f"{name} must explicitly reject stage claims")
    if document.get("official_score_claimed") is not False:
        raise ValueError(f"{name} must explicitly reject score claims")
    if document.get("benchmark_score", "missing") is not None:
        raise ValueError(f"{name} benchmark_score must be null")


def validate(directory: Path) -> dict[str, Any]:
    replay_path = directory / "replay_trace.json"
    metrics_path = directory / "metrics.json"
    manifest_path = directory / "manifest.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    finite_tree(replay)
    finite_tree(metrics)
    require_truth_boundary(replay, "replay")
    require_truth_boundary(metrics, "metrics")
    if replay.get("classification") != "measured_actuator_rehearsal":
        raise ValueError("Unexpected replay classification")
    if metrics.get("classification") != "participant_controller_rehearsal":
        raise ValueError("Unexpected metrics classification")
    if replay.get("task_objects_teleported") is not False:
        raise ValueError("Replay must reject task-object teleportation")
    if replay.get("robot_links_teleported") is not False:
        raise ValueError("Replay must reject robot-link teleportation")
    if replay.get("task_object_mutation_api_used") is not False:
        raise ValueError("Replay must reject task-object mutation APIs")

    dof_names = replay.get("dof_names")
    frames = replay.get("frames")
    if not isinstance(dof_names, list) or not dof_names:
        raise ValueError("Replay is missing DOF names")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Replay contains no frames")
    previous_step = -1
    observed_stages: set[int] = set()
    for index, frame in enumerate(frames):
        step = frame.get("simulation_step")
        if not isinstance(step, int) or step < previous_step:
            raise ValueError(f"Non-monotonic frame step at index {index}")
        previous_step = step
        stage_id = frame.get("stage_id")
        if isinstance(stage_id, int) and 1 <= stage_id <= 4:
            observed_stages.add(stage_id)
        joints = frame.get("joint_positions")
        if not isinstance(joints, list) or len(joints) != len(dof_names):
            raise ValueError(f"Joint vector mismatch at frame {index}")
        objects = frame.get("task_objects")
        if set(objects or {}) != {"plate", "cup", "bowl", "spoon"}:
            raise ValueError(f"Task-object snapshot mismatch at frame {index}")
    if observed_stages != {1, 2, 3, 4}:
        raise ValueError(f"Replay stage coverage mismatch: {observed_stages}")

    stage_results = metrics.get("stages")
    if not isinstance(stage_results, list) or [
        item.get("stage_id") for item in stage_results
    ] != [1, 2, 3, 4]:
        raise ValueError("Metrics must contain ordered stages 1 through 4")
    for stage in stage_results:
        if stage.get("classification") != "actuator_rehearsal":
            raise ValueError("Stage classification mismatch")
        if stage.get("official_stage_complete") is not False:
            raise ValueError("A rehearsal stage cannot claim completion")
        if stage.get("official_stage_score", "missing") is not None:
            raise ValueError("A rehearsal stage score must be null")

    files = manifest.get("files", {})
    for path in (replay_path, metrics_path):
        record = files.get(path.name)
        if not isinstance(record, dict):
            raise ValueError(f"Manifest missing {path.name}")
        if record.get("sha256") != sha256_file(path):
            raise ValueError(f"SHA-256 mismatch for {path.name}")
        if record.get("bytes") != path.stat().st_size:
            raise ValueError(f"Byte count mismatch for {path.name}")
    return {
        "passed": True,
        "directory": str(directory),
        "frame_count": len(frames),
        "stage_ids": sorted(observed_stages),
        "controller_motion_gate_passed": bool(metrics.get("passed")),
        "official_stage_completion_claimed": False,
        "official_score_claimed": False,
        "benchmark_score": None,
        "sha256": {
            replay_path.name: files[replay_path.name]["sha256"],
            metrics_path.name: files[metrics_path.name]["sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = validate(args.directory.expanduser().resolve())
    print(
        "FOUR_STAGE_REPLAY_VALIDATION "
        + json.dumps(result, sort_keys=True)
    )


if __name__ == "__main__":
    main()
