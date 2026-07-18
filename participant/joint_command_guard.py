#!/usr/bin/env python3
"""Import-safe fail-closed checks for named articulation commands."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _failure(reason: str, **record: Any) -> dict[str, Any]:
    return {"passed": False, "reason": reason, **record}


def joint_target_continuity_record(
    *,
    dof_names: Sequence[str],
    current_positions: Sequence[float],
    targets: Mapping[str, float],
    expected_names: Sequence[str],
    max_abs_delta_rad: float,
) -> dict[str, Any]:
    """Certify that one named revolute-joint command is locally continuous."""

    names = tuple(str(name) for name in dof_names)
    expected = tuple(str(name) for name in expected_names)
    base = {
        "threshold_rad": None,
        "current_by_name": {},
        "target_by_name": {},
        "signed_delta_rad_by_name": {},
        "max_abs_delta_rad": None,
        "max_abs_delta_joint": None,
        "missing_target_names": [],
        "unexpected_target_names": [],
    }
    if len(names) != len(set(names)):
        return _failure("articulation DOF names are not unique", **base)
    if len(expected) != len(set(expected)):
        return _failure("expected target names are not unique", **base)
    if not expected:
        return _failure("no expected arm target names were provided", **base)
    try:
        current = tuple(float(value) for value in current_positions)
        threshold = float(max_abs_delta_rad)
    except (TypeError, ValueError):
        return _failure("joint positions or threshold are not numeric", **base)
    if len(current) != len(names):
        return _failure(
            "joint-position vector length does not match DOF names", **base
        )
    if not math.isfinite(threshold) or threshold <= 0.0:
        return _failure(
            "joint continuity threshold must be finite and positive", **base
        )
    base["threshold_rad"] = threshold
    if any(not math.isfinite(value) for value in current):
        return _failure("current joint positions are not all finite", **base)

    target_names = {str(name) for name in targets}
    expected_set = set(expected)
    missing = sorted(expected_set - target_names)
    unexpected = sorted(target_names - expected_set)
    base["missing_target_names"] = missing
    base["unexpected_target_names"] = unexpected
    if missing or unexpected:
        return _failure(
            "IK target names do not exactly match the expected arm joints",
            **base,
        )

    try:
        target_by_name = {
            name: float(targets[name])
            for name in expected
        }
    except (KeyError, TypeError, ValueError):
        return _failure("IK targets are not all numeric", **base)
    if any(not math.isfinite(value) for value in target_by_name.values()):
        return _failure("IK targets are not all finite", **base)

    index_by_name = {name: index for index, name in enumerate(names)}
    absent_from_articulation = sorted(
        name for name in expected if name not in index_by_name
    )
    if absent_from_articulation:
        base["missing_target_names"] = absent_from_articulation
        return _failure(
            "expected arm joints are absent from the articulation", **base
        )

    current_by_name = {
        name: current[index_by_name[name]]
        for name in expected
    }
    deltas = {
        name: target_by_name[name] - current_by_name[name]
        for name in expected
    }
    max_joint = max(expected, key=lambda name: abs(deltas[name]))
    max_delta = abs(deltas[max_joint])
    record = {
        **base,
        "threshold_rad": threshold,
        "current_by_name": current_by_name,
        "target_by_name": target_by_name,
        "signed_delta_rad_by_name": deltas,
        "max_abs_delta_rad": max_delta,
        "max_abs_delta_joint": max_joint,
    }
    if max_delta > threshold:
        return _failure(
            "IK target exceeds the per-step joint continuity limit", **record
        )
    return {"passed": True, "reason": None, **record}


def arm_and_spine_effort_record(
    *,
    dof_names: Sequence[str],
    measured_efforts: Sequence[float],
    arm_joint_names: Sequence[str],
    spine_joint_name: str,
    arm_effort_abort_threshold: float,
) -> dict[str, Any]:
    """Classify revolute-arm torque separately from prismatic spine force."""

    names = tuple(str(name) for name in dof_names)
    arms = tuple(str(name) for name in arm_joint_names)
    base = {
        "arm_effort_by_name": {},
        "peak_abs_arm_effort": None,
        "peak_abs_arm_effort_joint": None,
        "spine_force_newtons": None,
        "arm_effort_abort_threshold": None,
        "arm_threshold_exceeded": False,
    }
    if len(names) != len(set(names)):
        return _failure("articulation DOF names are not unique", **base)
    if len(arms) != len(set(arms)):
        return _failure("arm joint names are not unique", **base)
    if not arms:
        return _failure("no revolute arm joint names were provided", **base)
    try:
        efforts = tuple(float(value) for value in measured_efforts)
        threshold = float(arm_effort_abort_threshold)
    except (TypeError, ValueError):
        return _failure("efforts or arm threshold are not numeric", **base)
    if len(efforts) != len(names):
        return _failure(
            "effort vector length does not match DOF names", **base
        )
    if not math.isfinite(threshold) or threshold <= 0.0:
        return _failure(
            "arm effort threshold must be finite and positive", **base
        )
    base["arm_effort_abort_threshold"] = threshold
    index_by_name = {name: index for index, name in enumerate(names)}
    missing = [
        name
        for name in (*arms, str(spine_joint_name))
        if name not in index_by_name
    ]
    if missing:
        return _failure(
            f"monitored articulation joints are missing: {sorted(missing)}",
            **base,
        )
    monitored = {
        name: efforts[index_by_name[name]]
        for name in (*arms, str(spine_joint_name))
    }
    if any(not math.isfinite(value) for value in monitored.values()):
        return _failure(
            "arm or spine effort telemetry is non-finite", **base
        )

    arm_effort_by_name = {name: monitored[name] for name in arms}
    peak_joint = max(arms, key=lambda name: abs(arm_effort_by_name[name]))
    peak_effort = abs(arm_effort_by_name[peak_joint])
    exceeded = peak_effort > threshold
    record = {
        **base,
        "arm_effort_by_name": arm_effort_by_name,
        "peak_abs_arm_effort": peak_effort,
        "peak_abs_arm_effort_joint": peak_joint,
        "spine_force_newtons": monitored[str(spine_joint_name)],
        "arm_effort_abort_threshold": threshold,
        "arm_threshold_exceeded": exceeded,
    }
    if exceeded:
        return _failure(
            "measured revolute-arm effort exceeded the abort threshold",
            **record,
        )
    return {"passed": True, "reason": None, **record}
