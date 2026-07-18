#!/usr/bin/env python3
"""Import-safe fail-closed checks for named articulation commands."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _failure(reason: str, **record: Any) -> dict[str, Any]:
    return {"passed": False, "reason": reason, **record}


def _named_finite_values(
    *,
    values: Mapping[str, float],
    expected_names: Sequence[str],
    value_label: str,
) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    base = {
        "missing_names": [],
        "unexpected_names": [],
    }
    try:
        expected = tuple(str(name) for name in expected_names)
    except (TypeError, ValueError):
        return None, _failure(
            "expected joint names are not a sequence", **base
        )
    if len(expected) != len(set(expected)):
        return None, _failure(
            "expected joint names are not unique", **base
        )
    if not expected:
        return None, _failure(
            "no expected arm joint names were provided", **base
        )
    try:
        names = {str(name) for name in values}
    except (TypeError, ValueError):
        return None, _failure(
            f"{value_label} must be a named mapping", **base
        )
    expected_set = set(expected)
    missing = sorted(expected_set - names)
    unexpected = sorted(names - expected_set)
    base["missing_names"] = missing
    base["unexpected_names"] = unexpected
    if missing or unexpected:
        return None, _failure(
            f"{value_label} names do not exactly match the expected arm joints",
            **base,
        )
    try:
        result = {name: float(values[name]) for name in expected}
    except (KeyError, TypeError, ValueError):
        return None, _failure(
            f"{value_label} values are not all numeric", **base
        )
    if any(not math.isfinite(value) for value in result.values()):
        return None, _failure(
            f"{value_label} values are not all finite", **base
        )
    return result, None


def joint_command_slew_record(
    *,
    previous_targets: Mapping[str, float],
    targets: Mapping[str, float],
    expected_names: Sequence[str],
    max_abs_delta_rad: float,
) -> dict[str, Any]:
    """Certify adjacent named commands without conflating servo tracking."""

    base = {
        "signal": "target_to_previous_command_slew",
        "threshold_rad": None,
        "previous_target_by_name": {},
        "target_by_name": {},
        "signed_slew_rad_by_name": {},
        "max_abs_delta_rad": None,
        "max_abs_delta_joint": None,
        "missing_names": [],
        "unexpected_names": [],
        "command_clamped": False,
    }
    try:
        expected = tuple(str(name) for name in expected_names)
    except (TypeError, ValueError):
        return _failure(
            "expected joint names are not a sequence", **base
        )
    try:
        threshold = float(max_abs_delta_rad)
    except (TypeError, ValueError):
        return _failure("command slew threshold is not numeric", **base)
    if not math.isfinite(threshold) or threshold <= 0.0:
        return _failure(
            "command slew threshold must be finite and positive", **base
        )
    base["threshold_rad"] = threshold

    previous_by_name, previous_failure = _named_finite_values(
        values=previous_targets,
        expected_names=expected,
        value_label="previous command",
    )
    if previous_failure is not None:
        return _failure(
            str(previous_failure["reason"]),
            **{
                **base,
                "missing_names": previous_failure["missing_names"],
                "unexpected_names": previous_failure["unexpected_names"],
            },
        )
    target_by_name, target_failure = _named_finite_values(
        values=targets,
        expected_names=expected,
        value_label="proposed command",
    )
    if target_failure is not None:
        return _failure(
            str(target_failure["reason"]),
            **{
                **base,
                "missing_names": target_failure["missing_names"],
                "unexpected_names": target_failure["unexpected_names"],
            },
        )
    assert previous_by_name is not None
    assert target_by_name is not None
    deltas = {
        name: target_by_name[name] - previous_by_name[name]
        for name in expected
    }
    max_joint = max(expected, key=lambda name: abs(deltas[name]))
    max_delta = abs(deltas[max_joint])
    record = {
        **base,
        "threshold_rad": threshold,
        "previous_target_by_name": previous_by_name,
        "target_by_name": target_by_name,
        "signed_slew_rad_by_name": deltas,
        "max_abs_delta_rad": max_delta,
        "max_abs_delta_joint": max_joint,
    }
    if max_delta > threshold:
        return _failure(
            "proposed command exceeds the adjacent-command slew limit",
            **record,
        )
    return {"passed": True, "reason": None, **record}


def joint_tracking_error_dwell_record(
    *,
    dof_names: Sequence[str],
    measured_positions: Sequence[float],
    targets: Mapping[str, float],
    expected_names: Sequence[str],
    max_abs_error_rad: float,
    dwell_steps: int,
    prior_consecutive_exceeded_steps: int,
) -> dict[str, Any]:
    """Observe target-to-measured error and abort only after its dwell."""

    base = {
        "signal": "target_to_measured_tracking_error",
        "threshold_rad": None,
        "dwell_steps": None,
        "prior_consecutive_exceeded_steps": None,
        "consecutive_exceeded_steps": None,
        "remaining_dwell_steps": None,
        "threshold_exceeded": False,
        "dwell_reached": False,
        "abort_required": True,
        "status": "invalid",
        "measured_by_name": {},
        "target_by_name": {},
        "signed_tracking_error_rad_by_name": {},
        "max_abs_error_rad": None,
        "max_abs_error_joint": None,
        "missing_names": [],
        "unexpected_names": [],
    }
    try:
        names = tuple(str(name) for name in dof_names)
        expected = tuple(str(name) for name in expected_names)
    except (TypeError, ValueError):
        return _failure(
            "DOF names or expected joint names are not sequences", **base
        )
    if len(names) != len(set(names)):
        return _failure("articulation DOF names are not unique", **base)
    try:
        measured = tuple(float(value) for value in measured_positions)
        threshold = float(max_abs_error_rad)
    except (TypeError, ValueError):
        return _failure(
            "measured positions or tracking threshold are not numeric",
            **base,
        )
    if len(measured) != len(names):
        return _failure(
            "joint-position vector length does not match DOF names", **base
        )
    if any(not math.isfinite(value) for value in measured):
        return _failure(
            "measured joint positions are not all finite", **base
        )
    if not math.isfinite(threshold) or threshold <= 0.0:
        return _failure(
            "tracking-error threshold must be finite and positive", **base
        )
    if not isinstance(dwell_steps, int) or isinstance(dwell_steps, bool):
        return _failure("tracking-error dwell must be an integer", **base)
    if dwell_steps < 1:
        return _failure(
            "tracking-error dwell must be at least one step", **base
        )
    if (
        not isinstance(prior_consecutive_exceeded_steps, int)
        or isinstance(prior_consecutive_exceeded_steps, bool)
        or prior_consecutive_exceeded_steps < 0
        or prior_consecutive_exceeded_steps >= dwell_steps
    ):
        return _failure(
            "prior tracking-error dwell state is invalid", **base
        )
    base.update(
        {
            "threshold_rad": threshold,
            "dwell_steps": dwell_steps,
            "prior_consecutive_exceeded_steps": (
                prior_consecutive_exceeded_steps
            ),
        }
    )

    target_by_name, target_failure = _named_finite_values(
        values=targets,
        expected_names=expected,
        value_label="active command",
    )
    if target_failure is not None:
        return _failure(
            str(target_failure["reason"]),
            **{
                **base,
                "missing_names": target_failure["missing_names"],
                "unexpected_names": target_failure["unexpected_names"],
            },
        )
    assert target_by_name is not None
    index_by_name = {name: index for index, name in enumerate(names)}
    absent = sorted(name for name in expected if name not in index_by_name)
    if absent:
        return _failure(
            "expected arm joints are absent from the articulation",
            **{**base, "missing_names": absent},
        )
    measured_by_name = {
        name: measured[index_by_name[name]] for name in expected
    }
    errors = {
        name: target_by_name[name] - measured_by_name[name]
        for name in expected
    }
    max_joint = max(expected, key=lambda name: abs(errors[name]))
    max_error = abs(errors[max_joint])
    exceeded = max_error > threshold
    consecutive = (
        prior_consecutive_exceeded_steps + 1 if exceeded else 0
    )
    dwell_reached = consecutive >= dwell_steps
    record = {
        **base,
        "consecutive_exceeded_steps": consecutive,
        "remaining_dwell_steps": max(0, dwell_steps - consecutive),
        "threshold_exceeded": exceeded,
        "dwell_reached": dwell_reached,
        "abort_required": dwell_reached,
        "status": (
            "dwell_reached"
            if dwell_reached
            else "dwell_pending"
            if exceeded
            else "within_threshold"
        ),
        "measured_by_name": measured_by_name,
        "target_by_name": target_by_name,
        "signed_tracking_error_rad_by_name": errors,
        "max_abs_error_rad": max_error,
        "max_abs_error_joint": max_joint,
    }
    if dwell_reached:
        return _failure(
            "target-to-measured tracking error exceeded its sustained dwell",
            **record,
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
