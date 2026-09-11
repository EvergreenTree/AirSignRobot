"""Rulebook 1.0 score arithmetic, separated from policy observations.

These are local score calculations from supplied evaluation evidence. They do
not replace organizer adjudication or infer physical success from commands.
Source: https://github.com/EBiM-Benchmark/ebim-benchmark.github.io/blob/f92510f87c71ed4c975e9a571bcbbd288913cf5d/src/docs/Autonomous_Robot_Benchmark_Rulebook_1.0.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

import numpy as np

OBJECTS = ("plate", "cup", "bowl", "spoon")


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _boolean(value, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def mask_iou(pad_mask, target_mask) -> float:
    """Binary-mask IoU, not the simulator aid's bounding-box approximation."""
    a, b = np.asarray(pad_mask), np.asarray(target_mask)
    if a.ndim != 2 or a.shape != b.shape or a.dtype != bool or b.dtype != bool:
        raise ValueError("IoU needs same-shaped 2D boolean masks in one registered camera frame")
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()/union) if union else 0.


@dataclass(frozen=True)
class Task2Score:
    valid_placement_iou: float
    raw_iou: float
    elapsed_seconds: float

    @property
    def rank_key(self):
        # Page1: valid IoU primary, completion time tie breaker. Raw IoU is
        # reported as a secondary metric, not invented as an extra tie break.
        return self.valid_placement_iou, -self.elapsed_seconds


def task2_score(*, pick_success: bool, orientation_success: bool,
                raw_iou: float, elapsed_seconds: float) -> Task2Score:
    raw_iou = _nonnegative(raw_iou, "raw_iou")
    if raw_iou > 1:
        raise ValueError("raw_iou must be in [0,1]")
    picked = _boolean(pick_success, "pick_success")
    oriented = _boolean(orientation_success, "orientation_success")
    valid = picked and oriented
    return Task2Score(raw_iou if valid else 0., raw_iou,
                      _nonnegative(elapsed_seconds, "elapsed_seconds"))


@dataclass
class FeedHoldLedger:
    """Counts only continuous observed loaded-spoon hold intervals.

    Repeated/stale timestamps cannot accrue hold time; missing frames longer
    than max_gap invalidate continuity. A verified three-second hold remains
    recorded after retraction so return-to-bowl evidence can be evaluated.
    """

    max_gap: float = 0.25
    last_timestamp: float | None = None
    started: float | None = None
    current_seconds: float = 0.
    best_seconds: float = 0.
    returned_after_hold: bool = False

    def __post_init__(self):
        if not math.isfinite(self.max_gap) or self.max_gap <= 0:
            raise ValueError("max_gap must be finite and positive")

    def update(self, timestamp: float, *, spoon_loaded: bool, in_feed_zone: bool,
               returned_to_bowl: bool = False) -> None:
        timestamp = _nonnegative(timestamp, "timestamp")
        loaded = _boolean(spoon_loaded, "spoon_loaded")
        in_zone = _boolean(in_feed_zone, "in_feed_zone")
        returned = _boolean(returned_to_bowl, "returned_to_bowl")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            self.started, self.current_seconds = None, 0.
            return
        gap = math.inf if self.last_timestamp is None else timestamp-self.last_timestamp
        self.last_timestamp = timestamp
        if loaded and in_zone:
            if self.started is None or gap > self.max_gap:
                self.started = timestamp
            self.current_seconds = timestamp-self.started
            self.best_seconds = max(self.best_seconds, self.current_seconds)
        else:
            self.started, self.current_seconds = None, 0.
        if returned and self.best_seconds >= 3.:
            self.returned_after_hold = True

    @property
    def complete(self) -> bool:
        return self.best_seconds >= 3. and self.returned_after_hold


@dataclass(frozen=True)
class Task3Evidence:
    assigned_placements: Mapping[str, bool] = field(default_factory=dict)
    feeding_hold_seconds: float = 0.
    feeding_loaded_throughout: bool = False
    beans_returned: bool = False
    original_bean_mass: float = 0.
    recovered_bean_mass: float = 0.
    sink_placements: Mapping[str, bool] = field(default_factory=dict)
    elapsed_seconds: float = 0.


@dataclass(frozen=True)
class Task3Score:
    table_setup: float
    feeding: float
    bean_recovery: float
    cleanup: float
    elapsed_seconds: float

    @property
    def total(self) -> float:
        return self.table_setup+self.feeding+self.bean_recovery+self.cleanup

    @property
    def highest_completed_stage(self) -> int:
        scores = (self.table_setup, self.feeding, self.bean_recovery, self.cleanup)
        return max((i for i, score in enumerate(scores, start=1) if score >= 4.), default=0)

    @property
    def rank_key(self):
        return self.highest_completed_stage, self.total, -self.elapsed_seconds


def task3_score(evidence: Task3Evidence) -> Task3Score:
    """Four utensils, loaded feeding+return, continuous recovered mass ratio.

    No tray point, no coarse 80/90/100% bins from the development scorer.
    Stage1 partial credit uses one point per correctly assigned utensil; the
    PDF specifies four points total but leaves this partial-credit split
    implicit, so report per-object evidence alongside the local estimate.
    """
    def count(mapping):
        if set(mapping)-set(OBJECTS):
            raise ValueError("only plate/cup/bowl/spoon are scored; tray is not a fifth object")
        return float(sum(_boolean(mapping.get(name, False), name) for name in OBJECTS))
    setup, cleanup = count(evidence.assigned_placements), count(evidence.sink_placements)
    hold = _nonnegative(evidence.feeding_hold_seconds, "feeding_hold_seconds")
    loaded = _boolean(evidence.feeding_loaded_throughout, "feeding_loaded_throughout")
    returned = _boolean(evidence.beans_returned, "beans_returned")
    feeding = 4. if hold >= 3. and loaded and returned else 0.
    original = _nonnegative(evidence.original_bean_mass, "original_bean_mass")
    recovered = _nonnegative(evidence.recovered_bean_mass, "recovered_bean_mass")
    if original == 0. and recovered > 0.:
        raise ValueError("positive recovered mass requires a measured initial bean mass")
    recovery = 4.*min(1., recovered/original) if original else 0.
    return Task3Score(setup, feeding, recovery, cleanup,
                      _nonnegative(evidence.elapsed_seconds, "elapsed_seconds"))
