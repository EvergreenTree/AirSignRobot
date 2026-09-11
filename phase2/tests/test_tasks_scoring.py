import numpy as np
import pytest

from ebim_phase2.scoring import FeedHoldLedger, Task3Evidence, mask_iou, task2_score, task3_score


def test_task2_pick_and_orientation_gate_actual_mask_iou():
    a = np.array([[1, 0], [1, 0]], dtype=bool)
    b = np.array([[0, 0], [1, 1]], dtype=bool)
    assert mask_iou(a, b) == pytest.approx(1/3)
    assert task2_score(pick_success=True, orientation_success=True, raw_iou=.8, elapsed_seconds=4).valid_placement_iou == .8
    assert task2_score(pick_success=True, orientation_success=False, raw_iou=.8, elapsed_seconds=4).valid_placement_iou == 0.
    assert task2_score(pick_success=False, orientation_success=True, raw_iou=.8, elapsed_seconds=4).raw_iou == .8
    with pytest.raises(ValueError):
        task2_score(pick_success=False, orientation_success="yes", raw_iou=.8, elapsed_seconds=4)


@pytest.mark.parametrize("ratio", [.0, .4, .8, .9, 1.])
def test_task3_mass_score_is_proportional_not_development_bins(ratio):
    score = task3_score(Task3Evidence(original_bean_mass=100., recovered_bean_mass=100*ratio))
    assert score.bean_recovery == pytest.approx(4*ratio)


def test_task3_cleanup_and_ranking_use_four_objects_no_tray():
    all_objects = {x: True for x in ("plate", "cup", "bowl", "spoon")}
    cleanup = task3_score(Task3Evidence(sink_placements=all_objects, elapsed_seconds=30))
    setup = task3_score(Task3Evidence(assigned_placements=all_objects, elapsed_seconds=10))
    assert cleanup.total == 4. and cleanup.highest_completed_stage == 4
    assert cleanup.rank_key > setup.rank_key
    with pytest.raises(ValueError, match="tray"):
        task3_score(Task3Evidence(sink_placements={"tray": True}))


def test_feed_ledger_requires_continuous_loaded_hold_then_return():
    ledger = FeedHoldLedger()
    for t in np.arange(0, 2.1, .1):
        ledger.update(float(t), spoon_loaded=True, in_feed_zone=True)
    ledger.update(2.2, spoon_loaded=False, in_feed_zone=True)
    for t in np.arange(2.3, 5.51, .1):
        ledger.update(float(t), spoon_loaded=True, in_feed_zone=True)
    assert ledger.best_seconds >= 3 and not ledger.complete
    ledger.update(5.7, spoon_loaded=False, in_feed_zone=False, returned_to_bowl=True)
    assert ledger.complete
    score = task3_score(Task3Evidence(feeding_hold_seconds=ledger.best_seconds,
                                     feeding_loaded_throughout=True, beans_returned=ledger.returned_after_hold))
    assert score.feeding == 4.
    assert task3_score(Task3Evidence(feeding_hold_seconds=3.5, feeding_loaded_throughout=True)).feeding == 0.


def test_duplicate_frames_and_long_gaps_cannot_certify_feeding():
    ledger = FeedHoldLedger()
    for _ in range(80):
        ledger.update(0., spoon_loaded=True, in_feed_zone=True)
    ledger.update(4., spoon_loaded=True, in_feed_zone=True)
    assert ledger.best_seconds == 0.
    assert not ledger.complete


@pytest.mark.parametrize("mass", [-1, float("nan"), float("inf")])
def test_invalid_mass_is_not_silently_scored(mass):
    with pytest.raises(ValueError):
        task3_score(Task3Evidence(original_bean_mass=1., recovered_bean_mass=mass))
