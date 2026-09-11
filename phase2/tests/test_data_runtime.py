import json
from pathlib import Path
import numpy as np
import pytest

from ebim_phase2.data import EpisodeDataset, nearest_indices, split_episodes
from ebim_phase2.runtime import TemporalEnsemble, CommandLimiter
from ebim_phase2.schema import action23


def test_nearest_uses_pts_not_nominal_fps():
    times = np.array([0., .034, .069, .12])
    assert nearest_indices(times, np.array([.0, .05, .10])).tolist() == [0, 1, 3]
    with pytest.raises(ValueError):
        nearest_indices(np.array([0., 0.]), np.array([0.]))


def test_episode_split_independent_of_input_order():
    ids = [f"part{i//10}_{i}" for i in range(30)]
    splits = split_episodes(ids)
    assert splits == split_episodes(ids[::-1])
    assert set(splits.values()) == {"train", "validation", "test"}
    with pytest.raises(ValueError):
        split_episodes(["a", "a", "b"])


def test_reusing_cache_does_not_bypass_requested_alignment_loss_limit(tmp_path):
    from ebim_phase2.data import prepare_episode
    folder=tmp_path/'episode';folder.mkdir()
    saved={'source_revision':'pinned','settings':{'max_skew_s':.075},'observations':17,
           'dropped_unaligned_observations':3,'max_unaligned_fraction':.2}
    (folder/'episode.json').write_text(json.dumps(saved))
    job={'raw':str(tmp_path),'output':str(tmp_path),'id':'episode','revision':'pinned','settings':saved['settings']}
    with pytest.raises(ValueError,match='alignment-loss'):prepare_episode(job)
    assert prepare_episode({**job,'max_unaligned_fraction':.2})==saved


def test_padding_never_crosses_episodes_and_train_statistics_only(tmp_path):
    records = []
    for index, split in enumerate(["train", "validation", "test"]):
        root = tmp_path / str(index); root.mkdir()
        np.save(root / "state.npy", np.full((3, 62), index, dtype=np.float32))
        np.save(root / "action.npy", np.full((3, 23), index*100, dtype=np.float32))
        np.save(root / "rgb.npy", np.zeros((3, 3, 32, 32, 3), dtype=np.uint8))
        np.save(root / "obs_frames.npy", np.arange(3))
        records.append({"id": str(index), "task": 1, "split": split, "observations": 3})
    (tmp_path / "manifest.json").write_text(json.dumps({"episodes": records}))
    train = EpisodeDataset(tmp_path, 1, "train", 4)
    assert train.stats["action_mean"] == [0]*23
    tail = train[2]
    assert tail["mask"].tolist() == [True, False, False, False]
    assert np.all(tail["action"] == 0)
    with pytest.raises(ValueError):
        EpisodeDataset(tmp_path, 1, "test", 4)
    test = EpisodeDataset(tmp_path, 1, "test", 4, train.stats)
    assert np.all(test[0]["action"] > 100)


def test_task2_mapping_preserves_spine_and_gripper_fraction():
    native = np.arange(17, dtype=float)
    native[7] = .8; native[15] = .6; native[16] = 434.
    with pytest.raises(ValueError, match="calibrated"):
        action23(native, 2)
    # Test fixture conversion only: no assertion about the release's physical units.
    full = action23(native, 2, spine_scale_m_per_native=.001, spine_offset_m=-.1)
    np.testing.assert_array_equal(full[:16], native[:16])
    assert np.all(full[16:22] == 0)
    assert full[22] == pytest.approx(.334)
    with pytest.raises(ValueError):
        action23(np.zeros(19), 2)


def test_native_joint_mapping_is_explicit_and_does_not_change_gripper_slots():
    native = np.arange(23, dtype=float)
    mapped = action23(native, 1, joint_action_scale=[-1.]*14, joint_action_offset_rad=[.1]*14)
    assert mapped[0] == .1 and mapped[8] == -7.9
    assert mapped[7] == native[7] and mapped[15] == native[15]
    with pytest.raises(ValueError):
        action23(native, 1, joint_action_scale=[0.]*14, joint_action_offset_rad=[0.]*14)


def test_temporal_ensemble_uses_elapsed_frames_and_resets():
    ensemble = TemporalEnsemble()
    first = np.arange(10)[:, None] * np.ones((1, 23))
    assert ensemble.step(0., first)[0] == 0
    value = ensemble.step(.1, np.full((10, 23), 10.))[0]
    assert 6 < value < 10  # old chunk's index2, with new observation favored
    with pytest.raises(ValueError):
        ensemble.step(.1, first)
    ensemble.reset()
    assert ensemble.step(0., first)[0] == 0
    # The final valid frame of an older chunk still contributes.
    ensemble=TemporalEnsemble(decay=0.)
    ensemble.step(0.,np.zeros((2,23)))
    np.testing.assert_allclose(ensemble.step(.05,np.ones((2,23))),.5)
    np.testing.assert_allclose(ensemble.step(.1,np.ones((2,23))),1.)


def calibration():
    return {"calibration_id": "test-only", "joint_lower": [-3]*14, "joint_upper": [3]*14,
            "joint_speed_rad_s": [.4]*14, "gripper_open_knuckle": [0,0], "gripper_closed_knuckle": [.8,.8],
            "gripper_open_command": [1,1], "gripper_closed_command": [0,0], "spine_lower_m": 0,
            "spine_upper_m": .5, "spine_speed_m_s": .1, "base_linear_speed_m_s": .1,
            "base_yaw_speed_rad_s": .2, "max_force_n": 20, "max_observation_age_s": .2, "max_camera_skew_s": .075}


def test_limiter_stale_and_force_failures_emit_no_stale_position_target():
    limiter = CommandLimiter(calibration())
    state = np.zeros(62); state[20] = .4; state[41] = .2; state[61] = .3
    record = {"timestamp_s": 5., "collision_imminent": False}
    action, reason = limiter.apply(np.ones(23), state, 1, record, 6.)
    assert reason == "stale_or_future_state"
    assert action is None
    np.testing.assert_allclose(limiter.gripper_commands(state, 1), [.5,.75])
    state[14] = 30
    action, reason = limiter.apply(np.ones(23), state, 1, record, 5.05)
    assert reason == "force_limit" and action is None


def test_limiter_task2_spine_requires_measured_input_and_base_locked():
    limiter = CommandLimiter(calibration())
    state = np.zeros(42)
    record = {"timestamp_s": 5., "collision_imminent": False}
    with pytest.raises(KeyError):
        limiter.apply(np.ones(23), state, 2, record, 5.05)
    record["spine_height_m"] = .3
    action, reason = limiter.apply(np.ones(23), state, 2, record, 5.05)
    assert reason == "bounded_proposal"
    assert np.max(np.abs(action[:7])) <= .020001
    assert action[22] == pytest.approx(.305)
    assert np.all(action[16:22] == 0)


@pytest.mark.parametrize("collision", [None, 0, "false", True])
def test_limiter_requires_explicit_false_collision_measurement(collision):
    limiter = CommandLimiter(calibration())
    action, reason = limiter.apply(np.ones(23), np.zeros(62), 1,
                                   {"timestamp_s": 0., "collision_imminent": collision}, 0.)
    assert action is None and reason == "collision_or_missing_collision_observation"


@pytest.mark.parametrize("opened,closed", [([100,100],[0,0]), ([-.1,1],[0,0]), ([0,1],[0,0])])
def test_limiter_rejects_invalid_gripper_scale_and_indistinct_endpoints(opened, closed):
    with pytest.raises(ValueError, match="native"):
        CommandLimiter({**calibration(), "gripper_open_command": opened, "gripper_closed_command": closed})


def test_limiter_uses_minimum_observation_and_wall_clock_elapsed_time():
    limiter = CommandLimiter(calibration())
    state = np.zeros(62); state[61] = .3
    first, _ = limiter.apply(np.ones(23), state, 1, {"timestamp_s": 9.85, "collision_imminent": False}, 10.)
    state[:7], state[21:28], state[61] = first[:7], first[8:15], first[22]
    # Camera/control timestamps advance 50ms, requests only 10ms: use 10ms.
    second, reason = limiter.apply(np.ones(23), state, 1, {"timestamp_s": 9.90, "collision_imminent": False}, 10.01)
    assert reason == "bounded_proposal"
    assert second[0]-state[0] == pytest.approx(.4*.01)
    assert second[22]-state[61] == pytest.approx(.1*.01)
    state[:7], state[21:28], state[61] = second[:7], second[8:15], second[22]
    # Observation advances only 5ms while wall clock advances 50ms: use 5ms.
    third, reason = limiter.apply(np.ones(23), state, 1, {"timestamp_s": 9.905, "collision_imminent": False}, 10.06)
    assert third[0]-state[0] == pytest.approx(.4*.005)
    assert third[22]-state[61] == pytest.approx(.1*.005)
    duplicate, reason = limiter.apply(np.ones(23), state, 1, {"timestamp_s": 9.905, "collision_imminent": False}, 10.07)
    assert duplicate is None and reason == "nonmonotonic_observation"
    backwards_clock, reason = limiter.apply(np.ones(23), state, 1, {"timestamp_s": 9.91, "collision_imminent": False}, 10.05)
    assert backwards_clock is None and reason == "nonmonotonic_command_time"


def test_high_rate_requests_cannot_multiply_joint_speed_limit():
    limiter = CommandLimiter(calibration())
    state = np.zeros(62); state[61] = .3
    first = None
    for i in range(100):
        stamp = i*.01
        action, reason = limiter.apply(np.ones(23)*10, state, 1,
                                       {"timestamp_s": stamp, "collision_imminent": False}, stamp)
        assert reason == "bounded_proposal"
        if first is None:
            first = action[0]
        state[:7], state[21:28] = action[:7], action[8:15]
    assert state[0]-first == pytest.approx(.4*.99)


def test_policy_stale_preflight_returns_null_action_without_loading_images():
    from types import SimpleNamespace
    import threading
    from ebim_phase2.runtime import Policy
    policy = Policy.__new__(Policy)
    policy.task, policy.episode_id, policy.lock = 1, None, threading.Lock()
    policy.config = SimpleNamespace(state_dim=62, action_dim=23)
    policy.schema = {"observation.state": {"names": [f"q{i}" for i in range(62)]}}
    policy.ensemble, policy.limiter = TemporalEnsemble(), CommandLimiter(calibration())
    record = {"task": 1, "episode_id": "test", "state": [0.]*62, "timestamp_s": 0.,
              "state_names": policy.schema["observation.state"]["names"], "collision_imminent": False}
    output = policy.infer(record, now=5.)
    assert output["status"] == "no_command" and output["reason"] == "stale_or_future_state"
    assert output["action"] is None and output["commands"] is None
    # Task1's 30-minute limit includes base approach and does not require a
    # route-progress detector to have already produced its first result.
    record['timestamp_s'] = 1805.
    output = policy.infer(record, now=1805.)
    assert output['action'] is None and output['reason'] == 'task1_time_limit'
    policy.reset('next-trial')
    assert policy.episode_started_at is None


def test_world_frame_base_command_uses_measured_heading():
    from ebim_phase2.runtime import base_command_in_body_frame
    state=np.zeros(62);state[47:49]=np.sqrt(.5)
    action=np.zeros(23);action[16]=.1;action[21]=.2
    result=base_command_in_body_frame(action,state,'world')
    np.testing.assert_allclose(result[16:18],[0,-.1],atol=1e-12)
    assert result[21]==.2
    np.testing.assert_array_equal(base_command_in_body_frame(action,state,'body'),action)
    with pytest.raises(ValueError):base_command_in_body_frame(action,state,'unknown')
    with pytest.raises(ValueError):base_command_in_body_frame(action,np.zeros(62),'world')


@pytest.mark.parametrize("workers", [0, 2])
def test_global_step_sampler_resume_matches_uninterrupted_despite_prefetch(workers):
    import torch
    from torch.utils.data import DataLoader
    from ebim_phase2.train import GlobalStepBatchSampler
    make = lambda start: DataLoader(range(97), batch_sampler=GlobalStepBatchSampler(97, 4, 42, start, 10),
                                    multiprocessing_context="spawn" if workers else None,
                                    num_workers=workers, generator=torch.Generator().manual_seed(77))
    baseline = [batch.tolist() for batch in make(0)]
    interrupted = iter(make(0))
    consumed = [next(interrupted).tolist() for _ in range(3)]
    resumed = [batch.tolist() for batch in make(3)]
    assert consumed+resumed == baseline
    assert list(GlobalStepBatchSampler(97, 4, 42, 10, 10)) == []
    assert list(GlobalStepBatchSampler(97, 4, 42, 11, 10)) == []


def test_dataloader_worker_seeding_does_not_change_training_rng():
    import torch
    from torch.utils.data import DataLoader
    from ebim_phase2.train import GlobalStepBatchSampler
    torch.manual_seed(123)
    expected = torch.get_rng_state().clone()
    loader = DataLoader(range(97), batch_sampler=GlobalStepBatchSampler(97, 4, 42, 0, 2),
                        generator=torch.Generator().manual_seed(77), num_workers=0)
    next(iter(loader))
    assert torch.equal(expected, torch.get_rng_state())


def resume_checkpoint(score=.4, historical=.2):
    return {"format": "airsign_act_v1", "task": 1, "dataset_sha256": "fixture", "model_config": {},
            "step": 10, "model": {"marker": 123}, "best_validation": historical,
            "validation": {"normalized_mae": score}}


def test_new_output_resume_uses_available_weights_score_and_creates_best(tmp_path):
    import torch
    from ebim_phase2.train import initialize_resume_best
    checkpoint = resume_checkpoint()
    score, note = initialize_resume_best(tmp_path, checkpoint)
    assert score == .4 and "historical_best_weights_unavailable" in note
    saved = torch.load(tmp_path/"best.pt", weights_only=False)
    assert saved["model"] == checkpoint["model"]
    assert saved["best_validation"] == .4 and saved["historical_best_validation"] == .2
    assert checkpoint["best_validation"] == .2  # Input/source artifact is unchanged.


def test_resume_preserves_existing_best_artifact(tmp_path):
    import torch
    from ebim_phase2.train import initialize_resume_best
    torch.save(resume_checkpoint(.1, .1), tmp_path/"best.pt")
    before = (tmp_path/"best.pt").read_bytes()
    score, note = initialize_resume_best(tmp_path, resume_checkpoint(.4, .1))
    assert score == .1 and note == "kept_existing_best_artifact"
    assert (tmp_path/"best.pt").read_bytes() == before


def test_completed_resume_exits_before_dataset_or_model_and_does_not_update(tmp_path, monkeypatch, capsys):
    import torch
    from ebim_phase2 import train as training
    dataset = tmp_path/"dataset"; dataset.mkdir()
    (dataset/"manifest.json").write_text('{}')
    checkpoint = {**resume_checkpoint(.2, .2), "dataset_sha256": training.sha256(dataset/"manifest.json")}
    resume = tmp_path/"source.pt"; torch.save(checkpoint, resume)
    output = tmp_path/"resumed"
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed resume must not construct training data/model")
    monkeypatch.setattr(training, "EpisodeDataset", forbidden)
    monkeypatch.setattr(training, "ACT", forbidden)
    monkeypatch.setattr("sys.argv", ["train", "--dataset", str(dataset), "--output", str(output),
                                     "--task", "1", "--steps", "10", "--device", "cpu", "--resume", str(resume)])
    training.main()
    message = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert message["optimizer_updates"] == 0 and message["step"] == 10
    assert torch.load(output/"best.pt", weights_only=False)["model"] == checkpoint["model"]
