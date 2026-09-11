import json
from dataclasses import replace

import numpy as np
import pytest

from ebim_phase2.tasks.control import Controller, Estimate, Observation, ServoCalibration, Target, Waypoint
from ebim_phase2.tasks.geometry import Pose, Rectangle, pose_error, route
from ebim_phase2.tasks.plans import task2_plan, task3_plan
from ebim_phase2.tasks.perception import RGBDCalibration, pad_goals, project_keypoint


def calibration(**overrides):
    fields = dict(calibration_id="synthetic-unit-test-only", joint_min=(-3.,)*14, joint_max=(3.,)*14,
                  gripper_open_rad=(1., 1.), gripper_closed_rad=(0., 0.),
                  gripper_open_command=(1., 1.), gripper_closed_command=(0., 0.))
    return ServoCalibration(**{**fields, **overrides})


def obs(t=0., *, signals=None, poses=None, **overrides):
    values = {"collision_clear": True, **(signals or {})}
    return Observation(t, np.zeros(62),
                       poses={k: Estimate(v, t, 1., "rgbd") for k, v in (poses or {}).items()},
                       signals={k: Estimate(v, t, 1., "force" if k == "head_force_n" else "vision") for k, v in values.items()},
                       **overrides)



def replay_step(controller, observation, *, now=None):
    """Synthetic fixtures explicitly select their replay clock."""
    return controller.step(observation, now=observation.timestamp if now is None else now)


def test_python_api_uses_receiver_clock_by_default(monkeypatch):
    monkeypatch.setattr('ebim_phase2.tasks.control.time.time',lambda:10.)
    controller=Controller((Waypoint('wait','test',require={'ready':True}),),calibration())
    stale=controller.step(obs(0.,signals={'ready':True}))
    assert stale.action is None and 'stale' in stale.reason
    fresh=controller.step(obs(10.,signals={'ready':True}))
    assert fresh.action is not None and fresh.status=='ACTIVE'


def test_dls_actuates_only_measured_arm_with_bounded_joint_steps():
    plan = (Waypoint("move", "test", {"left": Target("goal")}),)
    controller = Controller(plan, calibration())
    poses = {"left_tcp": Pose((0, 0, 0), (1, 0, 0, 0)), "goal": Pose((.2, 0, 0), (1, 0, 0, 0))}
    replay_step(controller, obs(0., poses=poses, jacobians={"left": np.eye(6, 7)}))
    decision = replay_step(controller, obs(.05, poses=poses, jacobians={"left": np.eye(6, 7)}))
    assert decision.status == "ACTIVE"
    assert 0 < decision.action[0] <= .4*.05
    assert np.count_nonzero(decision.action) == 1
    assert np.array_equal(decision.action[16:22], np.zeros(6))


def test_missing_pose_or_collision_clearance_never_moves():
    controller = Controller((Waypoint("move", "test", {"left": Target("goal")}),), calibration())
    decision = replay_step(controller, obs())
    assert decision.status == "WAITING" and "missing" in decision.reason
    assert np.array_equal(decision.action, np.zeros(23))
    decision = replay_step(controller, obs(.05, signals={"collision_clear": False}))
    assert decision.status == "WAITING" and "collision" in decision.reason


def test_timeout_retries_once_then_fails_without_fake_completion():
    waypoint = Waypoint("verify", "test", require={"placed": True}, hold_seconds=.1,
                        timeout_seconds=.2, max_retries=1)
    controller = Controller((waypoint,), calibration())
    assert replay_step(controller, obs(0., signals={"placed": False})).status == "ACTIVE"
    assert replay_step(controller, obs(.25, signals={"placed": False})).status == "RETRY"
    failed = replay_step(controller, obs(.50, signals={"placed": False}))
    assert failed.status == "FAILED" and "exhausted" in failed.reason
    assert replay_step(controller, obs(.55, signals={"placed": True})).status == "FAILED"
    controller.reset()
    assert replay_step(controller, obs(0., signals={"placed": True})).status == "ACTIVE"


def test_continuous_hold_resets_on_payload_loss_and_duplicate_frames():
    waypoint = Waypoint("feed", "feeding", require={"loaded": True}, hold_seconds=.3)
    controller = Controller((waypoint,), calibration())
    for t in (0., .1, .2):
        assert replay_step(controller, obs(t, signals={"loaded": True})).status == "ACTIVE"
    assert replay_step(controller, obs(.25, signals={"loaded": False})).status == "ACTIVE"
    assert replay_step(controller, obs(.3, signals={"loaded": True})).status == "ACTIVE"
    assert replay_step(controller, obs(.3, signals={"loaded": True})).status == "WAITING"
    for t in (.4, .5, .6):
        assert replay_step(controller, obs(t, signals={"loaded": True})).status == "ACTIVE"
    assert replay_step(controller, obs(.71, signals={"loaded": True})).status == "COMPLETE"


def test_stale_perception_and_clock_gaps_do_not_count_as_hold():
    controller = Controller((Waypoint("verify", "test", require={"placed": True}, hold_seconds=.3),), calibration())
    old = obs(0., signals={"placed": True})
    stale = replay_step(controller, old, now=1.)
    assert stale.status == "WAITING" and stale.action is None
    assert stale.to_dict()['action'] is None
    stale_signals = {**obs(.5).signals, "placed": Estimate(True, 0., 1., "vision")}
    assert replay_step(controller, replace(obs(.5), signals=stale_signals)).status == "WAITING"
    assert replay_step(controller, obs(.6, signals={"placed": True})).status == "ACTIVE"
    assert replay_step(controller, obs(1., signals={"placed": True})).status == "ACTIVE"
    assert controller.stable_seconds == 0.


def test_servo_rate_uses_wall_clock_and_rejects_stale_terminal_state():
    plan = (Waypoint('move', 'test', {'left': Target('goal')}),)
    controller = Controller(plan, calibration())
    poses = {'left_tcp': Pose((0,0,0), (1,0,0,0)), 'goal': Pose((.2,0,0), (1,0,0,0))}
    replay_step(controller, obs(9.85, poses=poses, jacobians={'left': np.eye(6,7)}), now=10.)
    decision = replay_step(controller, obs(9.9, poses=poses, jacobians={'left': np.eye(6,7)}), now=10.01)
    assert 0 < decision.action[0] <= calibration().max_joint_speed*.01
    repeated = replay_step(controller, obs(9.9, poses=poses), now=10.02)
    assert repeated.action is None
    controller.status = 'COMPLETE'
    assert replay_step(controller, obs(9.95), now=11.).action is None


def test_cli_uses_host_clock_by_default_for_stale_state(tmp_path, monkeypatch, capsys):
    from ebim_phase2.tasks.__main__ import main
    import io
    path = tmp_path/'calibration.json'
    path.write_text(json.dumps(vars(calibration())))
    monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps({'timestamp': 0, 'state': [0]*62, 'now': 0})+'\n'))
    assert main(['--task', 'task2', '--calibration', str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['action'] is None and result['reason'] == 'stale robot observation'


def test_rejects_ground_truth_and_invalid_geometry():
    with pytest.raises(ValueError, match="simulator truth"):
        Estimate(True, 0., 1., "simulator_ground_truth")
    with pytest.raises(ValueError):
        Pose((0, 0, 0), (0, 0, 0, 0))
    with pytest.raises(ValueError):
        Observation(0., np.zeros(61))
    with pytest.raises(ValueError):
        obs(jacobians={"left": np.full((6, 7), np.nan)})


def test_gripper_conversion_uses_explicit_site_endpoints_and_preserves_last_command():
    config = calibration(gripper_open_rad=(.7, .7), gripper_closed_rad=(-.1, -.1),
                         gripper_open_command=(0., 0.), gripper_closed_command=(1., 1.))
    assert config.gripper_command("left", .7) == pytest.approx(0.)
    assert config.gripper_command("left", -.1) == pytest.approx(1.)
    controller = Controller((Waypoint("grasp", "test", grippers={"left": "closed"},
                                      require={"holding": True}, gripper_feedback=False),), config)
    first = replay_step(controller, obs(0., signals={"holding": False}))
    second = replay_step(controller, obs(.05, signals={"holding": False}))
    assert second.action[7] > first.action[7]
    waiting = replay_step(controller, obs(.1, signals={"collision_clear": False}))
    assert waiting.action[7] == second.action[7]


def test_native_task2_state_layout_requires_real_spine_and_correct_gripper_indices():
    state = np.zeros(42)
    state[7], state[28], state[20], state[41] = .25, .75, 9., 9.
    controller = Controller((Waypoint("hold", "thermal_pad", require={"ready": True}),), calibration())
    native = Observation(.0, state, state_profile=2, spine_height_m=.12)
    held = controller.hold(native)
    assert held[7] == .25 and held[15] == .75 and held[22] == .12
    assert not np.any(held[16:22])
    with pytest.raises(ValueError, match="separately measured"):
        Observation(0., state, state_profile=2)


def test_near_head_requires_measured_force_and_site_limit():
    waypoint = Waypoint("feed", "feeding", near_head=True)
    controller = Controller((waypoint,), calibration())
    assert "force-limit calibration" in replay_step(controller, obs()).reason
    controller = Controller((waypoint,), calibration(head_force_limit_n=10.))
    assert replay_step(controller, obs()).status == "WAITING"
    decision = replay_step(controller, obs(.05, signals={"head_force_n": 10.}))
    assert decision.status == "FAILED" and "force" in decision.reason
    assert not np.any(decision.action[16:22])


def test_navigation_requires_current_stage_map_and_replans_changed_obstacles():
    obstacle = Rectangle((.8, -.1), (1.2, .1))
    path = route((0, 0), (2, 0), (obstacle,), .2)
    assert len(path) > 2
    assert all(not obstacle.intersects(a, b, .2) for a, b in zip(path, path[1:]))
    controller = Controller((Waypoint("navigate", "cleanup", navigate="goal"),),
                            calibration(base_radius=.1, payload_radius=.1))
    poses = {"goal": Pose((2, 0, 0), (1, 0, 0, 0))}
    kwargs = dict(poses=poses, base_xyyaw=(0, 0, 0), obstacles=(obstacle,), obstacle_timestamp=0.)
    assert replay_step(controller, obs(0., obstacle_stage="feeding", **kwargs)).status == "WAITING"
    decision = replay_step(controller, obs(.05, obstacle_stage="cleanup", **kwargs))
    assert decision.status == "ACTIVE" and decision.action[17] != 0.
    changed = (Rectangle((-.1, -.1), (.1, .1)),)
    decision = replay_step(controller, obs(.1, poses=poses, base_xyyaw=(0, 0, 0), obstacles=changed,
                                    obstacle_timestamp=.1, obstacle_stage="cleanup"))
    assert decision.status == "WAITING" and not np.any(decision.action[16:22])


def test_task_plans_follow_phase2_assignments_and_optional_peeling():
    assert "pad_peel" not in [x.name for x in task2_plan()]
    assert "pad_peel" in [x.name for x in task2_plan(peel_required=True)]
    seats = {"plate": "a", "cup": "b", "bowl": "c", "spoon": "c"}
    plan = task3_plan(seats)
    assert list(dict.fromkeys(x.stage for x in plan)) == ["table_setup", "feeding", "bean_recovery", "cleanup"]
    feed = next(x for x in plan if x.name == "feed_hold")
    assert feed.hold_seconds >= 3. and feed.near_head
    assert all("tray" not in x.name for x in plan)
    with pytest.raises(ValueError, match="three distinct"):
        task3_plan({**seats, "spoon": "d"})
    torn = replay_step(Controller(task2_plan(), calibration()), obs(signals={"pad_torn": True}))
    assert torn.status == "FAILED"


def test_depth_projection_and_world_vertical_goal_construction():
    camera = RGBDCalibration(100., 100., 2., 2., .001,
                             Pose((1., 2., 3.), (1., 0., 0., 0.)), "test-camera")
    depth = np.full((5, 5), 1000, dtype=np.uint16)
    assert project_keypoint(depth, (2, 2), camera) == pytest.approx((1., 2., 4.))
    depth[2, 2] = 500
    with pytest.raises(ValueError, match="discontinuity"):
        project_keypoint(depth, (2, 2), camera)
    down = Pose((0., 0., .5), (0., 1., 0., 0.))
    goals = pad_goals({"left": down, "right": down}, {"left": down, "right": down})
    assert goals["pad_lift_left"].xyz[2] == pytest.approx(.55)
    assert np.allclose(pose_error(down, Pose(down.xyz, tuple(-np.array(down.wxyz)))), 0.)


def test_jsonl_cli_rejects_invalid_input_without_emitting_action(tmp_path, monkeypatch, capsys):
    from ebim_phase2.tasks.__main__ import main
    import io
    path = tmp_path/"calibration.json"
    path.write_text(json.dumps(vars(calibration())))
    monkeypatch.setattr("sys.stdin", io.StringIO('{"timestamp":0,"state":[0]}\nnull\n[]\n42\n{"reset":true}\n'))
    assert main(["--task", "task2", "--calibration", str(path)]) == 0
    rows = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert all(row['action'] is None and row['status']=='INVALID_OBSERVATION' for row in rows[:4])
    assert rows[4]["status"] == "RESET"


def test_complete_task2_program_in_synthetic_kinematic_feedback_fixture():
    """An action/state integration test, not a deformable-physics success test."""
    plan = task2_plan()
    controller = Controller(plan, calibration())
    state = np.zeros(62)
    state[20] = state[41] = 1.
    key_goals = {}
    for item in plan:
        for arm, target in item.targets.items():
            key_goals[target.key] = Pose((.02, .01 if arm == "left" else -.01, .03), (1., 0., 0., 0.))
    seen = set()
    for tick in range(1000):
        t = tick*.05
        waypoint = plan[min(controller.index, len(plan)-1)]
        signals = {"collision_clear": True, "pad_torn": False,
                   **waypoint.require, **waypoint.require_while}
        poses = {**key_goals,
                 "left_tcp": Pose(tuple(state[:3]), (1., 0., 0., 0.)),
                 "right_tcp": Pose(tuple(state[21:24]), (1., 0., 0., 0.))}
        observation = replace(obs(t, signals=signals, poses=poses,
                                  jacobians={"left": np.eye(6, 7), "right": np.eye(6, 7)}), state=state.copy())
        decision = replay_step(controller, observation)
        assert decision.status in ("ACTIVE", "COMPLETE")
        assert decision.action.shape == (23,) and np.isfinite(decision.action).all()
        assert np.all(decision.action[16:22] == 0.)
        seen.add(decision.waypoint)
        state[:7], state[21:28] = decision.action[:7], decision.action[8:15]
        state[20], state[41] = decision.action[7], decision.action[15]
        if decision.status == "COMPLETE":
            break
    assert controller.status == "COMPLETE"
    assert seen == {item.name for item in plan}


def test_phase2_pad_default_does_not_require_an_undemonstrated_second_grasp():
    plan = task2_plan()
    assert all('left' not in waypoint.targets and 'left' not in waypoint.grippers for waypoint in plan)
    assert all('pad_grasped_left' not in waypoint.require and 'pad_grasped_left' not in waypoint.require_while for waypoint in plan)
    both = task2_plan(arms=('left','right'))
    assert both[1].require == {'pad_grasped_left': True, 'pad_grasped_right': True}
    pose = Pose((0,0,.5), (1,0,0,0))
    goals = pad_goals({'right': pose}, {'right': pose})
    assert 'pad_place_right' in goals and 'pad_place_left' not in goals


def test_complete_task3_program_with_synthetic_feedback_and_interrupted_outcomes():
    """Exercise all stages with invented Cartesian actuators and scripted sensors.

    This is a controller protocol fixture, not physics, recorded competition
    data, or evidence that any manipulation/force/perception skill works. The
    scripted predicates below are independent of the waypoint requirements.
    """
    assignments = {'plate': 'a', 'cup': 'b', 'bowl': 'c', 'spoon': 'c'}
    plan = task3_plan(assignments)
    assert len(plan) == 105
    config = calibration(head_force_limit_n=5., base_radius=.05, payload_radius=.02)
    controller = Controller(plan, config)
    state, base = np.zeros(62), np.zeros(3)
    state[20] = state[41] = 1.
    identity = (1., 0., 0., 0.)
    goals = {}
    for waypoint in plan:
        for arm, target in waypoint.targets.items():
            # Each arm has three Cartesian coordinates and one yaw coordinate.
            # Small distinct targets force measured convergence between skills.
            index = len(goals)
            angle = .2 if target.key.startswith('recovery_pour_') else 0.
            goals[target.key] = Pose((.12 + .01*(index % 3),
                                     .025 if arm == 'left' else -.025,
                                     .04 + .005*(index % 4)),
                                    (np.cos(angle/2), 0., 0., np.sin(angle/2)))
        if waypoint.navigate and waypoint.navigate not in goals:
            index = len(goals)
            angle = .15*(index % 3)
            goals[waypoint.navigate] = Pose((.04*(index % 5), .03*(index % 4), 0.),
                                             (np.cos(angle/2), 0., 0., np.sin(angle/2)))

    def feedback(t, *, loaded=True, returned=True, poured=True, mass_stable=True,
                 cleanup_verified=True, map_stage):
        # These are deliberately synthetic sensor responses. Contact is gated
        # by the fixture's measured gripper travel, not by a waypoint advancing.
        left_contact, right_contact = state[20] < .2, state[41] < .2
        signals = dict(collision_clear=True, arms_stowed=True, payload_stowed=True,
                       bowl_stabilized=left_contact, scoop_load_ok=True,
                       spoon_has_beans=loaded, head_motion_clear=True,
                       spoon_in_feed_zone=True, head_force_n=.2,
                       beans_returned=returned, bowl_grasped_bimanual=left_contact and right_contact,
                       bin_aligned=True, pour_complete=poured, recovered_mass_stable=mass_stable)
        for name, seat in assignments.items():
            signals.update({f'holding_{name}': right_contact, f'{name}_supported': True,
                            f'{name}_released': True, f'seat_{seat}_{name}_verified': True,
                            f'sink_{name}_verified': cleanup_verified})
        poses = dict(goals)
        for arm, start in (('left', 0), ('right', 21)):
            q = state[start:start+7]
            poses[f'{arm}_tcp'] = Pose(tuple(q[:3] + np.r_[base[:2], 0.]),
                                      (np.cos(q[5]/2), 0., 0., np.sin(q[5]/2)))
        return replace(obs(t, signals=signals, poses=poses,
                           jacobians={'left': np.eye(6, 7), 'right': np.eye(6, 7)},
                           base_xyyaw=tuple(base), obstacle_timestamp=t,
                           obstacle_stage=map_stage,
                           obstacles=(Rectangle((2., 2.), (3., 3.)),)), state=state.copy())

    visited, stage_order, refreshed_stages = set(), [], set()
    stage, map_stage, map_delay = None, 'uninitialized', 0
    entry_ticks = {}
    feed_interrupted, loss_ticks, load_resumed_at = False, 0, None
    feed_completed_at = None
    retries, navigation_commands, blocked_returns, blocked_mass, blocked_cleanup = 0, 0, 0, 0, set()
    for tick in range(12000):
        t = tick*.05
        index = controller.index
        waypoint = plan[index]
        visited.add(index)
        entry_ticks.setdefault(index, tick)
        elapsed_ticks = tick-entry_ticks[index]
        if waypoint.stage != stage:
            stage = waypoint.stage
            stage_order.append(stage)
            map_delay = 3
        stale_stage_map = bool(map_delay)
        if not stale_stage_map:
            map_stage = stage

        if waypoint.name == 'feed_hold' and controller.stable_seconds >= 1. and not feed_interrupted:
            feed_interrupted, loss_ticks = True, 3
        loaded = loss_ticks == 0
        if waypoint.name == 'feed_hold' and feed_interrupted and loaded and load_resumed_at is None:
            load_resumed_at = t
        returned = waypoint.name != 'feed_return' or elapsed_ticks >= 12
        # Withhold the recovery outcome through the original 30-second timeout.
        # One retry must reacquire evidence and then finish the same waypoint.
        poured = waypoint.name != 'recovery_pour' or controller.retries.get(index, 0) == 1
        mass_stable = waypoint.name != 'recovery_upright' or elapsed_ticks >= 10
        cleanup_verified = waypoint.stage != 'cleanup' or not waypoint.name.endswith('_retract') or elapsed_ticks >= 12
        observation = feedback(t, loaded=loaded, returned=returned, poured=poured,
                               mass_stable=mass_stable, cleanup_verified=cleanup_verified,
                               map_stage=map_stage)
        decision = replay_step(controller, observation)
        assert decision.status in ('ACTIVE', 'WAITING', 'RETRY', 'COMPLETE'), (index, decision)
        assert decision.action is not None and decision.action.shape == (23,)
        assert np.isfinite(decision.action).all()
        assert np.max(np.abs(decision.action[:7]-state[:7])) <= config.max_joint_speed*.05 + 1e-9
        assert np.max(np.abs(decision.action[8:15]-state[21:28])) <= config.max_joint_speed*.05 + 1e-9
        assert np.linalg.norm(decision.action[16:18]) <= config.max_base_speed + 1e-9
        assert abs(decision.action[21]) <= config.max_base_yaw_speed + 1e-9
        assert np.all(decision.action[[18, 19, 20, 22]] == 0.)
        if stale_stage_map:
            assert waypoint.navigate and decision.status == 'WAITING'
            assert 'refreshed for this stage' in decision.reason
            assert controller.index == index and not np.any(decision.action[16:22])
            map_delay -= 1
            if map_delay == 0:
                refreshed_stages.add(stage)
        if not loaded:
            assert decision.status == 'WAITING' and controller.index == index
            assert controller.stable_seconds == 0.
            loss_ticks -= 1
        if not returned:
            assert controller.index == index
            blocked_returns += 1
        if not mass_stable:
            assert controller.index == index
            blocked_mass += 1
        if not cleanup_verified:
            assert controller.index == index
            blocked_cleanup.add(waypoint.name)
        if decision.status == 'RETRY':
            assert waypoint.name == 'recovery_pour' and controller.index == index
            retries += 1
        if controller.index != index and waypoint.name == 'feed_hold':
            assert load_resumed_at is not None and t-load_resumed_at >= waypoint.hold_seconds
            feed_completed_at = t
        if waypoint.name == 'feed_return':
            assert feed_completed_at is not None and t > feed_completed_at
        if np.any(decision.action[16:22]):
            assert waypoint.navigate and map_stage == stage
            navigation_commands += 1

        # Apply the command to the invented kinematics; never teleport to goals.
        state[:7], state[21:28] = decision.action[:7], decision.action[8:15]
        state[20], state[41] = decision.action[7], decision.action[15]
        c, s = np.cos(base[2]), np.sin(base[2])
        body_vx, body_vy = decision.action[16:18]
        base[:2] += .05*np.array([c*body_vx-s*body_vy, s*body_vx+c*body_vy])
        base[2] += .05*decision.action[21]
        if decision.status == 'COMPLETE':
            break
    assert controller.status == 'COMPLETE', (controller.index, plan[controller.index].name)
    assert visited == set(range(len(plan)))
    assert stage_order == ['table_setup', 'feeding', 'bean_recovery', 'cleanup']
    assert refreshed_stages == set(stage_order)
    assert navigation_commands > 0 and feed_interrupted and feed_completed_at is not None
    assert retries == 1 and blocked_returns == 12 and blocked_mass == 10
    assert blocked_cleanup == {f'{name}_retract' for name in assignments}
