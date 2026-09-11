"""Competition stage programs; geometric poses and outcomes come from sensors.

Pose keys refer to *TCP* goals after calibrated tool/grasp offsets, not object
centers. They can be populated by the helper in perception.py or a site RGB-D
tracker. Each transfer gets independent evidence; no elapsed-time success.
"""

from __future__ import annotations

from .control import Target, Waypoint

OBJECTS = ("plate", "cup", "bowl", "spoon")


def _pair(prefix):
    return {arm: Target(f"{prefix}_{arm}") for arm in ("left", "right")}


def task2_plan(*, peel_required: bool = False, arms: tuple[str, ...] = ('right',)) -> tuple[Waypoint, ...]:
    """Secure→optional peel→carry→align→place sequence for one thermal pad.

    The supplied release's task2_real demos can train the shared ACT policy;
    this geometric controller is a separately usable, calibrated baseline.
    Enable ``peel_required`` only when the actual site presents an attached pad.
    The Phase II PDF defines pick-and-place; peeling is in the broader website
    description and is not assumed to be required at every Phase II site.
    Default to one arm, as seen in the Munich RAM-pad recordings. A bimanual
    grasp is an explicit site choice, not a Phase II scoring requirement.
    """
    if not arms or len(set(arms)) != len(arms) or set(arms)-{'left', 'right'}:
        raise ValueError('Select one or both distinct robot arms')
    targets = lambda prefix: {arm: Target(f'{prefix}_{arm}') for arm in arms}
    phase = "thermal_pad"
    opened, closed = {arm: 'open' for arm in arms}, {arm: 'closed' for arm in arms}
    grasped = {f'pad_grasped_{arm}': True for arm in arms}
    plan = [
        Waypoint("pad_pregrasp", phase, targets("pad_pregrasp"), opened,
                 require={"pad_detected": True, "target_detected": True, "pad_torn": False}),
        Waypoint("pad_grasp", phase, targets("pad_grasp"), closed,
                 require=grasped,
                 gripper_feedback=False),
    ]
    intact = {**grasped, 'pad_torn': False}
    if peel_required:
        plan.append(Waypoint("pad_peel", phase, targets("pad_peel"), closed,
                             require={"pad_detached": True},
                             require_while={**intact, "peel_load_ok": True},
                             gripper_feedback=False, timeout_seconds=45.))
    plan.extend((
        Waypoint("pad_lift", phase, targets("pad_lift"), closed,
                 require={"pick_success": True}, require_while=intact, gripper_feedback=False),
        Waypoint("pad_align", phase, targets("pad_hover"), closed,
                 require={"pad_orientation_correct": True}, require_while=intact, gripper_feedback=False),
        Waypoint("pad_lower", phase, targets("pad_place"), closed,
                 require={"pad_supported_by_target": True},
                 require_while={**intact, "pad_orientation_correct": True, "placement_load_ok": True},
                 gripper_feedback=False),
        Waypoint("pad_release", phase, targets("pad_place"), opened,
                 require={"pad_released": True}, require_while={"pad_supported_by_target": True}),
        Waypoint("pad_retract", phase, targets("pad_hover"), opened,
                 require={"pad_released": True}),
        Waypoint("pad_verify", phase, require={"pick_success": True, "pad_orientation_correct": True,
                                               "pad_torn": False, "placement_verified": True},
                 hold_seconds=0.5, timeout_seconds=10.),
    ))
    return tuple(plan)


def _transfer(name: str, stage: str, source: str, destination: str, station: str,
              *, arm: str = "right") -> list[Waypoint]:
    holding = {f"holding_{name}": True}
    return [
        Waypoint(f"{name}_navigate_pick", stage, navigate=source+"_station",
                 require_while={"arms_stowed": True}, timeout_seconds=120.),
        Waypoint(f"{name}_approach", stage, {arm: Target(source+"_pregrasp")}, {arm: "open"}),
        Waypoint(f"{name}_grasp", stage, {arm: Target(source+"_grasp")}, {arm: "closed"},
                 require=holding, gripper_feedback=False),
        Waypoint(f"{name}_lift", stage, {arm: Target(source+"_lift")}, {arm: "closed"},
                 require=holding, require_while=holding, gripper_feedback=False),
        Waypoint(f"{name}_stow_carry", stage, {arm: Target(f"carry_{arm}")},
                 require={"payload_stowed": True}, require_while=holding),
        Waypoint(f"{name}_navigate_place", stage, navigate=station,
                 require_while={**holding, "payload_stowed": True}, timeout_seconds=180.),
        Waypoint(f"{name}_place_hover", stage, {arm: Target(destination+"_hover")},
                 require_while=holding),
        Waypoint(f"{name}_place", stage, {arm: Target(destination+"_place")},
                 require={f"{name}_supported": True}, require_while=holding),
        Waypoint(f"{name}_release", stage, grippers={arm: "open"},
                 require={f"{name}_released": True}, require_while={f"{name}_supported": True}),
        Waypoint(f"{name}_retract", stage, {arm: Target(f"stow_{arm}")},
                 require={f"{destination}_verified": True, "arms_stowed": True}, hold_seconds=0.3),
    ]


def task3_plan(assignments: dict[str, str], *, bean_pour_tilt_key: str = "recovery_pour") -> tuple[Waypoint, ...]:
    """Four-stage assisted living program, with three organizer-assigned seats.

    ``assignments`` maps plate/cup/bowl/spoon to seats, bowl and spoon share a
    seat. Pose targets, force limits, grasp offsets and spoon trajectory must
    be calibrated; this function contains no hard-coded room/head coordinates.
    """
    if set(assignments) != set(OBJECTS) or any(not isinstance(x, str) or not x for x in assignments.values()):
        raise ValueError("assignments must contain plate, cup, bowl, spoon seat labels")
    if assignments["bowl"] != assignments["spoon"] or len(set(assignments.values())) != 3:
        raise ValueError("three distinct assigned seats required; bowl and spoon share one")
    plan: list[Waypoint] = []
    stage = "table_setup"
    # Unstack small objects before the supporting plate.
    for name in ("cup", "spoon", "bowl", "plate"):
        seat = assignments[name]
        plan.extend(_transfer(name, stage, f"kitchen_{name}", f"seat_{seat}_{name}", f"seat_{seat}_station"))
    stage, seat = "feeding", assignments["bowl"]
    plan.extend((
        Waypoint("feeding_navigate", stage, navigate=f"seat_{seat}_station",
                 require_while={"arms_stowed": True}, timeout_seconds=180.),
        Waypoint("feeding_approach", stage, _pair("feeding_pregrasp"), {"left": "open", "right": "open"}),
        Waypoint("feeding_grasp", stage, _pair("feeding_grasp"), {"left": "closed", "right": "closed"},
                 require={"bowl_stabilized": True, "holding_spoon": True}, gripper_feedback=False),
        Waypoint("scoop_entry", stage, _pair("scoop_entry"),
                 require_while={"bowl_stabilized": True, "holding_spoon": True}),
        Waypoint("scoop_sweep", stage, _pair("scoop_exit"),
                 require_while={"bowl_stabilized": True, "holding_spoon": True, "scoop_load_ok": True}),
        Waypoint("scoop_lift", stage, _pair("scoop_lift"),
                 require={"spoon_has_beans": True}, require_while={"bowl_stabilized": True, "holding_spoon": True}),
        Waypoint("feed_approach", stage, {"right": Target("head_feed_hover")},
                 require_while={"spoon_has_beans": True, "head_motion_clear": True, "bowl_stabilized": True}, near_head=True),
        Waypoint("feed_hold", stage, {"right": Target("head_feed_goal")},
                 require={"spoon_in_feed_zone": True, "spoon_has_beans": True},
                 require_while={"spoon_has_beans": True, "head_motion_clear": True, "bowl_stabilized": True},
                 hold_seconds=3.2, timeout_seconds=20., near_head=True),
        Waypoint("feed_retract", stage, {"right": Target("head_feed_hover")},
                 require_while={"head_motion_clear": True}, near_head=True),
        Waypoint("feed_return", stage, _pair("feeding_return"),
                 require={"beans_returned": True}, require_while={"bowl_stabilized": True}),
        Waypoint("feeding_replace", stage, _pair("feeding_grasp"),
                 require={"bowl_supported": True, "spoon_supported": True}),
        Waypoint("feeding_release", stage, grippers={"left": "open", "right": "open"},
                 require={"bowl_released": True, "spoon_released": True}),
        Waypoint("feeding_stow", stage, _pair("stow"), require={"arms_stowed": True}),
    ))
    stage = "bean_recovery"
    plan.extend((
        Waypoint("recovery_navigate_pick", stage, navigate=f"seat_{seat}_station",
                 require_while={"arms_stowed": True}, timeout_seconds=180.),
        Waypoint("recovery_approach", stage, _pair("bowl_pregrasp"), {"left": "open", "right": "open"}),
        Waypoint("recovery_grasp", stage, _pair("bowl_grasp"), {"left": "closed", "right": "closed"},
                 require={"bowl_grasped_bimanual": True}, gripper_feedback=False),
        Waypoint("recovery_lift", stage, _pair("bowl_lift"), require_while={"bowl_grasped_bimanual": True}),
        Waypoint("recovery_stow_carry", stage, _pair("bowl_carry"),
                 require={"payload_stowed": True}, require_while={"bowl_grasped_bimanual": True}),
        Waypoint("recovery_navigate_bin", stage, navigate="recycling_station", timeout_seconds=180.,
                 require_while={"bowl_grasped_bimanual": True, "payload_stowed": True}),
        Waypoint("recovery_align", stage, _pair("recovery_upright"),
                 require={"bin_aligned": True}, require_while={"bowl_grasped_bimanual": True}),
        Waypoint("recovery_pour", stage, _pair(bean_pour_tilt_key),
                 require={"pour_complete": True}, require_while={"bowl_grasped_bimanual": True, "bin_aligned": True},
                 hold_seconds=0.5, timeout_seconds=30.),
        Waypoint("recovery_upright", stage, _pair("recovery_upright"),
                 require={"recovered_mass_stable": True}, require_while={"bowl_grasped_bimanual": True}),
        Waypoint("recovery_set_down", stage, _pair("recovery_setdown"), require={"bowl_supported": True}),
        Waypoint("recovery_release", stage, grippers={"left": "open", "right": "open"},
                 require={"bowl_released": True}),
        Waypoint("recovery_stow", stage, _pair("stow"), require={"arms_stowed": True}),
    ))
    stage = "cleanup"
    for name in ("bowl", "spoon", "cup", "plate"):
        source = "recovery_bowl" if name == "bowl" else f"seat_{assignments[name]}_{name}"
        plan.extend(_transfer(name, stage, source, f"sink_{name}", "sink_station"))
    return tuple(plan)
