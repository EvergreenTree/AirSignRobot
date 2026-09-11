# AirSign — Phase II Task 2 and Task 3

This is new Phase II code. The older Task 3 project was read as a reference;
its simulator-truth policy and old trajectory-release assumptions are not
imported. Task 2 real demonstrations are now available at
[ebim_task2_realrobotdata](https://huggingface.co/datasets/ebim-benchmark/ebim_task2_realrobotdata).
The organizers will not provide Task 3 trajectories, according to the user's
current information. This implementation does not wait for that release and
does not relabel Task 1/2 demonstrations as Task 3 demonstrations.

## Rules researched before implementation

The [official Rulebook 1.0 PDF](https://github.com/EBiM-Benchmark/ebim-benchmark.github.io/blob/f92510f87c71ed4c975e9a571bcbbd288913cf5d/src/docs/Autonomous_Robot_Benchmark_Rulebook_1.0.pdf)
is the basis for scoring. The PDF was downloaded and all five pages extracted
locally on September 10, 2026. The
[current competition page](https://ebim-benchmark.github.io/competition.html)
was checked alongside the local official benchmark implementation.

| Track | Phase II definition used here |
|---|---|
| Task 2 | Pick a pad and place it at the randomly assigned target, with the correct orientation. Score = pick success × orientation success × IoU. Wrong orientation yields zero. Completion time breaks ties; raw IoU is reported separately. |
| Task 3 | Table Setup → Feeding → Bean Recovery → Cleanup, four points per stage. Plate, cup, bowl, spoon; no fifth tray point. Three assigned seats, with bowl and spoon together beside the head. Feeding requires beans in the spoon during a continuous hold of at least three seconds and their subsequent return. Recovery is **4 × recovered mass / original mass**, capped at four. Cleanup awards one point for each of the four utensils in the marked sink. |

Task 3 ranking is highest completed stage, then total score, then completion
time. The local implementation identifies the highest stage with its full
four points; it does not introduce an undocumented consecutive-stage ranking
rule. It reports individual stage evidence. Stage 1's one-point-per-object
partial-credit split is an explicit local interpretation: the PDF gives its
four-point maximum and placement requirements but does not spell out that
partial-credit table. Organizer adjudication remains authoritative.

The old simulator development helpers differ materially: their Task 3 scorer
can include the tray and use recovery bins. Those helpers are not used for
Phase II score arithmetic. Task 2's development scorer computes **bounding-box**
IoU from simulator annotators; `scoring.mask_iou` instead computes binary-mask
IoU from masks registered into the same image frame. The scorer accepts an
externally measured raw IoU too. Neither function is the organizer evaluator.

The broader website describes peeling a liner. The Phase II PDF specifies
pick-and-place without mandating a peel in every trial. Consequently peeling
is **opt-in**, based on the actual site's presented pad condition. No numeric
tear, force, or universal Task 2/3 time limit was found in that PDF. The CLI's
1,800-second timeout is a configurable local operational timeout, not a claim
about those tracks' official limits.

## Related work and resulting design choices

- [ReKep (CoRL 2024)](https://arxiv.org/abs/2409.01652) motivates sequences of
  geometric constraints and repeated observation-driven progress checks. Its
  [official implementation](https://github.com/huangwl18/ReKep) explicitly notes
  that its demo accesses simulator keypoints/SDFs and does not supply the real
  perception stack. Here, every policy estimate declares its sensor source,
  timestamp and confidence; simulator ground truth is rejected. This is a
  small hand-authored controller, not a reproduction of ReKep's VLM pipeline.
- [DeformerNet (2023/2024)](https://arxiv.org/abs/2305.04449) uses observed
  deformable shape and goal shape for iterative bimanual corrections. This
  supports checking actual pad support, orientation, loads and placement
  instead of advancing after a timed motion. DeformerNet weights and training
  are not included here; shared ACT training uses the actual Task 2 release.
- [CARBS (CoRL 2022)](https://arxiv.org/abs/2211.14652) motivates coordinated
  stabilization during acquisition and verification of the loaded spoon.
  Task 3 explicitly holds the bowl while the other arm scoops and feeds.
  Its spoon-entry/exit/tool poses are calibrated inputs, not an asserted
  universal scooping trajectory. No CARBS checkpoint is claimed.

## Implemented runtime

`ebim_phase2.tasks.Controller` executes named waypoints with a bounded
Cartesian servo. An injected IK callback may be used; otherwise, the controller
uses measured 6×7 Jacobians for damped least-squares steps. It clips Cartesian
and angular increments before IK, limits joint increments, applies calibrated
joint bounds, ramps the grippers, and emits this project's common physical
command layout.

Missing, uncertain, stale, backwards or duplicate observations cannot certify
progress. Grasp/release/outcome predicates require fresh boolean measurements.
Continuous-hold timers reset when evidence drops out. Waypoint timeouts retry
within a fixed budget and then fail; the controller does not claim a completed
skill merely because its command ran. `reset()` clears episode state.

Task 2 executes pregrasp, grasp, optional load-monitored peel, lift,
correct-face alignment, supported placement, release, retraction and outcome
verification. Material tears stop the episode. Task 3 unpacks and transfers the
four utensils, stabilizes the bowl for scooping, checks payload before entering
the head area, holds for 3.2 continuous observed seconds, retracts and returns
the beans, transports and pours the bowl, and returns the four utensils to the
sink. Object transfer uses explicit grasp, lift, stow, navigate, place, release
and verification states.

The default Task 2 program uses the right arm, matching the inspected Munich
RAM-pad recordings. The other arm holds its measured pose. Use `--pad-arms
left` or `--pad-arms both` when required by the actual site setup. Neither the
Phase II PDF nor these recordings justify making two simultaneous grasps a
universal prerequisite for pad placement.

Navigation uses a visibility graph around footprint-expanded measured
rectangles. It checks complete path segments, caches unchanged routes, and
replans when goals or obstacles change. Every stage requires a newly identified
obstacle map; stale prior-stage maps cannot actuate the base. The conservative
footprint includes payload clearance. Near the head, the base remains stopped
and a fresh force measurement plus a site-configured force limit is required.
This servo is not a substitute for the robot's low-level collision controller.

## Data and command boundary

The project's common physical command has 23 values in this exact order:

```
left_arm[7], left_gripper[1], right_arm[7], right_gripper[1],
base_linear_xyz[3], base_angular_xyz[3], spine[1]
```

Although organizer feature names say `percent`, the inspected real Task 1
release stores gripper commands on a **0–1 scale**. This implementation uses
that native scale. The calibrated open/closed command endpoints explicitly
define its direction, while separate measured-knuckle endpoints define the
radians-to-command conversion. No ×100 conversion is performed.

`Observation.state` defaults to the native Task 1 62-value state layout. Left arm
positions are `0:7`, left knuckle `20`, right positions `21:28`, right knuckle
`41`, spine `61`. Set `state_profile: 2` to use Task 2's native 42-value state
directly; a separate measured `spine_height_m` is then required because its
state has no spine feedback. No synthetic 62-value padding is needed.
Task 2 raw state is 42 values: left position `0:7`, left grip `7`, left torque
`8:15`, left wrench `15:21`, right position `21:28`, right grip `28`, right
torque `29:36`, right wrench `36:42`. Its native action is 17 values, with spine
last and no base command. Those schemas must not be treated as interchangeable.

All TCP/target poses and Jacobians use one explicitly calibrated world frame;
quaternions are `wxyz`. Jacobians map joint velocity to world-frame TCP
`[vx, vy, vz, wx, wy, wz]`. Every observation supplies fresh robot state, fresh
TCP poses, and either measured Jacobians or the site IK callback. Navigation
additionally needs `base_xyyaw`, `obstacles`, `obstacle_timestamp` and
`obstacle_stage`. Commands for the base are body-frame twists.

`RGBDCalibration` and `project_keypoint` deproject image keypoints using
registered depth, calibrated intrinsics and camera extrinsics. Depth edges or
missing depth are rejected. `pad_goals` builds fixed pregrasp/lift/place goals
from measured source and assigned-target TCP grasp poses. World +Z clearance
does not flip when the gripper points down. Snapshot these goals once per
attempt; do not continuously add a lift displacement to an already moving pad.
RGB keypoint detection, registration, face classification, grasp verification,
torque interpretation and calibration must be supplied by the site adapter.
No detector or physical validation result is fabricated.

## Running and connecting

From the project source directory:

```bash
python -m ebim_phase2.tasks --help
python -m ebim_phase2.tasks --task task2 --describe
python -m ebim_phase2.tasks --task task3 --describe
python -m ebim_phase2.tasks --task task2 --calibration /path/to/site.json < observations.jsonl > commands.jsonl
python -m ebim_phase2.tasks --task task3 --calibration /path/to/site.json --assignments /path/to/seats.json < observations.jsonl > commands.jsonl
```

Use `--peel-required` only for a site trial requiring it. `--describe` lists
every stage's required target keys and feedback predicates. Assignments JSON
has exactly `plate`, `cup`, `bowl`, `spoon`, with three distinct seat labels and
the last two sharing the organizer's head-adjacent seat.

Required calibration JSON fields are `calibration_id`, `joint_min` and
`joint_max` (14 each, left seven then right seven), `gripper_open_rad`,
`gripper_closed_rad`, `gripper_open_command`, `gripper_closed_command` (two each).
Task 3 near-head control also requires `head_force_limit_n` from the site's
approved controller configuration. Optional servo/footprint limits are listed
in `ServoCalibration`; the defaults are conservative development settings and
must be checked against the actual site calibration.

Each input JSON line has `timestamp`, `state`, `poses`, `signals` and optional
`jacobians`/navigation fields. A pose entry contains `xyz`, `wxyz`, `timestamp`,
`confidence`, `source`. A signal entry contains `value`, `timestamp`,
`confidence`, `source`. Allowed sources identify RGB-D/vision, robot state,
tactile/force/scale, calibration, localization or LiDAR. Static calibrated
target goals may use source `calibration`; measured predicates may not use
that exemption to bypass freshness. Send `{"reset": true}` between episodes.

Outputs include `action`, `status`, `stage`, `waypoint`, `reason` and `retries`.
`WAITING` with fresh robot state holds measured arm positions, retains the
previous gripper commands, holds the spine and stops the base. Stale, duplicate
or backwards robot/command clocks emit a null action. Invalid robot-state messages emit
`INVALID_OBSERVATION` with `action: null`; the transport must enforce its
deadman watchdog. The CLI emits JSONL and does not itself connect to hardware.
An integration can instead instantiate the controller directly. Its output
already contains physical joint coordinates and spine metres; forward it
through the site's transport and command limits without applying the learned
policy's native GELLO-to-robot conversion again. This common layout is not a
verified Task 3 organizer transport schema.

The JSONL CLI checks the host wall clock by default, so incoming timestamps
must use that clock domain. `--replay-clock` is only for offline fixtures.
The Python `Controller.step` API also uses the receiver's wall clock by default.
Synthetic tests may pass an explicit replay `now`; physical integrations must
use the actual receiver clock.

Task 2's official scene requires Isaac Sim 5.1/PhysX GPU deformables; a rigid
MuJoCo proxy is not an equivalent thermal-pad dynamics validation. Task 3 can
be evaluated in the current official Isaac environment once its perception and
actuator bindings are calibrated. This addition does not relaunch or reuse the
old Phase I simulator-truth controller.

## Validation and remaining physical work

The targeted tests exercise actual action layout, bounded DLS motion, measured
gripper conversion/hold behavior, stale observations, repeated frames,
continuous feed timing, retries, timeouts, measured-force stops, route
collision geometry, stage-map changes, RGB-D projection, world-frame pad
goals, JSONL input handling and Phase II score arithmetic. Test calibration and
sensor data are synthetic unit-test fixtures, not competition results.

A complete Task 3 integration test executes all 105 waypoints using invented
Cartesian actuators and scripted sensor responses. It checks each stage's map
refresh, an interrupted and restarted loaded-spoon hold, bean return, a recovery
timeout/retry, recovered-mass evidence and all four cleanup verification gates.
This checks controller transitions and command integration; it does not validate
physical grasping, scooping, perception or contact dynamics.

The runtime and task programs are runnable. They still need site camera/tool
calibration, task perception and collision signals, current simulator or real
transport binding, and closed-loop contact validation before a physical
success rate can be stated. Task 2 learned policies can use the shared real-data
training path. Task 3 has no real-trajectory checkpoint; its current executable
baseline is the measured-feedback program described here.
