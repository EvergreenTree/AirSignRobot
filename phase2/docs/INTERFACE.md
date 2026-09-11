# AirSign policy interface

This JSONL/loopback-HTTP interface is the project's integration boundary. It is
not a claim that the organizers expose this transport. Bind it to the site's
actual observation and actuator interfaces after checking the current robot
configuration. Joint targets, measured states and camera identities must keep
the exact order in the checkpoint schema.

## Learned Tasks 1 and 2

Every request contains:

| Field | Meaning |
|---|---|
| `task` | Integer 1 or 2, matching the checkpoint |
| `episode_id` | New identifier for each trial; changing it resets temporal aggregation |
| `timestamp_s` | Robot observation time, seconds in the host wall-clock domain |
| `state` | Native 62-value Task 1 or 42-value Task 2 state |
| `state_names` | Exact `observation.state.names` array stored in the checkpoint |
| `images` | Three camera objects keyed by the names below |
| `collision_imminent` | Fresh actual collision-controller evidence; must be JSON `false` to command |
| `spine_height_m` | Additional measured feedback required for Task 2, absent from its native state |

Camera keys are `observation.images.head`, `observation.images.wrist_left` and
`observation.images.wrist_right`. Each object has `timestamp_s` and either
`base64` (encoded PNG/JPEG) or `path` (local image file). Training and inference
both use RGB letterboxing. The local path option is intended for a trusted
site adapter; the HTTP service binds only to `127.0.0.1`.

Task 1 state slots 55–60 are the recorded base-command feedback. A live adapter
must populate them from commands already issued to the controller, never from
the next target action. These fields strongly correlate with the demonstration
base labels, so low offline base-action error alone does not demonstrate navigation.

Invoke the policy at the demonstrated 20 Hz cadence where practical. Temporal
aggregation aligns action chunks by elapsed observation time. Only the next
action is returned; future chunk entries are not sent open-loop. The release's
software timing checks replay already-letterboxed 320-pixel validation images.
Direct timings include local PNG decoding, preprocessing, inference and limits;
HTTP timings include base64 JSON over loopback. Neither measures native camera
acquisition/transport, full-resolution decoding, or the physical controller.
The learned Task 1 service stops commands after 30 minutes from its first
request for that episode, including the base-approach portion. The organizer's
trial clock remains authoritative.

Responses contain `team: "AirSign"`, task, episode, observation timestamp,
`status`, `reason`, `native_action_dim`, and the following payload fields:

- `native_action`: predicted values in the exact native demonstration layout,
  when inference ran. Task 2's spine remains in its recorded native scale here.
- `action`: 23 values in the common robot layout, or `null` when no such command
  is available. Order: left joints 7, left gripper 1, right joints 7, right
  gripper 1, linear base twist xyz 3, angular base twist xyz 3, spine metres 1.
  The common planar base velocity is in the robot body frame.
- `commands`: named representation of `action`, or `null`. Despite the
  organizer's `width_percent` feature names, gripper values are native **0–1**.

`shadow_native_proposal` means the native output has no configured conversion
to the physical command coordinates; its `action` and `commands` are null for
both tasks. `bounded_proposal` means the configured checks passed,
not that physical task success or collision freedom has been demonstrated.
`no_command` means the transport must stop updates and retain its own watchdog
and low-level hold behavior. Do not hold using an obsolete measured pose.

## Site calibration for learned policies

The `--calibration` JSON needs the following fields. Obtain values from the
actual robot setup; no physical limits or transformation are inferred from a
simulator's defaults.

| Fields | Shape / unit |
|---|---|
| `calibration_id` | Site/configuration identifier |
| `joint_lower`, `joint_upper` | 14 radians each, left seven then right seven |
| `joint_speed_rad_s` | 14 positive limits |
| `joint_action_scale`, `joint_action_offset_rad` | 14 each: `robot_target_rad = native_GELLO_target × scale + offset`; explicitly configure identity if verified |
| `gripper_open_knuckle`, `gripper_closed_knuckle` | Two measured knuckle radians each |
| `gripper_open_command`, `gripper_closed_command` | Two distinct native 0–1 endpoints each; define the actual direction |
| `spine_lower_m`, `spine_upper_m`, `spine_speed_m_s` | Physical spine bounds and rate |
| `base_linear_speed_m_s`, `base_yaw_speed_rad_s` | Planar base limits |
| `native_base_velocity_frame` | Task 1 requires `body` or `world`, verified against the recording/live interface; world-frame xy is rotated using measured base heading |
| `max_force_n` | Measured wrist-force stop threshold |
| `max_observation_age_s`, `max_camera_skew_s` | Clock/freshness limits |
| `spine_scale_m_per_native`, `spine_offset_m` | Required for learned policies: `metres = native × scale + offset`; Task 1 may use verified identity |

The Task 2 release contains spine targets near 434 while Task 1 contains values
near 0.3. The source documents do not establish the Task 2 scale and origin.
Do not assume millimetres solely from those numbers. The Task 2 base is stopped
because its demonstrations have no base actions. Task 1 permits only planar
base translation and yaw.

The recorded action fields are GELLO targets. Their names alone do not prove
that they equal the physical controller's final joint coordinates. Conversion
occurs before measured-joint limit checking. The geometric Task 2/3 programs
already output robot joint coordinates and must not receive this native-action
conversion a second time. A transport that expects GELLO coordinates must map
the bounded physical target back through its verified interface mapping.

The final data audit found a concrete mismatch in Task 1 part 2 episodes 5–8:
right-joint-5's recorded GELLO target stays at −2.8763 radians while its measured
position stays near −1.18 radians. This does not establish an executed motion
command or a universal calibration offset. The site's adapter must preserve
actual controller activation and command semantics; a coordinate conversion
alone cannot establish whether a recorded target was enabled. The result report
retains this unresolved issue and the unchanged held-out metrics.

The learned interface has no hidden access to simulator object poses or old
fixture coordinates. Its ACT network receives no separately annotated route
instruction; novel route generalization must therefore be evaluated and is
not claimed from the training data alone.

For supervised Task 1 execution, add `--route /site/current-route.json`. That
file contains the current `route_id`, ordered `fixtures` (unique occurrence
identifiers), and matching `directions` strings from the organizer assignment.
Each request must then include `route_progress` with that same `route_id`,
`timestamp_s`, a `source` of `vision`, `rgbd` or `tactile`, confidence at least
0.8, boolean `correct` flags keyed by fixture ID, and integer
`validated_through` (the count covered by a verified checkpoint). A correctness
flag means that fixture engagement **and the assigned direction** were verified.
Unknown fixtures must remain absent/false; a checkpoint does not repair an
incorrect earlier prefix.

The response's `route` object gives `next_fixture` and `next_direction` when
available. Freshness, prefix regression, a 30-second progress stall, an
unconfirmed final checkpoint, or completed routing suppresses further learned
commands. A new episode ID resets this state. `cable.RouteMonitor` and the
supervisor consume observed progress; they do not perform RGB cable segmentation,
certify physical checkpoints, synthesize a recovery grasp, or condition the ACT
network on a different route.

## Geometric Task 2/3 programs

The `ebim_phase2.tasks` JSONL interface uses richer measured pose, Jacobian,
navigation and boolean outcome observations. It has a separate
`ServoCalibration` configuration. See [TASKS23.md](TASKS23.md) for the complete
schema, source provenance and `--describe` commands. These programs require
site perception; adding fabricated object poses to a record is not validation.
