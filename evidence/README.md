# Validation evidence

These records document component and development checks against official
benchmark commit `cb5184574f33611f943ff42aae461678ccb538e9`.

- `actuator_ik_probe.json`: actuator-driven dual-arm IK and closed-loop base
  component validation.
- `robotiq_gripper_scene_gate2.json`: articulation-only Robotiq open/close/open
  validation plus read-only scene inventory.
- `development_integration_validation.json`: non-authoritative geometry/grader
  validation. This test moves prims directly and is not participant performance.

Every JSON record has `benchmark_score: null`. None is evidence of an official
stage completion or a competition score.
