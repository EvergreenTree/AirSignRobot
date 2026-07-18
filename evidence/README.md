# Validation evidence index

These records document component and development checks against official
benchmark commit `cb5184574f33611f943ff42aae461678ccb538e9`.

## Start here: canonical judge evidence

- [`four-stage-rehearsal/`](four-stage-rehearsal/): canonical measured actuator
  replay, stage metrics, manifest, and detached exit status for all four
  rulebook stage intents.

Its `passed: true` result applies only to measured navigation, IK, Robotiq, and
hold gates. It records zero task-object displacement, rejects every official
stage-completion and score claim, and reports `benchmark_score: null`.

## Supporting component evidence

- `actuator_ik_probe.json`: actuator-driven dual-arm IK and closed-loop base
  component validation.
- `robotiq_gripper_scene_gate2.json`: articulation-only Robotiq open/close/open
  validation plus read-only scene inventory.
- `development_integration_validation.json`: non-authoritative geometry/grader
  validation. This test moves prims directly and is not participant performance.

## Retained diagnostic — not representative judge evidence

- `run-20260718-c/`: retained failed diagnostic replay. Its Stage 3 base
  approach gate failed. It is preserved to document a measured failure boundary,
  but it is not the canonical result and should not be used as the submission
  demonstration.
- [`stage1-physical-development/`](stage1-physical-development/): two
  exact-entrypoint cup-route diagnostics with manifests, trajectories, detached
  container exits, and explicit failure boundaries. Both are failed development
  runs (`official_stage_complete: false`, `official_stage_score: null`), not
  evidence of a completed Stage 1 task.

Every JSON record has `benchmark_score: null`. None is evidence of an official
stage completion or a competition score.
