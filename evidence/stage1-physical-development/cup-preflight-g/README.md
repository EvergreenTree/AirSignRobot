# `cup-preflight-g` — fail-closed arm-continuity diagnostic

Status: **failed development diagnostic; not official or canonical evidence**.

This no-base-motion run used the current Stage 1 controller through the
submission entrypoint:

```text
/usr/local/bin/airsign stage1-table-setup \
  --gate cup-preflight \
  --output-dir /output \
  --head-placement A \
  --headless
```

The controller moved only the two arm articulations. It issued no base,
gripper, or task-object command. The compact-posture lift stopped before its
eighth target command when the measured-to-target joint gap reached
`0.026346 rad`, above the fail-closed `0.025000 rad` guard.

The diagnostic also disambiguated the earlier mixed-unit effort abort:

- initial prismatic spine force: `679.105469 N`;
- peak absolute prismatic spine force: `784.667236 N`;
- peak absolute revolute-arm effort: `54.288223`, at
  `left_fr3v2_joint3`;
- revolute-arm abort threshold: `180.0`.

Spine force is recorded separately and is not compared with the revolute-arm
threshold. No spine-force safety threshold is claimed. This result calls for
a rate-limited joint trajectory/tracking controller; it is not a task
completion, route success, official force result, or benchmark score.

Provenance:

- Runtime image tag: `airsignrobot:stage1-be057128`, with the current
  controller and four import-safe diagnostic modules bind-mounted read-only.
- Controller SHA-256:
  `778a387def82449cba9f67504b9afb7825c70249960e6d74e14fe4d944644239`
- Official benchmark commit:
  `cb5184574f33611f943ff42aae461678ccb538e9`
- Container exit: `1`
- `official_stage_complete`: `false`
- `official_stage_score`: `null`

Artifact SHA-256 values:

- `metrics.json`:
  `288dc7b78314de7b8ba247449f0c459f350b23b764f5f27277624bab4cf8f4f6`
- `trajectory.json`:
  `078193c301484d9c2515e686ca9daca9186e3f1bbce4545aa6e436d1f7295991`
- `manifest.json`:
  `1855ce52df7c6310a664d09f15e4d163fc86ef735a46455ef48ac1c21d5c9071`
- `container.exit`:
  `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`
