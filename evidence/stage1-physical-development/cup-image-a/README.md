# `cup-image-a` — failed exact-entrypoint diagnostic

Status: **failed development diagnostic; not official or canonical evidence**.

The exact rebuilt AirSign image was invoked through its public
`/usr/local/bin/airsign stage1-table-setup --gate cup` entrypoint. The
controller emitted `passed: false`, and the entrypoint correctly returned
container exit code `1`.

The first and only attempted physical gate was
`cup_route_north_clearance`. It ended at `(-4.600158, 2.808918)` for target
`(-4.600225, 2.85)`: position error `0.041082 m` against the unchanged
`0.040000 m` tolerance. The prerequisite failure stopped the run before any
grasp motion; all task objects remained at their settled starting poses.

Provenance:

- Image: `airsignrobot:stage1-973108eb`
- Image ID/digest:
  `sha256:924f63fbd30ef37fb7727862961df72358468ef996896a6e470345c98d7c047c`
- Controller SHA-256:
  `973108ebe0812661b6b1bf493105196b22dfca074b3f06d78cc1cb0c21d41ec6`
- Official benchmark commit:
  `cb5184574f33611f943ff42aae461678ccb538e9`
- Container exit: `1`
- `official_stage_complete`: `false`
- `official_stage_score`: `null`

Artifact SHA-256 values captured on the simulation host:

- `metrics.json`:
  `377704130b136c92005af1d93f7579f0adc17742860fc29551f93552c6ca683c`
- `trajectory.json`:
  `0103021d3ae3e5d7772030bf12d497614264ed528d0c74284a5f4fc03a4320e7`
- `manifest.json`:
  `ab8c205b12ac99bc5a34aa12a9aec8bb319414e8efb2d6417cdac1df9eb4c25f`
- `container.exit`:
  `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`
