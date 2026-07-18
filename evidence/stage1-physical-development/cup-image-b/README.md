# `cup-image-b` — final failed exact-entrypoint diagnostic

Status: **failed development diagnostic; not official or canonical evidence**.

This was the single independently reviewed follow-up run after adding a
bounded minimum planar base command. The exact rebuilt image was invoked
through `/usr/local/bin/airsign stage1-table-setup --gate cup`. The controller
emitted `passed: false`, and the public entrypoint correctly returned
container exit code `1`.

The minimum-command correction worked for the first route gate:

- `cup_route_north_clearance` passed at `(-4.600173, 2.812706)`.
- Position error was `0.037294 m` against the unchanged `0.040000 m`
  tolerance.
- The `0.08 m/s` floor engaged for 176 desired-command steps, from step 375
  through step 550.

The next gate failed closed:

- `cup_route_clear_table_north` triggered the one-second rolling
  collision/stall abort.
- Target: `(-0.6, 2.85)`.
- Final base position: `(-4.599498, 2.812883)`.
- Path length: `0.002921 m` over 480 steps.
- The command floor did not engage on this gate; the long transit command was
  already above the floor.

This measured stall is consistent with insufficient swept-volume clearance or
a steering/control limitation; the run does not disambiguate those causes.
The controller recorded the exact label `collision/obstruction safety abort`.
The prerequisite abort stopped the run before any grasp or task-object
manipulation.

Provenance:

- Image: `airsignrobot:stage1-be057128`
- Image ID/digest:
  `sha256:ef53a18c7280be84e8b3e998b774e7e10f027ed399f9e765287989cd148cd013`
- Controller SHA-256:
  `be057128666bb7419d2f3543a3a19e98cdf3d2de6319fdb2cdda5e7ff2af9ab5`
- Official benchmark commit:
  `cb5184574f33611f943ff42aae461678ccb538e9`
- Container exit: `1`
- `official_stage_complete`: `false`
- `official_stage_score`: `null`

Artifact SHA-256 values captured on the simulation host:

- `metrics.json`:
  `ca52c1928222dda04136245b764b803e89ac88d8143bebefd9da97a1280b760e`
- `trajectory.json`:
  `6ce39d6d622fe77d248ede4b1140a717547f8c8c64580129868c4e250ee72e78`
- `manifest.json`:
  `fe3b7e0dfe1a0a57ee9e979cd245182c556c0a4fef67cbccebdb8f9c864160fb`
- `container.exit`:
  `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`
