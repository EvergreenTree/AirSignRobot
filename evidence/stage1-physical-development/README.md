# Stage 1 physical development evidence

This directory contains diagnostic Isaac Sim runs for the public Stage 1
controller. These records are **not** official EBiM completion or score
evidence. Every bundle preserves the controller's truth boundary:

- `official_stage_complete` is `false`.
- `official_stage_score` is `null`.
- The pinned public benchmark does not expose the randomized organizer target
  provider, so no run here is presented as an assigned-target result.

Failed runs are retained because their measured first failure and container
exit status are useful reproducibility evidence. A failed diagnostic must not
be substituted for the canonical four-stage replay or described as a task
success.

Bundles:

- [`cup-image-a`](cup-image-a/README.md): exact-entrypoint failure at the
  initial north-clearance gate; retained to document the low-speed deadband.
- [`cup-image-b`](cup-image-b/README.md): final exact-entrypoint diagnostic.
  The north-clearance gate passed, then the lateral transit stopped with the
  controller's `collision/obstruction safety abort`. This result does not
  disambiguate physical clearance from steering/control limitations.
