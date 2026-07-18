# Failed diagnostic replay

This directory intentionally preserves GCP diagnostic run `run-20260718-c`.
It is not the canonical passing actuator rehearsal.

- Schema, file hashes, stage coverage, and truth-boundary validation: passed
- Stage 1 actuator gates: passed
- Stage 2 actuator gates: passed
- Stage 3 actuator gates: failed
- Stage 4 actuator gates: passed
- Failed gate: `stage3_base_approach`
- Final position error: 0.102805 m
- Official stage completion: not claimed
- Official benchmark score: null

The failed 0.30 m approach was reduced to a bounded 0.15 m approach for the
canonical run in `../four-stage-rehearsal/`. This diagnostic remains useful for
showing the measured controller boundary and avoiding survivorship-only
evidence.

The `controller.exit` value in this discarded run came from an early detached
launcher that did not propagate the simulator pipeline status. It is not a
success signal; `metrics.json` records the failed motion gate authoritatively.
