# Four-stage actuator rehearsal evidence

This is the canonical AirSign browser-replay bundle from GCP run
`run-20260718-e`.

## Result

- Runtime: Isaac Sim 5.1.0 on an NVIDIA L4
- Official benchmark commit:
  `cb5184574f33611f943ff42aae461678ccb538e9`
- Head placement: `A`
- Physics rate: 240 Hz
- Replay rate: 10 Hz
- Replay frames: 390
- Simulated duration: 35.404167 seconds
- Controller motion gates: passed for all four stage rehearsals
- Official stage completion: not claimed
- Official benchmark score: null

`passed: true` means only that the recorded navigation, IK, Robotiq driver, and
three-second hold gates met their measured tolerances. The rulebook's object
transport, spoon bean occupancy, recovered mass, and sink-placement predicates
were not established. The task objects had zero measured displacement in this
run.

## Files

- `replay_trace.json`: measured base, joint, gripper, TCP, and task-object
  states plus stage/phase intent markers.
- `metrics.json`: gate results, rulebook-outcome boundaries, and provenance.
- `manifest.json`: SHA-256 and byte count for the replay and metrics files.
- `controller.exit`: real detached-container exit status (`0`).

Key hashes:

- Controller:
  `18652a66db56ce9b9b06d532b7887943b5095a406de29a13ec25674b0bb1a6ab`
- Metrics:
  `75b1d842e5b13283ee09ed1315a449788bc7aba1915f7f7b14f01b44628775ed`
- Replay:
  `48123adf219fb9ca95cdca8f27f4029777bf210ac18c3280609d01750abb93c7`

Validate from the repository root:

```bash
python3 participant/validate_four_stage_replay.py \
  evidence/four-stage-rehearsal
```
