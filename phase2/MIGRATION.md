# AirSign Phase II: on-site migration

Run these commands from this repository's `phase2/` directory. Task 1/2
inference needs the code, Python dependencies and exported weights. Downloading
the demonstration datasets or installing a simulator is unnecessary for inference.

## Get code and weights

```bash
git clone https://github.com/EvergreenTree/AirSignRobot.git
cd AirSignRobot/phase2
python3 scripts/fetch_policies.py --output policies
```

For an existing checkout, run `git pull --ff-only` and then `cd phase2`.
The fetch script requires only Python's standard library. It verifies the pinned
132 MB release archive, both exported policy hashes and their schema manifests.
It refuses to overwrite different weights. To transfer by USB or another machine,
download [the release archive](https://github.com/EvergreenTree/AirSignRobot/releases/download/phase2-v0.2.0/AirSign-phase2-offline-v0.2.0.tar.gz)
and run `python3 scripts/fetch_policies.py --bundle /path/to/AirSign-phase2-offline-v0.2.0.tar.gz --output policies`.

## Install and start

Use an environment with a working PyTorch/CUDA build for the on-site GPU. The
development B300 was labelled L20D but exposed CUDA capability 10.3; that label
does not establish the hardware or driver on this machine. The tested development
environment is recorded in [environment-tested.json](docs/environment-tested.json).

```bash
python3.12 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e .
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -m ebim_phase2.runtime --checkpoint policies/task1/policy.pt --port 8080
```

From another terminal, `curl http://127.0.0.1:8080/` should report readiness and
`mode: "shadow"`. To run Task 2, use `policies/task2/policy.pt` and port 8081.
For a functional check without CUDA, add `--device cpu`; CPU timing has no real-time
guarantee. Optional perception is installed with `python -m pip install -e '.[vision]'`
and fetched with `python scripts/fetch_vision_models.py --output cache/vision`.

The service accepts JSON via `POST /` and binds to loopback. Its JSONL form is
`./run.sh --checkpoint policies/task1/policy.pt < observations.jsonl`.
The full observation and command schemas are in [INTERFACE.md](docs/INTERFACE.md).

## Bind the current robot interface

The repository provides a policy boundary; the organizer's actual ROS/RPC driver
must supply observations and accept commands. Shadow output intentionally has
`action: null` and `commands: null`. `native_action` is a recorded GELLO-space
proposal and is not a verified physical joint command.

For calibrated proposals, supply the site's verified configuration:

```bash
python -m ebim_phase2.runtime --checkpoint policies/task1/policy.pt \
  --calibration /site/policy.json --port 8080
```

The configuration needs explicit joint conversion, gripper endpoints/direction,
spine conversion, robot limits, and Task 1's native base-velocity frame. Task 2's
recorded spine values near 434 have no verified scale/origin in the release.
Task 1 also contains documented target/feedback mismatches; the adapter must
preserve actual controller activation semantics. Use the checkpoint's exact
`state_names`, three camera identities, acquisition timestamps in the receiver's
clock domain, episode ID, and real collision evidence. Task 2 also needs measured
`spine_height_m`. Null commands require the transport's watchdog/hold behavior.

Task 3 uses a separate measured-feedback program and calibration:

```bash
python -m ebim_phase2.tasks --task task3 --describe
python -m ebim_phase2.tasks --task task3 --calibration /site/servo.json \
  --assignments /site/seats.json < task3-observations.jsonl
```

Its output already uses physical joint coordinates and spine metres. Its TCP
goals, Jacobians/IK, grasp/outcome signals and obstacle maps are site observations;
see [TASKS23.md](docs/TASKS23.md). No Task 3 demonstration checkpoint is included.

The release passed 80 software tests, exported-policy and loopback HTTP checks.
[Results](docs/results/RESULTS.md) distinguish offline prediction from physical
performance. The current Dockerfile is present in this directory, but its build
remains unverified because the development machine could not pull the base image.
