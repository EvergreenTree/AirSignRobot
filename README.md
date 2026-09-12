# AirSign - EBiM Task 3 technical report

**Submission route: Technical Report.** The report is the primary deliverable;
`ebim-task3-phase2` is the accompanying installable Codex skill. No Docker image
is submitted. The [official submissions README](https://github.com/EBiM-Benchmark/submissions)
explicitly describes a report route without a Dockerfile, assessed on technical
maturity and readiness with 0.65 weighting. Filing and scoring remain organizer decisions.

- [Technical report](REPORT.md) and [PDF](REPORT.pdf)
- [Installable skill ZIP](release/ebim-task3-phase2.zip) and [checksums](release/SHA256SUMS)
- [Skill source](skills/ebim-task3-phase2/SKILL.md)
- [Measured results](evidence/results.json)

The on-site work demonstrated a partial supervised plate approach, including
17.64 cm net right-arm displacement. The later persistent gamepad interface is
mock-tested. Plate contact, grasp, lift and complete task stages were not verified.

## Install in Codex

Target: **Codex with GPT-6 Astra** (`gpt-6-astra`), using a host that can run local
commands, retain a process session and view local image files. The selected model,
Codex account/network access and robot credentials are host prerequisites; they
are not inside the skill. [Model reference](https://developers.openai.com/api/docs/models/gpt-6-astra).

From a checkout of the submission's pinned commit, copy the complete skill folder
into the supported user skill location. This command refuses to replace an existing
installation:

```bash
mkdir -p "$HOME/.agents/skills"
test ! -e "$HOME/.agents/skills/ebim-task3-phase2" && \
  cp -R skills/ebim-task3-phase2 "$HOME/.agents/skills/"
```

Alternatively, extract `release/ebim-task3-phase2.zip` into that directory after
checking that the destination does not already exist. The ZIP contains one
`ebim-task3-phase2/` folder, including scripts, build input, references and a
fixture image. It has no references to this checkout's private development folder.
Codex detects installed skills automatically; restart if needed. You may also ask
`$skill-installer` to install this repository's skill at the submitted commit.
[Official skill installation guidance](https://learn.chatgpt.com/docs/build-skills).

Select Astra in the host, then invoke:

```text
Use $ebim-task3-phase2 to read the setup notes and run the offline action-loop
example. Report its result and the prerequisites for a new attended robot session.
```

## Review the skill offline

```bash
SKILL_DIR="$HOME/.agents/skills/ebim-task3-phase2"
cmake -S "$SKILL_DIR" -B "$SKILL_DIR/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$SKILL_DIR/build" -j2
python3 "$SKILL_DIR/scripts/gamepad.py" --demo --output /tmp/airsign-frames
```

Requires Python 3.9+, CMake and a C++17 compiler. The default mock uses only
Python's standard library; no model call, GPU, ROS, Docker or robot is needed.
View the returned image and send `<frame-id> R:X+/fine`. Each pulse returns a
new frame. `LOOK` refreshes, `STOP` requests stopping, and `QUIT` closes the session.
A fixture is always labelled simulated; it does not show the effect of mock motion.

The skill contains the separate experimental live setup: two-host roles,
normal control ownership, RT build, SSH/camera loop, coordinate conventions and
fault boundaries. No physical movement is authorized by installation. The
operator must establish a new attended session and verify the actual configuration.

## Evidence and scope

Fifteen mock guard tests cover the packaged runtime, including a terminal-worker
restart hint. An independent copied-skill test completed one specified fine pulse;
the distributable archive is also checked from an isolated installation.
The retained ten-cycle benchmark reported a 924 ms median with a synthetic
100 ms decision delay and an 800 ms mock pulse. It does not measure Astra or
real SSH/camera/robot latency. The five-second bound is a frame freshness deadline,
not a guarantee of model response time.

The public [English project write-up](https://airsign-ebim-track3.chattytransformer.chatgpt.site/)
and [Chinese version](https://airsign-ebim-track3.chattytransformer.chatgpt.site/zh)
remain supplementary July research/simulator background. This report contains
the September hardware and skill work. Development tests, full logs, older code,
report generators and issue drafts are kept in ignored `.local/`; historical
commits preserve earlier evidence.
