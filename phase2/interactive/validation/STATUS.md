# Offline prototype validation - September 12, 2026

- CMake built the mock backend with AppleClang 17.0.0, C++17, on macOS.
- A separate compile with `-Wall -Wextra -Werror` passed.
- `python3 -m pytest -q --tb=short phase2/interactive/test_gamepad.py`: **14 passed in 7.89 s**.
- Ten benchmark cycles used one persistent mock process, one static image fixture
  served over loopback HTTP, 0.8 s pulses and a synthetic 0.1 s decision delay.
- Median complete cycle: **924.14 ms**; maximum: **927.08 ms**.
- Median fresh-frame/state request: **8.04 ms**; median command-to-result: **801.44 ms**.
- All ten mock cycles were below five seconds. No actual model reasoning, SSH,
  ROS camera pipeline or robot motion was part of this benchmark.

The sandbox initially refused loopback binding. The same tests then passed with
permission to run the local HTTP fixture. This was not a robot connection.

Raw metrics are in `benchmark.json` and `latency.jsonl`; top-level prototype source
hashes are in `source-hashes.json`. The live libfranka target and ROS camera mode
remain unbuilt/unverified on the site hardware. They must not be presented as
physically tested or as part of the earlier on-site result.
