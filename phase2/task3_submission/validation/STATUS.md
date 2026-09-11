# Validation scope

Preparation date: 2026-09-12. No hardware movement is performed by these checks.

| Check | Status |
| --- | --- |
| Existing Task 2/3 control and scoring suite | 31 passed on macOS, Python 3.9.6, NumPy 2.0.2, pytest 8.4.2 |
| Evidence integrity and metric recomputation | PASS; 79 hashed files; 0.1764476117476625 m and 57.95783339889058 degrees reproduced |
| Isolated Python 3.12 review / synthetic JSONL checks | PASS; Python 3.12.14, NumPy 2.0.2, pytest 8.4.2; 31 tests and 3 synthetic cases; output in `local-review.txt` |
| Skill frontmatter and required fields | Checked with skill-creator `quick_validate.py` |
| Report PDF rendering | PASS; all 3 pages rendered and visually inspected, selectable text checked |
| Byte identity of archived on-site sources | Checked against originals and manifest |
| Docker build and run | NOT RUN: no local Docker engine; remote SSH unreachable after departure |
| New CMake wrapper on RT host | NOT RUN after packaging; original source compiled and used on site |
| Full real-testbed Task 3 policy | NOT VALIDATED; live perception/transport and grasp integration incomplete |

No Docker or end-to-end hardware success is implied by passing Python tests.
Post-commit clean-export results and the exact candidate SHA belong in the
preparation handoff outside the pinned tree, to avoid self-referential evidence.
