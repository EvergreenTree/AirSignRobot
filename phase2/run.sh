#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${AIRSIGN_PYTHON:-python3}"
exec "$python_bin" -m ebim_phase2.runtime "$@"
