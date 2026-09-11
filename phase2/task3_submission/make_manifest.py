"""Authoring helper: regenerate hashes after deliberately reviewing changes.

Verification never invokes this script. Validation output is excluded to avoid
self-referential hashes; source, evidence and the report are included.
"""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[1]
files = set()
for directory in (here, root/'phase2/hardware/task3_onsite',
                  root/'skills/ebim-task3-phase2', root/'phase2/ebim_phase2'):
    for path in directory.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or '.pytest_cache' in path.parts:
            continue
        if path.name == 'manifest.json' or path.is_relative_to(here/'validation'):
            continue
        files.add(path)
for relative in ('phase2/scripts/probe_hardware.py', 'phase2/scripts/probe_franka_state.cpp',
                 'phase2/tests/test_tasks_control.py', 'phase2/tests/test_tasks_scoring.py',
                 'phase2/Dockerfile.task3', 'phase2/Dockerfile.task3.dockerignore'):
    files.add(root / relative)
manifest = {'algorithm': 'sha256', 'scope': 'source, report, skill and retained evidence; validation outputs excluded',
            'files': [{'path': path.relative_to(root).as_posix(),
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in sorted(files)]}
(here/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(f"Recorded {len(files)} file hashes")
