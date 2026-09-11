"""Package reviewed source, offline reports and tested Task 1/2 inference weights."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from ebim_phase2.download import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.workspace.resolve()
    source = base / 'source'
    if args.output.exists():
        raise FileExistsError('Use a new filename for a release archive')
    report_path = source / 'docs/results/results.json'
    report = json.loads(report_path.read_text())
    files = {}
    for name in ('README.md', 'pyproject.toml', 'Dockerfile', 'run.sh', '.gitignore', '.dockerignore'):
        path = source / name
        if path.exists():
            files[f'source/{name}'] = path
    for directory in ('ebim_phase2', 'scripts', 'tests', 'config', 'docs'):
        for path in sorted((source / directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                if path.is_symlink():
                    raise ValueError(f'Unexpected source symlink: {path}')
                files[f'source/{path.relative_to(source)}'] = path
    for task in (1, 2):
        exported = base / f'exports/task{task}'
        manifest = json.loads((exported / 'manifest.json').read_text())
        measured = next(item for item in report['tasks'] if item['task'] == task)
        if (manifest['dataset_coverage'] != 'full_official_release'
                or manifest != measured['export']
                or sha256(exported / 'policy.pt') != manifest['checkpoint_sha256']):
            raise ValueError(f'Task {task} export differs from the reviewed report')
        for name in ('manifest.json', 'policy.pt'):
            files[f'policies/task{task}/{name}'] = exported / name
    wheel = base / 'dist/airsign_ebim_phase2-0.2.0-py3-none-any.whl'
    checked = json.loads((base / 'runs/wheel-check.json').read_text())
    if sha256(wheel) != checked['wheel_sha256']:
        raise ValueError('Wheel differs from the separately installed artifact')
    files[f'wheel/{wheel.name}'] = wheel
    content = {'team': 'AirSign', 'phase': 2, 'physical_validation': None,
               'task3': 'measured-feedback program; no demonstration checkpoint',
               'optional_vision_models': 'Fetch exact snapshots with source/scripts/fetch_vision_models.py',
               'files': [{'path': name, 'size': path.stat().st_size, 'sha256': sha256(path)}
                         for name, path in sorted(files.items())]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.partial')
    with tarfile.open(temporary, 'w:gz') as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=f'AirSign-phase2/{name}', recursive=False)
        encoded = (json.dumps(content, indent=2) + '\n').encode()
        metadata = tarfile.TarInfo('AirSign-phase2/MANIFEST.json')
        metadata.size = len(encoded)
        archive.addfile(metadata, io.BytesIO(encoded))
    # Check every stored byte against the recorded hash before publishing.
    with tarfile.open(temporary, 'r:gz') as archive:
        for record in content['files']:
            stream = archive.extractfile(f"AirSign-phase2/{record['path']}")
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
            if digest.hexdigest() != record['sha256']:
                raise ValueError(f"Archive verification failed: {record['path']}")
    temporary.replace(args.output)
    checksum = sha256(args.output)
    args.output.with_suffix(args.output.suffix + '.sha256').write_text(f'{checksum}  {args.output.name}\n')
    print(json.dumps({'archive': str(args.output), 'sha256': checksum, 'files': len(files)}))


if __name__ == '__main__':
    main()
