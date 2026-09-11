"""Install the two verified release policies; Python standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def install(bundle, output, release):
    if sha256(bundle) != release['bundle_sha256']:
        raise ValueError('Bundle SHA256 differs from the pinned release')
    with tarfile.open(bundle, 'r:gz') as archive:
        for task, expected in release['policies'].items():
            prefix = f'AirSign-phase2/policies/task{task}'
            measured = json.load(archive.extractfile(f'{prefix}/manifest.json'))
            if measured != expected:
                raise ValueError(f'Task {task} manifest differs from the pinned export')
            folder = output/f'task{task}'
            folder.mkdir(parents=True, exist_ok=True)
            target = folder/'policy.pt'
            if target.exists() and sha256(target) != expected['checkpoint_sha256']:
                raise FileExistsError(f'Refusing to replace different weights: {target}')
            metadata = folder/'manifest.json'
            if metadata.exists() and json.loads(metadata.read_text()) != expected:
                raise FileExistsError(f'Refusing to replace a different manifest: {metadata}')
            if not target.exists():
                with tempfile.NamedTemporaryFile(dir=folder, prefix='policy-', suffix='.partial', delete=False) as temporary:
                    temp_path = Path(temporary.name)
                    try:
                        with archive.extractfile(f'{prefix}/policy.pt') as stream:
                            shutil.copyfileobj(stream, temporary, length=1024 * 1024)
                    except BaseException:
                        temp_path.unlink(missing_ok=True)
                        raise
                try:
                    if sha256(temp_path) != expected['checkpoint_sha256']:
                        raise ValueError(f'Task {task} policy SHA256 mismatch')
                    temp_path.replace(target)
                finally:
                    temp_path.unlink(missing_ok=True)
            metadata.write_text(json.dumps(expected, indent=2)+'\n')
            print(json.dumps({'task': int(task), 'policy': str(target.resolve()),
                              'sha256': expected['checkpoint_sha256']}), flush=True)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root/'policies')
    parser.add_argument('--bundle', type=Path, help='Use a previously downloaded release archive')
    args = parser.parse_args()
    release = json.loads((root/'config/policy-release.json').read_text())
    if args.bundle:
        install(args.bundle, args.output, release)
        return
    if all((args.output/f'task{task}/policy.pt').exists()
           and sha256(args.output/f'task{task}/policy.pt') == expected['checkpoint_sha256']
           and (args.output/f'task{task}/manifest.json').exists()
           and json.loads((args.output/f'task{task}/manifest.json').read_text()) == expected
           for task, expected in release['policies'].items()):
        print(json.dumps({'verified': True, 'output': str(args.output.resolve())}))
        return
    url = f"https://github.com/{release['repository']}/releases/download/{release['tag']}/{release['bundle']}"
    print(f'Downloading {url}', flush=True)
    with tempfile.TemporaryDirectory(prefix='airsign-phase2-release-') as directory:
        bundle = Path(directory)/release['bundle']
        request = urllib.request.Request(url, headers={'User-Agent': 'AirSign-phase2-migration'})
        with urllib.request.urlopen(request, timeout=120) as response, bundle.open('wb') as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        install(bundle, args.output, release)


if __name__ == '__main__':
    main()
