"""Select the audited dataset revisions before downloading into a new directory."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    locks = Path(__file__).resolve().parents[1] / 'config/data-locks'
    for source in sorted(locks.glob('*/source.json')):
        record = json.loads(source.read_text())
        destination = args.output / source.parent.name / 'source.json'
        if destination.exists():
            if json.loads(destination.read_text()) != record:
                raise ValueError(f'Existing data revision differs: {destination}')
        else:
            if destination.parent.exists() and any(destination.parent.iterdir()):
                raise ValueError(f'Cannot add a revision lock to unidentified existing data: {destination.parent}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'release': source.parent.name, 'revision': record['revision']}))


if __name__ == '__main__':
    main()
