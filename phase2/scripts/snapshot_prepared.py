"""Freeze an explicitly partial dataset without changing the full-release episode split."""
import argparse
from collections import Counter
import json
from pathlib import Path

from ebim_phase2.data import split_episodes, read_numeric
from ebim_phase2.schema import CAMERAS, check_metadata


def snapshot(raw_roots, prepared, output, task, seed=42, require_complete=False):
    ids, schema, revisions, expected_hashes = [], None, {}, {}
    for raw in raw_roots:
        source = json.loads((raw / 'source.json').read_text())
        if source['task'] != task:
            continue
        info = json.loads((raw / 'meta/info.json').read_text())
        check_metadata(info, task)
        current = {k: info['features'][k] for k in ('observation.state', 'action')}
        if schema is not None and schema != current:
            raise ValueError('Incompatible release schemas')
        schema = current
        episodes = [json.loads(line) for line in (raw / 'meta/episodes.jsonl').read_text().splitlines()]
        if len(episodes) != info['total_episodes']:
            raise ValueError('Full episode index required to preserve the held-out split')
        ids.extend(f"{source['release']}_{e['episode_index']:06d}" for e in episodes)
        revisions[source['release']] = source['revision']
        if require_complete:
            verified = json.loads((raw / 'download_manifest.json').read_text())
            if verified['revision'] != source['revision']:
                raise ValueError('Download manifest revision mismatch')
            prefix = source.get('prefix', '')
            expected_hashes[source['release']] = {
                x['path'][len(prefix)+1:] if prefix else x['path']: x['sha256']
                for x in verified['files']}
            for path in sorted(raw.glob('data/**/*.parquet')):
                read_numeric(path, info, task)
    splits = split_episodes(ids, seed)
    records, seen_sources = [], set()
    for path in sorted(prepared.glob('*/episode.json')):
        record = json.loads(path.read_text())
        if record['task'] != task:
            continue
        release = record['id'].rsplit('_', 1)[0]
        if record['id'] not in splits or record['source_revision'] != revisions.get(release):
            raise ValueError('Prepared episode is outside the pinned source index')
        if record['source_episode'] in seen_sources:
            raise ValueError('Duplicate original source episode across releases')
        seen_sources.add(record['source_episode'])
        if require_complete:
            if any(expected_hashes[release].get(path) != digest
                   for path, digest in record['source_hashes'].items()):
                raise ValueError('Prepared source hashes differ from verified download')
        records.append({**record, 'split': splits[record['id']]})
    if require_complete and {r['id'] for r in records} != set(ids):
        raise ValueError(f'Incomplete preparation: {len(records)}/{len(ids)} episodes')
    counts = Counter(r['split'] for r in records)
    if not counts['train'] or not counts['validation']:
        raise ValueError(f'Need prepared train and validation episodes; currently {dict(counts)}')
    settings = {tuple(sorted(r['settings'].items())) for r in records}
    if len(settings) != 1:
        raise ValueError('Incompatible preparation settings')
    config = dict(settings.pop())
    output.mkdir(parents=True, exist_ok=False)
    for record in records:
        (output / record['id']).symlink_to((prepared / record['id']).resolve(), target_is_directory=True)
    manifest = {'format': 'airsign_ebim_episodes_v1', 'image_size': config['image_size'],
                'stride': config['stride'], 'fps': 20, 'cameras': list(CAMERAS),
                'schemas': {str(task): schema}, 'split_seed': seed,
                'split_type': 'whole_episode; same partition as full release; not cross-site',
                'coverage': 'full_official_release' if require_complete else 'partial_download_smoke_only',
                'expected_episodes': len(ids),
                'source_revisions': revisions, 'episodes': records}
    manifest['alignment_policy']={'max_skew_s':config['max_skew_s'],
                                  'max_unaligned_fraction':max(r.get('max_unaligned_fraction',.05) for r in records)}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'coverage': manifest['coverage'], 'episodes': dict(counts)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, nargs='+', required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', type=int, choices=[1, 2], required=True)
    p.add_argument('--require-complete', action='store_true')
    args = p.parse_args()
    snapshot(args.raw, args.prepared, args.output, args.task, require_complete=args.require_complete)
