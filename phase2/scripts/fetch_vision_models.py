"""Fetch pinned, hash-checked safetensors for the optional RGB-D perception path."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from ebim_phase2.download import fetch_file, fetch_file_ranges, get_json

MODELS = {
    'grounding-dino-tiny': ('IDEA-Research/grounding-dino-tiny', 'a2bb814dd30d776dcf7e30523b00659f4f141c71'),
    'sam2.1-hiera-tiny': ('facebook/sam2.1-hiera-tiny', 'de431c4043854a71d8101e17995dfe596bf101a5'),
}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--range-workers', type=int, default=8)
    args = parser.parse_args()
    for name, (repo, revision) in MODELS.items():
        root = args.output/name
        root.mkdir(parents=True, exist_ok=True)
        metadata = get_json(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true')
        if metadata['sha'] != revision:
            raise ValueError('Model revision did not resolve to requested immutable commit')
        items = []
        for item in metadata['siblings']:
            path = item['rfilename']
            if '/' not in path and (path.endswith('.json') or path in ('vocab.txt', 'model.safetensors', 'README.md')):
                items.append({'path': path, 'size': item['size'],
                              'lfs': {'oid': item['lfs']['sha256']} if 'lfs' in item else {}})
        if not any(item['path'] == 'model.safetensors' for item in items):
            raise ValueError('A safetensors checkpoint is required')
        records = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_file_ranges,repo,revision,item,root/item['path'],args.range_workers,repo_type='model')
                       if item['size']>64 << 20 else
                       pool.submit(fetch_file, repo, revision, item, root/item['path'], repo_type='model') for item in items]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                print(json.dumps({'model': name, **record}), flush=True)
        manifest = {'repo_id': repo, 'revision': revision, 'files': sorted(records, key=lambda x: x['path'])}
        (root/'download_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
