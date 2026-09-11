"""Download immutable official releases, without copying uploaded HF caches."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time
import threading

import requests

RELEASES = {
    "task1_part1": ("ebim-benchmark/ebim-task1-realrobotdata-lerobot-part1", "", 1),
    "task1_part2": ("ebim-benchmark/ebim-task1-realrobotdata-lerobot-part2", "", 1),
    "task2_munich": ("ebim-benchmark/ebim_task2_realrobotdata", "task2_munich", 2),
}
_sessions = threading.local()


def session():
    if not hasattr(_sessions, "value"):
        _sessions.value = requests.Session()
    return _sessions.value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_json(url: str):
    response = session().get(url, timeout=90)
    response.raise_for_status()
    return response.json()


def tree(repo: str, revision: str, folder: str):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{folder}"
    while url:
        response = session().get(url, params={"recursive": "true", "limit": 1000}, timeout=90)
        response.raise_for_status()
        yield from response.json()
        url = response.links.get("next", {}).get("url")


def fetch_file(repo: str, revision: str, item: dict, target: Path, *, repo_type: str = "dataset") -> dict:
    if repo_type not in ("dataset", "model"):
        raise ValueError("repo_type must be dataset or model")
    repository_url = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo}"
    expected_hash = item.get("lfs", {}).get("oid")
    expected_size = item["size"]
    if not (target.exists() and target.stat().st_size == expected_size
            and (not expected_hash or sha256(target) == expected_hash)):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        for attempt in range(8):
            try:
                offset = temporary.stat().st_size if temporary.exists() else 0
                if offset >= expected_size:
                    if offset == expected_size and (not expected_hash or sha256(temporary) == expected_hash):
                        temporary.replace(target)
                        break
                    temporary.write_bytes(b"")
                    offset = 0
                with session().get(f"{repository_url}/resolve/{revision}/{item['path']}",
                                  headers={"Range": f"bytes={offset}-"} if offset else {},
                                  stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned a different range from requested")
                    with temporary.open("ab" if append else "wb") as f:
                        for block in response.iter_content(4 << 20):
                            f.write(block)
                if temporary.stat().st_size != expected_size:
                    raise ValueError(f"Truncated download: {item['path']}")
                if expected_hash and sha256(temporary) != expected_hash:
                    temporary.write_bytes(b"")
                    raise ValueError(f"Hash mismatch: {item['path']}")
                temporary.replace(target)
                break
            except (requests.RequestException, ValueError) as error:
                print(json.dumps({"retry_file": item["path"], "attempt": attempt+1,
                                  "error": type(error).__name__}), flush=True)
                if attempt == 7:
                    raise
                time.sleep(min(2 ** attempt, 30))
    return {"path": item["path"], "size": expected_size, "sha256": sha256(target)}


def fetch_file_ranges(repo, revision, item, target, workers=8, chunk_bytes=16 << 20, *, repo_type="dataset"):
    """Independent resumable HTTP ranges avoid a slow single large-file transfer."""
    if repo_type not in ('dataset','model'):raise ValueError('Unknown repository type')
    repository_url=f"https://huggingface.co/{'datasets/' if repo_type=='dataset' else ''}{repo}"
    size, expected_hash = item['size'], item['lfs']['oid']
    if target.exists() and target.stat().st_size == size and sha256(target) == expected_hash:
        return {'path': item['path'], 'size': size, 'sha256': expected_hash}
    temporary = target.with_name(target.name+'.partial')
    parts = target.with_name(target.name+'.ranges')
    parts.mkdir(parents=True, exist_ok=True)
    # Retain complete/partial bytes received by the earlier sequential downloader.
    if temporary.exists():
        with temporary.open('rb') as source:
            for start in range(0, size, chunk_bytes):
                data = source.read(min(chunk_bytes, size-start))
                if not data: break
                part = parts/f'{start:012d}.part'
                if not part.exists() or part.stat().st_size < len(data):part.write_bytes(data)
        temporary.unlink()

    def fetch_range(start):
        end = min(size, start+chunk_bytes)-1
        part = parts/f'{start:012d}.part'
        for attempt in range(8):
            try:
                received = part.stat().st_size if part.exists() else 0
                if received > end-start+1:
                    part.unlink();received=0
                if received == end-start+1:return part
                offset=start+received
                url=f'{repository_url}/resolve/{revision}/{item["path"]}'
                with session().get(url,headers={'Range':f'bytes={offset}-{end}'},stream=True,timeout=(30,180)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{size}':
                        raise ValueError('Remote server did not honor the exact model byte range')
                    with part.open('ab') as sink:
                        for block in response.iter_content(1 << 20):sink.write(block)
                if part.stat().st_size != end-start+1:raise ValueError('Truncated model range')
                return part
            except (requests.RequestException,ValueError) as error:
                print(json.dumps({'range_start':start,'path':item['path'],'attempt':attempt+1,'error':type(error).__name__}),flush=True)
                if attempt==7:raise
                time.sleep(min(2**attempt,30))
    starts=list(range(0,size,chunk_bytes))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fetch_range,start) for start in starts]
        for n,future in enumerate(as_completed(futures),1):
            future.result()
            print(json.dumps({'repo':repo,'path':item['path'],'ranges_complete':n,'ranges_total':len(starts)}),flush=True)
    with temporary.open('wb') as sink:
        for start in starts:
            with (parts/f'{start:012d}.part').open('rb') as source:
                for block in iter(lambda:source.read(8 << 20),b''):sink.write(block)
    if temporary.stat().st_size!=size or sha256(temporary)!=expected_hash:
        temporary.unlink()
        for start in starts:(parts/f'{start:012d}.part').unlink()
        raise ValueError('Model hash mismatch; damaged ranges removed for a clean retry')
    temporary.replace(target)
    for start in starts:(parts/f'{start:012d}.part').unlink()
    parts.rmdir()
    return {'path':item['path'],'size':size,'sha256':expected_hash}


def download(name: str, output: Path, workers: int, metadata_only: bool = False, range_workers: int = 1):
    repo, prefix, task = RELEASES[name]
    root = output / name
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "source.json"
    if lock.exists():
        source = json.loads(lock.read_text())
        if source["repo_id"] != repo:
            raise ValueError("Existing source lock disagrees with requested release")
    else:
        source = {"repo_id": repo, "revision": get_json(f"https://huggingface.co/api/datasets/{repo}")["sha"],
                  "prefix": prefix, "task": task, "release": name}
        lock.write_text(json.dumps(source, indent=2) + "\n")
    files = []
    for folder in (["meta"] if metadata_only else ["meta", "data", "videos"]):
        remote = f"{prefix}/{folder}" if prefix else folder
        files.extend(x for x in tree(repo, source["revision"], remote) if x["type"] == "file")
    # Finish all streams of early episodes together, enabling early synchronization audits.
    def order(item):
        match = re.search(r"episode_(\d+)", item["path"])
        return (int(match.group(1)) if match else -1, item["path"])
    files.sort(key=order)
    print(json.dumps({"release": name, "revision": source["revision"], "files": len(files),
                      "bytes": sum(x["size"] for x in files)}), flush=True)
    records, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={}
        for item in files:
            target=root/(item['path'][len(prefix)+1:] if prefix else item['path'])
            if range_workers>1 and item['size']>64 << 20 and item.get('lfs',{}).get('oid'):
                future=pool.submit(fetch_file_ranges,repo,source['revision'],item,target,range_workers)
            else:
                future=pool.submit(fetch_file,repo,source['revision'],item,target)
            futures[future]=item
        for n, future in enumerate(as_completed(futures), 1):
            try:
                records.append(future.result())
            except Exception as error:
                failures.append({"path": futures[future]["path"], "error": repr(error)})
                print(json.dumps({"release": name, "failed": failures[-1]}), flush=True)
            if n % 20 == 0 or n == len(files):
                print(json.dumps({"release": name, "downloaded": n, "total": len(files)}), flush=True)
    if failures:
        (root / "download_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError(f"{name}: {len(failures)} files failed; rerun to retry only missing/invalid files")
    path = root / ("metadata_manifest.json" if metadata_only else "download_manifest.json")
    path.write_text(json.dumps({**source, "files": sorted(records, key=lambda x: x["path"])}, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--releases", nargs="+", choices=list(RELEASES), default=list(RELEASES))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument('--range-workers',type=int,default=1,
                   help='Connections per large file; e.g. --workers 2 --range-workers 12 uses up to 24 per release')
    args = p.parse_args()
    if min(args.workers,args.range_workers)<1:p.error('Worker counts must be positive')
    with ThreadPoolExecutor(max_workers=len(args.releases)) as pool:
        futures = [pool.submit(download, name, args.output, args.workers, args.metadata_only,args.range_workers)
                   for name in args.releases]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
