"""Audit LeRobot v2.1 releases and build timestamp-aligned memory-mapped episodes."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import os

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset

from .download import sha256
from .schema import CAMERAS, PROFILES, check_metadata


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(image)
    ratio = min(size / image.width, size / image.height)
    resized = image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))), Image.Resampling.BILINEAR)
    result = Image.new("RGB", (size, size))
    result.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.asarray(result)


def nearest_indices(times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if len(times) == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Video timestamps must be nonempty and strictly increasing")
    right = np.searchsorted(times, targets).clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    return np.where(np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right)


def decode_camera(path: Path, targets: np.ndarray, size: int, max_skew_s: float):
    # Decode sequentially once. PTS, not the metadata's nominal FPS, determines correspondence.
    frames, times = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 2
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Video frame without timestamp: {path}")
            times.append(float(frame.pts * frame.time_base))
            frames.append(letterbox(frame.to_ndarray(format="rgb24"), size))
    times = np.asarray(times)
    indices = nearest_indices(times, targets)
    skew = np.abs(times[indices] - targets)
    valid = skew <= max_skew_s
    return np.stack([frames[i] for i in indices]), {
        "frames": len(times), "first_pts_s": float(times[0]), "last_pts_s": float(times[-1]),
        "median_frame_dt_s": float(np.median(np.diff(times))), "max_skew_s": float(np.max(skew)),
        "dropped_observations": int((~valid).sum()),
    }, valid


def split_episodes(ids: list[str], seed: int = 42) -> dict[str, str]:
    if len(ids) < 3 or len(set(ids)) != len(ids):
        raise ValueError("At least three distinct episodes required")
    order = sorted(ids, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    n = max(1, round(len(order) * 0.1))
    return {key: ("test" if i < n else "validation" if i < 2*n else "train") for i, key in enumerate(order)}


def read_numeric(path: Path, info: dict, task: int):
    table = pq.read_table(path)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    times = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64).reshape(-1)
    frames = np.asarray(table["frame_index"].to_pylist()).reshape(-1)
    episode = np.asarray(table["episode_index"].to_pylist()).reshape(-1)
    p = PROFILES[task]
    if state.shape != (len(times), p.state_dim) or action.shape != (len(times), p.action_dim):
        raise ValueError(f"Bad state/action shapes in {path}")
    if not np.isfinite(state).all() or not np.isfinite(action).all() or not np.isfinite(times).all():
        raise ValueError(f"Nonfinite values in {path}")
    components = {}
    for key, feature in info['features'].items():
        if not key.startswith('observation.state.'):
            continue
        values = np.asarray(table[key].to_pylist(), dtype=np.float32).reshape(len(times), -1)
        for index, name in enumerate(feature['names']):
            components[f"{key.removeprefix('observation.state.')}_{name}"] = values[:, index]
    reconstructed = np.stack([components[name] for name in info['features']['observation.state']['names']], axis=1)
    if not np.array_equal(reconstructed, state):
        raise ValueError(f"Flattened state disagrees with named component columns: {path}")
    if len(set(episode.tolist())) != 1 or not np.array_equal(frames, np.arange(len(times))):
        raise ValueError(f"Episode/frame index corruption: {path}")
    if len(times) < 2 or np.any(np.diff(times) <= 0) or np.max(np.abs(np.diff(times) - 0.05)) > 0.005:
        raise ValueError(f"Irregular 20 Hz timestamps in {path}")
    # Unknown annotation enums are retained for audit, never assumed to be success labels.
    annotations = {key: np.unique(np.asarray(table[key].to_pylist())).tolist()
                   for key in ("annotation.human.validity", "next.reward", "next.done") if key in table.column_names}
    return state, action, times, annotations


def prepare_episode(job: dict) -> dict:
    raw = Path(job["raw"])
    out = Path(job["output"]) / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "episode.json"
    max_drop_fraction=float(job.get('max_unaligned_fraction',.05))
    if not math.isfinite(max_drop_fraction) or not 0<=max_drop_fraction<1:
        raise ValueError('max_unaligned_fraction must be in [0,1)')
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["settings"] != job["settings"] or saved["source_revision"] != job["revision"]:
            raise ValueError(f"Existing cache has a different source/config: {out}")
        dropped=saved.get('dropped_unaligned_observations',0)
        if dropped>max_drop_fraction*(saved['observations']+dropped):
            raise ValueError(f'Cached episode exceeds requested alignment-loss threshold: {out}')
        return saved
    info = json.loads((raw / "meta/info.json").read_text())
    check_metadata(info, job["task"])
    path = raw / job["parquet"]
    state, action, times, annotations = read_numeric(path, info, job["task"])
    stride, size = job["settings"]["stride"], job["settings"]["image_size"]
    obs_frames = np.arange(0, len(times), stride, dtype=np.int64)
    camera_audit = {}
    valid_observations = np.ones(len(obs_frames), dtype=bool)
    rgb_path = out / "rgb.npy"
    rgb = np.lib.format.open_memmap(rgb_path, mode="w+", dtype=np.uint8,
                                  shape=(len(obs_frames), len(CAMERAS), size, size, 3))
    source_hashes = {job["parquet"]: sha256(path)}
    for i, camera in enumerate(CAMERAS):
        relative = info["video_path"].format(episode_chunk=job["episode_index"] // info["chunks_size"],
                                            episode_index=job["episode_index"], video_key=camera)
        video = raw / relative
        images, camera_audit[camera], valid = decode_camera(video, times[obs_frames], size, job["settings"]["max_skew_s"])
        valid_observations &= valid
        rgb[:, i] = images
        source_hashes[relative] = sha256(video)
    rgb.flush()
    dropped = int((~valid_observations).sum())
    if dropped > max_drop_fraction*len(obs_frames) or valid_observations.sum() < 2:
        raise ValueError(f"Too many unaligned observations ({dropped}/{len(obs_frames)}): {job['id']}")
    if dropped:
        filtered_rgb = np.array(rgb[valid_observations])
        del rgb
        np.save(rgb_path, filtered_rgb, allow_pickle=False)
        del filtered_rgb
        obs_frames = obs_frames[valid_observations]
    else:
        del rgb
    for name, array in (("state", state), ("action", action), ("timestamp", times), ("obs_frames", obs_frames)):
        np.save(out / f"{name}.npy", array, allow_pickle=False)
    record = {"id": job["id"], "task": job["task"], "source_revision": job["revision"],
              "source_hashes": source_hashes, "source_episode": job["source_episode"],
              "site": job["site"], "frames": len(times), "observations": len(obs_frames),
              "settings": job["settings"], "camera_audit": camera_audit, "annotations": annotations,
              "dropped_unaligned_observations": dropped,
              "max_unaligned_fraction":max_drop_fraction,
              "state_min": state.min(0).tolist(), "state_max": state.max(0).tolist(),
              "action_min": action.min(0).tolist(), "action_max": action.max(0).tolist()}
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    return record


def prepare(raw_roots: list[Path], output: Path, image_size: int, stride: int, workers: int,
            max_skew_s: float, seed: int, max_unaligned_fraction: float = .05):
    output.mkdir(parents=True, exist_ok=True)
    jobs, schemas, seen_sources = [], {}, set()
    for raw in raw_roots:
        source = json.loads((raw / "source.json").read_text())
        if not (raw / "download_manifest.json").exists():
            raise ValueError(f"Complete download with manifest required: {raw}")
        info = json.loads((raw / "meta/info.json").read_text())
        task = source["task"]
        check_metadata(info, task)
        schema = {key: info["features"][key] for key in ("observation.state", "action")}
        if task in schemas and schemas[task] != schema:
            raise ValueError("Cannot merge differing semantic schemas for one task")
        schemas[task] = schema
        source_episodes = info.get("creation_metadata", {}).get("source_episodes", [])
        # Task2 release has holes in episode indices. Enumerate actual files, not range(total_episodes).
        paths = sorted(raw.glob("data/chunk-*/*.parquet"))
        if len(paths) != info["total_episodes"]:
            raise ValueError(f"Expected {info['total_episodes']} parquet episodes, found {len(paths)} in {raw}")
        for path in paths:
            index = int(path.stem.split("_")[-1])
            original = source_episodes[index] if index < len(source_episodes) else f"{source['repo_id']}:{index}"
            if original in seen_sources:
                raise ValueError(f"Duplicate source episode across releases: {original}")
            seen_sources.add(original)
            jobs.append({"raw": str(raw.resolve()), "output": str(output.resolve()),
                         "id": f"{source['release']}_{index:06d}", "task": task,
                         "source_episode": original, "site": "munich" if task == 2 else "not_provided",
                         "revision": source["revision"], "episode_index": index,
                         "max_unaligned_fraction":max_unaligned_fraction,
                         "parquet": str(path.relative_to(raw)),
                         "settings": {"image_size": image_size, "stride": stride, "max_skew_s": max_skew_s}})
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(prepare_episode, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            print(json.dumps({"prepared": n, "total": len(jobs), "episode": records[-1]["id"]}), flush=True)
    split = {}
    for task in schemas:
        split.update(split_episodes([r["id"] for r in records if r["task"] == task], seed))
    for record in records:
        record["split"] = split[record["id"]]
    manifest = {"format": "airsign_ebim_episodes_v1", "image_size": image_size, "stride": stride,
                "fps": 20, "cameras": list(CAMERAS), "schemas": schemas, "split_seed": seed,
                "split_type": "whole_episode_within_release; not cross-site evaluation",
                "alignment_policy":{"max_skew_s":max_skew_s,"max_unaligned_fraction":max_unaligned_fraction},
                "episodes": sorted(records, key=lambda r: r["id"])}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(output / "manifest.json"), "episodes": len(records)}), flush=True)


class EpisodeDataset(Dataset):
    def __init__(self, root: Path, task: int, split: str, chunk_size: int, stats: dict | None = None):
        self.root, self.task, self.chunk_size = Path(root), task, chunk_size
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.episodes = [r for r in self.manifest["episodes"] if r["task"] == task and r["split"] == split]
        if not self.episodes:
            raise ValueError(f"Empty task{task} {split} split")
        self.cumulative = np.cumsum([r["observations"] for r in self.episodes])
        self.arrays = {}
        if stats is None:
            if split != "train":
                raise ValueError("Validation/test must use training-only statistics")
            stats = self.compute_stats()
        self.stats = stats
        self.state_mean = np.array(stats["state_mean"], dtype=np.float32)
        self.state_std = np.array(stats["state_std"], dtype=np.float32)
        self.action_mean = np.array(stats["action_mean"], dtype=np.float32)
        self.action_std = np.array(stats["action_std"], dtype=np.float32)

    def compute_stats(self):
        result = {}
        for name in ("state", "action"):
            count = 0
            total = square = None
            for r in self.episodes:
                values = np.load(self.root / r["id"] / f"{name}.npy", mmap_mode="r").astype(np.float64)
                if total is None:
                    total, square = np.zeros(values.shape[1]), np.zeros(values.shape[1])
                total += values.sum(0)
                square += np.square(values).sum(0)
                count += len(values)
            mean = total / count
            std = np.sqrt(np.maximum(square / count - mean ** 2, 0))
            result[f"{name}_mean"] = mean.tolist()
            result[f"{name}_std"] = np.maximum(std, 0.01).tolist()
        result["episodes"] = [r["id"] for r in self.episodes]
        return result

    def __len__(self):
        return int(self.cumulative[-1])

    def __getitem__(self, index):
        episode = int(np.searchsorted(self.cumulative, index, side="right"))
        row = index - (self.cumulative[episode - 1] if episode else 0)
        r = self.episodes[episode]
        if episode not in self.arrays:
            self.arrays[episode] = {k: np.load(self.root / r["id"] / f"{k}.npy", mmap_mode="r")
                                    for k in ("rgb", "state", "action", "obs_frames")}
        arrays = self.arrays[episode]
        frame = int(arrays["obs_frames"][row])
        ids = np.arange(frame, frame + self.chunk_size)
        mask = ids < len(arrays["action"])
        targets = arrays["action"][ids.clip(max=len(arrays["action"]) - 1)]
        return {"images": np.array(arrays["rgb"][row]).transpose(0, 3, 1, 2),
                "state": (arrays["state"][frame] - self.state_mean) / self.state_std,
                "action": (targets - self.action_mean) / self.action_std,
                "mask": mask, "episode": episode, "frame": frame}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=160)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-skew-s", type=float, default=0.075)
    p.add_argument('--max-unaligned-fraction',type=float,default=.05,
                   help='Maximum discarded observation fraction; Task2 uses .20 after auditing its slower-camera episodes')
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.stride < 1 or args.image_size < 32:
        p.error("stride must be positive and image-size >= 32")
    prepare(args.raw, args.output, args.image_size, args.stride, args.workers, args.max_skew_s, args.seed,args.max_unaligned_fraction)


if __name__ == "__main__":
    main()
