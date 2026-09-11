"""Read-only local scan registration. Diagnostic feedback, not a safety scanner."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree

def cloud(data):
    points = []
    stamps = []
    for side in ("front", "rear"):
        scan = data["scans"][side]
        stamps.append(float(scan["received_monotonic"]))
        if scan["frame"] != "lidar_" + side:
            raise ValueError("Unexpected lidar frame")
        ranges = np.array([np.nan if v is None else v for v in scan["filtered_ranges"]])
        angles = scan["angle_min"] + np.arange(len(ranges))*scan["angle_increment"]
        phi = (math.pi/4 if side == "front" else 5*math.pi/4) - 0.0007963268 - angles
        offset = np.array([0.3275, 0.2175]) * (1 if side == "front" else -1)
        ok = np.isfinite(ranges) & (ranges > 0.5) & (ranges < 3.0)
        xy = np.c_[ranges[ok]*np.cos(phi[ok]), ranges[ok]*np.sin(phi[ok])] + offset
        xy = xy[np.linalg.norm(xy, axis=1) > 0.8]
        points.append(xy[::3])
    result = np.vstack(points)
    if len(result) < 200:
        raise ValueError("Too few scene points")
    if np.min(np.linalg.eigvalsh(np.cov(result.T))) < 0.08:
        raise ValueError("Insufficient two-dimensional scene structure")
    return result, min(stamps)

def register(reference, current, initial=None, tree=None):
    if tree is None:
        tree = cKDTree(reference)
    if initial is None:
        rotation, translation = np.eye(2), np.zeros(2)
    else:
        angle = float(initial["yaw"])
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        translation = np.array([initial["x"], initial["y"]], dtype=float)
    for _ in range(18):
        moved = current @ rotation.T + translation
        # Exact neighbors outside this existing rejection radius are unused.
        distance, index = tree.query(moved, distance_upper_bound=0.045)
        valid = distance < 0.045
        if valid.sum() < 160 or valid.mean() < 0.7:
            raise ValueError("Poor scan overlap")
        a, b = moved[valid], reference[index[valid]]
        ca, cb = a.mean(0), b.mean(0)
        u, _, vh = np.linalg.svd((a-ca).T @ (b-cb))
        dr = vh.T @ u.T
        if np.linalg.det(dr) < 0:
            vh[-1] *= -1
            dr = vh.T @ u.T
        dt = cb - dr @ ca
        rotation, translation = dr @ rotation, dr @ translation + dt
        if np.linalg.norm(dt) < 1e-6 and abs(dr[1, 0]) < 1e-6:
            break
    residual, _ = tree.query(current @ rotation.T + translation, distance_upper_bound=0.025)
    inliers = residual < 0.025
    if inliers.mean() < 0.8 or np.median(residual) > 0.009:
        raise ValueError("Registration residual too high")
    angle = math.atan2(rotation[1, 0], rotation[0, 0])
    return dict(x=float(translation[0]), y=float(translation[1]), yaw=angle,
                inlier_ratio=float(inliers.mean()), median_residual=float(np.median(residual)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/tmp/airsign-live-lidar.json")
    parser.add_argument("--output", default="/tmp/airsign-lidar-motion.json")
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 90:
        parser.error("Observation duration must be 1..90 seconds")
    source, output = Path(args.input), Path(args.output)
    first = json.loads(source.read_text())
    reference, stamp = cloud(first)
    tree = cKDTree(reference)
    started = time.monotonic()
    if started-stamp > 0.1:
        raise RuntimeError("Stale reference scans")
    Path(str(output)+".reference.json").write_text(json.dumps(first))
    with Path(str(output)+".jsonl").open("w") as log:
        sequence = 0
        previous = None
        previous_mtime = None
        previous_sensor = stamp
        while time.monotonic()-started < args.seconds:
            cycle = time.monotonic()
            # Poll promptly for new relay data without re-registering the same
            # cloud. Never refresh a sensor timestamp when its data are reused.
            # Stale input still enters the existing rejection path below.
            current_mtime = source.stat().st_mtime_ns
            if current_mtime == previous_mtime and cycle-previous_sensor <= 0.12:
                time.sleep(0.005)
                continue
            previous_mtime = current_mtime
            result = {"valid": False, "reference_monotonic": started, "sequence": sequence}
            try:
                current, stamp = cloud(json.loads(source.read_text()))
                previous_sensor = stamp
                if time.monotonic()-stamp > 0.12:
                    raise ValueError("Stale lidar")
                result.update(register(reference, current, previous, tree))
                result.update(valid=True, sensor_monotonic=stamp)
                previous = result.copy()
            except Exception as exc:
                result["error"] = str(exc)
            result["written_monotonic"] = time.monotonic()
            result["compute_seconds"] = result["written_monotonic"]-cycle
            text = json.dumps(result)
            temporary = Path(str(output)+".new")
            temporary.write_text(text)
            temporary.replace(output)
            log.write(text+"\n")
            log.flush()
            sequence += 1
            time.sleep(max(0, 0.01-(time.monotonic()-cycle)))
    result["valid"] = False
    result["error"] = "Observer finished"
    output.write_text(json.dumps(result))

if __name__ == "__main__":
    main()
