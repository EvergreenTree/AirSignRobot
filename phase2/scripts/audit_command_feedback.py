"""Report static target/feedback disagreements without filtering or rewriting data."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from ebim_phase2.download import sha256
from ebim_phase2.schema import JOINT_ACTION_INDICES, JOINT_STATE_INDICES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.dataset/'manifest.json').read_text())
    flagged = []
    for episode in manifest['episodes']:
        folder = args.dataset/episode['id']
        frames = np.load(folder/'obs_frames.npy', mmap_mode='r')
        action = np.load(folder/'action.npy', mmap_mode='r')[frames][:, JOINT_ACTION_INDICES]
        state = np.load(folder/'state.npy', mmap_mode='r')[frames][:, JOINT_STATE_INDICES]
        names = manifest['schemas'][str(episode['task'])]['action']['names']
        for index in range(14):
            target, measured = action[:, index], state[:, index]
            difference = np.abs(target-measured)
            # These are diagnostic thresholds, not robot safety limits or a
            # definition of invalid demonstrations. An enable signal may be absent.
            if np.ptp(target) <= 1e-5 and np.ptp(measured) <= .05 and np.median(difference) >= .25:
                flagged.append({'episode': episode['id'], 'split': episode['split'],
                                'action_name': names[JOINT_ACTION_INDICES[index]],
                                'target_range_rad': [float(target.min()), float(target.max())],
                                'measured_range_rad': [float(measured.min()), float(measured.max())],
                                'mean_abs_difference_rad': float(difference.mean())})
    report = {'scope': 'diagnostic only; no automatic deletion, relabeling or calibration inference',
              'dataset_sha256': sha256(args.dataset/'manifest.json'),
              'thresholds': {'target_range_max_rad': 1e-5, 'measured_range_max_rad': .05,
                             'median_abs_difference_min_rad': .25},
              'episodes': len(manifest['episodes']), 'flagged_channels': flagged}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'flagged_channels': len(flagged),
                      'episode_counts_by_split': dict(Counter(split for _, split in
                           {(r['episode'], r['split']) for r in flagged}))}))


if __name__ == '__main__':
    main()
