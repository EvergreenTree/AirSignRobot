"""Exercise exported inference on recorded validation images using test-only calibration."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from PIL import Image
import torch

from ebim_phase2.data import EpisodeDataset
from ebim_phase2.download import sha256
from ebim_phase2.runtime import Policy
from ebim_phase2.schema import CAMERAS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    policy = Policy(args.checkpoint)
    data = EpisodeDataset(args.dataset, policy.task, 'validation', policy.config.chunk_size, policy.stats)
    episode = data.episodes[0]['id']
    root = args.dataset/episode
    frame = int(np.load(root/'obs_frames.npy', mmap_mode='r')[0])
    state = np.load(root/'state.npy', mmap_mode='r')[frame].tolist()
    pictures = np.load(root/'rgb.npy', mmap_mode='r')[0]
    # These limits only exercise software branches. They are deliberately not
    # exported as a usable physical site configuration.
    calibration = {'calibration_id': 'OFFLINE_TEST_FIXTURE_NOT_FOR_ROBOT',
                   'native_base_velocity_frame':'body',
                   'joint_action_scale': [1.]*14, 'joint_action_offset_rad': [0.]*14,
                   'joint_lower': [-10]*14, 'joint_upper': [10]*14, 'joint_speed_rad_s': [.4]*14,
                   'gripper_open_knuckle': [0,0], 'gripper_closed_knuckle': [.8,.8],
                   'gripper_open_command': [1,1], 'gripper_closed_command': [0,0],
                   'spine_lower_m': 0, 'spine_upper_m': 1, 'spine_speed_m_s': .02,
                   'base_linear_speed_m_s': .1, 'base_yaw_speed_rad_s': .2,
                   'max_force_n': 1e6, 'max_observation_age_s': 1, 'max_camera_skew_s': .075,
                   'spine_scale_m_per_native': .001 if policy.task == 2 else 1., 'spine_offset_m': 0.}
    with tempfile.TemporaryDirectory(prefix='airsign-inference-check-') as directory:
        images = {}
        for name, rgb in zip(CAMERAS, pictures):
            path = Path(directory)/f'{name}.png'
            Image.fromarray(rgb).save(path)
            images[name] = {'path': str(path), 'timestamp_s': 1000.}
        record = {'task': policy.task, 'episode_id': 'offline-contract-check', 'timestamp_s': 1000.,
                  'state': state, 'state_names': policy.schema['observation.state']['names'],
                  'images': images, 'collision_imminent': False, 'spine_height_m': .3}
        shadow = policy.infer(record, now=1000.)
        assert shadow['status'] == 'shadow_native_proposal' and shadow['action'] is None
        del policy
        bounded = Policy(args.checkpoint, calibration=calibration)
        latency = []
        for index in range(25):
            timestamp = 1000.+.05*index
            record['timestamp_s'] = timestamp
            for camera in images.values():
                camera['timestamp_s'] = timestamp
            before = time.perf_counter()
            result = bounded.infer(record, now=timestamp)
            latency.append((time.perf_counter()-before)*1000)
            assert result['status'] == 'bounded_proposal' and len(result['action']) == 23
            assert np.isfinite(result['action']).all()
        stale = bounded.infer(record, now=1010.)
        assert stale['status'] == 'no_command' and stale['action'] is None
        report = {'scope': 'offline recorded-image software integration; no physical robot validation',
                  'task': bounded.task, 'episode': episode, 'shadow_status': shadow['status'],
                  'checkpoint_sha256':sha256(args.checkpoint),
                  'dataset_sha256':sha256(args.dataset/'manifest.json'),
                  'bounded_status': result['status'], 'stale_status': stale['status'],
                  'test_only_spine_conversion': True,
                  'image_input':'already-letterboxed released validation images',
                  'input_image_shape':list(pictures[0].shape),
                  'warm_end_to_end_median_ms': float(np.median(latency[5:])),
                  'warm_end_to_end_p95_ms': float(np.percentile(latency[5:], 95)),
                  'timing_scope': 'prepared-image PNG read/decode, preprocessing, model, temporal ensemble, limits; no native camera acquisition/transport or robot IO'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
