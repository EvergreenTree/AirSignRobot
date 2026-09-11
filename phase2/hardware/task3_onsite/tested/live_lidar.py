"""Subscription-only lidar relay across local ROS transport configurations."""
import json
import math
import os
import time
from collections import deque
from pathlib import Path

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

DEST = Path('/tmp/airsign-live-lidar.json')
rclpy.init()
node = rclpy.create_node('airsign_lidar_observer')
scans = {}
history = {side: deque(maxlen=3) for side in ('front','rear')}


def receive(side, msg):
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    if side in scans and stamp <= scans[side]['stamp']:
        return
    raw = [float(v) if math.isfinite(v) and msg.range_min <= v <= msg.range_max
           else None for v in msg.ranges]
    window = history[side]
    if window and (len(window[-1][1]) != len(raw) or stamp-window[-1][0] > .08):
        window.clear()
    window.append((stamp,raw))
    filtered = None
    if len(window) == 3 and stamp-window[0][0] <= .1:
        filtered = []
        for values in zip(*(row for _,row in window)):
            median = sorted(v if v is not None else math.inf for v in values)[1]
            filtered.append(median if math.isfinite(median) else None)
    scans[side] = {
        'received_monotonic': time.monotonic(),
        'stamp': stamp,
        'frame': msg.header.frame_id,
        'angle_min': msg.angle_min,
        'angle_increment': msg.angle_increment,
        'range_min': msg.range_min,
        'range_max': msg.range_max,
        'ranges': raw,
        'filtered_ranges': filtered,
        'filter': 'three_scan_temporal_median',
        'filter_span_s': stamp-window[0][0],
    }


subscriptions = [node.create_subscription(
    LaserScan, '/lidar_' + side + '/scan',
    lambda msg, side=side: receive(side, msg), qos_profile_sensor_data,
) for side in ('front', 'rear')]


def save():
    temp = DEST.with_suffix('.tmp')
    temp.write_text(json.dumps({'written_monotonic': time.monotonic(),
                                'scans': scans}, allow_nan=False))
    os.replace(temp, DEST)


timer = node.create_timer(0.05, save)
try:
    rclpy.spin(node)
finally:
    node.destroy_node()
    rclpy.shutdown()
