#!/usr/bin/env python3
"""Stationary ROS 2 sensor probe. Creates subscriptions only; never commands hardware.

Run with the host's system Python after sourcing its ROS environment.
Output is an observation report, not certification that motion is safe.
"""
import argparse
import array
import json
import math
import socket
import statistics
import struct
import sys
import time
import zlib
from pathlib import Path


def save_image(msg, folder, name):
    """Encode camera data without the host's optional NumPy/OpenCV stack."""
    formats = {"rgb8": (3, 8, 2), "bgr8": (3, 8, 2),
               "rgba8": (4, 8, 6), "bgra8": (4, 8, 6),
               "mono8": (1, 8, 0), "8UC1": (1, 8, 0),
               "16UC1": (2, 16, 0), "mono16": (2, 16, 0),
               "32FC1": (4, None, None)}
    if msg.encoding not in formats:
        raise ValueError("Unsupported image encoding: " + msg.encoding)
    stride, bits, color = formats[msg.encoding]
    raw = bytes(msg.data)
    rows = [bytearray(raw[y * msg.step:y * msg.step + msg.width * stride])
            for y in range(msg.height)]
    extra = {}
    if msg.encoding in ("16UC1", "32FC1"):
        depths = array.array("H" if msg.encoding == "16UC1" else "f")
        depths.frombytes(b"".join(rows))
        if bool(msg.is_bigendian) != (sys.byteorder == "big"):
            depths.byteswap()
        sampled = depths[::max(1, len(depths) // 10000)]
        good = [v for v in sampled if math.isfinite(v) and v > 0]
        extra["sampled_valid_depth_fraction"] = len(good) / len(sampled) if sampled else 0
        extra["sampled_depth_median_native_units"] = statistics.median(good) if good else None
    if bits is None:
        path = folder / (name + ".bin")
        path.write_bytes(raw)
        extra.update(step=msg.step, is_bigendian=bool(msg.is_bigendian))
        return path, extra
    for row in rows:
        if msg.encoding in ("bgr8", "bgra8"):
            row[0::stride], row[2::stride] = row[2::stride], row[0::stride]
        elif bits == 16 and not msg.is_bigendian:
            row[0::2], row[1::2] = row[1::2], row[0::2]
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    header = struct.pack(">IIBBBBB", msg.width, msg.height, bits, color, 0, 0, 0)
    path = folder / (name + ".png")
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", zlib.compress(b"".join(b"\0" + row for row in rows)))
                     + chunk(b"IEND", b""))
    return path, extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--role", choices=("all", "arm", "base"), default="all",
                        help="Read high-bandwidth cameras only on their owning host")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 30:
        parser.error("--seconds must be between 1 and 30")
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, JointState, LaserScan
    from nav_msgs.msg import Odometry
    from diagnostic_msgs.msg import DiagnosticArray

    specs = {
        "/head_camera/zed/rgb/color/rect/image": Image,
        "/head_camera/zed/depth/depth_registered": Image,
        "/head_camera/zed/rgb/color/rect/camera_info": CameraInfo,
        "/wrist_camera_left/color/image_raw": Image,
        "/wrist_camera_right/color/image_raw": Image,
        "/wrist_camera_left/depth/image_rect_raw": Image,
        "/wrist_camera_right/depth/image_rect_raw": Image,
        "/wrist_camera_left/color/camera_info": CameraInfo,
        "/wrist_camera_right/color/camera_info": CameraInfo,
        "/left/joint_states": JointState,
        "/right/joint_states": JointState,
        "/left/gripper/joint_states": JointState,
        "/right/gripper/joint_states": JointState,
        "/swerve_drive_controller/odom": Odometry,
        "/lidar_front/scan": LaserScan,
        "/lidar_rear/scan": LaserScan,
        "/diagnostics": DiagnosticArray,
    }
    if args.role != "all":
        specs = {topic: kind for topic, kind in specs.items()
                 if topic == "/diagnostics" or
                 (args.role == "arm" and topic.startswith(("/wrist_", "/left/", "/right/"))) or
                 (args.role == "base" and topic.startswith(("/head_", "/lidar_", "/swerve_")))}
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"host": socket.gethostname(), "started_at_s": time.time(),
              "motion_commands_sent": False, "topics": {}}
    rclpy.init()
    node = rclpy.create_node("airsign_stationary_probe")
    subscriptions = []

    def callback(topic, msg):
        record = report["topics"][topic]
        record["messages"] += 1
        record["last_received_at_s"] = time.time()
        record.setdefault("first_received_at_s", record["last_received_at_s"])
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        record["last_message_stamp_s"] = stamp
        record["frame_id"] = msg.header.frame_id
        if isinstance(msg, Image):
            record.update(width=msg.width, height=msg.height, encoding=msg.encoding)
            if "image_file" not in record and "image_save_error" not in record:
                try:
                    name = topic.strip("/").replace("/", "__")
                    path, extra = save_image(msg, args.output, name)
                    record["image_file"] = str(path)
                    record["saved_image_stamp_s"] = stamp
                    record.update(extra)
                except Exception as error:
                    record["image_save_error"] = str(error)
        elif isinstance(msg, CameraInfo):
            record.update(width=msg.width, height=msg.height, k=list(msg.k), d=list(msg.d))
        elif isinstance(msg, LaserScan):
            if "scan_file" not in record:
                path = args.output / (topic.strip("/").replace("/", "__") + ".json")
                path.write_text(json.dumps({
                    "stamp_s": stamp, "frame_id": msg.header.frame_id,
                    "angle_min": msg.angle_min, "angle_increment": msg.angle_increment,
                    "range_min": msg.range_min, "range_max": msg.range_max,
                    "ranges": [float(v) if math.isfinite(v) else None for v in msg.ranges],
                }, allow_nan=False) + "\n")
                record["scan_file"] = str(path)
            valid = [(i, float(v)) for i, v in enumerate(msg.ranges)
                     if math.isfinite(v) and msg.range_min <= v <= msg.range_max]
            nearest = min(valid, key=lambda item: item[1]) if valid else None
            record.update(valid_ranges=len(valid), total_ranges=len(msg.ranges),
                          minimum_range_m=nearest[1] if nearest else None,
                          minimum_angle_in_sensor_frame_rad=msg.angle_min + nearest[0] * msg.angle_increment if nearest else None)
        elif isinstance(msg, JointState):
            record.update(names=list(msg.name), positions=list(msg.position),
                          velocities=list(msg.velocity), efforts=list(msg.effort))
        elif isinstance(msg, Odometry):
            record["child_frame_id"] = msg.child_frame_id
            record["linear_velocity"] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
            record["angular_velocity"] = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        elif isinstance(msg, DiagnosticArray):
            records = record.setdefault("statuses", {})
            for status in msg.status:
                level = status.level[0] if isinstance(status.level, bytes) else int(status.level)
                records[status.name] = {"level": level, "message": status.message}

    try:
        for topic, message_type in specs.items():
            report["topics"][topic] = {"messages": 0}
            subscriptions.append(node.create_subscription(
                message_type, topic, lambda msg, name=topic: callback(name, msg),
                qos_profile_sensor_data))
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        for topic, record in report["topics"].items():
            record["publishers"] = [
                {"node": info.node_name, "namespace": info.node_namespace}
                for info in node.get_publishers_info_by_topic(topic)]
        report["nodes"] = node.get_node_names_and_namespaces()
        report["services"] = node.get_service_names_and_types()
        report["finished_at_s"] = time.time()
        path = args.output / "report.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"report": str(path), "topics": report["topics"]}, allow_nan=False))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
