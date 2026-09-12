"""Persistent latest-frame server. ROS path is subscription-only; fixture path is labelled mock.

Bind to loopback and use an SSH tunnel. Never serve this unauthenticated endpoint
on a public/LAN interface. Camera source stamps must use the host wall clock.
"""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time


class LatestFrame:
    def __init__(self, fixture=None):
        self.condition = threading.Condition()
        self.fixture = Path(fixture) if fixture else None
        self.fixture_bytes = self.fixture.read_bytes() if fixture else None
        self.sequence = 0
        self.message = None
        self.received = 0.
        self.stamp = 0.

    def receive(self, message):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        with self.condition:
            if stamp <= self.stamp:
                return
            self.message, self.stamp, self.received = message, stamp, time.monotonic()
            self.sequence += 1
            self.condition.notify_all()

    def frame(self):
        if self.fixture:
            with self.condition:
                self.sequence += 1
                seq = self.sequence
            return dict(frame_id=str(seq), simulated=True, age_ms=0., encoding_ms=0.,
                        image_base64=base64.b64encode(self.fixture_bytes).decode(),
                        suffix=self.fixture.suffix, source='static fixture, no live camera')
        requested = time.time()
        deadline = time.monotonic() + .8
        # Demand a newly captured frame, not a backlog/unchanged cached image.
        with self.condition:
            while self.message is None or self.stamp < requested:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('no new camera frame; check source clock and ROS publisher')
                self.condition.wait(remaining)
            msg, stamp, received, seq = self.message, self.stamp, self.received, self.sequence
        now = time.time()
        if not 0 <= now-stamp <= .25 or time.monotonic()-received > .25:
            raise ValueError('camera stamp/receipt not fresh in host clock domain')
        import cv2
        import numpy as np
        if msg.encoding not in ('rgb8', 'bgr8') or msg.width <= 0 or msg.height <= 0:
            raise ValueError('expected RGB8/BGR8 camera data')
        start = time.monotonic()
        rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
        rgb = rows[:, :msg.width*3].reshape(msg.height, msg.width, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else rgb
        ok, encoded = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise ValueError('JPEG encoding failed')
        age = max(time.time()-stamp, time.monotonic()-received)
        if not 0 <= age <= .25:
            raise ValueError('frame expired during encoding')
        return dict(frame_id=str(seq), simulated=False, age_ms=age*1000,
                    encoding_ms=(time.monotonic()-start)*1000, source_stamp_s=stamp,
                    image_base64=base64.b64encode(encoded).decode(), suffix='.jpg',
                    width=msg.width, height=msg.height, source=msg.header.frame_id)


def server(cache, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/frame':
                self.send_error(404)
                return
            try:
                result, code = cache.frame(), 200
            except (ValueError, TimeoutError) as error:
                result, code = {'error': str(error)}, 503
            data = json.dumps(result).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--fixture', type=Path)
    source.add_argument('--topic', help='e.g. /wrist_camera_right/color/image_raw')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    cache = LatestFrame(args.fixture)
    node = None
    if args.topic:
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import Image
        rclpy.init()
        node = rclpy.create_node('airsign_latest_wrist')
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        node.create_subscription(Image, args.topic, cache.receive, qos)
        threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    httpd = server(cache, args.port)
    print(json.dumps({'camera_url': f'http://127.0.0.1:{httpd.server_port}/frame',
                      'simulated': bool(args.fixture)}), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        if node:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
