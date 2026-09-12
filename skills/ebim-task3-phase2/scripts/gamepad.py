"""One persistent worker plus a latest-frame camera; no queued actions or implicit retries."""
import argparse
import base64
from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
import uuid


class Worker:
    def __init__(self, argv):
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1)
        self.lock = threading.Lock()
        self.disconnected = threading.Event()
        self.pending = {0: queue.Queue()}
        self.sequence = 0
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        try:
            ready = self.pending[0].get(timeout=10)
            if ready['event'] != 'ready':
                raise RuntimeError('worker did not become ready')
            self.backend = ready['backend']
        except Exception:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                event = json.loads(line)
                event['received_ns'] = time.monotonic_ns()
                with self.lock:
                    recipient = self.pending.get(event['id'])
                if recipient:
                    recipient.put(event)
        finally:
            self.disconnected.set()
            with self.lock:
                for recipient in self.pending.values():
                    recipient.put({'event': 'disconnected'})

    def is_running(self):
        return not self.disconnected.is_set() and self.process.poll() is None

    def request(self, op, arguments='', timeout=3.):
        with self.lock:
            if not self.is_running():
                raise RuntimeError('worker exited')
            self.sequence += 1
            seq = self.sequence
            recipient = self.pending[seq] = queue.Queue()
            sent = time.monotonic_ns()
            self.process.stdin.write(f'{op} {seq} {arguments}'.rstrip()+'\n')
            self.process.stdin.flush()
        started = None
        try:
            deadline = time.monotonic() + timeout
            while True:
                event = recipient.get(timeout=max(.001, deadline-time.monotonic()))
                if event['event'] == 'started':
                    started = event
                    continue
                if event['event'] in ('fault', 'rejected', 'disconnected'):
                    raise RuntimeError(json.dumps(event))
                event.update(sent_ns=sent, started=started)
                return event
        finally:
            with self.lock:
                self.pending.pop(seq, None)

    def close(self):
        if self.process.poll() is None:
            try:
                self.request('QUIT', timeout=.5)
            except Exception:
                pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=3)
        if self.process.stdin:
            self.process.stdin.close()


@dataclass(frozen=True)
class View:
    id: str
    anchor: int
    deadline_ns: int
    remote_offset_ns: int
    capture_lower_ns: int
    issued_ns: int


class Gamepad:
    def __init__(self, worker, camera_url, output, arm='R', live=False):
        if arm not in ('L', 'R'):
            raise ValueError('arm must be L or R')
        if worker.backend == 'franka' and not live:
            raise ValueError('live worker requires explicit live mode')
        if live and worker.backend != 'franka':
            raise ValueError('live mode cannot use a mock worker')
        self.worker, self.camera_url, self.output, self.arm = worker, camera_url, Path(output), arm
        self.output.mkdir(parents=True, exist_ok=True)
        self.view = None
        self.guard = threading.Lock()
        self.stop_generation = 0
        self.metrics = []

    def _look(self):
        self.view = None
        epoch = self.stop_generation
        start = time.monotonic_ns()
        with urllib.request.urlopen(self.camera_url, timeout=1.2) as response:
            data = json.load(response)
        age = float(data['age_ms'])
        if not math.isfinite(age) or not 0 <= age <= 250:
            raise ValueError('camera age invalid')
        if self.worker.backend == 'franka' and data.get('simulated') is not False:
            raise ValueError('live control refuses fixture images')
        # Request start minus reported source age is a conservative capture lower bound.
        # It includes the round trip; no camera/worker clock synchronization is assumed.
        capture = start - int(age*1e6)
        state = self.worker.request('READ')
        pong = self.worker.request('PING')
        rtt = pong['received_ns']-pong['sent_ns']
        if rtt > 250_000_000:
            raise ValueError('worker round trip too slow for a short frame lease')
        # Remote timestamp minus local receive is the conservative end of the offset interval.
        # Subtract another 20 ms for clock-rate/measurement margin; refreshed every view.
        offset = pong['clock_ns']-pong['received_ns']-20_000_000
        deadline = capture + 5_000_000_000
        now = time.monotonic_ns()
        if now >= deadline or epoch != self.stop_generation:
            raise ValueError('view expired or cancelled while being acquired')
        frame_id = uuid.uuid4().hex[:12]
        suffix = data['suffix']
        if suffix not in ('.jpg', '.jpeg', '.png'):
            raise ValueError('unsupported image suffix')
        path = self.output / (frame_id+suffix)
        path.write_bytes(base64.b64decode(data['image_base64'], validate=True))
        self.view = View(frame_id, state['id'], deadline, offset, capture, now)
        return dict(frame=frame_id, image=str(path.resolve()), simulated=data['simulated'],
                    expires_in_ms=max(0,(deadline-now)/1e6), frame_ms=(now-start)/1e6,
                    pose=state['pose'], frame_axes=f'{self.arm} arm-base XYZ (not image directions)',
                    actions=[f'{self.arm}:{axis}{sign}' for axis in 'XYZ' for sign in '+-'],
                    fine_suffix='/fine', stop='STOP')

    def look(self):
        if not self.guard.acquire(blocking=False):
            raise ValueError('busy; actions are never queued')
        try:
            return self._look()
        finally:
            self.guard.release()

    def stop(self):
        self.stop_generation += 1
        self.view = None
        return self.worker.request('STOP', timeout=.5)

    def act(self, frame, action):
        if action == 'STOP':
            return self.stop()
        if not self.guard.acquire(blocking=False):
            raise ValueError('busy; actions are never queued')
        try:
            view, self.view = self.view, None  # Consume once, even on a rejected action.
            now = time.monotonic_ns()
            if view is None or frame != view.id or now >= view.deadline_ns:
                raise ValueError('missing, consumed or expired frame; LOOK again')
            match = re.fullmatch(r'([LR]):([XYZ])([+-])(?:/(fine))?', action)
            if not match or match[1] != self.arm:
                raise ValueError('choose one configured arm-base axis action or STOP')
            distance = (.001 if match[4] else .002) * (1 if match[3] == '+' else -1)
            expiry = view.deadline_ns + view.remote_offset_ns
            result = self.worker.request('MOVE', f'{view.anchor} {expiry} {"XYZ".index(match[2])} {distance} 0.8')
            if result.get('status') != 'complete':
                raise RuntimeError('pulse stopped; inspect state before restarting the worker')
            move_end = time.monotonic_ns()
            metrics = dict(frame=frame, action=action, backend=self.worker.backend,
                           decision_ms=(now-view.issued_ns)/1e6,
                           frame_to_dispatch_ms=(now-view.capture_lower_ns)/1e6,
                           dispatch_to_result_ms=(move_end-now)/1e6,
                           native_pulse_ms=result['pulse_ms'])
            # No blind command chaining: obtaining a post-pulse frame is part of the action API.
            try:
                following = self._look()
                metrics.update(next_frame_ms=following['frame_ms'],
                               cycle_ms=(time.monotonic_ns()-view.issued_ns)/1e6,
                               next_frame_ok=True)
            except Exception:
                metrics['next_frame_ok'] = False
                self._log(metrics)
                raise
            self._log(metrics)
            return dict(result=result, latency=metrics, next=following)
        except Exception:
            self.view = None
            raise
        finally:
            self.guard.release()

    def _log(self, item):
        self.metrics.append(item)
        with (self.output/'latency.jsonl').open('a') as log:
            log.write(json.dumps(item)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--worker', default=json.dumps([str(root/'build/gamepad_mock'), '--mock']),
                        help='JSON array of process argv; never passed to a local shell')
    parser.add_argument('--demo', action='store_true', help='Start an in-process static camera fixture; mock worker only')
    parser.add_argument('--camera', default='http://127.0.0.1:8765/frame')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=('L','R'), default='R')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    argv = json.loads(args.worker)
    if not isinstance(argv,list) or not argv or any(not isinstance(x,str) or not x for x in argv):
        parser.error('--worker must be a nonempty JSON string array')
    if args.demo and (args.live or argv != [str(root/'build/gamepad_mock'), '--mock']):
        parser.error('--demo requires the default mock worker, without --live')
    httpd = worker = None
    try:
        if args.demo:
            from camera_server import LatestFrame, server
            httpd = server(LatestFrame(root/'assets/approach.png'), 0)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            args.camera = f'http://127.0.0.1:{httpd.server_port}/frame'
        worker = Worker(argv)
        pad = Gamepad(worker,args.camera,args.output,args.arm,args.live)
        print(json.dumps(pad.look()),flush=True)
        for line in sys.stdin:
            parts = line.split()
            if parts == ['QUIT']:
                break
            try:
                if parts == ['LOOK']:
                    result = pad.look()
                elif parts == ['STOP']:
                    result = pad.stop()
                elif len(parts) == 2:
                    result = pad.act(*parts)
                else:
                    raise ValueError('Use LOOK, STOP, QUIT or <frame> R:X+ (optionally /fine)')
                print(json.dumps(result),flush=True)
            except Exception as error:
                exited = not worker.is_running()
                print(json.dumps({'error': str(error), 'next_action':
                    'Inspect the worker exit, then restart the bridge session' if exited else 'LOOK or STOP'}),flush=True)
                if exited:
                    return 1
    finally:
        if worker:
            worker.close()
        if httpd:
            httpd.shutdown()
            httpd.server_close()


if __name__ == '__main__':
    raise SystemExit(main())
