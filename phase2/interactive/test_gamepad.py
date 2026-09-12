import dataclasses
import json
from pathlib import Path
import subprocess
import threading
import time

import pytest

from camera_server import LatestFrame, server
from gamepad import Gamepad, Worker


@pytest.fixture
def worker():
    import os
    binary = os.environ.get('AIRSIGN_GAMEPAD_MOCK', '/private/tmp/airsign-gamepad-build/gamepad_mock')
    w = Worker([binary, '--mock'])
    yield w
    w.close()


@pytest.fixture
def pad(worker, tmp_path):
    fixture = Path(__file__).resolve().parents[1]/'task3_submission/evidence/images/before-final-step.png'
    httpd = server(LatestFrame(fixture), 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    result = Gamepad(worker, f'http://127.0.0.1:{httpd.server_port}/frame', tmp_path)
    yield result
    httpd.shutdown()
    httpd.server_close()


def test_persistent_worker_and_post_motion_image(pad):
    pid = pad.worker.process.pid
    first = pad.look()
    result = pad.act(first['frame'], 'R:X+')
    assert result['result']['pose'][12] == pytest.approx(.002)
    assert result['next']['frame'] != first['frame']
    back = pad.act(result['next']['frame'], 'R:X-/fine')
    assert back['result']['pose'][12] == pytest.approx(.001)
    assert pad.worker.process.pid == pid
    assert back['latency']['frame_to_dispatch_ms'] < 5000


def test_stale_frame_never_moves(pad):
    frame = pad.look()['frame']
    pad.view = dataclasses.replace(pad.view, deadline_ns=time.monotonic_ns()-1)
    with pytest.raises(ValueError, match='expired'):
        pad.act(frame, 'R:X+')
    assert pad.worker.request('READ')['pose'][12:15] == [0,0,0]


def test_one_use_even_when_action_invalid(pad):
    frame = pad.look()['frame']
    with pytest.raises(ValueError, match='configured arm'):
        pad.act(frame, 'L:X+')
    with pytest.raises(ValueError, match='consumed'):
        pad.act(frame, 'R:X+')
    assert pad.worker.request('READ')['pose'][12:15] == [0,0,0]


def test_old_frame_cannot_replay(pad):
    frame = pad.look()['frame']
    pad.act(frame, 'R:Y+')
    with pytest.raises(ValueError, match='consumed'):
        pad.act(frame, 'R:Y+')
    assert pad.worker.request('READ')['pose'][13] == pytest.approx(.002)


def test_stop_does_not_require_a_frame(pad):
    assert pad.act('nonexistent', 'STOP')['event'] == 'stop_requested'
    frame = pad.look()['frame']
    pad.stop()
    with pytest.raises(ValueError, match='consumed'):
        pad.act(frame,'R:X+')


@pytest.mark.parametrize('arguments', [
    '{anchor} {future} 0 0.05 0.8',
    '{anchor} {past} 0 0.002 0.8',
    '{anchor} {future} 0 nan 0.8',
    '{anchor} {future} 0 0.002 10',
])
def test_native_rejects_bad_bounds_and_expiry(worker, arguments):
    anchor = worker.request('READ')['id']
    now = worker.request('PING')['clock_ns']
    with pytest.raises(RuntimeError, match='bounds_or_expired'):
        worker.request('MOVE',arguments.format(anchor=anchor,future=now+2_000_000_000,past=now-1))
    assert worker.request('READ')['pose'][12:15] == [0,0,0]


def test_native_stop_invalidates_old_anchor(worker):
    anchor = worker.request('READ')['id']
    worker.request('STOP')
    now = worker.request('PING')['clock_ns']
    with pytest.raises(RuntimeError, match='anchor_or_deadline'):
        worker.request('MOVE',f'{anchor} {now+2_000_000_000} 0 .002 .8')


def test_busy_is_rejected_without_queued_move(worker):
    anchor = worker.request('READ')['id']
    now = worker.request('PING')['clock_ns']
    result = []
    thread = threading.Thread(target=lambda: result.append(worker.request('MOVE',f'{anchor} {now+2_000_000_000} 0 .002 .8')))
    thread.start()
    time.sleep(.12)
    with pytest.raises(RuntimeError, match='busy_no_queue'):
        worker.request('READ')
    thread.join()
    assert result[0]['status'] == 'complete'
    assert worker.request('READ')['pose'][12] == pytest.approx(.002)


def test_eof_interrupts_native_pulse():
    import os
    binary = os.environ.get('AIRSIGN_GAMEPAD_MOCK', '/private/tmp/airsign-gamepad-build/gamepad_mock')
    p = subprocess.Popen([binary,'--mock'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    assert json.loads(p.stdout.readline())['event'] == 'ready'
    p.stdin.write('READ 1\n'); p.stdin.flush()
    state = json.loads(p.stdout.readline())
    p.stdin.write(f"MOVE 2 1 {state['clock_ns']+2_000_000_000} 0 .005 1\n"); p.stdin.flush()
    assert json.loads(p.stdout.readline())['event'] == 'started'
    p.stdin.close()
    rows = [json.loads(x) for x in p.stdout]
    assert p.wait(timeout=3) == 3
    result = next(x for x in rows if x['event']=='result')
    assert result['status'] == 'stopped' and result['pose'][12] < .005


def test_fixture_is_refused_for_live_worker(pad):
    original = pad.worker.backend
    pad.worker.backend = 'franka'  # Exercise the gate without ever constructing a live backend.
    try:
        with pytest.raises(ValueError, match='refuses fixture'):
            pad.look()
    finally:
        pad.worker.backend = original


def test_no_blind_action_after_camera_loss(pad):
    frame = pad.look()['frame']
    pad.camera_url += '/missing'
    with pytest.raises(Exception):
        pad.act(frame,'R:Z-/fine')
    assert pad.view is None
    assert pad.metrics[-1]['next_frame_ok'] is False
    assert pad.worker.request('READ')['pose'][14] == pytest.approx(-.001)
