"""Finite authenticated session for individually reviewed, bounded arm steps."""
import base64
import getpass
import json
import select
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

auth = 'Basic ' + base64.b64encode(('franka:' + getpass.getpass('Desk password: ')).encode()).decode()
ctx = ssl._create_unverified_context()
token = None
last_read_time = 0
can_step = True

def call(path, method='GET', body=None):
    headers = {'Authorization': auth, 'Content-Type': 'application/json;charset=utf-8'}
    if token:
        headers['X-Control-Token'] = token
    request = urllib.request.Request('https://172.16.16.11' + path, method=method,
        data=None if body is None else json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(request, context=ctx, timeout=10) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()

def read():
    global last_read_time
    p = subprocess.run(['/tmp/airsign-probe-franka-state-v2', '172.16.16.11'],
                       capture_output=True, text=True, timeout=8)
    if p.returncode:
        raise RuntimeError(p.stderr)
    state = json.loads(p.stdout)
    Path('/tmp/airsign-live-right-state.json').write_text(p.stdout)
    Path('/tmp/airsign-live-right-expected.txt').write_text(' '.join(map(str, state['O_T_EE'])))
    last_read_time = time.monotonic()
    print('STATE', p.stdout, flush=True)

try:
    code, body = call('/api/system/control-token:take', 'POST', {'owner': 'AirSign SSH', 'timeout': 2})
    print('take_http', code, flush=True)
    if code != 200:
        print('reason', body, flush=True)
        sys.exit(2)
    token = body['token']
    print('activate_fci', call('/api/fci:activate', 'POST'), flush=True)
    time.sleep(1)
    read()
    print('READY: READ, STEP dx dy dz seconds, KEEP, RELEASE; 180s idle timeout', flush=True)
    while select.select([sys.stdin], [], [], 180)[0]:
        line = sys.stdin.readline().strip().split()
        if not line or line[0] == 'RELEASE':
            break
        if line == ['KEEP']:
            print('SESSION_ALIVE', flush=True)
        elif line == ['READ']:
            read()
        elif len(line) == 5 and line[0] == 'STEP':
            if not can_step or time.monotonic() - last_read_time > 90:
                print('STEP_REFUSED: state stale or previous step failed', flush=True)
                continue
            values = [str(float(x)) for x in line[1:]]
            p = subprocess.run(['timeout', '--signal=TERM', '--kill-after=3s', '11s',
                '/tmp/airsign-native-arm-translation', '172.16.16.11', *values,
                '/tmp/airsign-live-right-expected.txt', '--execute'],
                capture_output=True, text=True, timeout=15)
            print('STEP_RESULT', p.returncode, p.stdout, p.stderr, flush=True)
            if p.returncode:
                can_step = False
            read()
        else:
            print('UNRECOGNIZED_COMMAND', flush=True)
finally:
    if token:
        print('release', call('/api/system/control-token:release', 'POST'), flush=True)
