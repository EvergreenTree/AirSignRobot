"""Exercise the actual loopback HTTP service with recorded validation images."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import requests
import torch

from ebim_phase2.schema import CAMERAS
from ebim_phase2.download import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    episode=next(e for e in manifest['episodes'] if e['split']=='validation')
    folder=args.dataset/episode['id']
    frame=int(np.load(folder/'obs_frames.npy',mmap_mode='r')[0])
    record={'task':checkpoint['task'],'episode_id':'offline-http-fixture',
            'state_names':checkpoint['schema']['observation.state']['names'],
            'state':np.load(folder/'state.npy',mmap_mode='r')[frame].tolist(),'images':{}}
    for camera,rgb in zip(CAMERAS,np.load(folder/'rgb.npy',mmap_mode='r')[0]):
        buffer=BytesIO();Image.fromarray(rgb).save(buffer,format='PNG')
        record['images'][camera]={'base64':base64.b64encode(buffer.getvalue()).decode('ascii')}
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    session=requests.Session();session.trust_env=False
    endpoint=f'http://127.0.0.1:{port}'
    with tempfile.TemporaryFile(mode='w+') as log:
        process=subprocess.Popen([sys.executable,'-m','ebim_phase2.runtime','--checkpoint',str(args.checkpoint.resolve()),
                                  '--port',str(port)],stdout=log,stderr=log)
        try:
            ready=False
            for _ in range(120):
                if process.poll() is not None:
                    log.seek(0);raise RuntimeError(log.read())
                try:
                    health=session.get(endpoint,timeout=.5)
                    if health.status_code==200:
                        ready=health.json()['ready'];break
                except requests.RequestException:pass
                time.sleep(.25)
            if not ready:raise TimeoutError('Policy HTTP service did not become ready')
            durations=[]
            for _ in range(15):
                stamp=time.time();record['timestamp_s']=stamp
                for camera in record['images'].values():camera['timestamp_s']=stamp
                start=time.perf_counter()
                response=session.post(endpoint,json=record,timeout=20);response.raise_for_status()
                result=response.json();durations.append((time.perf_counter()-start)*1000)
                assert result['status']=='shadow_native_proposal' and result['action'] is None and result['commands'] is None
            invalid=session.post(endpoint,data='{',timeout=10)
            assert invalid.status_code==400 and invalid.json()['action'] is None
            for malformed in ('null', '[]', '42'):
                bad_shape=session.post(endpoint,data=malformed,timeout=10)
                assert bad_shape.status_code==400 and bad_shape.json()['action'] is None
            # Invalid input resets temporal history; a fresh subsequent request works.
            record['timestamp_s']=time.time()
            for camera in record['images'].values():camera['timestamp_s']=record['timestamp_s']
            assert session.post(endpoint,json=record,timeout=10).json()['status']=='shadow_native_proposal'
            report={'team':'AirSign','task':checkpoint['task'],'scope':'actual loopback HTTP and RGB replay; no physical robot',
                    'validation_episode':episode['id'],'requests':15,'mode':'shadow',
                    'checkpoint_sha256':sha256(args.checkpoint),
                    'dataset_sha256':sha256(args.dataset/'manifest.json'),
                    'malformed_request_status':invalid.status_code,'recovery_after_invalid_input':True,
                    'rejected_non_object_json':['null','[]','42'],
                    'image_input':'already-letterboxed released validation images',
                    'input_image_shape':list(rgb.shape),
                    'timing_scope':'prepared-image base64 JSON over loopback HTTP; no native camera acquisition/transport or robot IO',
                    'warm_http_median_ms':float(np.median(durations[5:])),
                    'warm_http_p95_ms':float(np.percentile(durations[5:],95))}
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
