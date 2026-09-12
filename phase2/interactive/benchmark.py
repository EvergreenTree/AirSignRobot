"""Measure the persistent mock + local fixture pipeline; never drives a robot."""
import argparse
import json
from pathlib import Path
import statistics
import threading
import time

from camera_server import LatestFrame, server
from gamepad import Gamepad, Worker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--decision-delay', type=float, default=.1)
    args=parser.parse_args()
    if not 2 <= args.steps <= 100 or not 0 <= args.decision_delay <= 3:
        parser.error('steps 2-100; synthetic decision delay 0-3 seconds')
    fixture=Path(__file__).resolve().parents[1]/'task3_submission/evidence/images/before-final-step.png'
    httpd=server(LatestFrame(fixture),0)
    threading.Thread(target=httpd.serve_forever,daemon=True).start()
    worker=Worker([str(args.worker),'--mock'])
    try:
        pad=Gamepad(worker,f'http://127.0.0.1:{httpd.server_port}/frame',args.output)
        view=pad.look()
        for i in range(args.steps):
            time.sleep(args.decision_delay)  # Synthetic scheduling delay, not measured model reasoning.
            view=pad.act(view['frame'],'R:X+' if i%2==0 else 'R:X-')['next']
        report={'scope':'local timing mock and static image fixture; no model reasoning, SSH, ROS or robot',
                'steps':args.steps,'worker_processes':1,'synthetic_decision_delay_ms':args.decision_delay*1000,
                'all_cycles_below_5s':all(x['cycle_ms']<5000 for x in pad.metrics)}
        for key in ('decision_ms','frame_to_dispatch_ms','dispatch_to_result_ms','next_frame_ms','cycle_ms'):
            values=sorted(x[key] for x in pad.metrics)
            report[key]={'median':statistics.median(values),'max':max(values),
                         'p95':values[min(len(values)-1,int(.95*len(values)))]}
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        worker.close()
        httpd.shutdown(); httpd.server_close()


if __name__=='__main__':
    main()
