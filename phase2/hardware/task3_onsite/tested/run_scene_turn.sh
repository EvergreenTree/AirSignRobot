#!/usr/bin/env bash
# One finite, operator-supervised step. Caller inspects fresh views beforehand.
label=$1
target=$2
if [[ ! "$label" =~ ^airsign-[a-z0-9-]+$ ]] || [[ -z "$target" ]]; then exit 2; fi
if pgrep -f '^/tmp/airsign-native-scene-turn( |$)' >/dev/null || pgrep -f '^python3 /tmp/airsign-lidar-motion-watch.py( |$)' >/dev/null; then
  echo 'Another scene observer or base session is present'; exit 3
fi
source /opt/ros/humble/setup.bash
source "$HOME/ros2_ws/install/setup.bash"
PYTHONNOUSERSITE=1 python3 /tmp/airsign-lidar-motion-watch.py --seconds 14 >"/tmp/$label-observer.log" 2>&1 &
observer_pid=$!
python3 - <<'PY'
import json,time
start=time.monotonic()
while time.monotonic()-start<2:
    try:
        x=json.load(open('/tmp/airsign-lidar-motion.json'))
        if x['valid'] and x['reference_monotonic']>start-.5 and x['sequence']>=5:
            break
    except Exception:
        pass
    time.sleep(.02)
else:
    raise SystemExit('No new scene reference')
PY
ready=$?
if [ "$ready" -eq 0 ]; then
  /tmp/airsign-native-scene-turn --supervised-right-turn "$target" >"/tmp/$label.log" 2>&1
  result=$?
  cat "/tmp/$label.log"
  cp /tmp/airsign-scene-turn-feedback.jsonl "/tmp/$label-feedback.jsonl"
  cp /tmp/airsign-scene-monitor.jsonl "/tmp/$label-monitor.jsonl"
else
  result=$ready
fi
# Capture post-state even when a gate stops the motion.
ROS_DOMAIN_ID=31 ROS_LOCALHOST_ONLY=1 python3 /tmp/airsign-probe-hardware.py --role base --seconds 1 --output "/tmp/$label-after" >"/tmp/$label-after.log" 2>&1
cp /tmp/airsign-live-lidar.json "/tmp/$label-after/lidar-live.json"
wait "$observer_pid"
cp /tmp/airsign-lidar-motion.json.jsonl "/tmp/$label-lidar.jsonl"
exit "$result"
