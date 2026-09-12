# Minimal action prompt

Use with the persistent gamepad bridge after the operator has established an
attended session, verified the arm-base axes, and reviewed the allowed workspace.
Keep planning and calibration outside this fast loop.

```text
Goal: approach the already identified plate edge. Do not grasp or lift yet.
You receive one fresh wrist image, its frame ID, measured pose and expiry.
Choose exactly one action: R:X+, R:X-, R:Y+, R:Y-, R:Z+, R:Z-, or STOP.
Axes are the RIGHT ARM BASE axes, not image up/down or room directions.
Default pulse: 2 mm over 0.8 s. Append /fine for 1 mm.
Output only: <frame-id> <action>
Use STOP for uncertainty, contact, obstruction, an unsuitable orientation or lost view.
One frame permits one pulse. Inspect the returned new image before another action.
Decision target: 3 seconds. Five seconds after the captured view, dispatch expires.
If the frame expires, request LOOK. Never resend the same frame or batch actions.
```

For a left-arm session use the configured `L:` vocabulary and verified left-base
axes. This prototype selects one arm for the session; it does not coordinate
simultaneous bimanual motion. Closing the gripper, wrist rotation, base motion and
contact approach require separately implemented and validated primitives.

An agent tool wrapper can send a line to the already-running bridge and display
the `next.image` file in the same tool invocation. Avoid starting a new shell,
SSH connection, camera subscription or native process for each decision. A model
timeout is enforced by rejecting the action; this prompt cannot guarantee how
long a hosted model or its tool system takes to return.
