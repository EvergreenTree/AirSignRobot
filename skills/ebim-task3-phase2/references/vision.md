# Frames, wrist vision and limits of the calibration

Two D405 wrist cameras provided 640x480 RGB and raw 16UC1 depth (millimetres)
at approximately 30 Hz on the sensor host. Current launch mapped
`wrist_camera_left` to serial 409122272639 and `wrist_camera_right` to
409122274492. The older guide's mapping differed. We resolved current labels
with measured arm poses and cross-camera projection: left was lower, right higher.

Live TF contained an identity transform from one wrist camera link to the other.
It was a placeholder, not physical hand-eye calibration. Keep within-camera
factory extrinsics distinct from camera-to-arm and arm-to-world transforms.
Raw depth was not RGB registered. Reproject with factory calibration before
using RGB keypoints. `tasks/perception.py` expects already registered depth.

| Depth camera | fx = fy | cx | cy |
| --- | --- | --- | --- |
| Left | 387.386047 | 319.823212 | 238.512772 |
| Right | 384.862213 | 317.712128 | 240.703033 |

Read intrinsics live rather than assuming these historical values. Right RGB
fx/fy were 392.054047/391.133423 with nonzero distortion, different from depth.

The vendor model used was `franka_vision_and_manipulation_kit`'s
`robots/vision_and_manipulation_kit.urdf.xacro`, with mobile platform enabled,
ROS control disabled, FR3 v2_1 arms. It is not redistributed here. The older
minimal `robot.urdf` lacked the actual gripper and camera geometry.

Nominal arm mounting poses, relative to spine zero, were:

| Arm | xyz (m) | roll, pitch, yaw (rad) |
| --- | --- | --- |
| Right | 0.442, -0.05018, 0.5 | 0.8933480924, -0.1745607426, 0.4633450616 |
| Left | 0.442, 0.05018, 0.5 | -0.8933480924, -0.1745607426, -0.4633450616 |

Actual spine height was unavailable during the plate approach. The common
unknown height cancels in relative table/tool comparisons, but these coordinates
are not absolute room heights. Arm state matrices are column-major. The measured
flange-to-configured-EE transform had identity rotation and z=0.174 m; this is
not proof that the EE exactly equals the finger contact point.

Nominal camera chain: flange to gripper adapter Rz=-90 degrees, z=0.011 m;
camera casing xyz=(0,0.027,0.0363); mount xyz=(0,0.0186,0.008),
rpy=(0,-70,-90) degrees; bottom-screw to camera link
xyz=(0.01085,0.009,0.021); optical rpy=(-90,0,-90) degrees.
Measured FK and cross-camera feature projection supported the mapping, but
no full contact or mesh-collision calibration was completed.

Historical plate center estimate in the spine-zero frame was approximately
(1.004,-0.154,0.334) m. It came from agent-selected image/depth regions and
nominal geometry. It must never be used as a fresh target. The right tool
approached with its existing tilted orientation; thin-rim contact, gripper
rotation and two-arm grasp remained unresolved. Wall fitting and joint-center
clearance estimates are partial geometry checks, not full swept-volume proofs.
