# Hardware, startup and recovery knowledge

Observed September 11-12, 2026. Reconfirm on each session.

| Role | Address | Environment observed |
| --- | --- | --- |
| Base / real-time host | 172.16.0.50 (SSH), 172.16.16.50 wired | ROS Humble; kernel 5.15.148-rt-tegra; libfranka 0.20.4 |
| Arm / sensor host | 172.16.0.100 (SSH) | ROS Jazzy; kernel 6.17.0-1032-oem; no real-time kernel |
| Mobile platform | 172.16.16.10 | TMR drive/spine interfaces |
| Robot-right arm | 172.16.16.11 | Franka Desk / FCI |
| Robot-left arm | 172.16.16.12 | Franka Desk / FCI |

On the RT host, source `/opt/ros/humble/setup.bash` and
`~/ros2_ws/install/setup.bash`. On the sensor host, source `~/tmr_env.sh`;
observed ROS domain 0 with CycloneDDS. The base LiDAR relay used domain 31 and
localhost transport. Do not assume that cross-host ROS discovery is working
because SSH or HTTPS works. Mullvad local-network sharing was necessary on the
operator laptop; keep network exceptions confined to the authorized LAN.

## REST and control ownership

Inspect `/admin/api/system-status`, `/api/fci`, `/api/system/control-token` and
the installed OpenAPI at `/api/deskapi.json`. Use the site's credentials through
a password prompt. The tested normal API sequence was:

1. `POST /api/system/control-token:take`, JSON owner and a 2-second timeout.
2. Use returned token in `X-Control-Token`, without logging it.
3. `POST /api/fci:activate`; verify activation and read measured state.
4. `POST /api/system/control-token:release` in cleanup; observed HTTP 204.

Locked brakes / Safe Torque Off require the site's normal initialization.
Release of the emergency button alone does not establish full readiness.
`POST /api/arm/joints:unlock` once timed out while the operation continued;
read status before considering another request. `/api/arm/joints:lock` is a
normal API endpoint; inspect its installed schema before use. A successful token
release does not prove brakes are locked.

A Desk owner can block normal SSH control. Prefer its normal release. The
vendor's supported force-request procedure requires physical confirmation on
the arm's blue Pilot button; it is not a software bypass. We found this path
but did not execute a force takeover. Never forge another owner's token or
disable the ownership/safety mechanism.

If drive and spine both report internal-interface timeouts while the dashboard
and SSH respond, diagnose the internal services/bus separately from Wi-Fi.
Do not repeatedly command motion to test service health. Use the guide's
startup/recovery instructions; do not independently alter firmware or safety settings.

## Arm execution and grippers

The non-RT host refused `RealtimeConfig::kEnforce` before movement. The same
motion helper ran on the RT host without weakening that requirement. Measured
pose and error gates precede each finite step. Preserve this distinction from
the read-only probe's `kIgnore`.

The installed Robotiq configuration maps LEFT to FTDI `DAANVRU5` and RIGHT to
`DAANTK6Q`, via `/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_<serial>-if00-port0`.
Confirm side and exclusive serial ownership before using it.

Observed Modbus RTU: 115200 baud, 8N1, slave 9; function 3 reads six registers
at 0x07D0, with response length and CRC checked. Writes use function 0x10,
three registers at 0x03E8. Payloads observed in the session:

| Operation | Six data bytes (decimal) |
| --- | --- |
| Reset | 0, 0, 0, 0, 0, 0 |
| Activate | 1, 0, 0, 0, 0, 0 |
| Open at low speed / minimum force setting | 9, 0, 0, 0, 20, 0 |

Activation itself moves the fingers through a cycle. A successful right activation
and open was observed (gSTA=3, fault=0, position=3); no plate grasp was commanded.
Fault 0x09 indicated a communication timeout, not automatically a need to cycle
activation again. The original transactions were inline terminal operations;
these notes are not a physically retested packaged serial driver.

Vendor reference: [Robotiq control registers](https://assets.robotiq.com/website-assets/support_documents/document/online/2F-85_2F-140_Instruction_Manual_Gen_HTML_20190524.zip/2F-85_2F-140_Instruction_Manual_Gen_HTML/Content/4.%20Control.htm).
