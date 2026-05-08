# R1 Pro DiMOS Integration — Setup & Connection Guide

## Overview

This directory contains test scripts for validating DiMOS connectivity to the
Galaxea R1 Pro humanoid robot over ethernet. The robot runs ROS2 Humble on a
Jetson Orin (Ubuntu 22.04 / L4T). The laptop runs Ubuntu 24.04 with ROS2 Jazzy.

**Current status**: Chassis movement, arm control, and keyboard teleop all
working end-to-end through DiMOS adapters. Dual-arm manipulation planning is
in progress.

---

## Network Setup

### Physical Connection
- Connect laptop to robot via ethernet cable
- Robot ethernet port: `eth1` on the robot

### Robot IP (persistent after netplan config)
- Robot `eth1`: `192.168.123.150/24`
- Laptop ethernet (`enxf8e43bb7046c`): `192.168.123.100/24`

### Set laptop ethernet IP (if not already set)
```bash
sudo ip addr add 192.168.123.100/24 dev enxf8e43bb7046c
```

### SSH into robot
```bash
ssh nvidia@192.168.123.150
# password: nvidia
```

### Make robot IP persistent across reboots (already done)
Edit `/etc/netplan/50-cloud-init.yaml` on the robot, add `192.168.123.150/24`
to eth1 addresses:
```yaml
eth1:
  dhcp4: true
  addresses: [192.168.2.150/24, 192.168.123.150/24]
```
Then: `sudo netplan apply`

---

## Robot Startup Procedure

Run these commands on the robot via SSH every session:

```bash
# Step 1: Start CAN bus driver
bash ~/can.sh

# Step 2: Launch full robot stack (ros2_discovery, mobiman, hdas, tools)
cd ~/galaxea/install/startup_config/share/startup_config/script
./robot_startup.sh boot ../sessions.d/ATCStandard/R1PROBody.d/

# Step 3: Wait ~30 seconds for HDAS to fully init (arms open/close = healthy)

# Step 4: Launch Livox MID360 LiDAR driver
#   The R1PROBody.d session config does NOT include the lidar launch — you have
#   to start it by hand each session, otherwise /hdas/lidar_chassis_left has
#   zero publishers and the chassis adapter sits subscribed to silence.
#   Hardware is at 192.168.2.100; verify reachable with `ping 192.168.2.100`.
bash ~/galaxea/install/startup_config/share/startup_config/script/boot/modules/hdas/start_livox_lidar.sh

# Step 5: Verify the head depth stream
#   The signal_camera_head launch publishes RGB but sometimes does NOT publish
#   /hdas/camera_head/depth/depth_registered. If `ros2 topic info` shows 0
#   publishers on that topic, restart the head signal camera launch:
#   bash ~/galaxea/install/startup_config/share/startup_config/script/boot/modules/hdas/start_signal_camera_head.sh

# Step 6: Start chassis gatekeeper (required for chassis control from laptop)
source ~/galaxea/install/setup.bash
export ROS_DOMAIN_ID=41
python3 ~/chassis_gatekeeper.py
```

```bash
# Step 7: Verify everything is publishing (run on the robot or laptop)
source ~/galaxea/install/setup.bash
export ROS_DOMAIN_ID=41
ros2 topic list --no-daemon | grep -E 'hdas|lidar' | head -20
# Expected: /hdas/feedback_arm_left, /hdas/feedback_arm_right, /hdas/lidar_chassis_left, etc.

# Spot-check rates on the streams the chassis adapter consumes:
ros2 topic hz /hdas/lidar_chassis_left                                    # ~10 Hz
ros2 topic hz /hdas/camera_wrist_left/color/image_raw/compressed          # ~15 Hz
ros2 topic hz /hdas/camera_wrist_right/color/image_raw/compressed         # ~15 Hz
ros2 topic hz /hdas/camera_head/left_raw/image_raw_color/compressed       # ~15 Hz
```

### Sensors that should auto-start (and what to do when they don't)

`robot_startup.sh` reads sessions from `R1PROBody.d/` and runs each entry's
launch script. On a clean boot the session brings up:

- **HDAS** (CAN-side: arms, torso, chassis, grippers, IMUs) via
  `start_hdas_r1pro.sh`
- **RealSense wrist cameras** (left + right D405) via
  `start_realsense_camera_r1pro.sh` — reads serials from
  `/opt/galaxea/sensor/realsense/RS_LEFT` and `RS_RIGHT`
- **Head signal camera** (head RGB stereo + depth) via
  `start_signal_camera_head.sh`

The session does NOT bring up the **Livox MID360 LiDAR** — Step 4 above is the
manual workaround. If you want it to launch automatically, add a session
entry under `~/galaxea/install/startup_config/share/startup_config/script/sessions.d/ATCStandard/R1PROBody.d/`
that invokes
`~/galaxea/install/startup_config/share/startup_config/script/boot/modules/hdas/start_livox_lidar.sh`
(this hasn't been pushed upstream — keep the manual step in sync until that
session config gets fixed on the robot).

If a wrist camera reports `RS2_USB_STATUS_BUSY` or repeatedly disconnects
(check `~/.ros/log/realsense2_camera_node_*_*.log`), it's a USB-layer fault
— reseat the cable on that camera and rerun
`start_realsense_camera_r1pro.sh`.

### Robot tmux sessions
| Session | Purpose |
|---|---|
| `ros_discovery` | FastDDS discovery server on port 11811 (for VR/WiFi, not needed for ethernet) |
| `mobiman` | Main motion control stack |
| `hdas` | Hardware abstraction — arms, chassis, torso, grippers |
| `tools` | Utilities |

Check session health: `tmux attach -t hdas` (Ctrl+B D to detach)

---

## Laptop Setup (every session)

```bash
cd ~/Downloads/dimos

source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=$(pwd)/scripts/r1pro_test/fastdds_r1pro.xml
```

Tip: add these to a shell script `scripts/r1pro_test/env.sh` and `source` it.

---

## Chassis Gatekeeper (Key Concept)

The R1 Pro `chassis_control_node` has **three internal gates** that all must be
unlocked simultaneously for chassis movement to work. The gatekeeper runs on the
robot and handles all three, exposing a simple `/cmd_vel` topic for the laptop.

### The 3 Gates

| Gate | What blocks it | How gatekeeper fixes it |
|---|---|---|
| **Gate 1**: Subscriber count | Node skips IK if nobody subscribes to `/motion_control/chassis_speed` | Subscribes to the topic |
| **Gate 2**: `breaking_mode_` flag | HDAS publishes `mode=2` at 200Hz on `/controller`, setting `breaking_mode_=1` | Launch file remaps `/controller` → `/controller_unused`; gatekeeper publishes `mode=5` on `/controller_unused` |
| **Gate 3**: `acc_limit` defaults to zero | `calculateNextVelocity` uses `acc_limit * dt` which stays 0 | Publishes nonzero `TwistStamped` on `/motion_target/chassis_acc_limit` |

### Prerequisites (one-time on robot)
1. Edit `~/galaxea/src/mobiman/launch/r1_pro_chassis_control_launch.py`
2. Uncomment/add: `remappings=[('/controller', '/controller_unused')]`
3. Rebuild and restart mobiman

### Running
```bash
# On robot:
source ~/galaxea/install/setup.bash && export ROS_DOMAIN_ID=41
python3 ~/chassis_gatekeeper.py

# From laptop (test):
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}" --rate 20
```

---

## Verification Tests

Run in order after startup:

```bash
# Test 1: Topic discovery (70 topics expected, ~10s)
python3 scripts/r1pro_test/test_01_topic_discovery.py

# Test 2: Read live arm joint data (safe, read-only)
python3 scripts/r1pro_test/test_02_read_arm_feedback.py

# Test 4: Arm movement (moves joint 0 by 0.3 rad, then returns home)
python3 scripts/r1pro_test/test_04_arm_joint_command.py

# Test 3: Chassis movement — requires chassis_gatekeeper on robot
python3 scripts/r1pro_test/test_03_chassis_command.py

# Test 5: DiMOS ROS layer integration
python3 scripts/r1pro_test/test_05_dimos_ros_layer.py
```

### Test status
| Test | Status | Notes |
|---|---|---|
| 01 topic discovery | PASS | 70 topics visible |
| 02 arm feedback | PASS | 7-joint positions/velocities/efforts streaming |
| 03 chassis command | PASS | Works via chassis_gatekeeper → `/cmd_vel` |
| 04 arm movement | PASS | Joint 0 moves 0.3 rad and returns home |
| 05 DiMOS ROS layer | PASS | DiMOS adapters communicate with robot |

**Important**: Do NOT run tests individually back-to-back with separate
`rclpy.init()`/`rclpy.shutdown()` cycles. FastDDS 3.x (Jazzy) creates new DDS
participants each cycle, which corrupts the robot's Humble DDS nodes. Use
`run_all_tests.py` for sequential testing, or wait 30+ seconds between runs.

---

## DiMOS Integration Architecture

### Adapters
| Component | File | Pattern |
|---|---|---|
| Chassis | `dimos/hardware/drive_trains/r1pro/adapter.py` | `TwistBaseAdapter` — publishes `Twist` to `/cmd_vel` (via gatekeeper) |
| Arms | `dimos/hardware/manipulators/r1pro/adapter.py` | `ManipulatorAdapter` — parameterized by side (left/right) |
| ROS env | `dimos/hardware/r1pro_ros_env.py` | Sets ROS_DOMAIN_ID=41, FastDDS, rmw_fastrtps_cpp |

### Blueprints
| Blueprint | File | Components |
|---|---|---|
| `coordinator_r1pro` | `dimos/control/blueprints/r1pro.py` | Arms + chassis |
| `coordinator_r1pro_arms` | `dimos/control/blueprints/r1pro.py` | Arms only |

### Keyboard Teleop
`dimos/robot/galaxea/r1pro/blueprints/r1pro_keyboard_teleop.py` — keyboard
control of chassis and arms through DiMOS.

---

## Key Topics

| Topic | Type | Direction |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | laptop → gatekeeper (RELIABLE QoS) |
| `/hdas/feedback_arm_left` | `sensor_msgs/JointState` | robot → laptop |
| `/hdas/feedback_arm_right` | `sensor_msgs/JointState` | robot → laptop |
| `/hdas/feedback_chassis` | `sensor_msgs/JointState` | robot → laptop |
| `/hdas/feedback_torso` | `sensor_msgs/JointState` | robot → laptop |
| `/motion_target/target_speed_chassis` | `geometry_msgs/TwistStamped` | gatekeeper → chassis_control_node |
| `/motion_target/target_joint_state_arm_left` | `sensor_msgs/JointState` | laptop → robot |
| `/motion_target/target_joint_state_arm_right` | `sensor_msgs/JointState` | laptop → robot |
| `/motion_target/target_joint_state_torso` | `sensor_msgs/JointState` | laptop → robot |
| `/motion_target/target_position_gripper_left` | `sensor_msgs/JointState` | laptop → robot |
| `/motion_target/target_position_gripper_right` | `sensor_msgs/JointState` | laptop → robot |

---

## Challenges & How We Solved Them

### 1. Finding the robot's IP
Robot had no known IP when connected via ethernet. Used `tcpdump` and `arp -a`
to discover it. Robot's `eth1` had no IPv4 assigned by default — manually
assigned `192.168.123.150/24` with `sudo ip addr add`, then made it persistent
via netplan.

### 2. ROS2 topic discovery failing across machines
**Root causes found (in order):**

**a) `ROS_LOCALHOST_ONLY=1` set in robot's `~/.bashrc`**
The robot was configured to only accept local DDS connections. Changed to
`ROS_LOCALHOST_ONLY=0` in `~/.bashrc` so tmux sessions (which source bashrc)
inherit the correct setting.

**b) CycloneDDS ↔ FastDDS EDP incompatibility**
Tried CycloneDDS on the laptop (ROS2 Jazzy default) thinking it would
interoperate with FastDDS on the robot (ROS2 Humble). Peer discovery (PDP)
worked — tcpdump confirmed packets flowing both ways — but endpoint discovery
(EDP) failed silently. Topics never appeared.

Fix: switch laptop to FastDDS (`RMW_IMPLEMENTATION=rmw_fastrtps_cpp`) to match
the robot.

**c) FastDDS using wrong network interface on laptop**
Laptop has WiFi (`192.168.1.68`), ethernet (`192.168.123.100`), and Tailscale
(`100.78.x.x`). FastDDS multicast was going out the wrong interface.

Fix: `fastdds_r1pro.xml` — a FastDDS profile that binds to `192.168.123.100`
(ethernet) and sets `192.168.123.150:17650` as explicit unicast peer. This
bypasses multicast entirely.

**d) `interfaceWhiteList` renamed in FastDDS 3.x (Jazzy)**
The original XML used `<interfaceWhiteList>` which is FastDDS 2.x syntax.
FastDDS 3.x (shipped with Jazzy) renamed it to `allowlist`. The element was
silently ignored, so interface restriction never applied.

Fix: switched from transport-level interface restriction to locator-based
config (`metatrafficUnicastLocatorList`, `defaultUnicastLocatorList`,
`initialPeersList`) which works in both FastDDS 2.x and 3.x.

**e) Robot's FastDDS discovery server (port 11811)**
The robot runs a FastDDS discovery server via `start_discover_server.sh`.
Initially thought we needed to use `ROS_DISCOVERY_SERVER` to connect to it.
Investigation revealed the mobiman/hdas nodes do NOT connect to the discovery
server — they use standard multicast. The discovery server is for VR/WiFi
remote control only. Using `ROS_DISCOVERY_SERVER` on either side broke topic
visibility.

**f) HDAS process crashing (exit code -9)**
After restarting the robot stack, HDAS sometimes crashes on startup. Cause:
HDAS needs ~30 seconds to initialize and communicate with the arm motors over
CAN. If you check topics too early, only chassis topics appear. The arm
open/close cycle during boot confirms hardware is healthy. Always wait for this
before checking topics.

### 3. FastDDS 2.x/3.x DDS participant corruption
Running test scripts back-to-back with separate `rclpy.init()`/`rclpy.shutdown()`
cycles created new FastDDS 3.x participants each time. The `ParticipantEntitiesInfo`
wire format differs between FastDDS 2.x (Humble) and 3.x (Jazzy), corrupting the
robot's DDS participant state and causing topics to disappear.

Fix: `run_all_tests.py` calls `rclpy.init()` once, runs all tests, then calls
`rclpy.shutdown()` once. Each test exposes a `main() -> bool` function that
assumes rclpy is already initialized.

### 4. Chassis control node ignoring commands (the 3-gate problem)
Publishing `TwistStamped` to `/motion_target/target_speed_chassis` had no effect.
Binary analysis of `chassis_control_node` revealed three independent gates that
all block motion when unsatisfied. See the "Chassis Gatekeeper" section above
for the full solution.

This was the hardest problem — took multiple sessions of investigation including
binary disassembly of the node to identify the three gates.

### 5. ROS2 daemon unreliable on robot
The ros2 daemon on the robot has slow discovery and often shows only 2 topics
(`/parameter_events`, `/rosout`) even when 70+ topics are active. Always use
`ros2 topic list --no-daemon` on the robot for accurate results.

### 6. Docker migration (Humble container on the laptop)

We moved the laptop-side DimOS runtime into a Docker container
(`docker/dev/docker-compose-ros.yaml`) so the environment is reproducible and
decoupled from the host's ROS2 Jazzy install. Container base is Ubuntu 22.04 +
ROS2 Humble to match the robot exactly (removes the FastDDS cross-version
variable). Getting the container to actually talk to the robot and run
`dimos run r1pro-full` surfaced several issues that weren't obvious from the
host-side setup.

**a) Python 3.10 ↔ Python 3.12 mismatch with Humble's rclpy (the actual blocker)**

ROS2 Humble's `rclpy` C extension is compiled for Python 3.10 (Ubuntu 22.04's
system Python). DimOS's default `uv` venv targets Python 3.12 (DimOS's
`.python-version` file pins 3.12). Starting DimOS in the Humble container
crashes with:

```
ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'
The C extension '/opt/ros/humble/lib/python3.10/site-packages/
_rclpy_pybind11.cpython-312-x86_64-linux-gnu.so' isn't present
```

The path literally shows the mismatch — `lib/python3.10` and `cpython-312` in
the same filename. Humble ships the `.cpython-310-...so` variant; Python 3.12
won't load it.

First attempt — `uv venv --python 3.10 .venv` then `uv sync` — silently
replaced the 3.10 venv with a 3.12 one because `uv sync` honours
`.python-version` (pinned to `3.12`).

**Fix** (what actually worked):

```bash
cd /app
rm -rf .venv
uv venv --python 3.10 .venv
source .venv/bin/activate
uv sync --python 3.10            # override .python-version for this sync
```

DimOS's `pyproject.toml` declares `requires-python = ">=3.10"`, so all deps
resolve cleanly under 3.10. After that, `dimos run r1pro-full` started
successfully.

**b) Docker dropped the DDS-XML unicast workaround — multicast works**

The host-side setup forces unicast discovery via `fastdds_r1pro.xml`
(because the original mixed Humble↔Jazzy setup had cross-version quirks).
In the Humble↔Humble container setup, default multicast PDP over
`--network=host` reaches the robot on the direct ethernet link without any
XML. We removed `FASTRTPS_DEFAULT_PROFILES_FILE` from the compose file and
`devcontainer.json`; the XML stays in `scripts/r1pro_test/` as a fallback for
networks that filter multicast.

**c) `ros2` daemon cache looked identical to broken discovery (~1 hour lost)**

In the container, `ros2 topic list` returned 0–2 topics repeatedly, even
though tcpdump showed PDP traffic flowing both directions between container
and robot. We chased netns inode mismatches, multicast-broken-in-Docker, and
unicast peer-port lists before realising the `ros2` CLI daemon had cached an
empty discovery from an earlier session (when the old XML pinned bad locators)
and was serving that indefinitely.

**Fix**: any time `ros2 topic list` behaves weirdly in the container (or after
changing any RMW / FastDDS env var), run:

```bash
ros2 daemon stop
# then use --no-daemon for fresh discovery:
ros2 topic list --no-daemon | wc -l   # expect ~51 from R1PROBody.d
```

This only affects the CLI. `rclpy` used by DimOS adapters does fresh
discovery per participant init, so runtime is unaffected.

**d) Host-only kernel config that the container can't apply itself**

DimOS's system_configurator wants to enable loopback multicast (for LCM),
add a route for `224.0.0.0/4` via `lo`, and raise `net.core.rmem_max` for
large-frame UDP. Inside the container all of those fail — `/proc/sys/net`
is read-only, `net.core.*` sysctls aren't namespaced, and even with
`NET_ADMIN` the container can't modify them.

**Fix** — apply once on the laptop host (the container inherits via shared
netns under `--network=host`):

```bash
sudo ip link set lo multicast on
sudo ip route add 224.0.0.0/4 dev lo   # ignore "exists" errors
sudo sysctl -w net.core.rmem_max=67108864
sudo sysctl -w net.core.rmem_default=67108864
```

Persist via `/etc/sysctl.d/60-r1pro-ros2.conf`:
```
net.core.rmem_max = 67108864
net.core.rmem_default = 67108864
```

The compose file now declares `cap_add: NET_ADMIN` so `ip link`/`ip route`
work from the container when they do need to run (e.g. the LCM configurator
re-running).

**e) Python version pinning file (`.python-version`)**

`.python-version` in the repo pins uv to 3.12. This is correct for the host
Jazzy setup but broke the Humble container. Keep the pin — override per-command
with `uv sync --python 3.10` inside the Humble container only. Don't commit a
change to that file.

---

### 7. Rerun visualization with X11/Wayland forwarding from the container

The compose/devcontainer config already mounts `/tmp/.X11-unix`, the Wayland
socket dir (`XDG_RUNTIME_DIR`), and `/dev/dri` (GPU). What's needed is
(a) host permission for the container to talk to the display server,
(b) the right venv inside the container, and (c) an LCM-aware bridge with
the r1pro layout. The `dimos rerun-bridge` CLI uses the generic default
blueprint — for r1pro you want the wrapper at
[scripts/r1pro_test/run_rerun_bridge.py](run_rerun_bridge.py), which wires
[`r1pro_rerun_blueprint`](../../dimos/robot/humanoids/r1pro/blueprints.py)
into a `RerunBridgeModule` over `LCM()`.

**a) Host one-time setup (run on the laptop, NOT inside the container)**

```bash
# Allow local containers to open windows on your X server (X11 or XWayland).
# This grants any local user — fine for a dev laptop, do not run on shared
# hosts. To revoke later: `xhost -local:`
xhost +local:

# Avoid Docker creating a stray .Xauthority *directory* on Wayland-only hosts.
# Compose binds ${HOME}/.Xauthority into the container; if the file is
# missing, Docker will helpfully create a directory there. Pre-create it
# so the bind resolves to a regular file even on pure-Wayland sessions.
touch ~/.Xauthority

# (Already documented in section 6d) host kernel settings for ROS2 sensor
# streams — re-listed here because Rerun is useless without sensor data.
sudo ip link set lo multicast on
sudo ip route add 224.0.0.0/4 dev lo 2>/dev/null || true
sudo sysctl -w net.core.rmem_max=67108864
sudo sysctl -w net.core.rmem_default=67108864
```

**b) Inside the container — verify the display socket, GPU, and venv**

```bash
docker compose -f docker/dev/docker-compose-ros.yaml exec ros-dev bash

# Display + Wayland environment should be inherited from the host.
echo "DISPLAY=$DISPLAY  WAYLAND_DISPLAY=$WAYLAND_DISPLAY  XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
ls -l /tmp/.X11-unix/                 # expect X0 (or X1) socket present
ls -l /dev/dri/                       # expect renderD128 / card0 visible

# Quick GUI smoke test (one-time install if missing).
apt-get update && apt-get install -y x11-apps mesa-utils libvulkan1 vulkan-tools
xeyes &                               # window should pop up on the laptop
glxinfo -B | head -10                 # OpenGL renderer (e.g. llvmpipe / iris)
vulkaninfo --summary 2>&1 | head -20  # Vulkan driver — preferred by Rerun's wgpu

# 3.10 venv (per section 6a) — required for both rclpy AND rerun-sdk in this container.
cd /app
[ -d .venv ] || uv venv --python 3.10 .venv
source .venv/bin/activate
uv sync --python 3.10
python -c "import rerun, rclpy; print('rerun', rerun.__version__, 'rclpy ok')"
```

If `xeyes` opens, X11 forwarding is good. If not, see troubleshooting below.

**c) Three-terminal workflow (all inside the same container)**

Open three `docker compose ... exec ros-dev bash` shells (or use tmux from
the same shell — `tmux.conf` is already wired in the image).

```bash
# Terminal 1 — coordinator + adapters publishing LCM sensor streams
source /app/.venv/bin/activate
dimos run r1pro-full

# Terminal 2 — Rerun bridge with the r1pro layout (this opens the native viewer
# on your laptop via X11/XWayland)
source /app/.venv/bin/activate
python scripts/r1pro_test/run_rerun_bridge.py
# Optional: --mode web  (browser viewer at http://localhost:9090, no X needed)
# Optional: --memory-limit 8GB

# Terminal 3 — manipulation client / teleop / REPL
source /app/.venv/bin/activate
python -i -m dimos.manipulation.planning.examples.manipulation_client
```

The bridge subscribes to **all** LCM topics and renders any message whose
type implements `to_rerun()`. The r1pro adapters publish to topic names
that map 1:1 onto the blueprint entity paths (prefix `world` is added by
the bridge):

| LCM topic | Rerun entity path |
|---|---|
| `/r1pro/left_arm/wrist_color` | `world/r1pro/left_arm/wrist_color` |
| `/r1pro/right_arm/wrist_color` | `world/r1pro/right_arm/wrist_color` |
| `/r1pro/chassis/head` | `world/r1pro/chassis/head` |
| `/r1pro/chassis/lidar` | `world/r1pro/chassis/lidar` |

**d) Troubleshooting**

| Symptom | Likely cause | Fix |
|---|---|---|
| `Authorization required, but no authorization protocol specified` (or `cannot open display`) | Host X server rejecting the container's UID | Run `xhost +local:` on the host |
| `xeyes` works but Rerun window is blank / crashes immediately | wgpu can't find a working GPU backend | Install `libvulkan1 mesa-utils` in container; for NVIDIA proprietary, add `runtime: nvidia` + `NVIDIA_DRIVER_CAPABILITIES=all` to compose |
| Rerun opens but no camera/lidar panels populate | LCM bridge not receiving messages — coordinator not publishing yet, or different process group | Confirm `dimos run r1pro-full` is in the same container; check adapter logs for `N callbacks, N frames broadcast` |
| `ModuleNotFoundError: rerun` | Wrong venv (3.12) — section 6a applies | `uv venv --python 3.10 .venv && uv sync --python 3.10` |
| Window appears on the host but is laggy/tearing under Wayland | Rerun is going through XWayland | Set `WAYLAND_DISPLAY=` (empty) before launch to force pure X path; or `unset DISPLAY` to let wgpu try Wayland directly (less battle-tested) |
| `~/.Xauthority` becomes a directory on the host | Docker auto-created it because the file didn't exist | Stop the container, `rmdir ~/.Xauthority`, `touch ~/.Xauthority`, restart |

**e) Wayland note (Ubuntu 24.04 GNOME default)**

GNOME on Ubuntu 24.04 ships an XWayland server alongside the Wayland
compositor, so the X11 path above works without any extra config — the
window manager just routes the X client through XWayland. Native Wayland
mode (using `WAYLAND_DISPLAY` directly) is mounted in the compose file
for completeness but isn't required for Rerun and is the less-tested path.

---

### 8. Why `rr.spawn()` silently fails inside the dev container

**Symptom** — `python scripts/r1pro_test/run_rerun_bridge.py` (the older
version that called `rr.spawn(port=9876)`) runs without error, prints "Rerun
viewer should open", but no window appears. Switching to non-default ports
(19876, 28765, …) doesn't help; each one is also "already in use" the
moment we pick it. The bridge logs a backpressure warning a few seconds
later (`Sender has been blocked for over 5 seconds…`) — data is going
nowhere.

**Diagnostic** — on the laptop host:

```bash
sudo ss -tlnp '( sport = :9876 )'
# users:(("code",pid=358077,fd=44))   ← VS Code extension host has the port
```

The same `code` PID (a `code --type=utility` Node service) shows up holding
*every* rerun-ish port we've ever used — 9876, 9090, 19876, 28765, etc. —
including ones we made up moments earlier. Fresh, never-seen ports (e.g.
41234) are NOT held: `sudo ss -tlnp '( sport = :41234 )'` after running
`rerun --port 41234` shows `users:(("rerun",pid=…))`.

**Root cause** — VS Code extension hosts running on the laptop
(`urdf-visualizer`, `rde-ros-2`, or any webview that has ever interacted
with a Rerun viewer) cache TCP ports they've previously seen Rerun bind, and
re-bind those ports on `127.0.0.1` on subsequent VS Code launches. With
`network_mode: host`, the container shares the host's port namespace, so
those binds collide with our viewer subprocess. When `rr.spawn(port=9876)`
sees the port taken, it falsely concludes a viewer is already there and
silently routes data into a black hole — no window, no error.

This is **not** the standard `remote.autoForwardPorts` Dev Containers
behavior — those forwards would show up in the VS Code Ports panel; these
do not. It's an extension binding the port directly for its own use.

**Fix is built into [scripts/r1pro_test/run_rerun_bridge.py](run_rerun_bridge.py).** The wrapper now:

1. Picks a fresh ephemeral port at runtime via
   `socket.bind(("127.0.0.1", 0))` — never one VS Code has cached.
2. Launches the `rerun` CLI binary itself (`subprocess.Popen`) on that port,
   inheriting `DISPLAY` / `WAYLAND_DISPLAY` / `XAUTHORITY` for X11/XWayland.
3. Polls the port until the viewer is accepting connections.
4. Calls `rr.connect_grpc(f"rerun+http://127.0.0.1:{port}/proxy")` to attach
   the bridge's recording stream to the launched viewer.
5. On Ctrl-C, terminates the viewer subprocess so no zombie holds the port.

Single command, no flags, no port hygiene:

```bash
python scripts/r1pro_test/run_rerun_bridge.py
```

Native window opens via X11/XWayland within ~2 seconds, all R1 Pro sensor
streams populate per the `r1pro_rerun_blueprint` layout. Each subsequent
launch picks a different ephemeral port, so the VS Code cache never matches.

**Escape hatches preserved:**
- `--mode web` — runs `rr.serve_grpc()` + `rr.serve_web_viewer()` and
  prints the browser URL (default `http://localhost:9090`, override with
  `--web-port`). Useful when X11 forwarding isn't available.
- `--mode connect --connect-url URL` — bridge connects to a viewer the user
  launched themselves. Useful for remote viewers or for debugging the
  connection layer.

### 9. Manipulation blueprint blocked on missing deps (Drake + trimesh)

Running `dimos run r1pro-planner-full` surfaced two missing dependencies in
sequence. Neither was declared as a hard dep; the runtime failed loudly at
`ManipulationModule.start()` for Drake and silently at URDF load for trimesh.

**a) Drake not installed**

```
Exception in RPC handler for ManipulationModule/start:
  Drake is not installed. Install with: pip install drake
```

Drake is declared under `[project.optional-dependencies].manipulation` in
`pyproject.toml:200-214`, not in the base deps. Base `uv sync` skipped it.

Fix (one-shot):
```bash
uv sync --python 3.10 --extra manipulation
```

**b) `uv pip install` bypassing the venv**

After realising trimesh was also missing (below), `uv pip install trimesh`
reported `Using Python 3.10.12 environment at: /usr` and installed into the
system Python — not `/app/.venv`, even though the venv was activated.

Root cause: the Dockerfile sets `ENV UV_SYSTEM_PYTHON=1` (line 28), which
tells uv to ignore `VIRTUAL_ENV` and use the system Python by default.

Fix (one-shot):
```bash
uv pip install --python /app/.venv/bin/python trimesh
# or just:
pip install trimesh       # venv is activated so this uses the venv's pip
```

Or, per-session:
```bash
unset UV_SYSTEM_PYTHON
```

**c) trimesh not declared anywhere → silent STL→OBJ skip → Drake rejection**

After installing Drake, the planner then crashed at URDF load with:
```
trimesh not installed, skipping mesh conversion        (warning)
MakeConvexHull only applies to .obj, .vtk, and .gltf meshes;
  unsupported extension '.stl' for geometry data: /app/data/.../base_link.STL
```

Drake 1.40+ dropped `.STL` for collision geometry. DimOS's
`dimos/manipulation/planning/utils/mesh_utils.py` has a conversion pass that
transforms STL → OBJ before feeding the URDF to Drake, but it soft-fails
(`logger.warning + return`) when `trimesh` isn't importable. `trimesh` is
referenced only via that fallback import; it's not declared in
`pyproject.toml` under any extra.

Fix:
```bash
pip install trimesh
rm -rf /tmp/dimos_urdf_cache   # cache key doesn't distinguish "conversion
                               # ran" from "trimesh missing", so purge once
dimos run r1pro-planner-full
```

**Left as a to-do**: add `trimesh` to the `manipulation` extra in
`pyproject.toml` alongside drake so `uv sync --extra manipulation` pulls it
automatically. Filing separately.

### 10. Sensor spin loop hot-looping on an invalidated rclpy context

**Symptom** — terminal floods with tens of thousands of copies of:

```
rclpy._rclpy_pybind11.RCLError: failed to create timer: the given context
  is not valid, either rcl_init() was not called or rcl_shutdown() was called
R1 Pro <side> sensor executor exception (continuing): ...
```

Appeared within seconds of any module failing to start (e.g. the Drake/trimesh
errors above, or a Ctrl-C during startup).

**Root cause** — the per-adapter sensor spin loops in
`dimos/hardware/manipulators/r1pro/adapter.py:319-329` and
`dimos/hardware/drive_trains/r1pro/adapter.py:339-349` wrap
`sensor_executor.spin_once()` in a blanket `try / except Exception: log;
continue`. The intent is to survive transient callback errors. But when
rclpy's SIGINT handler (or a sibling module's `rclpy.shutdown()`) invalidates
the sensor's isolated Context, **every** subsequent `spin_once()` raises
`RCLError: context is not valid` — and the recovery loop catches and retries
forever, at wall-clock speed. One terminal flood per sensor participant.

**Fix** — add `sensor_context.ok()` to both loop conditions, and inside the
except block detect the "context is not valid" string to break out explicitly
rather than spin:

```python
while not sensor_stop.is_set() and sensor_context.ok():
    try:
        sensor_executor.spin_once(timeout_sec=0.1)
    except Exception as exc:
        if not sensor_context.ok() or "context is not valid" in str(exc):
            log.warning(
                "R1 Pro %s sensor context invalidated, exiting spin loop: %s",
                side, exc,
            )
            break
        log.warning("R1 Pro %s sensor executor exception (continuing): %s",
                    side, exc, exc_info=True)
```

Applied to both adapter files. Clean shutdowns now produce one exit line per
sensor thread instead of a flood. As a side benefit this is also **more
diagnostic** for the unresolved sensor-dropout issue: if the context dies mid
run, we now see it once with a timestamp rather than losing it in the flood.

### 11. Chassis sensors (lidar + auxiliary cameras) missing from Rerun tree

**Symptom** — after Rerun native + `r1pro-planner-full` came up cleanly,
wrist cameras and the head camera rendered, but the Rerun entity tree had no
`world/r1pro/chassis/lidar`, `chassis_front_left/right`, `chassis_left/right`,
`chassis_rear`, or `head_depth` entries.

**Classification** — same signature as the sensor-dropout problem now
resolved in the "Session Log" section below: IMU and wrist streams (smaller
packets, separate DDS participants) stay alive; large fragmented UDP payloads
(PointCloud2 lidar + chassis camera JPEGs) silently stop flowing into the
chassis adapter after ~5-30 s.

**Resolution** — Linux IP fragment reassembly buffer
(`net.ipv4.ipfrag_high_thresh`) defaults to 4 MB and overflows under
concurrent camera + lidar load. Bump it on the host to 64 MB; see Session
Log below for the full fix.

Diagnostic path when this recurs:

1. **Robot-side first** — SSH the robot and run:
   ```bash
   ros2 topic list | grep -i lidar
   ros2 topic info -v /hdas/lidar_chassis_left
   ros2 topic hz /hdas/lidar_chassis_left
   ```
   The adapter subscribes to exactly `/hdas/lidar_chassis_left` in
   `dimos/hardware/drive_trains/r1pro/adapter.py:312`.

2. **Laptop-side, container, ROS** — confirm the same topic is reaching the
   container:
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 daemon stop
   ros2 topic hz /hdas/lidar_chassis_left --no-daemon
   ```
   If ~10 Hz on the robot but 0 Hz in the container, the fragmented-UDP path
   is broken — bump `net.core.rmem_max` further (see section 6d).

3. **Laptop-side, LCM** — confirm whether the chassis adapter *did* publish to
   LCM (bridge-independent):
   ```bash
   python - <<'PY'
   from dimos.protocol.pubsub.impl.lcmpubsub import LCM
   import time
   lcm = LCM(); lcm.start()
   seen: dict[str, int] = {}
   def on_msg(msg, topic):
       name = getattr(topic, "name", str(topic))
       seen[name] = seen.get(name, 0) + 1
   lcm.subscribe_all(on_msg)
   time.sleep(5)
   for k, v in sorted(seen.items()):
       print(f"{v:5d}  {k}")
   PY
   ```
   If `/r1pro/chassis/lidar` is missing from this output, the chassis adapter
   never broadcast — section 2. If it's present with a non-zero count, the
   bridge isn't rendering — check `PointCloud2.to_rerun()` path.

4. **In-app** — in the Rerun left panel, click the entity if it exists and
   raise "Radius" or hit `F` to frame it. Rate-limited to 10 Hz by the bridge
   (`bridge.py:54-57`), which is expected.

Unresolved root cause is tracked in section "Session Log — The sensor dropout
problem" below.

---

## Robot Architecture Notes

- **Platform**: Jetson Orin (aarch64), Ubuntu 22.04, L4T (Jetpack)
- **ROS2**: Humble, FastDDS (rmw_fastrtps_cpp)
- **ROS_DOMAIN_ID**: 41
- **CAN bus**: arms and torso communicate via CAN (`can.sh` starts the driver)
- **HDAS**: Hardware abstraction layer — publishes all sensor feedback, receives
  all motion commands
- **mobiman**: Motion manager — handles kinematics, IK, safety limits
- **Custom message package**: `hdas_msg` — used for motor control, BMS, LED,
  version info. Standard ROS2 types used for joint states and geometry
- **Chassis type**: W1 (3-wheel swerve drive), from `/opt/galaxea/body/hardware.json`

---

## Next Steps

- [x] Topic discovery and DDS connectivity over ethernet
- [x] Arm feedback reading
- [x] Arm joint movement
- [x] Chassis movement (via gatekeeper)
- [x] DiMOS adapters (chassis + arms)
- [x] Keyboard teleop through DiMOS
- [x] Sensor stream integration (wrist cameras, chassis cameras, LiDAR, IMUs)
- [x] Full ControlCoordinator integration with dual-arm + chassis blueprint
- [x] Whole-body adapter
- [x] Sensor dropout under coordinator load — kernel IP fragment reassembly buffer too small (resolved 2026-05-08, see below)
- [ ] Torso control adapter (4-DOF, deferred)

---

## Session Log — Sensor Streams & Dual-Arm Coordinator Integration

### What was built

**Sensor streams on adapters** (`dimos/hardware/manipulators/r1pro/adapter.py`,
`dimos/hardware/drive_trains/r1pro/adapter.py`)

Each adapter now subscribes to all sensors physically attached to its hardware
and publishes decoded frames to independent LCM transports on `connect()`.
No changes to `ControlCoordinator` — it remains fully generic.

| Adapter | Sensors → LCM transports |
|---|---|
| `R1ProArmAdapter` (left) | `/r1pro/left_arm/wrist_color`, `/r1pro/left_arm/wrist_depth` |
| `R1ProArmAdapter` (right) | `/r1pro/right_arm/wrist_color`, `/r1pro/right_arm/wrist_depth` |
| `R1ProChassisAdapter` | `/r1pro/chassis/head`, `/r1pro/chassis/chassis_front_left`, `/r1pro/chassis/chassis_front_right`, `/r1pro/chassis/chassis_left`, `/r1pro/chassis/chassis_right`, `/r1pro/chassis/chassis_rear`, `/r1pro/chassis/head_depth`, `/r1pro/chassis/lidar`, `/r1pro/chassis/imu_chassis`, `/r1pro/chassis/imu_torso` |

**Async worker pattern** (prevents blocking the ROS spin thread):

1. ROS spin thread callback → enqueue raw `msg` object (zero-copy, no GIL pressure)
2. Dedicated worker thread per sensor → `bytes(msg.data)` + decode + `transport.broadcast()`
3. All queues are `maxsize=1` (latest-frame semantics — stale frames are replaced)

**Separate rclpy context for sensor subscriptions** (isolated DDS participant):

Sensor subscriptions use a completely separate `rclpy.Context` with its own
`MultiThreadedExecutor` and DDS participant. This prevents control traffic
(arm commands at ~100 Hz) from saturating the shared DDS receive threads and
dropping large camera frames that require UDP fragmentation.

**Crash-resilient spin loop**:

```python
# spin_once in a loop instead of spin() so any callback exception is
# logged and recovered from rather than killing the entire spin thread.
while not sensor_stop.is_set():
    try:
        sensor_executor.spin_once(timeout_sec=0.1)
    except Exception as exc:
        log.warning("sensor executor exception (continuing): %s", exc)
```

**Callback counters in every worker log line** (every 5 seconds):
```
R1 Pro left wrist_color: 150 callbacks, 148 frames broadcast in last 5.0s
```
When sensors drop: `0 callbacks` = DDS stopped delivering; `N callbacks, 0 frames` = decode/broadcast failing.

**Blueprints added** (`dimos/robot/humanoids/r1pro/blueprints.py`,
`dimos/robot/catalog/galaxea.py`, `dimos/robot/all_blueprints.py`):

- `r1pro-dual-mock` — dual-arm + chassis with mock adapters (runs offline)
- `r1pro-full` — dual-arm + chassis with real R1Pro adapters

**Whole-body adapter** — created during this session to unify all robot
subsystems (arms + torso + chassis + sensors) behind a single interface.

---

### The sensor dropout problem (resolved 2026-05-08)

**Symptom**: Sensor LCM topics (`/r1pro/*/wrist_color`, `/r1pro/chassis/head`,
etc.) stop publishing as soon as the ControlCoordinator tick loop starts writing
joint commands (~100 Hz). IMU topics keep working. Happens after 5–30 seconds.
Sometimes fails immediately on the second launch.

**What works**: IMU (small messages, single UDP packet). **What stops**: all
cameras and LiDAR (large messages, require UDP fragment reassembly).

**Root cause: Linux IP fragment reassembly buffer too small.**

`net.ipv4.ipfrag_high_thresh` defaults to 4 MB. Six chassis cameras × ~100 KB
JPEGs × ~30 Hz = ~18 MB/s of fragmented UDP, plus the lidar PointCloud2.
The reassembly pool fills in <300 ms and partially-assembled datagrams evict
each other before reassembly completes, so the kernel drops them and the
adapter's subscription never sees them. IMUs are unaffected because each
datagram fits in a single packet — no reassembly needed. This signature
("small messages survive, large fragmented messages die") was correctly
captured in the symptom but kept getting attributed to higher layers
(rclpy, FastDDS, GIL).

Smoking gun (host `nstat -az | grep -iE 'Reasm|Frag'` after a few sessions):
```
IpReasmFails       13,951,250
IpReasmOKs          1,360,584     ← 10× more failures than successes
ipfrag_high_thresh    4,194,304   ← only 4 MB
```

This is a *separate* kernel ceiling from `net.core.rmem_max` (which had already
been raised to 64 MB during Phase 3 and was not the bottleneck on its own).

**Fix — apply on the laptop host** (`net.ipv4.ipfrag_*` is net-namespaced;
`network_mode: host` means the container inherits):

```bash
sudo sysctl -w net.ipv4.ipfrag_high_thresh=67108864
sudo sysctl -w net.ipv4.ipfrag_low_thresh=50331648
sudo sysctl -w net.ipv4.ipfrag_time=60
```

Persist via `/etc/sysctl.d/60-r1pro-ros2.conf`:
```
net.core.rmem_max = 67108864
net.core.rmem_default = 67108864
net.ipv4.ipfrag_high_thresh = 67108864
net.ipv4.ipfrag_low_thresh = 50331648
net.ipv4.ipfrag_time = 60
```
Then `sudo sysctl --system`. Verify after restart:
```bash
sysctl net.ipv4.ipfrag_high_thresh net.core.rmem_max
nstat -n; sleep 30; nstat | grep -iE 'Reasm'   # IpReasmFails delta should be ~0
```

**Diagnostic that pinned it** (in case it recurs in a different shape):

```bash
# On the host. Run twice with the load on between.
nstat -az | grep -iE 'Reasm|FragOK|FragFail'
```
Look for `IpReasmFails` climbing while `IpReasmOKs` stays flat. If those are
the two counters that move, this is the exact same problem and the same fix
applies — possibly with a higher ceiling if more sensors are added.

**Fixes attempted while chasing this** (kept for future sessions — all are
correctness improvements but none are the *cause*):

| Fix | Rationale | Result |
|---|---|---|
| Move `bytes(msg.data)` copy off spin thread | Reduce GIL contention | No effect on dropout (kernel-level drop, never reaches Python) |
| Separate `rclpy.Context` for sensors | Independent DDS participant + sockets | No effect on dropout (same kernel netns shares the ipfrag pool) |
| Lambda wrappers for callback signatures | Fixed `TypeError: missing argument '_topic'` crashing spin thread | Real fix for an unrelated crash |
| `spin_once` loop with try/except + `context.ok()` check | Survive transient exceptions; exit cleanly on context shutdown | Real fix for hot-loop floods at shutdown (§10) |
| Raise `net.core.rmem_max` to 64 MB (Phase 3) | UDP socket receive buffer | Necessary but not sufficient — `rmem_max` is a different ceiling from `ipfrag_*` |
| Add `ReentrantCallbackGroup` to sensor subs (Phase 5b) | Parallel callback dispatch | Tested — no effect (drop is below rclpy). Reverted to keep adapters simple. |
