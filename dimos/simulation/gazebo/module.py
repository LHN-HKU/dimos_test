# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gazebo Harmonic native simulation module.

Provides the same I/O contract as the Unity sim module
(`dimos/simulation/unity/module.py`), backed by gz-sim. A C++ bridge
(`cpp/main.cpp`) spawns gz-sim, subscribes to its gz-transport sensor
topics, and republishes them on LCM as dimos messages. The bridge also
forwards LCM `cmd_vel` back to the robot's gz-transport cmd topic.
"""

from __future__ import annotations

from pathlib import Path

from dimos.core.core import rpc
from dimos.core.native_module import NativeModule, NativeModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_WORLD = str(_PKG_DIR / "worlds" / "dimos_bot.sdf")


class GazeboConfig(NativeModuleConfig):
    """Config for the Gazebo native module.

    Defaults match the bundled `worlds/dimos_bot.sdf` (a small differential-
    drive robot with a 360-sample 2D lidar and a 320x240 RGB camera).
    """

    cwd: str | None = "cpp"
    executable: str = "result/bin/gazebo_native"
    build_command: str | None = "nix build .#gazebo_native"

    # Path to the .sdf world file. Defaults to the bundled dimos_bot world.
    # The bridge embeds gz::sim::Server in-process to run this world.
    world: str = _DEFAULT_WORLD
    # Reserved for future GUI/rendering toggles. Currently unused — the
    # embedded server runs without a GUI; sensor rendering still happens.
    headless: bool = True
    # Frame ID prefix used for outbound headers.
    frame_id: str = "base_link"

    # gz-transport topic mapping (keep in sync with the world file).
    gz_cmd_vel: str = "/model/dimos_bot/cmd_vel"
    gz_odom: str = "/model/dimos_bot/odometry"
    gz_lidar: str = "/lidar/points"
    gz_camera: str = "/camera"
    gz_caminfo: str = "/camera_info"


class Gazebo(NativeModule):
    """Gazebo Harmonic simulation module.

    Ports (matching the Unity bridge's I/O contract):
        cmd_vel (In[Twist]):           Velocity commands forwarded to gz.
        terrain_map (In[PointCloud2]): Declared for parity with Unity; gz
                                       simulates its own terrain so this
                                       input is ignored.
        odometry (Out[Odometry]):           Robot odometry from gz DiffDrive.
        registered_scan (Out[PointCloud2]): Lidar from gz gpu_lidar.
        color_image (Out[Image]):           RGB camera from gz.
        semantic_image (Out[Image]):        Declared for parity with Unity;
                                            no gz publisher in the default
                                            world. Stays empty unless a
                                            semantic-segmentation plugin is
                                            added.
        camera_info (Out[CameraInfo]):      Camera intrinsics from gz.
    """

    config: GazeboConfig

    cmd_vel: In[Twist]
    terrain_map: In[PointCloud2]
    odometry: Out[Odometry]
    registered_scan: Out[PointCloud2]
    color_image: Out[Image]
    semantic_image: Out[Image]
    camera_info: Out[CameraInfo]

    @rpc
    def start(self) -> None:
        super().start()

    @rpc
    def stop(self) -> None:
        super().stop()
