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

"""Pure FAST-LIO2 NativeModule wrapper.

Consumes raw IMU (`sensor_msgs.Imu`) and raw lidar
(`sensor_msgs.RawLidarScan`) over LCM and runs FAST-LIO2 EKF-LOAM SLAM
in a native subprocess.  No hardware SDK linkage — pair with a sensor
module (e.g. the `livox` Mid-360 module) that publishes the raw streams.

Usage::

    from dimos.hardware.sensors.lidar.livox.module import Mid360
    from dimos.hardware.sensors.lidar.fastlio2.module import FastLio2
    from dimos.core.coordination.blueprints import autoconnect
    from dimos.core.coordination.module_coordinator import ModuleCoordinator

    ModuleCoordinator.build(autoconnect(
        Mid360.blueprint(lidar_ip="192.168.1.107"),
        FastLio2.blueprint(),
        SomeConsumer.blueprint(),
    )).loop()
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import TYPE_CHECKING, Annotated

from pydantic.experimental.pipeline import validate_as
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.native_module import NativeModule, NativeModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.sensor_msgs.RawLidarScan import RawLidarScan
from dimos.navigation.nav_stack.frames import FRAME_BODY, FRAME_ODOM
from dimos.spec import mapping, perception

_CONFIG_DIR = Path(__file__).parent / "config"


class FastLio2Config(NativeModuleConfig):
    cwd: str | None = "cpp"
    executable: str = "result/bin/fastlio2_native"
    build_command: str | None = "nix build .#fastlio2_native"

    # Sensor mount pose — position + orientation of the sensor relative to ground.
    # Converted to init_pose CLI arg [x, y, z, qx, qy, qz, qw] in model_post_init.
    mount: Pose = Pose()

    # Frame IDs for output messages.  "odom" reflects that FastLio2 provides
    # locally-smooth, continuous odometry (no loop-closure jumps).  PGO
    # publishes the map→odom correction via TF.
    frame_id: str = FRAME_ODOM
    child_frame_id: str = FRAME_BODY

    # FAST-LIO internal processing rates
    msr_freq: float = 50.0
    main_freq: float = 5000.0

    # Output publish rates (Hz)
    pointcloud_freq: float = 10.0
    odom_freq: float = 30.0

    # Point cloud filtering
    voxel_size: float = 0.1
    sor_mean_k: int = 50
    sor_stddev: float = 1.0

    # Global voxel map (disabled when map_freq <= 0)
    map_freq: float = 0.0
    map_voxel_size: float = 0.1
    map_max_range: float = 100.0

    # FAST-LIO YAML config (relative to config/ dir, or absolute path)
    # C++ binary reads YAML directly via yaml-cpp
    config: Annotated[
        Path, validate_as(...).transform(lambda p: p if p.is_absolute() else _CONFIG_DIR / p)
    ] = Path("mid360.yaml")

    # Resolved in __post_init__, passed as --config_path to the binary
    config_path: str | None = None

    # init_pose is computed from mount; config is resolved to config_path
    init_pose: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    cli_exclude: frozenset[str] = frozenset({"config", "mount"})

    def model_post_init(self, __context: object) -> None:
        """Resolve config_path and compute init_pose from mount."""
        super().model_post_init(__context)
        cfg = self.config
        if not cfg.is_absolute():
            cfg = _CONFIG_DIR / cfg
        self.config_path = str(cfg.resolve())
        m = self.mount
        self.init_pose = [
            m.x,
            m.y,
            m.z,
            m.orientation.x,
            m.orientation.y,
            m.orientation.z,
            m.orientation.w,
        ]


class FastLio2(NativeModule, perception.Lidar, perception.Odometry, mapping.GlobalPointcloud):
    config: FastLio2Config

    # Raw sensor inputs (subscribed by the native binary over LCM).
    raw_imu: In[Imu]
    raw_lidar: In[RawLidarScan]

    # SLAM outputs.
    lidar: Out[PointCloud2]
    odometry: Out[Odometry]
    global_map: Out[PointCloud2]

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(
            Disposable(self.odometry.transport.subscribe(self._on_odom_for_tf, self.odometry))
        )

    def _on_odom_for_tf(self, msg: Odometry) -> None:
        self.tf.publish(
            Transform(
                frame_id=FRAME_ODOM,
                child_frame_id=FRAME_BODY,
                translation=Vector3(
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                ),
                rotation=Quaternion(
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ),
                ts=msg.ts or time.time(),
            )
        )

    @rpc
    def stop(self) -> None:
        super().stop()


# Verify protocol port compliance (mypy will flag missing ports)
if TYPE_CHECKING:
    FastLio2()
