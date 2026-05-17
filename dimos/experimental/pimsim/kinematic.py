# Copyright 2025-2026 Dimensional Inc.
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

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time

import numpy as np

from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Odometry import Odometry


@dataclass(frozen=True)
class KinematicBaseSnapshot:
    x: float
    y: float
    z: float
    yaw: float
    fwd_speed: float
    left_speed: float
    yaw_rate: float
    vertical_speed: float

    @property
    def quaternion(self) -> Quaternion:
        return Quaternion.from_euler(Vector3(0.0, 0.0, self.yaw))

    def base_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        quat = self.quaternion
        return (
            np.array([self.x, self.y, self.z], dtype=np.float64),
            np.array([quat.w, quat.x, quat.y, quat.z], dtype=np.float64),
        )

    def to_odometry(self) -> Odometry:
        quat = self.quaternion
        return Odometry(
            ts=time.time(),
            frame_id="map",
            child_frame_id="sensor",
            pose=Pose(
                position=[self.x, self.y, self.z],
                orientation=[quat.x, quat.y, quat.z, quat.w],
            ),
            twist=Twist(
                linear=[self.fwd_speed, self.left_speed, self.vertical_speed],
                angular=[0.0, 0.0, self.yaw_rate],
            ),
        )


class KinematicBaseSim:
    def __init__(
        self,
        *,
        init_x: float,
        init_y: float,
        init_z: float,
        init_yaw: float,
        vehicle_height: float,
        sim_rate: float,
        lock_z: bool,
    ) -> None:
        self._init_x = init_x
        self._init_y = init_y
        self._init_z = init_z
        self._init_yaw = init_yaw
        self._vehicle_height = vehicle_height
        self._sim_rate = sim_rate
        self._lock_z = lock_z
        self._lock = threading.Lock()
        self._fwd_speed = 0.0
        self._left_speed = 0.0
        self._yaw_rate = 0.0
        self._x = init_x
        self._y = init_y
        self._z = init_z + vehicle_height
        self._yaw = init_yaw

    @property
    def vehicle_height(self) -> float:
        return self._vehicle_height

    def set_command(self, twist: Twist) -> None:
        with self._lock:
            self._fwd_speed = float(twist.linear.x)
            self._left_speed = float(twist.linear.y)
            self._yaw_rate = float(twist.angular.z)

    def reset(self) -> KinematicBaseSnapshot:
        with self._lock:
            self._fwd_speed = 0.0
            self._left_speed = 0.0
            self._yaw_rate = 0.0
            self._x = self._init_x
            self._y = self._init_y
            self._z = self._init_z + self._vehicle_height
            self._yaw = self._init_yaw
            return self._snapshot_locked(vertical_speed=0.0)

    def snapshot(self) -> KinematicBaseSnapshot:
        with self._lock:
            return self._snapshot_locked(vertical_speed=0.0)

    def step(self, dt: float) -> KinematicBaseSnapshot:
        with self._lock:
            prev_z = self._z
            self._yaw += dt * self._yaw_rate
            if self._yaw > math.pi:
                self._yaw -= 2 * math.pi
            elif self._yaw < -math.pi:
                self._yaw += 2 * math.pi
            cos_yaw, sin_yaw = math.cos(self._yaw), math.sin(self._yaw)
            self._x += dt * (cos_yaw * self._fwd_speed - sin_yaw * self._left_speed)
            self._y += dt * (sin_yaw * self._fwd_speed + cos_yaw * self._left_speed)
            if not self._lock_z:
                self._z = self._init_z + self._vehicle_height
            return self._snapshot_locked(
                vertical_speed=(self._z - prev_z) * self._sim_rate,
            )

    def _snapshot_locked(self, *, vertical_speed: float) -> KinematicBaseSnapshot:
        return KinematicBaseSnapshot(
            x=self._x,
            y=self._y,
            z=self._z,
            yaw=self._yaw,
            fwd_speed=self._fwd_speed,
            left_speed=self._left_speed,
            yaw_rate=self._yaw_rate,
            vertical_speed=vertical_speed,
        )
