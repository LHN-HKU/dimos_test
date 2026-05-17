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
from typing import Protocol

import mujoco
import numpy as np

from dimos.experimental.pimsim.robot_meshes import RobotMeshes


@dataclass(frozen=True)
class SyntheticLidarConfig:
    n_azimuth: int
    n_elevation: int
    elevation_min_deg: float
    elevation_max_deg: float
    max_range: float


class RaycastScene(Protocol):
    def raycast(self, origin: np.ndarray, direction: np.ndarray, max_range: float) -> float | None:
        """Return hit distance in meters, or None when the ray misses."""


class MujocoRaycastScene:
    def __init__(self, robot: RobotMeshes) -> None:
        self._robot = robot
        self._geomid_buf = np.zeros(1, dtype=np.int32)

    def raycast(self, origin: np.ndarray, direction: np.ndarray, max_range: float) -> float | None:
        dist = mujoco.mj_ray(
            self._robot.model,
            self._robot.data,
            origin,
            direction,
            None,
            1,
            -1,
            self._geomid_buf,
        )
        if dist < 0 or dist > max_range:
            return None
        return float(dist)


class SyntheticLidar:
    def __init__(self, config: SyntheticLidarConfig) -> None:
        self._config = config
        self._directions: np.ndarray | None = None

    def scan(self, scene: RaycastScene, origin: np.ndarray, yaw: float) -> np.ndarray | None:
        directions = self._ensure_directions()
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        rotation = np.array(
            [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        world_dirs = directions @ rotation.T

        hits: list[tuple[float, float, float]] = []
        for direction in world_dirs:
            dist = scene.raycast(origin, direction, self._config.max_range)
            if dist is None:
                continue
            hit = origin + direction * dist
            hits.append((float(hit[0]), float(hit[1]), float(hit[2])))

        if not hits:
            return None
        return np.array(hits, dtype=np.float32)

    def _ensure_directions(self) -> np.ndarray:
        if self._directions is not None:
            return self._directions

        azimuths = np.linspace(0.0, 2.0 * math.pi, self._config.n_azimuth, endpoint=False)
        elevations = np.linspace(
            math.radians(self._config.elevation_min_deg),
            math.radians(self._config.elevation_max_deg),
            self._config.n_elevation,
        )
        az, el = np.meshgrid(azimuths, elevations, indexing="xy")
        cos_el = np.cos(el)
        dirs = np.stack([cos_el * np.cos(az), cos_el * np.sin(az), np.sin(el)], axis=-1).reshape(
            -1, 3
        )
        self._directions = np.ascontiguousarray(dirs, dtype=np.float64)
        return self._directions
