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

from __future__ import annotations

import math
import threading
import time
from typing import Any

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.navigation.nav_stack.modules.mls_planner.planner import (
    MLS,
    Node,
    astar,
    max_step_in_voxels,
    points_to_mls,
    robot_height_in_voxels,
    snap_to_surface,
    surface_centers,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class MlsPlannerConfig(ModuleConfig):
    world_frame: str = "world"
    voxel_size: float = 0.1
    robot_height: float = 0.5  # m
    max_step_height: float = 0.15  # m — largest step the robot can climb between adjacent surfaces


class MlsPlanner(Module):
    """3D multi-level surface planner: extracts an MLS from the global voxel map
    and runs surface-graph A* between the robot's current surface and the goal.
    """

    config: MlsPlannerConfig

    global_map: In[PointCloud2]
    odometry: In[Odometry]
    goal: In[PoseStamped]
    path: Out[Path]
    surfaces: Out[PointCloud2]  # debug: extracted MLS surface centers

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._latest_odom: Odometry | None = None
        self._latest_mls: MLS | None = None
        self._latest_goal: PoseStamped | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.odometry.subscribe(self._on_odom)))
        self.register_disposable(Disposable(self.global_map.subscribe(self._on_map)))
        self.register_disposable(Disposable(self.goal.subscribe(self._on_goal)))

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_odom(self, msg: Odometry) -> None:
        with self._lock:
            self._latest_odom = msg

    def _on_map(self, msg: PointCloud2) -> None:
        points = msg.points_f32()
        rh_voxels = robot_height_in_voxels(self.config.robot_height, self.config.voxel_size)
        mls = points_to_mls(points, self.config.voxel_size, rh_voxels)
        with self._lock:
            self._latest_mls = mls
        self._publish_surfaces(mls)
        # Re-plan against the cached goal in case it arrived before this map
        # finished processing (race when multi-scene tests cycle inputs).
        self._maybe_replan()

    def _on_goal(self, goal: PoseStamped) -> None:
        with self._lock:
            self._latest_goal = goal
        self._maybe_replan()

    def _maybe_replan(self) -> None:
        with self._lock:
            odom = self._latest_odom
            mls = self._latest_mls
            goal = self._latest_goal
        if odom is None or mls is None or goal is None:
            return
        path = self._plan(mls, odom, goal)
        if path is not None:
            self.path.publish(path)

    def _publish_surfaces(self, mls: MLS) -> None:
        centers = surface_centers(mls, self.config.voxel_size)
        cloud = PointCloud2.from_numpy(
            points=centers,
            frame_id=self.config.world_frame,
            timestamp=time.time(),
        )
        self.surfaces.publish(cloud)
        logger.info("MlsPlanner extracted %d surfaces across %d columns", len(centers), len(mls))

    def _plan(self, mls: MLS, odom: Odometry, goal: PoseStamped) -> Path | None:
        vs = self.config.voxel_size
        start_node = snap_to_surface(
            mls, math.floor(odom.x / vs), math.floor(odom.y / vs), odom.z, vs
        )
        goal_node = snap_to_surface(
            mls,
            math.floor(goal.position.x / vs),
            math.floor(goal.position.y / vs),
            goal.position.z,
            vs,
        )
        if start_node is None or goal_node is None:
            logger.warning(
                "MlsPlanner: could not snap start/goal to MLS surface (start=%s goal=%s)",
                start_node,
                goal_node,
            )
            return None

        max_step_voxels = max_step_in_voxels(self.config.max_step_height, vs)
        nodes = astar(mls, start_node, goal_node, max_step_voxels)
        if nodes is None:
            logger.warning("MlsPlanner: no path from %s to %s", start_node, goal_node)
            return None

        logger.info("MlsPlanner: path with %d waypoints", len(nodes))
        return Path(
            ts=time.time(),
            frame_id=self.config.world_frame,
            poses=[self._node_to_pose(n) for n in nodes],
        )

    def _node_to_pose(self, node: Node) -> PoseStamped:
        kx, ky, kz = node
        vs = self.config.voxel_size
        half = 0.5 * vs
        # z is the TOP of the supporting voxel — where feet/wheels actually sit,
        # not the voxel center.
        return PoseStamped(
            ts=time.time(),
            frame_id=self.config.world_frame,
            position=[kx * vs + half, ky * vs + half, (kz + 1) * vs],
            orientation=[0.0, 0.0, 0.0, 1.0],
        )
