#!/usr/bin/env python3
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

"""Unitree Go2 onboard navigation blueprint — uses the Go2's WebRTC API
for control + sensors instead of the G1's DDS + FastLio2 setup.

Same shape as :mod:`unitree_g1_nav_onboard`: GO2Connection (WebRTC)
streams lidar and pose into the rosnav-style ``create_nav_stack`` (simple
planner + terrain analysis + path follower + PGO), MovementManager
mediates teleop vs autonomous goals, vis_module renders Rerun.

The one delta vs G1 is that Go2's WebRTC odometry comes out as
``PoseStamped`` instead of ``nav_msgs.Odometry``. The G1 gets clean
``Odometry`` from FastLio2; Go2 doesn't have a separate SLAM front-end
on this path, so :class:`PoseStampedToOdometry` adapts the WebRTC pose
stream into the ``Odometry`` topic the SLAM module (PGO/RtabMap) and
the rest of the nav stack consume.

Usage:
    dimos run unitree-go2-nav-onboard
"""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.navigation.nav_stack.main import create_nav_stack, nav_stack_rerun_config
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.go2.odom_adapter import PoseStampedToOdometry
from dimos.visualization.vis_module import vis_module

# Standing height of the Go2; used by terrain_analysis / planners to filter
# obstacle vs ground points and to size the local planner footprint.
_GO2_HEIGHT_CLEARANCE = 0.42
_GO2_MAX_SPEED = 0.6

unitree_go2_nav_onboard = (
    autoconnect(
        GO2Connection.blueprint(),
        PoseStampedToOdometry.blueprint(),
        create_nav_stack(
            planner="simple",
            vehicle_height=_GO2_HEIGHT_CLEARANCE,
            max_speed=_GO2_MAX_SPEED,
            terrain_analysis={
                "obstacle_height_threshold": 0.05,
                "ground_height_threshold": 0.05,
                "sensor_range": 30,
            },
            simple_planner={
                "cell_size": 0.15,
                "obstacle_height_threshold": 0.10,
                "inflation_radius": 0.35,
                "lookahead_distance": 1.5,
                "replan_rate": 5.0,
                "replan_cooldown": 2.0,
            },
        ),
        MovementManager.blueprint(),
        vis_module(
            viewer_backend=global_config.viewer,
            rerun_config=nav_stack_rerun_config(
                {"memory_limit": "1GB"},
                vis_throttle=0.5,
            ),
        ),
    )
    .remappings(
        [
            # The Go2 publishes its WebRTC pointcloud on "lidar" — the
            # nav_stack modules expect that exact stream under the name
            # "registered_scan".
            (GO2Connection, "lidar", "registered_scan"),
            # WebRTC pose → adapter input. The adapter publishes
            # nav_msgs.Odometry as "odometry", which PGO + terrain modules
            # already autoconnect to.
            (GO2Connection, "odom", "pose"),
            # Planner owns way_point — disconnect MovementManager's click
            # relay (same as G1 onboard).
            (MovementManager, "way_point", "_mgr_way_point_unused"),
        ]
    )
    .global_config(n_workers=10, robot_model="unitree_go2")
)


__all__ = ["unitree_go2_nav_onboard"]
