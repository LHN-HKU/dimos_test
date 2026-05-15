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

"""Convert the Go2 WebRTC connection's ``PoseStamped`` odom into a
``nav_msgs/Odometry`` topic the nav_stack SLAM modules (PGO, RtabMap)
consume.

The G1 onboard blueprint uses :class:`FastLio2` for both LiDAR
registration and odometry, and FastLio2 publishes
``odometry: Out[Odometry]`` natively. The Go2 WebRTC channel hands us
``odom: Out[PoseStamped]`` instead, which the rest of the stack
(simple_planner, replanning_a_star) already speaks. This module bridges
those two type-equivalent representations so a Go2 onboard blueprint
can wire the rosnav-style nav_stack just like the G1 does.
"""

from __future__ import annotations

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.PoseWithCovariance import PoseWithCovariance
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.navigation.nav_stack.frames import FRAME_BODY, FRAME_ODOM


class PoseStampedToOdometryConfig(ModuleConfig):
    """Frame names baked into the emitted ``Odometry`` messages."""

    odom_frame: str = FRAME_ODOM
    body_frame: str = FRAME_BODY


class PoseStampedToOdometry(Module):
    """Subscribes to ``pose`` (PoseStamped) and re-publishes as
    ``odometry`` (nav_msgs Odometry).

    Twist is not filled in — the nav_stack consumers only read the
    pose half of the message. Frame names default to ``odom`` /
    ``body`` matching the rosnav stack's frame conventions.
    """

    config: PoseStampedToOdometryConfig

    pose: In[PoseStamped]
    odometry: Out[Odometry]

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.pose.subscribe(self._on_pose)))

    def _on_pose(self, msg: PoseStamped) -> None:
        odom_pose = Pose(
            msg.x,
            msg.y,
            msg.z,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        self.odometry.publish(
            Odometry(
                ts=msg.ts,
                frame_id=self.config.odom_frame,
                child_frame_id=self.config.body_frame,
                pose=PoseWithCovariance(odom_pose),
            )
        )
