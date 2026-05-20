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

from dataclasses import dataclass
import math
import threading
import time
from typing import Any

import numpy as np
from reactivex import interval
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


@dataclass
class Scene:
    """Hand-crafted test world for the planner."""

    voxels: np.ndarray  # (N, 3) float32 world-frame coordinates of occupied voxel centers
    voxel_size: float
    start_position: tuple[float, float, float]
    start_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    goal_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    name: str = "scene"


def _cell_centers(low: float, high: float, voxel_size: float) -> np.ndarray:
    """World-frame voxel-center positions for cells whose centers lie in [low, high].

    Generated via integer cell indices to avoid floating-point drift in
    ``np.arange(step=voxel_size)``, which would otherwise mis-bucket points
    at certain x and y values and produce missing stripes downstream.
    """
    i_min = math.floor(low / voxel_size)
    i_max = math.floor(high / voxel_size)
    return (np.arange(i_min, i_max + 1) + 0.5) * voxel_size


def _flat_floor(
    voxel_size: float,
    extent: tuple[float, float, float, float],
    z: float = 0.0,
    holes: list[tuple[float, float, float, float]] | None = None,
) -> np.ndarray:
    """Single-layer floor at height ``z`` over ``extent=(xmin, xmax, ymin, ymax)``,
    with rectangular ``holes`` cut out (e.g. footprints of objects sitting on it)."""
    xmin, xmax, ymin, ymax = extent
    xs = _cell_centers(xmin, xmax, voxel_size)
    ys = _cell_centers(ymin, ymax, voxel_size)
    z_center = (math.floor(z / voxel_size) + 0.5) * voxel_size
    fx, fy = np.meshgrid(xs, ys, indexing="ij")
    mask = np.ones(fx.shape, dtype=bool)
    for hx_min, hx_max, hy_min, hy_max in holes or []:
        mask &= ~((fx >= hx_min) & (fx <= hx_max) & (fy >= hy_min) & (fy <= hy_max))
    return np.stack([fx[mask], fy[mask], np.full(int(mask.sum()), z_center)], axis=1)


def _box_shell(
    voxel_size: float,
    bounds: tuple[float, float, float, float, float, float],
    include_bottom: bool = False,
) -> np.ndarray:
    """Hollow axis-aligned box: top face + 4 side walls. No interior.

    ``bounds=(xmin, xmax, ymin, ymax, zmin, zmax)``. ``include_bottom`` defaults
    False since boxes sitting on a floor occlude their bottom face from lidar.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    xs = _cell_centers(xmin, xmax, voxel_size)
    ys = _cell_centers(ymin, ymax, voxel_size)
    zs = _cell_centers(zmin, zmax, voxel_size)

    def _grid(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        ga, gb, gc = np.meshgrid(a, b, c, indexing="ij")
        return np.stack([ga.ravel(), gb.ravel(), gc.ravel()], axis=1)

    faces = [
        _grid(xs, ys, zs[-1:]),  # top
        _grid(xs[:1], ys, zs),  # -x wall
        _grid(xs[-1:], ys, zs),  # +x wall
        _grid(xs, ys[:1], zs),  # -y wall
        _grid(xs, ys[-1:], zs),  # +y wall
    ]
    if include_bottom:
        faces.append(_grid(xs, ys, zs[:1]))
    return np.concatenate(faces, axis=0)


def spiral_staircase_scene(voxel_size: float = 0.1) -> Scene:
    """Three-flight spiral staircase; flight 3 sits directly above flight 1.

    Floor at z=0. Flight 1 climbs +y from z_voxel=1 to 10 in x∈[0,1]. Landing 1
    is a flat platform at z_voxel=10 spanning x∈[-1,1]. Flight 2 reverses, climbing
    -y from z_voxel=11 to 20 in x∈[-1,0]. Landing 2 at z_voxel=20 spans x∈[-1,1].
    Flight 3 reverses again, climbing +y from z_voxel=21 to 30 in x∈[0,1] — same
    (x,y) footprint as flight 1, two flights' worth of z above it.

    Each step is a single-voxel-thick floating patch (visible from above and
    below via lidar). Robot starts on the floor and must spiral up to the goal
    at the top of flight 3.

    Flight 2's x range is shifted to overlap the +x half of flights 1 and 3 by
    0.5m, so columns in x∈[0, 0.5] carry voxels from all three flights at
    different z levels — a true 3-level overhang test for the column walker.
    """
    step_depth = 0.2  # 2 voxels per step in y
    flight_x_a = (0.0, 1.0)  # flights 1 and 3 (same footprint)
    flight_x_b = (-0.5, 0.5)  # flight 2 (overlaps +x half of A by 0.5m)

    def _flight(
        x_extent: tuple[float, float], y0: float, climb_sign: int, z_base_voxel: int
    ) -> list[np.ndarray]:
        """10 single-voxel-layer steps; climb_sign=+1 climbs +y, -1 climbs -y."""
        out: list[np.ndarray] = []
        for k in range(1, 11):
            y_lo = y0 + climb_sign * (k - 1) * step_depth
            y_hi = y0 + climb_sign * k * step_depth
            ymin, ymax = (y_lo, y_hi) if y_lo < y_hi else (y_hi, y_lo)
            z_voxel = z_base_voxel + k  # step k's top voxel
            out.append(
                _flat_floor(
                    voxel_size,
                    extent=(*x_extent, ymin, ymax),
                    z=z_voxel * voxel_size,
                )
            )
        return out

    parts: list[np.ndarray] = []
    # Flight 1: from y=0 climbing +y to y=2, z_voxel 1..10.
    parts.extend(_flight(flight_x_a, y0=0.0, climb_sign=+1, z_base_voxel=0))
    # Landing 1: y in [2, 3] at z_voxel=10, spans both flights' x ranges.
    parts.append(_flat_floor(voxel_size, extent=(-1.0, 1.0, 2.0, 3.0), z=10 * voxel_size))
    # Flight 2: from y=3 climbing -y to y=1, z_voxel 11..20.
    parts.extend(_flight(flight_x_b, y0=3.0, climb_sign=-1, z_base_voxel=10))
    # Landing 2: y in [0, 1] at z_voxel=20.
    parts.append(_flat_floor(voxel_size, extent=(-1.0, 1.0, 0.0, 1.0), z=20 * voxel_size))
    # Flight 3: from y=0 climbing +y to y=2, z_voxel 21..30 — directly above flight 1.
    parts.extend(_flight(flight_x_a, y0=0.0, climb_sign=+1, z_base_voxel=20))

    # Floor everywhere. No holes — lidar sees through gaps between floating steps.
    parts.append(_flat_floor(voxel_size, extent=(-5.0, 5.0, -5.0, 5.0)))

    voxels = np.concatenate(parts, axis=0).astype(np.float32)
    return Scene(
        voxels=voxels,
        voxel_size=voxel_size,
        start_position=(-3.0, 0.0, 0.5),
        # Top of flight 3 (step 10) at (x=0.5, y=1.9, z_voxel=30).
        goal_position=(0.5, 1.9, 3.5),
        name="spiral_staircase",
    )


def default_scene(voxel_size: float = 0.1) -> Scene:
    """Lidar-realistic shell scene: floor + tall central box + ramp + bridge.

    Robot starts at (-3, 0) and the goal is at (3, 0). A 2m square, 1m-tall
    box sits at the origin — too tall to climb. To its -y side, a 5-step ramp
    stretches from the floor's -y edge to the box; each step is 1 voxel
    (0.1m) tall, traversable. To its +y side, a 2m-wide bridge spans from
    the box to the floor's +y edge; its underside is at z=0.8m, leaving only
    0.7m of clearance under it — less than the robot's 0.75m height, so the
    walker should filter out the floor underneath the bridge as unreachable.

    Expected path from (-3,0) to (3,0): around the central box, using step 1
    of the ramp at the -y end as a low bridge to cross the obstacle strip
    (step 1 is the only step with a 1-voxel delta to the floor).
    """
    # Central tall box. zmax=0.95 → top voxel at z_voxel=9 (cleanly aligned).
    big_box = (-1.0, 1.0, -1.0, 1.0, 0.0, 0.95)

    # Five 1-voxel-tall steps along the -y side of the box, each 0.8m deep in y.
    # zmax = (k + 0.5) * voxel_size keeps floor-of-zmax/voxel_size = k.
    # Step 5 is split: left half (x in [-1, 0]) stays flat at z_voxel=5, and
    # the right half (x in [0, 1]) is sliced into 5 sub-steps climbing from
    # z_voxel=5 to z_voxel=9 (= box top), so the robot can reach the box top.
    step_x = (-1.0, 1.0)
    step_5_y = (-1.8, -1.0)
    steps = [
        (*step_x, -5.0, -4.2, 0.0, 0.15),  # step 1: top voxel z_voxel=1
        (*step_x, -4.2, -3.4, 0.0, 0.25),  # step 2: z_voxel=2
        (*step_x, -3.4, -2.6, 0.0, 0.35),  # step 3
        (*step_x, -2.6, -1.8, 0.0, 0.45),  # step 4
        # Step 5 left half (flat at z_voxel=5).
        (-1.0, 0.0, *step_5_y, 0.0, 0.55),
        # Step 5 right half: 5 sub-steps climbing in -x from z=5 to z=9.
        (0.8, 1.0, *step_5_y, 0.0, 0.55),  # sub A: z=5 (entry at +x edge)
        (0.6, 0.8, *step_5_y, 0.0, 0.65),  # sub B: z=6
        (0.4, 0.6, *step_5_y, 0.0, 0.75),  # sub C: z=7
        (0.2, 0.4, *step_5_y, 0.0, 0.85),  # sub D: z=8
        (0.0, 0.2, *step_5_y, 0.0, 0.95),  # sub E: z=9 (= box top)
    ]

    # Bridge on +y side of box. Top voxel at z_voxel=9 (matches box top); 2
    # voxels thick (underside at z_voxel=8, z=0.8m).
    bridge = (-1.0, 1.0, 1.0, 5.0, 0.85, 0.95)

    # Floor holes: the strip from ramp through central box (no floor visible).
    # No hole under the bridge — lidar sees the floor through the gap on its
    # sides, and the column walker will filter it as unreachable.
    floor = _flat_floor(
        voxel_size,
        extent=(-5.0, 5.0, -5.0, 5.0),
        holes=[(-1.0, 1.0, -5.0, 1.0)],
    )
    box_voxels = _box_shell(voxel_size, big_box)
    step_voxels = [_box_shell(voxel_size, s) for s in steps]
    # include_bottom=True: lidar would see the bridge's underside from below,
    # so emit voxels there. Without this, interior columns under the bridge
    # only have the top voxel and the column walker computes too generous a
    # gap (8 voxels) to the floor and emits a phantom-reachable floor surface.
    bridge_voxels = _box_shell(voxel_size, bridge, include_bottom=True)
    voxels = np.concatenate([floor, box_voxels, *step_voxels, bridge_voxels], axis=0).astype(
        np.float32
    )

    return Scene(
        voxels=voxels,
        voxel_size=voxel_size,
        start_position=(-3.0, 0.0, 0.5),
        # Goal at the +y end of the bridge: forces the planner to climb the
        # ramp + sub-staircase, traverse the box top, and walk the bridge.
        goal_position=(0.0, 4.5, 1.4),
        name="default_floor_box_ramp_bridge",
    )


class EvaluatorConfig(ModuleConfig):
    world_frame: str = "world"
    body_frame: str = "body"
    publish_period: float = 5.0  # s — republish all messages this often


class Evaluator(Module):
    """Publishes a synthetic scene and evaluates the planner's returned path.

    Outputs the three inputs a planner expects (global_map, odometry, goal);
    subscribes to the planner's path output and logs basic metrics.
    """

    config: EvaluatorConfig

    global_map: Out[PointCloud2]
    odometry: Out[Odometry]
    goal: Out[PoseStamped]
    path: In[Path]

    def __init__(self, scenes: list[Scene] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._scenes: list[Scene] = scenes if scenes else [default_scene()]
        self._index: int = 0
        self._lock = threading.Lock()

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.path.subscribe(self._on_path)))
        self.register_disposable(interval(self.config.publish_period).subscribe(self._publish_next))
        logger.info("Evaluator started with %d scene(s)", len(self._scenes))

    @rpc
    def stop(self) -> None:
        super().stop()

    def _publish_next(self, _: Any) -> None:
        if self._index >= len(self._scenes):
            return  # All scenes exhausted; stop publishing.
        scene = self._scenes[self._index]
        logger.info(
            "Evaluator publishing scene %d/%d: %s",
            self._index + 1,
            len(self._scenes),
            scene.name,
        )
        self._publish_map(scene)
        self._publish_odom(scene)
        self._publish_goal(scene)
        self._index += 1

    def _publish_map(self, scene: Scene) -> None:
        cloud = PointCloud2.from_numpy(
            points=scene.voxels,
            frame_id=self.config.world_frame,
            timestamp=time.time(),
        )
        self.global_map.publish(cloud)

    def _publish_odom(self, scene: Scene) -> None:
        x, y, z = scene.start_position
        qx, qy, qz, qw = scene.start_orientation
        odom = Odometry(
            ts=time.time(),
            frame_id=self.config.world_frame,
            child_frame_id=self.config.body_frame,
            pose=Pose(position=Vector3(x, y, z), orientation=Quaternion(qx, qy, qz, qw)),
        )
        self.odometry.publish(odom)

    def _publish_goal(self, scene: Scene) -> None:
        x, y, z = scene.goal_position
        qx, qy, qz, qw = scene.goal_orientation
        goal = PoseStamped(
            ts=time.time(),
            frame_id=self.config.world_frame,
            position=Vector3(x, y, z),
            orientation=Quaternion(qx, qy, qz, qw),
        )
        self.goal.publish(goal)

    def _on_path(self, path: Path) -> None:
        n = len(path.poses)
        if n == 0:
            logger.warning("Evaluator received empty path")
            return
        total_xy = 0.0
        total_z = 0.0
        for a, b in zip(path.poses, path.poses[1:], strict=False):
            dx = b.position.x - a.position.x
            dy = b.position.y - a.position.y
            dz = b.position.z - a.position.z
            total_xy += (dx * dx + dy * dy) ** 0.5
            total_z += abs(dz)
        logger.info(
            "Evaluator received path: %d poses, xy_len=%.2fm, z_traveled=%.2fm",
            n,
            total_xy,
            total_z,
        )
