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

"""Benchmark runners for sim and hardware.

Two cores, one file:

* **Sim** (`_run_path_follower_sim`) - boots a real ControlCoordinator in-process
  with the Go2SimTwistBaseAdapter, drives the controller through a path, records
  the trajectory. Used by `test_controllers.py` and the sim battery.

* **Hardware** (`_run_path_follower_hw`) - LCM pub/sub on `/cmd_vel` + `/go2/odom`,
  pose-velocity estimator (Go2 over WebRTC publishes pose only, no twist),
  rigid path-anchoring to the robot's current pose, SIGINT-safe shutdown.

Per-controller wrappers (one pair per algorithm: `run_<algo>_sim` and
`run_<algo>_hw`) are thin factories that build the controller's task and hand
it to the matching core loop.

Hardware caveats: onboard Go2 odom drifts on long/curvy paths - use short
paths for trustworthy CTE numbers. Pre-roll is manual.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import signal
import threading
import time
from typing import TYPE_CHECKING, Protocol

from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.control.task import CoordinatorState, JointStateSnapshot
from dimos.control.tasks.baseline_path_follower_task import (
    BaselinePathFollowerTask,
    BaselinePathFollowerTaskConfig,
)
from dimos.control.tasks.feedforward_gain_compensator import FeedforwardGainConfig
from dimos.control.tasks.lyapunov_path_controller import LyapunovPathControllerConfig
from dimos.control.tasks.path_follower_task import (
    PathFollowerTask,
    PathFollowerTaskConfig,
)
from dimos.control.tasks.pure_pursuit_path_follower_task import (
    PurePursuitPathFollowerTask,
    PurePursuitPathFollowerTaskConfig,
)
from dimos.control.tasks.reactive_path_follower_task import (
    ReactivePathFollowerTask,
    ReactivePathFollowerTaskConfig,
)

# Lyapunov on hw spins on cornered paths because k_theta * sin(e_theta) directly
# amplifies heading-estimate noise from /go2/odom. Default k_theta=1.5 saturated
# wz on corner_90 (cte=234cm in the first hw run). 0.6 keeps convergence usable
# while attenuating noise amplification.
_LYAPUNOV_HW_KTHETA = 0.6

# RPP on hw cuts corners at high speed because PurePursuit's adaptive lookahead
# grows with speed (default max_lookahead=1.0). On a 90° corner with 2m legs the
# lookahead point ends up past the corner on the second leg, giving a gentle
# sweep instead of a sharp turn. wz peak was ~0.55 rad/s (no saturation) at all
# speeds — the controller wasn't trying, the geometry told it not to. Capping
# max_lookahead at 0.5 forces the controller to commit to the turn.
_RPP_MAX_LOOKAHEAD = 0.5
from dimos.control.tasks.velocity_tracking_pid import VelocityTrackingConfig
from dimos.core.global_config import global_config
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Path import Path
from dimos.utils.benchmarking.scoring import ExecutedTrajectory, TrajectoryTick
from dimos.utils.benchmarking.sim_blueprint import GO2_TICK_RATE_HZ, _base_joints, _go2_sim_base
from dimos.utils.trigonometry import angle_diff

if TYPE_CHECKING:
    from dimos.hardware.drive_trains.go2_sim.adapter import Go2SimTwistBaseAdapter


# Hardware safety limits (from Rung 1 saturation envelope)
VX_MAX = 1.0  # m/s
WZ_MAX = 1.5  # rad/s

# Any of these task states means "we're done"
_ARRIVED_STATES = frozenset({"arrived", "completed"})
_FAILED_STATES = frozenset({"aborted"})


class _PathFollowerLike(Protocol):
    """Common contract every controller task must satisfy for either core loop."""

    def start_path(self, path: Path, current_odom: PoseStamped) -> bool: ...
    def update_odom(self, odom: PoseStamped) -> None: ...
    def compute(self, state) -> object: ...  # CoordinatorState -> JointCommandOutput|None
    def get_state(self) -> str: ...


@dataclass
class HwRunOptions:
    """Runtime knobs for the hw core loop and its per-controller wrappers."""

    timeout_s: float = 30.0
    speed: float = 0.55
    k_angular: float = 0.5
    pid_config: VelocityTrackingConfig | None = None
    ff_config: FeedforwardGainConfig | None = None
    odom_warmup_s: float = 2.0
    cmd_topic: str = "/cmd_vel"
    odom_topic: str = "/go2/odom"


# --- Sim helpers ---


def _odom_to_pose(odom: list[float]) -> PoseStamped:
    return PoseStamped(
        position=Vector3(odom[0], odom[1], 0.0),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, odom[2])),
    )


def _vels_to_twist(v: list[float]) -> Twist:
    return Twist(linear=Vector3(v[0], v[1], 0.0), angular=Vector3(0.0, 0.0, v[2]))


# --- Hardware helpers ---


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _twist_clamped(vx: float, vy: float, wz: float) -> Twist:
    return Twist(
        linear=Vector3(_clamp(vx, -VX_MAX, VX_MAX), 0.0, 0.0),  # vy ignored on Go2
        angular=Vector3(0.0, 0.0, _clamp(wz, -WZ_MAX, WZ_MAX)),
    )


class _PoseVelocityEstimator:
    """Differentiate consecutive PoseStamped to derive body-frame (vx, vy, wz).

    Go2 over WebRTC publishes pose only - no velocity. EMA smooths the raw
    differences (single-pole, alpha tuned for 10 Hz to follow real motion
    while suppressing sample-noise jitter).
    """

    def __init__(self, alpha: float = 0.5) -> None:
        self._prev_pose: PoseStamped | None = None
        self._prev_t: float | None = None
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0
        self._alpha = alpha

    def update(self, pose: PoseStamped, t: float) -> tuple[float, float, float]:
        if self._prev_pose is None or self._prev_t is None:
            self._prev_pose = pose
            self._prev_t = t
            return 0.0, 0.0, 0.0
        dt = t - self._prev_t
        if dt <= 0:
            return self._vx, self._vy, self._wz

        dx = pose.position.x - self._prev_pose.position.x
        dy = pose.position.y - self._prev_pose.position.y
        dyaw = angle_diff(pose.orientation.euler[2], self._prev_pose.orientation.euler[2])

        wx = dx / dt
        wy = dy / dt
        yaw = pose.orientation.euler[2]
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        bx = wx * cos_y + wy * sin_y
        by = -wx * sin_y + wy * cos_y
        bw = dyaw / dt

        self._vx = self._alpha * bx + (1 - self._alpha) * self._vx
        self._vy = self._alpha * by + (1 - self._alpha) * self._vy
        self._wz = self._alpha * bw + (1 - self._alpha) * self._wz

        self._prev_pose = pose
        self._prev_t = t
        return self._vx, self._vy, self._wz


def _shift_path_to_start_at_pose(path: Path, start_pose: PoseStamped) -> Path:
    """Rigid-transform `path` so its first pose aligns with `start_pose`.

    Reference paths are defined in a robot-centric frame (path[0] at origin
    facing +x). On hardware we need them in the Go2 odom frame, anchored
    to wherever the robot is now.
    """
    px0 = path.poses[0].position.x
    py0 = path.poses[0].position.y
    pyaw0 = path.poses[0].orientation.euler[2]
    sx = start_pose.position.x
    sy = start_pose.position.y
    syaw = start_pose.orientation.euler[2]

    dyaw = syaw - pyaw0
    cos_d, sin_d = math.cos(dyaw), math.sin(dyaw)

    new_poses = []
    for p in path.poses:
        rx = p.position.x - px0
        ry = p.position.y - py0
        nx = rx * cos_d - ry * sin_d
        ny = rx * sin_d + ry * cos_d
        new_poses.append(
            PoseStamped(
                position=Vector3(sx + nx, sy + ny, 0.0),
                orientation=Quaternion.from_euler(Vector3(0.0, 0.0, p.orientation.euler[2] + dyaw)),
            )
        )
    return Path(poses=new_poses)


# --- Sim core loop ---


def _run_path_follower_sim(
    task_factory: Callable[[], _PathFollowerLike],
    path: Path,
    timeout_s: float,
    sample_rate_hz: float,
) -> ExecutedTrajectory:
    """Generic sim loop: any task implementing the path-follower contract."""
    coord = ControlCoordinator(
        tick_rate=GO2_TICK_RATE_HZ,
        hardware=[_go2_sim_base()],
        tasks=[
            TaskConfig(
                name="vel_base",
                type="velocity",
                joint_names=_base_joints,
                priority=10,
            ),
        ],
    )
    task = task_factory()

    coord.start()
    try:
        adapter: Go2SimTwistBaseAdapter = coord._hardware["base"].adapter
        # Reset plant to path start so sim odom matches path[0].
        start = path.poses[0]
        adapter.set_initial_pose(start.position.x, start.position.y, start.orientation.euler[2])
        adapter.connect()

        coord.add_task(task)
        task.start_path(path, _odom_to_pose(adapter.read_odometry()))

        ticks: list[TrajectoryTick] = []
        period = 1.0 / sample_rate_hz
        t0 = time.perf_counter()
        next_sample = t0
        arrived = False

        while True:
            now = time.perf_counter()
            t_rel = now - t0
            if t_rel > timeout_s:
                break

            pose = _odom_to_pose(adapter.read_odometry())
            task.update_odom(pose)

            ticks.append(
                TrajectoryTick(
                    t=t_rel,
                    pose=pose,
                    cmd_twist=_vels_to_twist(adapter._cmd),
                    actual_twist=_vels_to_twist(adapter.read_velocities()),
                )
            )

            s = task.get_state()
            if s in _ARRIVED_STATES:
                arrived = True
                break
            if s in _FAILED_STATES:
                break

            next_sample += period
            sleep_for = next_sample - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

        return ExecutedTrajectory(ticks=ticks, arrived=arrived)
    finally:
        coord.stop()


# --- Hardware core loop ---


def _run_path_follower_hw(
    task_factory: Callable[[], _PathFollowerLike],
    path: Path,
    opts: HwRunOptions,
    *,
    interactive: bool,
    label: str,
) -> tuple[Path, ExecutedTrajectory]:
    """Generic hardware loop: any task implementing the path-follower contract.

    Handles odom warm-up, pre-roll, path anchoring, the 10 Hz tick loop,
    SIGINT safety, and zero-Twist on exit.
    """
    cmd_pub = LCMTransport(opts.cmd_topic, Twist)
    odom_sub = LCMTransport(opts.odom_topic, PoseStamped)

    latest_pose: list[PoseStamped | None] = [None]
    last_odom_t: list[float] = [0.0]
    odom_lock = threading.Lock()

    def _on_odom(msg: PoseStamped) -> None:
        with odom_lock:
            latest_pose[0] = msg
            last_odom_t[0] = time.perf_counter()

    odom_sub.subscribe(_on_odom)

    # Wait for odom warmup so we have a real pose before starting.
    print(f"[hw {label}] waiting up to {opts.odom_warmup_s:.1f}s for odom on {opts.odom_topic}...")
    deadline = time.perf_counter() + opts.odom_warmup_s
    while time.perf_counter() < deadline:
        with odom_lock:
            if latest_pose[0] is not None:
                break
        time.sleep(0.05)
    with odom_lock:
        if latest_pose[0] is None:
            cmd_pub.broadcast(None, _twist_clamped(0, 0, 0))
            raise RuntimeError(
                f"No odom received on {opts.odom_topic} within {opts.odom_warmup_s}s. "
                "Is `dimos run unitree-go2-webrtc-keyboard-teleop` running?"
            )
        start_pose = latest_pose[0]

    if interactive:
        first = path.poses[0]
        print(
            f"[hw {label}] PRE-ROLL: park robot at path start "
            f"({first.position.x:.2f}, {first.position.y:.2f}, "
            f"yaw={first.orientation.euler[2]:.2f}) then press Enter..."
        )
        input()
        with odom_lock:
            start_pose = latest_pose[0]

    task = task_factory()
    path_world = _shift_path_to_start_at_pose(path, start_pose)
    print(
        f"[hw {label}] anchored path: start=({path_world.poses[0].position.x:.2f},"
        f"{path_world.poses[0].position.y:.2f}) "
        f"goal=({path_world.poses[-1].position.x:.2f},"
        f"{path_world.poses[-1].position.y:.2f})"
    )
    task.start_path(path_world, start_pose)

    # SIGINT -> clean stop
    stop_flag = {"stop": False}

    def _sigint_handler(_signum, _frame):  # type: ignore[no-untyped-def]
        stop_flag["stop"] = True
        print(f"\n[hw {label}] SIGINT - stopping")

    prev = signal.signal(signal.SIGINT, _sigint_handler)

    ticks: list[TrajectoryTick] = []
    arrived = False
    period = 1.0 / GO2_TICK_RATE_HZ
    t0 = time.perf_counter()
    next_tick = t0
    vel_est = _PoseVelocityEstimator()

    try:
        while True:
            now = time.perf_counter()
            t_rel = now - t0

            if stop_flag["stop"]:
                break
            if t_rel > opts.timeout_s:
                print(f"[hw {label}] timeout after {opts.timeout_s:.1f}s")
                break

            with odom_lock:
                pose = latest_pose[0]
                last_pose_age = now - last_odom_t[0]
            if pose is None or last_pose_age > 1.0:
                print(f"[hw {label}] ABORT: stale odom ({last_pose_age:.2f}s)")
                break

            task.update_odom(pose)
            est_vx, est_vy, est_wz = vel_est.update(pose, now)
            state = CoordinatorState(
                joints=JointStateSnapshot(
                    joint_velocities={
                        "base/vx": est_vx,
                        "base/vy": est_vy,
                        "base/wz": est_wz,
                    },
                    timestamp=now,
                ),
                t_now=now,
                dt=period,
            )
            cmd = task.compute(state)
            if cmd is not None and cmd.velocities is not None:
                vx, vy, wz = cmd.velocities[0], cmd.velocities[1], cmd.velocities[2]
            else:
                vx = vy = wz = 0.0

            twist = _twist_clamped(vx, vy, wz)
            cmd_pub.broadcast(None, twist)

            ticks.append(
                TrajectoryTick(
                    t=t_rel,
                    pose=pose,
                    cmd_twist=twist,
                    actual_twist=Twist(
                        linear=Vector3(est_vx, est_vy, 0.0),
                        angular=Vector3(0.0, 0.0, est_wz),
                    ),
                )
            )

            s = task.get_state()
            if s in _ARRIVED_STATES:
                arrived = True
                print(f"[hw {label}] arrived in {t_rel:.2f}s")
                break
            if s in _FAILED_STATES:
                print(f"[hw {label}] task aborted")
                break

            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        for _ in range(3):
            cmd_pub.broadcast(None, _twist_clamped(0, 0, 0))
            time.sleep(0.05)
        signal.signal(signal.SIGINT, prev)

    return path_world, ExecutedTrajectory(ticks=ticks, arrived=arrived)


# --- Per-controller wrappers (sim) ---


def run_baseline_sim(
    path: Path,
    timeout_s: float = 60.0,
    sample_rate_hz: float = GO2_TICK_RATE_HZ,
    speed: float = 0.55,
    k_angular: float = 0.5,
    pid_config: VelocityTrackingConfig | None = None,
    ff_config: FeedforwardGainConfig | None = None,
    task_name: str = "baseline_follower",
) -> ExecutedTrajectory:
    """Production LocalPlanner P-controller in sim."""

    def _make() -> BaselinePathFollowerTask:
        return BaselinePathFollowerTask(
            name=task_name,
            config=BaselinePathFollowerTaskConfig(
                speed=speed,
                k_angular=k_angular,
                pid_config=pid_config,
                ff_config=ff_config,
            ),
            global_config=global_config,
        )

    return _run_path_follower_sim(_make, path, timeout_s, sample_rate_hz)


def run_lyapunov_sim(
    path: Path,
    timeout_s: float = 60.0,
    sample_rate_hz: float = GO2_TICK_RATE_HZ,
    pid_config: VelocityTrackingConfig | None = None,
    ff_config: FeedforwardGainConfig | None = None,
    task_name: str = "lyapunov_follower",
) -> ExecutedTrajectory:
    """Lyapunov reactive controller in sim."""

    def _make() -> ReactivePathFollowerTask:
        return ReactivePathFollowerTask(
            name=task_name,
            config=ReactivePathFollowerTaskConfig(
                joint_names=list(_base_joints),
                pid_config=pid_config,
                ff_config=ff_config,
            ),
        )

    return _run_path_follower_sim(_make, path, timeout_s, sample_rate_hz)


def run_pure_pursuit_sim(
    path: Path,
    timeout_s: float = 60.0,
    sample_rate_hz: float = GO2_TICK_RATE_HZ,
    speed: float = 0.55,
    ff_config: FeedforwardGainConfig | None = None,
    task_name: str = "pure_pursuit_follower",
) -> ExecutedTrajectory:
    """Classic Pure Pursuit (constant lookahead, no PID) in sim."""

    def _make() -> PurePursuitPathFollowerTask:
        return PurePursuitPathFollowerTask(
            name=task_name,
            config=PurePursuitPathFollowerTaskConfig(
                joint_names=list(_base_joints),
                target_speed=speed,
                ff_config=ff_config,
            ),
            global_config=global_config,
        )

    return _run_path_follower_sim(_make, path, timeout_s, sample_rate_hz)


def run_rpp_sim(
    path: Path,
    timeout_s: float = 60.0,
    sample_rate_hz: float = GO2_TICK_RATE_HZ,
    speed: float = 0.55,
    ff_config: FeedforwardGainConfig | None = None,
    task_name: str = "rpp_follower",
    *,
    max_lookahead: float | None = None,
    ct_kp: float = 0.0,
    ct_ki: float = 0.0,
    ct_kd: float = 0.0,
) -> ExecutedTrajectory:
    """Regulated Pure Pursuit (PathFollowerTask: PurePursuit + adaptive
    lookahead + curvature-aware velocity profiler + cross-track PID) in sim.

    ``max_lookahead`` defaults to ``_RPP_MAX_LOOKAHEAD`` (0.5m). Pass an
    explicit value to override per-call (e.g., for a geometric tuning sweep).
    ``ct_kp/ct_ki/ct_kd`` default to 0 (PID disabled); pass non-zero to
    engage the cross-track PID with explicit gains.
    """
    eff_lookahead = max_lookahead if max_lookahead is not None else _RPP_MAX_LOOKAHEAD

    def _make() -> PathFollowerTask:
        return PathFollowerTask(
            name=task_name,
            config=PathFollowerTaskConfig(
                joint_names=list(_base_joints),
                max_linear_speed=speed,
                max_lookahead=eff_lookahead,
                ct_kp=ct_kp,
                ct_ki=ct_ki,
                ct_kd=ct_kd,
                ff_config=ff_config,
            ),
            global_config=global_config,
        )

    return _run_path_follower_sim(_make, path, timeout_s, sample_rate_hz)


def run_rpp_tuned_sim(
    path: Path,
    timeout_s: float = 60.0,
    sample_rate_hz: float = GO2_TICK_RATE_HZ,
    speed: float = 0.55,
    ff_config: FeedforwardGainConfig | None = None,
    task_name: str = "rpp_tuned_follower",
) -> ExecutedTrajectory:
    """RPP in sim with config from :func:`tuning.tune_rpp_for_path`.

    Sim parity for :func:`run_rpp_tuned_hw` — useful for pre-flight on
    a new path before burning robot time.
    """
    from dimos.utils.benchmarking.tuning import tune_rpp_for_path

    cfg = tune_rpp_for_path(path, speed)
    cfg.pop("_diagnostics", None)
    return run_rpp_sim(
        path,
        timeout_s=timeout_s,
        sample_rate_hz=sample_rate_hz,
        speed=speed,
        ff_config=ff_config,
        task_name=task_name,
        **cfg,
    )


def run_setpoint_sim(
    path: Path,
    timeout_s: float = 60.0,
    sample_rate_hz: float = GO2_TICK_RATE_HZ,
    speed: float = 0.55,
    task_name: str = "setpoint_follower",
) -> ExecutedTrajectory:
    """Pose-PID setpoint controller (drives to ``path.poses[-1]``) in sim.

    Uses :class:`SetpointControlTask` which wraps :class:`SetpointController`
    as the same path-follower protocol the battery runner expects. ``speed``
    becomes ``max_vx`` on the underlying controller.
    """
    from dimos.utils.benchmarking.setpoint_benchmark import (
        SetpointControllerConfig,
        SetpointControlTask,
    )

    def _make() -> SetpointControlTask:
        return SetpointControlTask(
            name=task_name,
            setpoint_config=SetpointControllerConfig(max_vx=speed),
            joint_names=list(_base_joints),
        )

    return _run_path_follower_sim(_make, path, timeout_s, sample_rate_hz)


def run_mpc_sim(
    path: Path,
    timeout_s: float = 60.0,
    sample_rate_hz: float = GO2_TICK_RATE_HZ,
    speed: float = 0.55,
    task_name: str = "mpc_follower",
) -> ExecutedTrajectory:
    """MPC stub - raises NotImplementedError on first compute()."""
    # Lazy import: mpc_path_follower_task isn't committed on this branch,
    # so at module-import time we can't see it. The stub is meant to raise
    # on call anyway; keep the stub callable signature, fail at call time.
    from dimos.control.tasks.mpc_path_follower_task import (
        MPCPathFollowerTask,
        MPCPathFollowerTaskConfig,
    )
    from dimos.utils.benchmarking.plant_models import GO2_PLANT_FITTED

    def _make() -> MPCPathFollowerTask:
        return MPCPathFollowerTask(
            name=task_name,
            config=MPCPathFollowerTaskConfig(
                joint_names=list(_base_joints),
                target_speed=speed,
                plant=GO2_PLANT_FITTED,
            ),
        )

    return _run_path_follower_sim(_make, path, timeout_s, sample_rate_hz)


# --- Per-controller wrappers (hardware) ---


def run_baseline_hw(
    path: Path,
    opts: HwRunOptions | None = None,
    *,
    interactive: bool = True,
) -> tuple[Path, ExecutedTrajectory]:
    """Production LocalPlanner P-controller (BaselinePathFollowerTask) on hw."""
    opts = opts or HwRunOptions()

    def _make() -> BaselinePathFollowerTask:
        return BaselinePathFollowerTask(
            name="baseline_hw_follower",
            config=BaselinePathFollowerTaskConfig(
                speed=opts.speed,
                k_angular=opts.k_angular,
                pid_config=opts.pid_config,
                ff_config=opts.ff_config,
            ),
            global_config=global_config,
        )

    return _run_path_follower_hw(_make, path, opts, interactive=interactive, label="baseline")


def run_lyapunov_hw(
    path: Path,
    opts: HwRunOptions | None = None,
    *,
    interactive: bool = True,
) -> tuple[Path, ExecutedTrajectory]:
    """Lyapunov reactive controller (ReactivePathFollowerTask) on hw."""
    opts = opts or HwRunOptions()

    def _make() -> ReactivePathFollowerTask:
        return ReactivePathFollowerTask(
            name="lyapunov_hw_follower",
            config=ReactivePathFollowerTaskConfig(
                joint_names=list(_base_joints),
                controller=LyapunovPathControllerConfig(k_theta=_LYAPUNOV_HW_KTHETA),
                pid_config=opts.pid_config,
                ff_config=opts.ff_config,
            ),
        )

    return _run_path_follower_hw(_make, path, opts, interactive=interactive, label="lyapunov")


def run_pure_pursuit_hw(
    path: Path,
    opts: HwRunOptions | None = None,
    *,
    interactive: bool = True,
) -> tuple[Path, ExecutedTrajectory]:
    """Classic Pure Pursuit (constant lookahead, no PID) on hw."""
    opts = opts or HwRunOptions()

    def _make() -> PurePursuitPathFollowerTask:
        return PurePursuitPathFollowerTask(
            name="pure_pursuit_hw_follower",
            config=PurePursuitPathFollowerTaskConfig(
                joint_names=list(_base_joints),
                target_speed=opts.speed,
                ff_config=opts.ff_config,
            ),
            global_config=global_config,
        )

    return _run_path_follower_hw(_make, path, opts, interactive=interactive, label="pp")


def run_rpp_hw(
    path: Path,
    opts: HwRunOptions | None = None,
    *,
    interactive: bool = True,
    max_lookahead: float | None = None,
    ct_kp: float = 0.0,
    ct_ki: float = 0.0,
    ct_kd: float = 0.0,
) -> tuple[Path, ExecutedTrajectory]:
    """Regulated Pure Pursuit (PathFollowerTask: PurePursuit + adaptive
    lookahead + curvature-aware velocity profiler + cross-track PID) on hw.

    ``max_lookahead`` defaults to ``_RPP_MAX_LOOKAHEAD`` (0.5m); pass an
    explicit value to override per-call (e.g. from
    :func:`tuning.tune_rpp_for_path`). ``ct_kp/ct_ki/ct_kd`` default to 0
    (PID disabled); pass non-zero to engage with explicit gains.
    """
    opts = opts or HwRunOptions()
    eff_lookahead = max_lookahead if max_lookahead is not None else _RPP_MAX_LOOKAHEAD

    def _make() -> PathFollowerTask:
        return PathFollowerTask(
            name="rpp_hw_follower",
            config=PathFollowerTaskConfig(
                joint_names=list(_base_joints),
                max_linear_speed=opts.speed,
                max_lookahead=eff_lookahead,
                ct_kp=ct_kp,
                ct_ki=ct_ki,
                ct_kd=ct_kd,
                ff_config=opts.ff_config,
            ),
            global_config=global_config,
        )

    return _run_path_follower_hw(_make, path, opts, interactive=interactive, label="rpp")


def run_rpp_tuned_hw(
    path: Path,
    opts: HwRunOptions | None = None,
    *,
    interactive: bool = True,
) -> tuple[Path, ExecutedTrajectory]:
    """RPP on hw with config from :func:`tuning.tune_rpp_for_path`.

    Calls the tuner with ``(path, opts.speed)``, gets back ``max_lookahead``
    and ``(ct_kp, ct_kd)``, then delegates to :func:`run_rpp_hw`. No
    hand-tuned constants — the config falls out of path geometry + the
    desired closed-loop bandwidth.
    """
    from dimos.utils.benchmarking.tuning import tune_rpp_for_path

    opts = opts or HwRunOptions()
    cfg = tune_rpp_for_path(path, opts.speed)
    diag = cfg.pop("_diagnostics", {})
    if diag.get("infeasible"):
        # Surface this in logs — operator should know the geometric tuner says
        # this (path, speed) pair is impossible. We still try (the run won't
        # hurt anything; cte will just be high).
        from dimos.utils.logging_config import setup_logger

        setup_logger().warning(
            f"run_rpp_tuned_hw: geometric tuner reports INFEASIBLE for path "
            f"(R_curve={diag.get('R_curve_min', float('nan')):.3f}m, "
            f"R_robot={diag.get('R_robot_min', float('nan')):.3f}m at v={opts.speed}m/s). "
            f"Running anyway; expect large cte."
        )
    return run_rpp_hw(path, opts, interactive=interactive, **cfg)


def run_setpoint_hw(
    path: Path,
    opts: HwRunOptions | None = None,
    *,
    interactive: bool = True,
) -> tuple[Path, ExecutedTrajectory]:
    """Pose-PID setpoint controller (drives to ``path.poses[-1]``) on hardware."""
    from dimos.utils.benchmarking.setpoint_benchmark import (
        SetpointControllerConfig,
        SetpointControlTask,
    )

    opts = opts or HwRunOptions()

    def _make() -> SetpointControlTask:
        return SetpointControlTask(
            name="setpoint_hw_follower",
            setpoint_config=SetpointControllerConfig(max_vx=opts.speed),
            joint_names=list(_base_joints),
        )

    return _run_path_follower_hw(_make, path, opts, interactive=interactive, label="setpoint")


def run_mpc_hw(
    path: Path,
    opts: HwRunOptions | None = None,
    *,
    interactive: bool = True,
) -> tuple[Path, ExecutedTrajectory]:
    """MPC stub on hardware - raises NotImplementedError on first compute()."""
    opts = opts or HwRunOptions()
    # Same lazy-import dance as run_mpc_sim.
    from dimos.control.tasks.mpc_path_follower_task import (
        MPCPathFollowerTask,
        MPCPathFollowerTaskConfig,
    )
    from dimos.utils.benchmarking.plant_models import GO2_PLANT_FITTED

    def _make() -> MPCPathFollowerTask:
        return MPCPathFollowerTask(
            name="mpc_hw_follower",
            config=MPCPathFollowerTaskConfig(
                joint_names=list(_base_joints),
                target_speed=opts.speed,
                plant=GO2_PLANT_FITTED,
            ),
        )

    return _run_path_follower_hw(_make, path, opts, interactive=interactive, label="mpc")


__all__ = [
    "HwRunOptions",
    "run_baseline_hw",
    "run_baseline_sim",
    "run_lyapunov_hw",
    "run_lyapunov_sim",
    "run_mpc_hw",
    "run_mpc_sim",
    "run_pure_pursuit_hw",
    "run_pure_pursuit_sim",
    "run_rpp_hw",
    "run_rpp_sim",
    "run_rpp_tuned_hw",
    "run_rpp_tuned_sim",
    "run_setpoint_hw",
    "run_setpoint_sim",
]
