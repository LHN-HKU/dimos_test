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

"""Heuristic arm pointing for the Unitree G1.

Closed-form joint angles for "aim the fingertip at a world point".
Pointing is geometrically a 2-DOF task (the fingertip ray) on a
7-DOF arm. We pick the redundancy explicitly to keep every pose
predictable and natural-looking, and use the three shoulder DOF
to supply the rotation that maps the at-zero fingertip direction
onto the target ray.

At the G1's all-zeros pose, the elbow has a built-in ~90° forward
bend (URDF: upper arm hangs along -z from the shoulder, but the
elbow joint origin offsets the forearm to +x). So the at-zero
fingertip direction in the torso frame is **+x**, not -z. (Verified
empirically against the running sim — see commit message.)

Algorithm:
    1. d = unit(target_torso - shoulder_torso)
    2. R = minimal rotation mapping (1,0,0) → d (Rodrigues axis-angle)
    3. Decompose R = Ry(pitch) Rx(roll) Rz(yaw) (YXZ Euler), assigning
       the three angles directly to shoulder_pitch, shoulder_roll,
       shoulder_yaw.
    4. elbow = +ELBOW_FLEX, wrist_pitch = -ELBOW_FLEX (an aesthetic
       bend at the elbow that the wrist immediately undoes — both
       joints rotate about local +y, so they cancel along the chain).
    5. wrist_roll = wrist_yaw = 0.

The wrist_roll about the pointing axis is the genuinely redundant
DOF; we leave it at zero (palm-down by convention).

Frame conventions (G1 URDF):
    +x = forward, +y = robot's left, +z = up (ROS REP-103)
    Both shoulders use axis (0,1,0) for pitch, (1,0,0) for roll,
    (0,0,1) for yaw — same formulas for left and right.

Approximations:
    * torso ≈ pelvis (waist joints ~0 in upright stance)
    * elbow flex shortens reach but doesn't change fingertip direction
    * 0.279 rad pre-roll baked into shoulder_pitch origin is ignored

For sub-degree accuracy or 6-DOF EE pose tracking, use the Drake-
based ``solve_pointing`` / ``solve`` path in ``drake_optimization_ik``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

# Shoulder origin in torso_link frame, from g1.urdf (left/right symmetric in y).
_SHOULDER_TORSO_X = 0.004
_SHOULDER_TORSO_Y = 0.100
_SHOULDER_TORSO_Z = 0.248

# Aesthetic constants. Small elbow flex looks more natural than a stiff
# straight arm; wrist_pitch counter-rotates to keep the fingertip on the
# pointing ray.
ELBOW_FLEX = 0.40  # rad ~ 23 deg

# G1 arm joint names (URDF).
LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# URDF joint limits (lower, upper) in rad. Mirrored on roll for right arm.
_LIMITS_LEFT = {
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint": (-1.5882, 2.2515),
    "left_shoulder_yaw_joint": (-2.6180, 2.6180),
    "left_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.9722, 1.9722),
    "left_wrist_pitch_joint": (-1.6144, 1.6144),
    "left_wrist_yaw_joint": (-1.6144, 1.6144),
}
_LIMITS_RIGHT = {
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "right_shoulder_yaw_joint": (-2.6180, 2.6180),
    "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.9722, 1.9722),
    "right_wrist_pitch_joint": (-1.6144, 1.6144),
    "right_wrist_yaw_joint": (-1.6144, 1.6144),
}

Side = Literal["left", "right"]


@dataclass(frozen=True)
class PointingSolution:
    side: Side
    joints: dict[str, float]  # URDF joint name -> angle (rad)
    direction_torso: tuple[float, float, float]  # debug: target dir in torso frame


def _quat_rotate_inverse(q_xyzw: tuple[float, float, float, float], v: np.ndarray) -> np.ndarray:
    """Rotate v by the inverse of quaternion q (q in (x,y,z,w) order)."""
    qx, qy, qz, qw = q_xyzw
    # Inverse of unit quaternion = conjugate
    qx, qy, qz = -qx, -qy, -qz
    # Standard quaternion-vector rotation: v' = q*v*q^-1
    # Using the optimized form:
    t = 2.0 * np.cross([qx, qy, qz], v)
    return v + qw * t + np.cross([qx, qy, qz], t)


def _world_to_torso(target_world: np.ndarray, pelvis_pose: PoseStamped) -> np.ndarray:
    """Express ``target_world`` in the torso (≈pelvis) frame."""
    p = np.array([pelvis_pose.position.x, pelvis_pose.position.y, pelvis_pose.position.z])
    q = (
        pelvis_pose.orientation.x,
        pelvis_pose.orientation.y,
        pelvis_pose.orientation.z,
        pelvis_pose.orientation.w,
    )
    return _quat_rotate_inverse(q, target_world - p)


def pick_arm_side(target_world: np.ndarray, pelvis_pose: PoseStamped) -> Side:
    """Pick the arm whose shoulder is closer to (and forward of) the target.

    Cross-body pointing (left arm reaching to robot's right) is ugly
    and often near a joint limit, so prefer the natural side: target
    on the robot's left -> left arm; on the right -> right arm.
    Targets near the centerline (|y| < shoulder_offset) default to
    left, matching the existing G1ManipulationModule fallback.
    """
    target_torso = _world_to_torso(target_world, pelvis_pose)
    if target_torso[1] >= 0.0:
        return "left"
    return "right"


def compute_pointing_joints(
    target_world: np.ndarray,
    pelvis_pose: PoseStamped,
    side: Side,
    *,
    elbow_flex: float = ELBOW_FLEX,
) -> PointingSolution | None:
    """Compute joint angles to point the chosen arm at a world point.

    Returns None if the target is unreachable as a pointing direction
    (behind the torso, or so close to the shoulder that direction is
    undefined).

    Args:
        target_world: (3,) world-frame target position in meters.
        pelvis_pose: latest /odom pose for the pelvis.
        side: which arm to use.
        elbow_flex: aesthetic elbow bend, cancelled by wrist_pitch.
    """
    target_world = np.asarray(target_world, dtype=np.float64).reshape(3)

    # Shoulder origin in torso frame (mirror y for right arm).
    sy = _SHOULDER_TORSO_Y if side == "left" else -_SHOULDER_TORSO_Y
    shoulder_torso = np.array([_SHOULDER_TORSO_X, sy, _SHOULDER_TORSO_Z])

    target_torso = _world_to_torso(target_world, pelvis_pose)
    delta = target_torso - shoulder_torso
    dist = float(np.linalg.norm(delta))
    if dist < 1e-3:
        return None
    d = delta / dist

    # Reject targets behind the torso (would require the arm to fold
    # through the body).
    if d[0] < -0.30:
        return None

    if side == "left":
        joint_names = LEFT_ARM_JOINTS
        limits = _LIMITS_LEFT
    else:
        joint_names = RIGHT_ARM_JOINTS
        limits = _LIMITS_RIGHT

    pitch, roll, yaw = _shoulder_yxz_to_align(d)

    # Wrist_pitch cancels the elbow's pitch contribution to the
    # fingertip direction (both rotate about local +y along the chain).
    raw = {
        joint_names[0]: pitch,  # shoulder_pitch
        joint_names[1]: roll,  # shoulder_roll
        joint_names[2]: yaw,  # shoulder_yaw
        joint_names[3]: elbow_flex,  # elbow
        joint_names[4]: 0.0,  # wrist_roll
        joint_names[5]: -elbow_flex,  # wrist_pitch
        joint_names[6]: 0.0,  # wrist_yaw
    }
    clipped: dict[str, float] = {}
    excess = 0.0
    for name, value in raw.items():
        lo, hi = limits[name]
        clamped = float(np.clip(value, lo, hi))
        excess = max(excess, abs(clamped - value))
        clipped[name] = clamped

    # > 0.10 rad past a limit means the direction is genuinely outside
    # the arm's workspace.
    if excess > 0.10:
        return None

    return PointingSolution(
        side=side,
        joints=clipped,
        direction_torso=(float(d[0]), float(d[1]), float(d[2])),
    )


def _shoulder_yxz_to_align(d: np.ndarray) -> tuple[float, float, float]:
    """Solve (pitch, roll, yaw) so Ry(p) Rx(r) Rz(y) maps (1,0,0) to d.

    The G1 at-zero pose has the elbow pre-bent ~90° forward via the
    URDF link offsets, so the at-zero fingertip direction in the
    shoulder frame is +x (not -z as the upper-arm orientation might
    suggest). The minimal rotation has a unique YXZ Euler decomposition
    when |R[1][2]| < 1 (no gimbal singularity), which holds everywhere
    in the front hemisphere we accept.
    """
    # Minimal rotation: axis-angle from cross/dot.
    v0 = np.array([1.0, 0.0, 0.0])
    cos_a = float(np.clip(np.dot(v0, d), -1.0, 1.0))
    axis = np.cross(v0, d)
    sin_a = float(np.linalg.norm(axis))
    if sin_a < 1e-9:
        # Parallel (cos≈+1) -> identity. Antiparallel impossible here
        # because targets behind the torso (d_x ≤ -0.3) are rejected
        # upstream.
        return 0.0, 0.0, 0.0
    axis = axis / sin_a
    angle = float(np.arctan2(sin_a, cos_a))
    # Rodrigues: R = I + sin(θ) K + (1-cos(θ)) K²
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    # YXZ Euler decomposition of R = Ry(p) Rx(r) Rz(y):
    #   R[1][2] = -sin(r)
    #   R[1][0] = cos(r) sin(y)
    #   R[1][1] = cos(r) cos(y)
    #   R[0][2] = sin(p) cos(r)
    #   R[2][2] = cos(p) cos(r)
    roll = float(np.arcsin(np.clip(-R[1, 2], -1.0, 1.0)))
    yaw = float(np.arctan2(R[1, 0], R[1, 1]))
    pitch = float(np.arctan2(R[0, 2], R[2, 2]))
    return pitch, roll, yaw


def solve_pointing(
    target: PoseStamped | np.ndarray | tuple[float, float, float],
    pelvis_pose: PoseStamped,
    *,
    side: Side | Literal["auto"] = "auto",
) -> PointingSolution | None:
    """Top-level entry: pick a side and compute joints.

    On auto, tries the natural side first; if it returns None (e.g.
    cross-body target), falls back to the other side.
    """
    if isinstance(target, PoseStamped):
        target_world = np.array([target.position.x, target.position.y, target.position.z])
    else:
        target_world = np.asarray(target, dtype=np.float64).reshape(3)

    if side == "auto":
        primary = pick_arm_side(target_world, pelvis_pose)
        result = compute_pointing_joints(target_world, pelvis_pose, primary)
        if result is not None:
            return result
        other: Side = "right" if primary == "left" else "left"
        return compute_pointing_joints(target_world, pelvis_pose, other)

    return compute_pointing_joints(target_world, pelvis_pose, side)


__all__ = [
    "ELBOW_FLEX",
    "LEFT_ARM_JOINTS",
    "RIGHT_ARM_JOINTS",
    "PointingSolution",
    "Side",
    "compute_pointing_joints",
    "pick_arm_side",
    "solve_pointing",
]
