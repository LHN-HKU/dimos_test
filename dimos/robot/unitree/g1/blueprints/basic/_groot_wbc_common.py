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

"""Shared GR00T WBC config — joint lists, gain tables, default arm pose.

Kept for older experiments that imported the shared constants directly.
The runnable stack is now the single ``g1-groot-wbc`` blueprint, with
``--simulation`` selecting the MuJoCo backend.
"""

from __future__ import annotations

from dimos.control.components import make_humanoid_joints

g1_joints = make_humanoid_joints("g1")
g1_legs_waist = g1_joints[:15]  # indices 0..14 — legs (12) + waist (3)
g1_arms = g1_joints[15:]  # indices 15..28 — left arm (7) + right arm (7)

# Per-joint PD gains, 29 entries in DDS motor order.  Lifted verbatim
# from g1-control-api/configs/g1_groot_wbc.yaml, which itself copies
# GR00T-WBC's g1_29dof_gear_wbc.yaml reference config.  These gains
# are the ones the balance / walk ONNX policies were trained against —
# diverging from them on real hardware risks instability.
G1_GROOT_KP: list[float] = [
    150.0,
    150.0,
    150.0,
    200.0,
    40.0,
    40.0,  # left leg
    150.0,
    150.0,
    150.0,
    200.0,
    40.0,
    40.0,  # right leg
    250.0,
    250.0,
    250.0,  # waist
    100.0,
    100.0,
    40.0,
    40.0,
    20.0,
    20.0,
    20.0,  # left arm
    100.0,
    100.0,
    40.0,
    40.0,
    20.0,
    20.0,
    20.0,  # right arm
]
G1_GROOT_KD: list[float] = [
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    2.0,  # left leg
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    2.0,  # right leg
    5.0,
    5.0,
    5.0,  # waist
    5.0,
    5.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,  # left arm
    5.0,
    5.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,  # right arm
]

# Relaxed arms-down pose.  Values taken from
# g1_control/backends/groot_wbc_backend.py:DEFAULT_29[15:] (all zeros),
# the zero-offset pose the policy was trained against.  Operators can
# override at runtime by publishing joint targets on the arms via the
# coordinator's ``joint_command`` transport.
ARM_DEFAULT_POSE: list[float] = [0.0] * 14

__all__ = [
    "ARM_DEFAULT_POSE",
    "G1_GROOT_KD",
    "G1_GROOT_KP",
    "g1_arms",
    "g1_joints",
    "g1_legs_waist",
]
