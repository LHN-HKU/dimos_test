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

"""Compatibility shims for the old sim-specific G1 GR00T WBC module."""

from __future__ import annotations

from pathlib import Path

from dimos.core.stream import In
from dimos.memory2.module import Recorder, RecorderConfig
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.robot.unitree.g1.blueprints.basic.g1_groot_wbc import (
    g1_groot_wbc as unitree_g1_groot_wbc_sim,
)

_G1_MEMORY_DB_PATH = "recording_g1.db"


class G1MemoryConfig(RecorderConfig):
    db_path: str | Path = _G1_MEMORY_DB_PATH


class G1Memory(Recorder):
    """Recorder for G1 visual, lidar, and odom streams."""

    color_image: In[Image]
    lidar: In[PointCloud2]
    odom: In[PoseStamped]
    config: G1MemoryConfig


__all__ = ["G1Memory", "G1MemoryConfig", "unitree_g1_groot_wbc_sim"]
