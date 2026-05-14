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

"""Compatibility import for the canonical ``g1-groot-wbc`` blueprint.

Keep this module importable for experiments that referenced the old
``unitree_g1_groot_wbc`` symbol. The runnable CLI entry is now only
``g1-groot-wbc``; pass ``--simulation`` to select the MuJoCo backend.
"""

from __future__ import annotations

from dimos.robot.unitree.g1.blueprints.basic.g1_groot_wbc import (
    g1_groot_wbc as unitree_g1_groot_wbc,
)

__all__ = ["unitree_g1_groot_wbc"]
