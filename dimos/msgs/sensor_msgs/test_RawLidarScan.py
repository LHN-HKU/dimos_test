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

import numpy as np
import pytest

from dimos.msgs.sensor_msgs.RawLidarScan import MAGIC, RawLidarScan


def _sample_scan(n: int = 32, ts: float = 1234.567_891) -> RawLidarScan:
    rng = np.random.default_rng(seed=0)
    xyz = rng.standard_normal((n, 3)).astype(np.float32)
    return RawLidarScan.from_arrays(
        xyz=xyz,
        reflectivity=rng.integers(0, 256, size=n, dtype=np.uint16),
        offset_time_ns=np.arange(n, dtype=np.uint32) * 10_000,
        line=rng.integers(0, 4, size=n, dtype=np.uint16),
        tag=rng.integers(0, 256, size=n, dtype=np.uint8),
        frame_id="mid360",
        ts=ts,
    )


def test_encode_decode_roundtrip() -> None:
    scan = _sample_scan()
    blob = scan.lcm_encode()
    assert blob[:4] == MAGIC
    decoded = RawLidarScan.lcm_decode(blob)
    assert decoded.frame_id == scan.frame_id
    assert decoded.ts == pytest.approx(scan.ts, abs=1e-6)
    assert decoded.points.dtype == scan.points.dtype
    assert decoded.points.shape == scan.points.shape
    np.testing.assert_array_equal(decoded.points, scan.points)


def test_zero_points_roundtrip() -> None:
    scan = _sample_scan(n=0)
    blob = scan.lcm_encode()
    decoded = RawLidarScan.lcm_decode(blob)
    assert len(decoded) == 0
    assert decoded.frame_id == "mid360"


def test_bad_magic_rejected() -> None:
    bad = b"XXXX" + b"\x00" * 10
    with pytest.raises(ValueError, match="bad magic"):
        RawLidarScan.lcm_decode(bad)


def test_truncated_payload_rejected() -> None:
    scan = _sample_scan(n=8)
    blob = scan.lcm_encode()
    with pytest.raises(ValueError, match="truncated"):
        RawLidarScan.lcm_decode(blob[:-5])


def test_from_arrays_dtype_enforced() -> None:
    with pytest.raises(TypeError, match="dtype"):
        RawLidarScan(points=np.zeros(4, dtype=np.float32), frame_id="x")


def test_wire_layout_packed_21_bytes() -> None:
    """Per-point block is 21 bytes, no padding, little-endian.

    Roundtrip must produce the same point bytes the C++ codec would.
    """
    n = 3
    dtype = np.dtype(
        {
            "names": ["x", "y", "z", "reflectivity", "offset_time_ns", "line", "tag"],
            "formats": ["<f4", "<f4", "<f4", "<u2", "<u4", "<u2", "u1"],
            "offsets": [0, 4, 8, 12, 14, 18, 20],
            "itemsize": 21,
        }
    )
    points = np.array(
        [
            (1.0, 2.0, 3.0, 200, 12_345, 1, 0x10),
            (0.0, 0.0, 0.0, 0, 0, 0, 0x00),
            (-1.5, 0.25, 7.0, 255, 99_999, 3, 0x30),
        ],
        dtype=dtype,
    )
    scan = RawLidarScan(points=points, frame_id="lidar", ts=0.0)
    blob = scan.lcm_encode()
    expected_tail = points.tobytes()
    assert len(expected_tail) == 21 * n
    assert blob[-len(expected_tail) :] == expected_tail
