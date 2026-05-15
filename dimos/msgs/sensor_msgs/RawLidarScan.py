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

"""Raw lidar scan message — per-point timing/line preserved.

Hand-rolled binary layout published over LCM as opaque bytes; no dimos-lcm
schema involved. The format intentionally matches the per-point block in
`dimos-module-fastlio2/src/raw_dump.hpp` so recorded scans can be replayed
into FastLio's `feed_lidar` without any intermediate translation.

Wire format (little-endian, packed):

    magic        : 4 bytes  b"RLS\\x01"
    ts_sec       : i32      ROS-style timestamp seconds
    ts_nsec      : u32      ROS-style timestamp nanoseconds
    frame_len    : u16      length of frame_id in bytes
    frame_id     : frame_len bytes (UTF-8)
    point_count  : u32
    points       : point_count * (
                       x: f32, y: f32, z: f32,
                       reflectivity: u16,
                       offset_time_ns: u32,
                       line: u16,
                       tag: u8,
                   )  = 21 bytes per point (packed, no trailing pad)

`tag` carries Livox-style return-quality bits; FastLio's preprocessor uses
`(tag & 0x30) ∈ {0x00, 0x10}` for noise filtering. Non-Livox sensors that
don't have an analogue should publish `tag=0` (matches the "accept" bin).
"""

from __future__ import annotations

import struct
import time

import numpy as np

from dimos.types.timestamped import Timestamped

MAGIC = b"RLS\x01"
_HEADER_FMT = "<4sIIH"  # magic, ts_sec(u32 reinterpret of i32), ts_nsec, frame_len
_HEADER_LEN = struct.calcsize(_HEADER_FMT)
_POINT_DTYPE = np.dtype(
    {
        "names": ["x", "y", "z", "reflectivity", "offset_time_ns", "line", "tag"],
        "formats": ["<f4", "<f4", "<f4", "<u2", "<u4", "<u2", "u1"],
        "offsets": [0, 4, 8, 12, 14, 18, 20],
        "itemsize": 21,
    }
)
assert _POINT_DTYPE.itemsize == 21


class RawLidarScan(Timestamped):
    """Raw lidar scan with per-point offset_time + ring/line index.

    `points` is an (N,) structured numpy array with dtype
    `(x f32, y f32, z f32, reflectivity u16, offset_time_ns u32, line u16)`.
    Use the helper `from_arrays(...)` to build one from plain numpy arrays.
    """

    msg_name = "sensor_msgs.RawLidarScan"

    def __init__(
        self,
        points: np.ndarray,
        frame_id: str = "lidar",
        ts: float | None = None,
    ) -> None:
        if points.dtype != _POINT_DTYPE:
            raise TypeError(
                f"RawLidarScan.points must have dtype {_POINT_DTYPE!r}, got {points.dtype!r}"
            )
        if points.ndim != 1:
            raise ValueError(f"RawLidarScan.points must be 1-D, got shape {points.shape}")
        self.ts = ts if ts is not None else time.time()
        self.frame_id = frame_id
        self.points = points

    @classmethod
    def from_arrays(
        cls,
        xyz: np.ndarray,
        reflectivity: np.ndarray,
        offset_time_ns: np.ndarray,
        line: np.ndarray,
        tag: np.ndarray | None = None,
        frame_id: str = "lidar",
        ts: float | None = None,
    ) -> RawLidarScan:
        n = xyz.shape[0]
        if xyz.shape != (n, 3):
            raise ValueError(f"xyz must be (N,3), got {xyz.shape}")
        if tag is None:
            tag = np.zeros(n, dtype=np.uint8)
        for name, arr in (
            ("reflectivity", reflectivity),
            ("offset_time_ns", offset_time_ns),
            ("line", line),
            ("tag", tag),
        ):
            if arr.shape != (n,):
                raise ValueError(f"{name} must be (N,), got {arr.shape}")
        points = np.empty(n, dtype=_POINT_DTYPE)
        points["x"] = xyz[:, 0].astype(np.float32, copy=False)
        points["y"] = xyz[:, 1].astype(np.float32, copy=False)
        points["z"] = xyz[:, 2].astype(np.float32, copy=False)
        points["reflectivity"] = reflectivity.astype(np.uint16, copy=False)
        points["offset_time_ns"] = offset_time_ns.astype(np.uint32, copy=False)
        points["line"] = line.astype(np.uint16, copy=False)
        points["tag"] = tag.astype(np.uint8, copy=False)
        return cls(points=points, frame_id=frame_id, ts=ts)

    def lcm_encode(self) -> bytes:
        sec, nsec = self.ros_timestamp()
        frame_bytes = self.frame_id.encode("utf-8")
        header = struct.pack(
            _HEADER_FMT,
            MAGIC,
            sec & 0xFFFFFFFF,
            nsec,
            len(frame_bytes),
        )
        count = np.uint32(self.points.shape[0]).tobytes()
        return header + frame_bytes + count + self.points.tobytes()

    @classmethod
    def lcm_decode(cls, data: bytes) -> RawLidarScan:
        if len(data) < _HEADER_LEN:
            raise ValueError(f"RawLidarScan: payload too short ({len(data)} bytes)")
        magic, sec_raw, nsec, frame_len = struct.unpack_from(_HEADER_FMT, data, 0)
        if magic != MAGIC:
            raise ValueError(f"RawLidarScan: bad magic {magic!r}, expected {MAGIC!r}")
        offset = _HEADER_LEN
        if len(data) < offset + frame_len + 4:
            raise ValueError("RawLidarScan: payload truncated in frame_id/point_count")
        frame_id = data[offset : offset + frame_len].decode("utf-8")
        offset += frame_len
        (point_count,) = struct.unpack_from("<I", data, offset)
        offset += 4
        expected_points_bytes = point_count * _POINT_DTYPE.itemsize
        if len(data) < offset + expected_points_bytes:
            raise ValueError(
                f"RawLidarScan: payload truncated, expected {expected_points_bytes} bytes of "
                f"points after offset {offset}, got {len(data) - offset}"
            )
        points = np.frombuffer(data, dtype=_POINT_DTYPE, count=point_count, offset=offset).copy()
        sec_signed = sec_raw if sec_raw < 0x80000000 else sec_raw - 0x100000000
        ts = sec_signed + nsec / 1_000_000_000
        return cls(points=points, frame_id=frame_id, ts=ts)

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __repr__(self) -> str:
        return (
            f"RawLidarScan(ts={self.ts:.6f}, frame_id={self.frame_id!r}, "
            f"points={self.points.shape[0]})"
        )
