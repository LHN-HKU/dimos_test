#!/usr/bin/env python3
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

"""Record a short live capture from Mid-360 + FastLio2 into a memory2 fixture.

Run from a host with the Mid-360 at 192.168.1.107 on enp2s0 (192.168.1.5/24).
Starts the two native binaries, subscribes to all four streams, writes a
SqliteStore at the requested path, and cleans up.

Streams written:
    raw_imu       sensor_msgs.Imu          (input to FastLio2)
    raw_lidar     sensor_msgs.RawLidarScan (input to FastLio2)
    odometry      nav_msgs.Odometry        (FastLio2 output)
    world_cloud   sensor_msgs.PointCloud2  (FastLio2 output)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import select
import subprocess
import sys
import threading
import time
from typing import Any

import lcm as lcmlib

REPO_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO_ROOT))

from dimos.memory2.store.sqlite import SqliteStore
from dimos.memory2.stream import Stream
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.sensor_msgs.RawLidarScan import RawLidarScan

LIVOX_DIR = REPO_ROOT / "dimos/hardware/sensors/lidar/livox/cpp"
FASTLIO_DIR = REPO_ROOT / "dimos/hardware/sensors/lidar/fastlio2/cpp"

# Wire-format topic strings (the C++ binaries take these verbatim as --<arg>).
TOPICS = {
    "raw_imu": "/imu#sensor_msgs.Imu",
    "raw_lidar": "/raw_lidar#sensor_msgs.RawLidarScan",
    "odometry": "/odom#nav_msgs.Odometry",
    "world_cloud": "/world_cloud#sensor_msgs.PointCloud2",
}

PAYLOAD_TYPES: dict[str, type[Any]] = {
    "raw_imu": Imu,
    "raw_lidar": RawLidarScan,
    "odometry": Odometry,
    "world_cloud": PointCloud2,
}


class StreamRecorder:
    def __init__(self, store: SqliteStore) -> None:
        self.streams: dict[str, Stream[Any, Any]] = {
            name: store.stream(name, PAYLOAD_TYPES[name]) for name in TOPICS
        }
        self.counts = {name: 0 for name in TOPICS}
        self._lock = threading.Lock()

    def _on(self, name: str, data: bytes) -> None:
        try:
            payload = PAYLOAD_TYPES[name].lcm_decode(data)
        except Exception as exc:
            print(f"[record] {name}: decode failed ({len(data)} bytes): {exc}", flush=True)
            return
        ts = getattr(payload, "ts", None) or time.time()
        try:
            with self._lock:
                self.streams[name].append(payload, ts=ts)
                self.counts[name] += 1
        except Exception as exc:
            print(f"[record] {name}: append failed: {exc}", flush=True)


def start_proc(cwd: Path, args: list[str]) -> subprocess.Popen[bytes]:
    print(f"[record] launching: {' '.join(args)}", flush=True)
    return subprocess.Popen(
        args, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
    )


def _wait_for_stdin(bus: lcmlib.LCM) -> None:
    """Pump the LCM bus until a line shows up on stdin (so messages keep
    flowing while we wait at a prompt)."""
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0.05)
        if ready:
            sys.stdin.readline()
            return
        bus.handle_timeout(50)


def stop_proc(proc: subprocess.Popen[bytes], timeout: float = 3.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "-o",
        default="tests/data/fastlio2_replay.db",
        help="memory2 SqliteStore path (default: %(default)s)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="recording length in seconds. If omitted, interactively prompts to start and to stop.",
    )
    parser.add_argument(
        "--lidar-ip", default="192.168.1.107", help="Mid-360 IP (default: %(default)s)"
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=2.0,
        help="seconds to let the pipeline stabilize before the start prompt",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"[record] removing existing fixture at {output_path}")
        output_path.unlink()

    mid360 = start_proc(
        LIVOX_DIR,
        [
            str(LIVOX_DIR / "result/bin/mid360_native"),
            "--raw_lidar",
            TOPICS["raw_lidar"],
            "--imu",
            TOPICS["raw_imu"],
            "--frame_id",
            "lidar",
            "--imu_frame_id",
            "imu",
            "--lidar_ip",
            args.lidar_ip,
        ],
    )
    fastlio = start_proc(
        FASTLIO_DIR,
        [
            str(FASTLIO_DIR / "result/bin/fastlio2_native"),
            "--raw_imu",
            TOPICS["raw_imu"],
            "--raw_lidar",
            TOPICS["raw_lidar"],
            "--lidar",
            TOPICS["world_cloud"],
            "--odometry",
            TOPICS["odometry"],
            "--config_path",
            "config/mid360.yaml",
            "--frame_id",
            "odom",
            "--child_frame_id",
            "base_link",
        ],
    )

    store = SqliteStore(path=str(output_path))
    recorder = StreamRecorder(store)
    bus = lcmlib.LCM()
    subs = []
    for name, topic in TOPICS.items():

        def _cb(_chan: str, data: bytes, n: str = name) -> None:
            recorder._on(n, data)

        subs.append(bus.subscribe(topic, _cb))

    print(
        f"[record] warming up for {args.warmup}s (FastLio IMU init, no data saved yet) …",
        flush=True,
    )
    warmup_end = time.time() + args.warmup
    while time.time() < warmup_end:
        bus.handle_timeout(50)

    if args.duration is None:
        print("[record] READY.  Press Enter to START recording.", flush=True)
        _wait_for_stdin(bus)
        recorder.counts = {name: 0 for name in TOPICS}
        capture_start = time.time()
        print("[record] RECORDING.  Press Enter (or Ctrl+C) to STOP.", flush=True)
        try:
            _wait_for_stdin(bus)
        except KeyboardInterrupt:
            print("\n[record] caught Ctrl+C, stopping…", flush=True)
        capture_duration = time.time() - capture_start
    else:
        recorder.counts = {name: 0 for name in TOPICS}
        capture_start = time.time()
        capture_end = capture_start + args.duration
        print(f"[record] RECORDING for {args.duration:.1f}s …", flush=True)
        while time.time() < capture_end:
            bus.handle_timeout(50)
        capture_duration = args.duration
    print("[record] capture done", flush=True)

    for s in subs:
        bus.unsubscribe(s)
    stop_proc(fastlio)
    stop_proc(mid360)
    store.stop()

    print(f"\n[record] message counts (over {capture_duration:.2f}s):")
    for name, n in recorder.counts.items():
        rate = n / capture_duration if capture_duration > 0 else 0.0
        print(f"  {name:14s} {n:6d}  ({rate:6.1f} Hz)")
    print(
        f"\n[record] fixture written to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)"
    )

    bad = [name for name, n in recorder.counts.items() if n == 0]
    if bad:
        print(f"[record] WARNING: no messages on: {', '.join(bad)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
