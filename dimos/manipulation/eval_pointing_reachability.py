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

"""Pointing reachability eval — sweep azimuth x elevation around a robot.

Drives ``ManipulationModule.point_at`` over a grid of (azimuth, elevation)
directions and records, for each cell, whether the named arm could
produce a plan + execute it.  The arm is sent home before every cell so
each trial is independent of the last; cells where ``go_home`` left the
arm wedged at joint limits are flagged ``WEDGED`` and skipped (so we
don't conflate IK infeasibility with stale state).

The eval is online — it talks to a running ``ManipulationModule`` (and
``ControlCoordinator``) over LCM RPC, and uses ``/odom`` for the live
pelvis pose so the same script works whether the robot is stationary,
walking around, or in a different facing.

Outputs a JSON file with per-cell results; render with
``visualize_pointing_reachability.py``.

Typical session::

    # Terminal 1: bring the sim up (must expose point_at and go_home)
    dimos --simulation mujoco run g1-groot-wbc

    # Terminal 2: run the eval per arm
    uv run python -m dimos.manipulation.eval_pointing_reachability \\
        --arm left_arm --out /tmp/reach_left.json
    uv run python -m dimos.manipulation.eval_pointing_reachability \\
        --arm right_arm --out /tmp/reach_right.json

    # Terminal 2 (cont.): render the dual-arm comparison
    uv run python -m dimos.manipulation.visualize_pointing_reachability \\
        --left /tmp/reach_left.json --right /tmp/reach_right.json \\
        --out /tmp/dual_arm_heatmap.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Any

import lcm

from dimos.core.rpc_client import RPCClient
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

# LCM publishes geometry_msgs/PoseStamped under "<topic>#<msg_type>".
ODOM_CHANNEL = "/odom#geometry_msgs.PoseStamped"

# Coarse default grid — kept tight so the eval finishes in ~5–10 min.
# Elevation range is the "useful pointing" band: -30°..+30°.  The full
# +/- 90° dome isn't worth exercising — the arm physically can't aim
# straight up or straight down with current orientation strategies.
_DEFAULT_AZIMUTHS_DEG = list(range(-180, 180, 30))  # 12 cells, every 30°
_DEFAULT_ELEVATIONS_DEG = [-30, -15, 0, 15, 30]  # 5 cells


def _get_pelvis(timeout_s: float = 3.0) -> dict | None:
    """Snapshot one ``/odom`` PoseStamped synchronously."""
    got: list[PoseStamped | None] = [None]

    def on(_channel: str, data: bytes) -> None:
        got[0] = PoseStamped.lcm_decode(data)

    bus = lcm.LCM()
    bus.subscribe(ODOM_CHANNEL, on)
    deadline = time.time() + timeout_s
    while got[0] is None and time.time() < deadline:
        bus.handle_timeout(100)
    if got[0] is None:
        return None
    msg = got[0]
    return {
        "pos": (float(msg.position.x), float(msg.position.y), float(msg.position.z)),
        "quat": (
            float(msg.orientation.x),
            float(msg.orientation.y),
            float(msg.orientation.z),
            float(msg.orientation.w),
        ),
    }


def _yaw_deg(q_xyzw: tuple[float, float, float, float]) -> float:
    x, y, z, w = q_xyzw
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny, cosy))


def _resolve_manip_module(module_name: str | None) -> Any:
    """Resolve a manipulation-module class for the RPC client.

    By default we look up ``G1ManipulationModule`` (the only sim subclass
    with point_at right now).  Allow ``--module`` to override for other
    robots that subclass ``ManipulationModule``.
    """
    if module_name is None or module_name == "g1":
        from dimos.robot.unitree.g1.g1_manipulation import G1ManipulationModule

        return G1ManipulationModule
    if "." in module_name:
        mod_path, _, cls_name = module_name.rpartition(".")
        import importlib

        return getattr(importlib.import_module(mod_path), cls_name)
    raise ValueError(f"unknown --module {module_name!r}; pass 'g1' or a full 'pkg.mod.ClassName'")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else "",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--module",
        default="g1",
        help="manipulation module class — 'g1' or a full 'pkg.mod.ClassName'",
    )
    ap.add_argument(
        "--arm", default="left_arm", help="robot_name registered in the manipulation module"
    )
    ap.add_argument(
        "--distance", type=float, default=1.5, help="meters from pelvis to the world-frame target"
    )
    ap.add_argument(
        "--reach", type=float, default=0.45, help="point_at extension (meters from shoulder anchor)"
    )
    ap.add_argument(
        "--azimuths",
        type=str,
        default=None,
        help=(
            "comma-separated azimuth degrees relative to robot facing "
            f"(default: {','.join(str(a) for a in _DEFAULT_AZIMUTHS_DEG)})"
        ),
    )
    ap.add_argument(
        "--elevations",
        type=str,
        default=None,
        help=(
            "comma-separated elevation degrees "
            f"(default: {','.join(str(a) for a in _DEFAULT_ELEVATIONS_DEG)})"
        ),
    )
    ap.add_argument(
        "--out", default="/tmp/eval_pointing_reachability.json", help="JSON output path"
    )
    ap.add_argument(
        "--wedge-threshold",
        type=float,
        default=0.5,
        help="max |joint angle| (rad) considered 'home' before a trial",
    )
    args = ap.parse_args()

    azimuths = (
        [int(s) for s in args.azimuths.split(",")] if args.azimuths else _DEFAULT_AZIMUTHS_DEG
    )
    elevations = (
        [int(s) for s in args.elevations.split(",")] if args.elevations else _DEFAULT_ELEVATIONS_DEG
    )

    manip_cls = _resolve_manip_module(args.module)
    client = RPCClient(None, manip_cls)

    pelvis = _get_pelvis()
    if not pelvis:
        print(f"FAIL: no pose on {ODOM_CHANNEL} after 3s — is the sim running?", file=sys.stderr)
        return 1
    pelvis_yaw = math.radians(_yaw_deg(pelvis["quat"]))
    print(f"pelvis pos={pelvis['pos']}  yaw={math.degrees(pelvis_yaw):.1f}°")
    print(f"distance={args.distance}m  reach={args.reach}m  arm={args.arm}")
    print(f"grid={len(azimuths)}az x {len(elevations)}el = {len(azimuths) * len(elevations)} cells")
    print()

    client.reset()
    rows: list[list[dict]] = []
    t_start = time.time()
    cell_idx = 0
    total = len(azimuths) * len(elevations)

    for el_deg in elevations:
        row: list[dict] = []
        el_rad = math.radians(el_deg)
        for az_deg in azimuths:
            cell_idx += 1
            try:
                client.go_home(robot_name=args.arm)
            except Exception:
                pass
            client.reset()
            joints = client.get_current_joints(robot_name=args.arm) or []
            wedged = bool(joints) and max(abs(j) for j in joints) > args.wedge_threshold

            az_world = pelvis_yaw + math.radians(az_deg)
            dx = args.distance * math.cos(el_rad) * math.cos(az_world)
            dy = args.distance * math.cos(el_rad) * math.sin(az_world)
            dz = args.distance * math.sin(el_rad)
            target = (
                pelvis["pos"][0] + dx,
                pelvis["pos"][1] + dy,
                pelvis["pos"][2] + dz,
            )

            if wedged:
                cell = {
                    "az": az_deg,
                    "el": el_deg,
                    "ok": False,
                    "mode": "-",
                    "elapsed": 0.0,
                    "recover": "n/a",
                    "result": f"WEDGED (max|q|={max(abs(j) for j in joints):.2f})",
                    "wedged_before": True,
                    "joints_before": joints,
                }
                row.append(cell)
                print(f"  [{cell_idx:>3}/{total}] az={az_deg:+4d}° el={el_deg:+3d}° → W (wedged)")
                continue

            t0 = time.time()
            try:
                pt = PointStamped(
                    x=float(target[0]),
                    y=float(target[1]),
                    z=float(target[2]),
                    frame_id="map",
                )
                result = client.point_at(target=pt, robot_name=args.arm)
            except Exception as exc:
                result = f"Error: {exc!r}"
            elapsed = time.time() - t0
            ok = "Pointing at" in result
            mode = (
                "look-at"
                if "(look-at)" in result
                else "preserve-orient"
                if "(preserve-orient)" in result
                else "soft-pointing"
                if "(soft-pointing)" in result
                else "-"
            )

            recover = "n/a"
            if ok:
                try:
                    rh = client.go_home(robot_name=args.arm)
                    recover = "ok" if "Reached" in rh else "FAIL"
                except Exception:
                    recover = "FAIL"
                client.reset()

            cell = {
                "az": az_deg,
                "el": el_deg,
                "ok": ok,
                "mode": mode,
                "elapsed": elapsed,
                "recover": recover,
                "result": result,
                "wedged_before": False,
            }
            row.append(cell)
            sym = (
                "L"
                if (ok and mode == "look-at")
                else "S"
                if (ok and mode == "soft-pointing")
                else "P"
                if (ok and mode == "preserve-orient")
                else "."
            )
            print(
                f"  [{cell_idx:>3}/{total}] az={az_deg:+4d}° el={el_deg:+3d}° → "
                f"{sym} ({elapsed:.1f}s) recover={recover}"
            )
        rows.append(row)

    # ---- ASCII heatmap (quick summary in the terminal) ----
    print("\n" + "=" * 60)
    print(f"REACHABILITY MAP — {time.time() - t_start:.0f}s wall ({args.arm})")
    print("=" * 60)
    print("L look-at  S soft-pointing  P preserve-orient  . IK fail  W skipped (wedged)\n")

    print("       " + " ".join(f"{a:>+4d}" for a in azimuths) + "  azimuth (deg, 0=facing)")
    print("       " + " ".join(["----"] * len(azimuths)))
    for row, el_deg in zip(rows, elevations, strict=False):
        cells: list[str] = []
        for c in row:
            if c.get("wedged_before"):
                cells.append("  W ")
            elif c["ok"]:
                cells.append(
                    "  L "
                    if c["mode"] == "look-at"
                    else "  S "
                    if c["mode"] == "soft-pointing"
                    else "  P "
                )
            else:
                cells.append("  . ")
        print(f"el={el_deg:+3d}° " + " ".join(cells))

    flat = [c for row in rows for c in row]
    n_wedged = sum(c.get("wedged_before") for c in flat)
    n_tested = len(flat) - n_wedged
    n_ok = sum(c["ok"] for c in flat)
    n_recover_fail = sum(c["ok"] and c["recover"] == "FAIL" for c in flat)
    print(
        f"\nTOTAL reachable: {n_ok}/{n_tested} ({100 * n_ok / max(1, n_tested):.0f}% of tested) "
        f"  [skipped {n_wedged} wedged]"
    )
    print(f"  wedged after success (go_home failed): {n_recover_fail}/{n_ok}")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)
    print(f"\nWrote {args.out}")
    print(
        "Render with: uv run python -m dimos.manipulation.visualize_pointing_reachability "
        f"--left {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
