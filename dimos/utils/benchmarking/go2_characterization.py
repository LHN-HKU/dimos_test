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

"""Tool 1 of the Go2 tuning deliverable: characterization.

Runs a space-cheap system-ID sequence (per-channel velocity steps — no
long paths), fits FOPDT per axis (vx, vy, wz), then runs the DERIVE step
and emits the versioned config artifact.

    uv run python -m dimos.utils.benchmarking.go2_characterization \\
        --mode sim --surface mujoco

**One harness, plant swapped by ``--mode``** — exactly the same SI loop
and fitter; only *where the robot behaves* differs:

* ``sim``: the plant is the in-process FOPDT ``Go2PlantSim`` seeded with
  the vendored ``GO2_PLANT_FITTED`` ground truth. A healthy run recovers
  the injected model (printed "recovered vs injected" table) — this
  self-tests the whole measure->fit->derive pipeline without a robot.
* ``hw``: the plant is the real Go2 over LCM (publish ``/cmd_vel``,
  differentiate ``/go2/odom`` to body-frame velocity). Wired; not
  exercised by CI/sim.

Both modes record cmd-vs-measured per channel and fit with the same
``fit_fopdt``; there is no separate hardware data-acquisition pipeline.
The tail (sections 1-4 + 6; section 5 left ``None`` for the benchmark
tool) is the pure ``derive_config``.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import time

import numpy as np

from dimos.utils.benchmarking.go2_tuning import Provenance, derive_config, git_sha
from dimos.utils.benchmarking.plant import (
    GO2_PLANT_FITTED,
    FopdtChannelParams,
    Go2PlantParams,
    Go2PlantSim,
)
from dimos.utils.characterization.modeling.fopdt import fit_fopdt

# Space-cheap SI plan: a few amplitudes per channel. vy is a real channel
# (the Go2 base strafes) so it gets its own sweep — not a copy of vx.
_SI_AMPLITUDES: dict[str, list[float]] = {
    "vx": [0.3, 0.6, 0.9],
    "vy": [0.2, 0.4],
    "wz": [0.4, 0.8, 1.2],
}
_DT = 0.02  # 50 Hz sample period
_PRE_ROLL_S = 1.0
_STEP_S = 5.0  # >> max (tau + L) so the channel fully settles

_CHANNELS = ("vx", "vy", "wz")
REPORTS_DIR = Path(__file__).parent / "reports"


# --- the swap point: a plant you step with a velocity command ------------


class _SimPlant:
    """Sim: the in-process FOPDT ``Go2PlantSim`` (vendored ground truth)."""

    is_hw = False

    def __init__(self) -> None:
        self._p = Go2PlantSim(GO2_PLANT_FITTED)

    def reset(self) -> None:
        self._p.reset(0.0, 0.0, 0.0, _DT)

    def step(self, cmd: dict[str, float]) -> dict[str, float]:
        self._p.step(cmd["vx"], cmd["vy"], cmd["wz"], _DT)
        return {"vx": self._p.vx, "vy": self._p.vy, "wz": self._p.wz}


class _HwPlant:
    """HW: the real Go2 over LCM — publish ``/cmd_vel``, differentiate
    ``/go2/odom`` to body-frame velocity. Wired; not run by CI/sim.

    Same ``reset``/``step`` surface as :class:`_SimPlant`, so the SI loop
    is identical regardless of mode.
    """

    is_hw = True

    def __init__(self) -> None:
        import math

        from dimos.core.transport import LCMTransport
        from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
        from dimos.msgs.geometry_msgs.Twist import Twist
        from dimos.msgs.geometry_msgs.Vector3 import Vector3

        self._math = math
        self._Twist, self._Vector3 = Twist, Vector3
        self._cmd_pub = LCMTransport("/cmd_vel", Twist)
        self._odom_sub = LCMTransport("/go2/odom", PoseStamped)
        self._pose = None
        self._odom_sub.subscribe(self._on_odom)
        self._prev = None

    def _on_odom(self, msg) -> None:
        self._pose = msg

    def reset(self) -> None:
        self._prev = None

    def step(self, cmd: dict[str, float]) -> dict[str, float]:
        m = self._math
        self._cmd_pub.broadcast(
            None,
            self._Twist(
                linear=self._Vector3(cmd["vx"], cmd["vy"], 0.0),
                angular=self._Vector3(0.0, 0.0, cmd["wz"]),
            ),
        )
        time.sleep(_DT)
        p = self._pose
        if p is None:
            return {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        yaw = p.orientation.euler[2]
        cur = (p.position.x, p.position.y, yaw, time.perf_counter())
        if self._prev is None:
            self._prev = cur
            return {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        dx, dy = cur[0] - self._prev[0], cur[1] - self._prev[1]
        dyaw = (cur[2] - self._prev[2] + m.pi) % (2 * m.pi) - m.pi
        dt = max(cur[3] - self._prev[3], 1e-3)
        self._prev = cur
        c, s = m.cos(yaw), m.sin(yaw)
        return {
            "vx": (dx * c + dy * s) / dt,
            "vy": (-dx * s + dy * c) / dt,
            "wz": dyaw / dt,
        }


# --- the one SI loop (mode-independent) ----------------------------------


def _run_si(plant) -> tuple[Go2PlantParams, dict]:
    """Step every channel/amplitude through ``plant``, fit FOPDT per
    channel, pool. Identical for sim and hw — only ``plant`` differs."""
    n_pre = int(_PRE_ROLL_S / _DT)
    n_step = int(_STEP_S / _DT)
    pooled: dict[str, FopdtChannelParams] = {}
    per_amplitude: dict[str, list[dict]] = {}

    for channel in _CHANNELS:
        fits = []
        per_amplitude[channel] = []
        for amp in _SI_AMPLITUDES[channel]:
            plant.reset()
            cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
            for _ in range(n_pre):
                plant.step(cmd)
            cmd[channel] = amp
            ys = [plant.step(cmd)[channel] for _ in range(n_step)]
            t = np.arange(len(ys), dtype=float) * _DT  # rel. to step edge
            fp = fit_fopdt(t, np.asarray(ys, dtype=float), u_step=amp)
            if not fp.converged or not np.isfinite([fp.K, fp.tau, fp.L]).all():
                print(f"  [warn] {channel}@{amp}: fit failed ({fp.reason})")
                continue
            fits.append(fp)
            per_amplitude[channel].append(
                {"amplitude": amp, "direction": "forward", "K": fp.K, "tau": fp.tau, "L": fp.L}
            )
        if not fits:
            raise RuntimeError(f"SI: no converged fits for channel {channel!r}")
        pooled[channel] = FopdtChannelParams(
            K=float(np.mean([f.K for f in fits])),
            tau=float(np.mean([f.tau for f in fits])),
            L=float(np.mean([f.L for f in fits])),
        )

    fitted = Go2PlantParams(vx=pooled["vx"], vy=pooled["vy"], wz=pooled["wz"])
    if not plant.is_hw:
        _print_selftest(fitted)
    return fitted, per_amplitude


def _print_selftest(fitted: Go2PlantParams) -> None:
    """sim only: recovered vs injected ground truth — should match."""
    print("\nSI self-test (recovered vs injected FOPDT ground truth):")
    print(f"  {'chan':4} {'K fit/true':>20} {'tau fit/true':>20} {'L fit/true':>20}")
    for ch in _CHANNELS:
        f, g = getattr(fitted, ch), getattr(GO2_PLANT_FITTED, ch)
        print(
            f"  {ch:4} {f.K:8.3f}/{g.K:<8.3f}   {f.tau:8.3f}/{g.tau:<8.3f}   {f.L:8.3f}/{g.L:<8.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Go2 characterization -> tuning config artifact")
    ap.add_argument("--mode", choices=["sim", "hw"], default="sim")
    ap.add_argument("--out", default=str(REPORTS_DIR), help="output dir for the artifact")
    ap.add_argument("--robot-id", default="go2")
    ap.add_argument("--surface", default="mujoco")
    ap.add_argument("--gait-mode", default="default", help="locomotion gait mode")
    args = ap.parse_args()

    plant = _SimPlant() if args.mode == "sim" else _HwPlant()
    fitted, per_amplitude = _run_si(plant)

    provenance = Provenance(
        robot_id=args.robot_id,
        surface=args.surface,
        mode=args.gait_mode,
        date=date.today().isoformat(),
        git_sha=git_sha(),
        sim_or_hw=args.mode,
        characterization_session_dir="(in-process SI)" if args.mode == "sim" else "(hw LCM SI)",
    )
    cfg = derive_config(fitted, provenance, per_amplitude=per_amplitude)

    out_path = (
        Path(args.out).expanduser()
        / f"go2_config_{args.mode}_{args.surface}_{provenance.date}_{provenance.git_sha}.json"
    )
    cfg.to_json(out_path)
    print("\nWrote config artifact (sections 1-4,6; section 5 pending benchmark):")
    print(out_path.resolve())


if __name__ == "__main__":
    main()
