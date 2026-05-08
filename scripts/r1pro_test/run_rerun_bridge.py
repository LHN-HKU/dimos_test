"""Launch RerunBridgeModule with the R1 Pro viewer layout.

Default behavior: pick a fresh ephemeral port, launch the `rerun` CLI binary
as a managed subprocess on that port, then connect the bridge's recording
stream to it via gRPC. Single command, no flags, native window opens.

WHY a fresh port every run:
    VS Code extensions on the host (urdf-visualizer, rde-ros-2, …) cache TCP
    ports they have ever observed Rerun bind, and re-bind those ports on
    127.0.0.1 on subsequent VS Code launches — even after a window reload.
    When `rr.spawn(port=9876)` runs and finds 9876 taken, it silently assumes
    a viewer is already there and pushes data into a black hole; no window
    opens, no error. Picking a never-before-used port from the OS at runtime
    sidesteps the cache entirely.

Usage (inside the dev container, with `dimos run r1pro-full` running):

    python scripts/r1pro_test/run_rerun_bridge.py
    python scripts/r1pro_test/run_rerun_bridge.py --mode web
    python scripts/r1pro_test/run_rerun_bridge.py --mode connect --connect-url URL
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

import typer

from dimos.protocol.pubsub.impl.lcmpubsub import LCM
from dimos.protocol.service.lcmservice import autoconf
from dimos.robot.humanoids.r1pro.blueprints import r1pro_rerun_blueprint
from dimos.visualization.rerun.bridge import RerunBridgeModule


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, proc: subprocess.Popen, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"rerun viewer subprocess exited with code {proc.returncode} "
                f"before opening port {port}"
            )
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"rerun viewer did not open {host}:{port} within {timeout:.1f}s")


def _spawn_viewer(port: int, memory_limit: str) -> subprocess.Popen:
    rerun_bin = shutil.which("rerun")
    if rerun_bin is None:
        raise RuntimeError("`rerun` not on PATH; expected /app/.venv/bin/rerun")
    return subprocess.Popen(
        [
            rerun_bin,
            "--port", str(port),
            "--memory-limit", memory_limit,
            "--hide-welcome-screen",
        ],
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
    )


def main(
    mode: str = typer.Option(
        "native",
        help="native (managed window, default), web (browser), connect (existing viewer)",
    ),
    memory_limit: str = typer.Option("25%", help="Viewer memory cap, e.g. '4GB' or '25%'"),
    connect_url: str = typer.Option(
        "rerun+http://127.0.0.1:9876/proxy",
        help="Only used in --mode connect.",
    ),
    web_port: int = typer.Option(9090, help="Browser port for --mode web."),
) -> None:
    import rerun as rr

    autoconf(check_only=True)

    bridge_mode = "none" if mode in ("native", "web") else mode

    bridge = RerunBridgeModule(
        viewer_mode=bridge_mode,
        memory_limit=memory_limit,
        blueprint=r1pro_rerun_blueprint,
        pubsubs=[LCM()],
        connect_url=connect_url,
    )
    bridge.start()  # rr.init("dimos") happens here; must precede connect_grpc.

    viewer_proc: subprocess.Popen | None = None

    if mode == "native":
        port = _free_port()
        viewer_proc = _spawn_viewer(port, memory_limit)
        try:
            _wait_for_port("127.0.0.1", port, viewer_proc, timeout=10.0)
        except (TimeoutError, RuntimeError) as exc:
            viewer_proc.terminate()
            raise SystemExit(f"Failed to launch rerun viewer: {exc}") from exc
        rr.connect_grpc(f"rerun+http://127.0.0.1:{port}/proxy")
        print(f"\n  Rerun viewer running on 127.0.0.1:{port} (X11/XWayland window)\n")
    elif mode == "web":
        grpc_port = _free_port()
        uri = rr.serve_grpc(grpc_port=grpc_port)
        rr.serve_web_viewer(connect_to=uri, web_port=web_port, open_browser=False)
        print(f"\n  Open in laptop browser: http://localhost:{web_port}\n")
    elif mode == "connect":
        # bridge.start() already routed via connect_url for connect mode.
        print(f"\n  Pushing data to existing viewer at {connect_url}\n")
    else:
        raise typer.BadParameter(f"unknown --mode {mode!r}")

    stopping = threading.Event()

    def _handler(*_: object) -> None:
        if stopping.is_set():
            sys.stderr.write("\nForce exit.\n")
            sys.stderr.flush()
            os._exit(1)
        stopping.set()
        sys.stderr.write("\nShutting down (Ctrl-C again to force)…\n")
        sys.stderr.flush()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    try:
        while not stopping.wait(timeout=1.0):
            if viewer_proc is not None and viewer_proc.poll() is not None:
                sys.stderr.write(
                    f"\nrerun viewer exited (code {viewer_proc.returncode}); shutting down.\n"
                )
                break
    finally:
        bridge.stop()
        if viewer_proc is not None and viewer_proc.poll() is None:
            viewer_proc.terminate()
            try:
                viewer_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                viewer_proc.kill()
                viewer_proc.wait(timeout=2.0)


if __name__ == "__main__":
    typer.run(main)
