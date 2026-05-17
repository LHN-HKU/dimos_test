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

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
from reactivex.disposable import Disposable
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect
import uvicorn

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.experimental.pimsim.browser import HTML
from dimos.experimental.pimsim.config import (
    BabylonSceneViewerConfig,
    CoordinatorControlSpec,
    HumanoidControlSpec,
    MujocoRespawnSpec,
)
from dimos.experimental.pimsim.geometry import (
    WS_MSG_CAMERA,
    canonical_joint_name,
    compose_scene_mesh_wxyz,
    dimos_joint_to_mjcf,
    media_type,
    path_contains,
)
from dimos.experimental.pimsim.kinematic import KinematicBaseSim
from dimos.experimental.pimsim.lidar import (
    MujocoRaycastScene,
    RaycastScene,
    SyntheticLidar,
    SyntheticLidarConfig,
)
from dimos.experimental.pimsim.robot_meshes import (
    RobotMeshes,
    apply_robot_state,
    load_robot_meshes,
)
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.nav_msgs.Path import Path as PathMsg
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.visualization_msgs.EntityMarkers import EntityMarkers, Marker
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class BabylonSceneViewerModule(Module):
    config: BabylonSceneViewerConfig
    joint_state: In[JointState]
    odom: In[PoseStamped]
    path: In[PathMsg]
    pointcloud_overlay: In[PointCloud2]
    camera_image: In[Image]
    clicked_point: Out[PointStamped]
    point_goal: Out[PointStamped]
    labeled_point: Out[EntityMarkers]
    cmd_vel: In[Twist]
    odometry: Out[Odometry]
    registered_scan: Out[PointCloud2]
    _mujoco_sim: MujocoRespawnSpec | None = None
    _robot_ctrl: HumanoidControlSpec | None = None
    _coordinator_ctrl: CoordinatorControlSpec | None = None

    def __init__(
        self,
        *,
        assets: dict[str, bytes] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        cfg = self.config
        if not cfg.mjcf_path:
            raise ValueError("BabylonSceneViewerConfig.mjcf_path must be set")
        self._mjcf_path = Path(cfg.mjcf_path)
        self._assets = assets
        # Resolved scene path — None when no scene is configured OR when
        # disable_scene is set (fast-load mode).
        self._scene_path: Path | None = (
            cfg.scene.mesh_path if (cfg.scene is not None and not cfg.disable_scene) else None
        )
        self._broadcast_dt = 1.0 / cfg.broadcast_hz
        self._pointcloud_min_dt = 1.0 / cfg.pointcloud_hz
        self._camera_min_dt = 1.0 / cfg.camera_hz
        self._sim_dt = 1.0 / cfg.sim_rate

        self._state_lock = threading.Lock()
        self._latest_joints: dict[str, float] = {}
        self._latest_base_pos: np.ndarray | None = None
        self._latest_base_wxyz: np.ndarray | None = None
        self._latest_path: list[list[float]] = []
        self._pointcloud_lock = threading.Lock()
        self._latest_pointcloud_payload: dict[str, Any] | None = None
        self._last_pointcloud_sent = 0.0

        self._kinematic = KinematicBaseSim(
            init_x=cfg.init_x,
            init_y=cfg.init_y,
            init_z=cfg.init_z,
            init_yaw=cfg.init_yaw,
            vehicle_height=cfg.vehicle_height,
            sim_rate=cfg.sim_rate,
            lock_z=cfg.lock_z,
        )
        self._sim_thread: threading.Thread | None = None

        if cfg.enable_sim:
            self._latest_base_pos, self._latest_base_wxyz = self._kinematic.snapshot().base_arrays()

        # _turbo_jpeg is lazy-initialised so the viewer still imports cleanly
        # on machines without PyTurboJPEG (it's an optional dep).
        self._camera_lock = threading.Lock()
        self._last_camera_sent = 0.0
        self._turbo_jpeg: Any = None

        self._robot: RobotMeshes | None = None
        self._raycast_scene: RaycastScene | None = None
        self._uvicorn_server: uvicorn.Server | None = None
        self._server_thread: threading.Thread | None = None
        self._broadcast_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._server_loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[WebSocket] = set()

        self._lidar_min_dt = 1.0 / cfg.lidar_hz if cfg.lidar_hz > 0 else float("inf")
        self._last_lidar_sent = 0.0
        self._lidar = SyntheticLidar(
            SyntheticLidarConfig(
                n_azimuth=cfg.lidar_n_azimuth,
                n_elevation=cfg.lidar_n_elevation,
                elevation_min_deg=cfg.lidar_elevation_min_deg,
                elevation_max_deg=cfg.lidar_elevation_max_deg,
                max_range=cfg.lidar_max_range,
            )
        )

    @rpc
    def start(self) -> None:
        super().start()

        self._robot = load_robot_meshes(self._mjcf_path, assets=self._assets)
        self._raycast_scene = MujocoRaycastScene(self._robot)
        app = self._create_app()
        config = uvicorn.Config(app, host="0.0.0.0", port=self.config.port, log_level="warning")
        self._uvicorn_server = uvicorn.Server(config)
        self._server_thread = threading.Thread(
            target=self._uvicorn_server.run,
            name="babylon-viewer-server",
            daemon=True,
        )
        self._server_thread.start()

        self.register_disposable(Disposable(self.joint_state.subscribe(self._on_joint_state)))
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self.register_disposable(Disposable(self.path.subscribe(self._on_path)))
        self.register_disposable(
            Disposable(self.pointcloud_overlay.subscribe(self._on_pointcloud_overlay))
        )
        self.register_disposable(Disposable(self.camera_image.subscribe(self._on_camera_image)))

        if self.config.enable_sim:
            self.register_disposable(Disposable(self.cmd_vel.subscribe(self._on_cmd_vel)))
            self._sim_thread = threading.Thread(
                target=self._sim_loop,
                name="babylon-viewer-sim",
                daemon=True,
            )
            self._sim_thread.start()

        self._broadcast_thread = threading.Thread(
            target=self._broadcast_loop,
            name="babylon-viewer-broadcast",
            daemon=True,
        )
        self._broadcast_thread.start()
        logger.info("Babylon scene viewer: http://localhost:%s/", self.config.port)

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        if self._broadcast_thread and self._broadcast_thread.is_alive():
            self._broadcast_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        super().stop()

    def _create_app(self) -> Starlette:
        @asynccontextmanager
        async def _lifespan(_app: Starlette) -> Any:
            self._server_loop = asyncio.get_running_loop()
            yield

        return Starlette(
            routes=[
                Route("/", self._index),
                Route("/config.json", self._config),
                Route("/robot.json", self._robot_json),
                Route("/arms.json", self._arms_json),
                Route("/assets/{asset_name:path}", self._asset),
                Route("/snapshot.jpg", self._snapshot_get, methods=["GET"]),
                Route("/snapshot.jpg", self._snapshot_post, methods=["POST"]),
                WebSocketRoute("/ws", self._websocket),
            ],
            lifespan=_lifespan,
        )

    async def _snapshot_post(self, request: Request) -> Response:
        body = await request.body()
        self._latest_snapshot = bytes(body)
        return Response(status_code=204)

    async def _snapshot_get(self, request: Request) -> Response:
        snap = getattr(self, "_latest_snapshot", b"")
        if not snap:
            return Response("no snapshot yet", status_code=404)
        return Response(snap, media_type="image/jpeg")

    async def _arms_json(self, request: Request) -> JSONResponse:
        """Joint-limit catalogue so the page can build sliders with real range."""
        if self._robot_ctrl is None:
            return JSONResponse({"joints": []})
        try:
            limits = self._robot_ctrl.arm_joint_limits()
        except Exception as exc:
            logger.warning("BabylonViewer: arm_joint_limits() failed: %s", exc)
            return JSONResponse({"joints": []})
        joints = [{"name": name, "min": float(lo), "max": float(hi)} for (name, lo, hi) in limits]
        return JSONResponse({"joints": joints})

    async def _index(self, request: Request) -> HTMLResponse:
        return HTMLResponse(HTML)

    async def _config(self, request: Request) -> JSONResponse:
        scene = self.config.scene
        scene_file = None
        scene_bytes = 0
        scale = 1.0
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
        wxyz = compose_scene_mesh_wxyz(y_up=True, rotation_zyx_deg=(0.0, 0.0, 0.0))
        if scene is not None and self._scene_path is not None and self._scene_path.exists():
            scene_file = f"scene{self._scene_path.suffix.lower()}"
            scene_bytes = self._scene_path.stat().st_size
            scale = scene.scale
            translation = scene.translation
            wxyz = compose_scene_mesh_wxyz(
                y_up=scene.y_up,
                rotation_zyx_deg=scene.rotation_zyx_deg,
            )
        return JSONResponse(
            {
                "sceneFile": scene_file,
                "sceneBytes": scene_bytes,
                "sceneScale": scale,
                "scenePosition": list(translation),
                "sceneWxyz": list(wxyz),
            }
        )

    async def _robot_json(self, request: Request) -> JSONResponse:
        robot = self._robot
        if robot is None:
            return JSONResponse({"bodyNames": [], "geoms": []})

        geoms: list[dict[str, Any]] = []
        for index, geom in enumerate(robot.geoms):
            geoms.append(
                {
                    "id": index,
                    "body": geom.body_name,
                    "vertices": geom.vertices.astype(np.float32).reshape(-1).tolist(),
                    "indices": geom.faces.astype(np.int32).reshape(-1).tolist(),
                    "position": geom.local_pos.astype(np.float32).tolist(),
                    "wxyz": geom.local_wxyz.astype(np.float32).tolist(),
                    "rgba": [float(value) for value in geom.rgba],
                }
            )
        return JSONResponse({"bodyNames": robot.body_names, "geoms": geoms})

    async def _asset(self, request: Request) -> Response:
        if self._scene_path is None or not self._scene_path.exists():
            return Response("scene asset not configured", status_code=404)

        asset_name = request.path_params["asset_name"]
        scene_asset_name = f"scene{self._scene_path.suffix.lower()}"
        if asset_name == scene_asset_name:
            return FileResponse(self._scene_path, media_type=media_type(self._scene_path))

        if self._scene_path.suffix.lower() == ".gltf":
            candidate = self._scene_path.parent / asset_name
            if path_contains(self._scene_path.parent, candidate) and candidate.exists():
                return FileResponse(candidate, media_type=media_type(candidate))

        return Response("asset not found", status_code=404)

    async def _websocket(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        try:
            await websocket.send_json(self._make_state_payload())
            with self._pointcloud_lock:
                pointcloud_payload = self._latest_pointcloud_payload
            if pointcloud_payload is not None:
                await websocket.send_json(pointcloud_payload)
            while True:
                message = await websocket.receive_json()
                self._handle_client_message(message)
        except WebSocketDisconnect:
            pass
        finally:
            self._clients.discard(websocket)

    def _handle_client_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "respawn":
            if self._mujoco_sim is not None:
                self._mujoco_sim.respawn()
            self._on_cmd_vel(Twist.zero())
            if self.config.enable_sim:
                self._reset_kinematic_state()
            return
        if message_type == "cmd_vel":
            twist = self._parse_twist(message)
            if twist is not None:
                self._on_cmd_vel(twist)
            return
        if message_type == "arm_joint":
            name = message.get("name")
            position = message.get("position")
            if (
                self._robot_ctrl is None
                or not isinstance(name, str)
                or not isinstance(position, (int, float))
            ):
                return
            self._robot_ctrl.set_arm_joint(name, float(position))
            return
        if message_type == "release_arms":
            if self._robot_ctrl is not None:
                self._robot_ctrl.release_arms()
            return
        if message_type == "set_activated":
            engaged = bool(message.get("engaged", False))
            if self._coordinator_ctrl is None:
                logger.warning("BabylonViewer: set_activated requested but no coordinator wired")
                return
            logger.info("BabylonViewer: set_activated=%s", engaged)
            self._coordinator_ctrl.set_activated(engaged=engaged)
            return
        if message_type == "set_dry_run":
            enabled = bool(message.get("enabled", False))
            if self._coordinator_ctrl is None:
                logger.warning("BabylonViewer: set_dry_run requested but no coordinator wired")
                return
            logger.info("BabylonViewer: set_dry_run=%s", enabled)
            self._coordinator_ctrl.set_dry_run(enabled=enabled)
            return
        if message_type == "labeled_point":
            self._publish_labeled_point(message)
            return
        if message_type == "fps":
            self._record_browser_fps(message.get("value"))
            return
        if message_type not in {"clicked_point", "point_goal"}:
            return
        point = message.get("point")
        if not isinstance(point, list) or len(point) != 3:
            return
        try:
            x, y, z = (float(value) for value in point)
        except (TypeError, ValueError):
            return
        stamped = PointStamped(x=x, y=y, z=z, frame_id="map")
        if message_type == "clicked_point":
            self.clicked_point.publish(stamped)
        else:
            self.point_goal.publish(stamped)

    def _record_browser_fps(self, value: Any) -> None:
        try:
            fps = float(value)
        except (TypeError, ValueError):
            return
        self._last_browser_fps = fps
        now = time.monotonic()
        # 1 line every ~5 s to keep the log readable.
        if now - getattr(self, "_last_fps_log", 0.0) > 5.0:
            logger.info("BabylonViewer: browser FPS=%.0f", fps)
            self._last_fps_log = now

    def _publish_labeled_point(self, message: dict[str, Any]) -> None:
        point = message.get("point")
        label = message.get("label")
        if not isinstance(point, list) or len(point) != 3 or not isinstance(label, str):
            return
        label = label.strip()
        if not label:
            return
        try:
            x, y, z = (float(value) for value in point)
        except (TypeError, ValueError):
            return
        entity_id = message.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            entity_id = f"wp_{int(time.time() * 1000)}"
        entity_type = message.get("entity_type")
        if not isinstance(entity_type, str) or not entity_type:
            entity_type = "location"
        marker = Marker(entity_id=entity_id, label=label, entity_type=entity_type, x=x, y=y, z=z)
        # repr() so the user can paste a working Python expression back into a REPL.
        logger.info("BabylonViewer: labeled_point %s", repr(marker))
        self.labeled_point.publish(EntityMarkers(markers=[marker]))

    @staticmethod
    def _parse_twist(message: dict[str, Any]) -> Twist | None:
        linear = message.get("linear", [0.0, 0.0, 0.0])
        angular = message.get("angular", [0.0, 0.0, 0.0])
        if not isinstance(linear, list) or not isinstance(angular, list):
            return None
        if len(linear) != 3 or len(angular) != 3:
            return None
        try:
            return Twist(
                linear=Vector3(*(float(value) for value in linear)),
                angular=Vector3(*(float(value) for value in angular)),
            )
        except (TypeError, ValueError):
            return None

    def _broadcast_loop(self) -> None:
        while not self._stop_event.is_set():
            loop = self._server_loop
            if loop is not None and self._clients:
                payload = self._make_state_payload()
                asyncio.run_coroutine_threadsafe(self._broadcast(payload), loop)
            time.sleep(self._broadcast_dt)

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for websocket in tuple(self._clients):
            try:
                await websocket.send_json(payload)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self._clients.discard(websocket)

    def _broadcast_from_thread(self, payload: dict[str, Any]) -> None:
        loop = self._server_loop
        if loop is None or not self._clients:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), loop)

    def _make_state_payload(self) -> dict[str, Any]:
        robot = self._robot
        if robot is None:
            return {"type": "state", "time": time.time(), "bodies": [], "path": []}

        with self._state_lock:
            joints = dict(self._latest_joints)
            base_pos = None if self._latest_base_pos is None else self._latest_base_pos.copy()
            base_wxyz = None if self._latest_base_wxyz is None else self._latest_base_wxyz.copy()
            path_points = [point[:] for point in self._latest_path]

        apply_robot_state(robot, base_pos, base_wxyz, joints)

        bodies = []
        for body_id, body_name in enumerate(robot.body_names):
            bodies.append(
                {
                    "name": body_name,
                    "position": robot.data.xpos[body_id].astype(np.float32).tolist(),
                    "wxyz": robot.data.xquat[body_id].astype(np.float32).tolist(),
                }
            )

        joint_positions = {canonical_joint_name(k): float(v) for k, v in joints.items()}
        return {
            "type": "state",
            "time": time.time(),
            "bodies": bodies,
            "path": path_points,
            "joints": joint_positions,
        }

    def _on_joint_state(self, msg: JointState) -> None:
        with self._state_lock:
            self._latest_joints = {
                dimos_joint_to_mjcf(name): float(position)
                for name, position in zip(msg.name, msg.position, strict=False)
            }

    def _on_odom(self, msg: PoseStamped) -> None:
        if self.config.enable_sim:
            return
        with self._state_lock:
            self._latest_base_pos = np.array([msg.x, msg.y, msg.z], dtype=np.float64)
            self._latest_base_wxyz = np.array(
                [
                    msg.orientation.w,
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                ],
                dtype=np.float64,
            )

    def _on_cmd_vel(self, twist: Twist) -> None:
        self._kinematic.set_command(twist)

    def _reset_kinematic_state(self) -> None:
        snapshot = self._kinematic.reset()
        base_pos, base_wxyz = snapshot.base_arrays()
        with self._state_lock:
            self._latest_base_pos = base_pos
            self._latest_base_wxyz = base_wxyz

    def _sim_step(self, dt: float) -> None:
        snapshot = self._kinematic.step(dt)
        self.odometry.publish(snapshot.to_odometry())
        base_pos, base_wxyz = snapshot.base_arrays()
        with self._state_lock:
            self._latest_base_pos = base_pos
            self._latest_base_wxyz = base_wxyz

    def _sim_loop(self) -> None:
        dt = self._sim_dt
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                self._sim_step(dt)
                if self.config.lidar_hz > 0 and t0 - self._last_lidar_sent >= self._lidar_min_dt:
                    self._publish_lidar_scan()
                    self._last_lidar_sent = t0
            except Exception as exc:
                logger.warning("BabylonViewer: sim step failed: %s", exc)
            sleep_for = dt - (time.monotonic() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _publish_lidar_scan(self) -> None:
        scene = self._raycast_scene
        if scene is None:
            return

        snapshot = self._kinematic.snapshot()
        origin = np.array(
            [
                snapshot.x,
                snapshot.y,
                snapshot.z - self._kinematic.vehicle_height + self.config.lidar_z_offset,
            ],
            dtype=np.float64,
        )
        points = self._lidar.scan(scene, origin, snapshot.yaw)
        if points is None:
            return
        self.registered_scan.publish(
            PointCloud2.from_numpy(points, frame_id="map", timestamp=time.time())
        )

    def _on_path(self, msg: PathMsg) -> None:
        with self._state_lock:
            self._latest_path = [[pose.x, pose.y, pose.z] for pose in msg.poses]

    def _on_pointcloud_overlay(self, msg: PointCloud2) -> None:
        now = time.monotonic()
        if now - self._last_pointcloud_sent < self._pointcloud_min_dt:
            return

        payload = self._make_pointcloud_payload(msg)
        if payload is None:
            return

        with self._pointcloud_lock:
            self._latest_pointcloud_payload = payload
            self._last_pointcloud_sent = now
        self._broadcast_from_thread(payload)

    def _on_camera_image(self, msg: Image) -> None:
        # Rate-limit to avoid saturating the websocket with multi-MB frames
        # when the publisher pushes at 30+ Hz.
        now = time.monotonic()
        with self._camera_lock:
            if now - self._last_camera_sent < self._camera_min_dt:
                return
            self._last_camera_sent = now

        try:
            jpeg = self._encode_jpeg(msg)
        except Exception as exc:
            logger.warning("BabylonViewer: camera JPEG encode failed: %s", exc)
            return
        if jpeg is None:
            return

        # Binary frame layout:
        #   byte 0:      WS_MSG_CAMERA (0x01)
        #   bytes 1-2:   name length (big-endian uint16)
        #   bytes 3..:   utf-8 camera name, then JPEG payload
        name = self.config.camera_name.encode("utf-8")[:65535]
        header = bytes([WS_MSG_CAMERA]) + len(name).to_bytes(2, "big") + name
        self._broadcast_bytes_from_thread(header + jpeg)

    def _encode_jpeg(self, msg: Image) -> bytes | None:
        if self._turbo_jpeg is None:
            from turbojpeg import TurboJPEG

            self._turbo_jpeg = TurboJPEG()

        from turbojpeg import TJPF_BGR, TJPF_GRAY, TJPF_RGB

        data = msg.data
        if data is None:
            return None
        match msg.format:
            case ImageFormat.RGB:
                pixel_format = TJPF_RGB
            case ImageFormat.BGR:
                pixel_format = TJPF_BGR
            case ImageFormat.GRAY:
                pixel_format = TJPF_GRAY
            case _:
                # RGBA/BGRA: drop alpha to keep encode cheap.
                if data.ndim == 3 and data.shape[2] == 4:
                    data = data[:, :, :3]
                    pixel_format = TJPF_BGR if msg.format == ImageFormat.BGRA else TJPF_RGB
                else:
                    return None

        encoded: bytes = self._turbo_jpeg.encode(
            np.ascontiguousarray(data),
            quality=self.config.camera_jpeg_quality,
            pixel_format=pixel_format,
        )
        return encoded

    def _broadcast_bytes_from_thread(self, payload: bytes) -> None:
        loop = self._server_loop
        if loop is None or not self._clients:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_bytes(payload), loop)

    async def _broadcast_bytes(self, payload: bytes) -> None:
        dead: list[WebSocket] = []
        for websocket in tuple(self._clients):
            try:
                await websocket.send_bytes(payload)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self._clients.discard(websocket)

    def _make_pointcloud_payload(self, msg: PointCloud2) -> dict[str, Any] | None:
        points = msg.points_f32()
        if points.size == 0:
            return None

        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return None

        if len(points) > self.config.pointcloud_max_points:
            indices = np.linspace(0, len(points) - 1, self.config.pointcloud_max_points).astype(
                np.int64
            )
            points = points[indices]

        z_values = points[:, 2]
        if len(z_values) >= 20:
            z_min, z_max = np.quantile(z_values, [0.02, 0.98])
        else:
            z_min, z_max = float(z_values.min()), float(z_values.max())
        normalized = np.clip((z_values - z_min) / (z_max - z_min + 1e-6), 0.0, 1.0)
        # Blue (low z) → green (high z) gradient.
        rgb = np.zeros((len(points), 3), dtype=np.float32)
        rgb[:, 1] = normalized * 255.0
        rgb[:, 2] = (1.0 - normalized) * 255.0
        colors = rgb.astype(np.uint8)

        return {
            "type": "pointcloud",
            "count": len(points),
            "positions": np.round(points, 3).reshape(-1).tolist(),
            "colors": colors.reshape(-1).tolist(),
        }
