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

"""Interactive camera calibration for dimos (ROS CameraInfo YAML output)."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import typer
import yaml

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg"})


class Source(str, Enum):
    """Frame source supported by the calibration CLI."""

    webcam = "webcam"
    folder = "folder"


app = typer.Typer(
    help="Calibrate camera intrinsics and write ROS CameraInfo YAML.",
    no_args_is_help=True,
)


def write_camera_info_yaml(
    path: str,
    *,
    image_width: int,
    image_height: int,
    camera_name: str,
    frame_id: str,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray | None = None,
    P: np.ndarray | None = None,
    distortion_model: str = "plumb_bob",
) -> None:
    """Write ROS-style sensor_msgs/CameraInfo YAML for use with ``load_camera_info``.

    ``frame_id`` is part of the keyword API for symmetry with ``load_camera_info``; the
    ROS YAML schema does not store it (pass ``frame_id`` when calling ``load_camera_info``).
    """
    k = np.asarray(K, dtype=np.float64).reshape(3, 3)
    d = np.asarray(D, dtype=np.float64).ravel()
    k_flat = k.ravel(order="C").tolist()
    d_flat = d.tolist()

    if R is None:
        r_flat = np.eye(3, dtype=np.float64).ravel(order="C").tolist()
    else:
        r_flat = np.asarray(R, dtype=np.float64).reshape(3, 3).ravel(order="C").tolist()

    if P is None:
        fx = k_flat[0]
        fy = k_flat[4]
        cx = k_flat[2]
        cy = k_flat[5]
        p_flat = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    else:
        p_flat = np.asarray(P, dtype=np.float64).reshape(3, 4).ravel(order="C").tolist()

    n_dist = len(d_flat)
    payload = {
        "image_width": int(image_width),
        "image_height": int(image_height),
        "camera_name": camera_name,
        "distortion_model": distortion_model,
        "camera_matrix": {"rows": 3, "cols": 3, "data": k_flat},
        "distortion_coefficients": {"rows": 1, "cols": int(n_dist), "data": d_flat},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": r_flat},
        "projection_matrix": {"rows": 3, "cols": 4, "data": p_flat},
    }

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)


def load_frames_from_folder(path: str) -> list[np.ndarray]:
    """Load ``*.png``, ``*.jpg``, and ``*.jpeg`` images from a directory.

    Files are ordered by filename (lexicographic sort of basenames). Raises if the path
    is not a directory or if any matching file fails to decode with ``cv2.imread``.
    """
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {path}")

    paths = sorted(
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    out: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Could not read image: {p}")
        out.append(img)
    return out


_CAMERACALIBRATE_WINDOW = "dimos cameracalibrate"


def capture_frames_from_webcam(
    device_index: int,
    target_count: int,
    cols: int,
    rows: int,
    *,
    no_display: bool = False,
) -> list[np.ndarray]:
    """Capture ``target_count`` BGR frames from a webcam when the board is visible.

    Shows a live preview (unless ``no_display`` is True, for headless runs and CI).
    When a chessboard is detected, press SPACE to accept the current frame. Press
    ``q`` to quit early (raises if fewer than ``target_count`` frames were accepted).

    ``no_display`` mirrors the CLI ``--no-display`` flag: no ``cv2.imshow`` or window
    teardown; ``cv2.waitKey`` is still used so automated tests can inject key codes.
    """
    if target_count < 1:
        raise ValueError("target_count must be >= 1")

    accepted: list[np.ndarray] = []
    cap: cv2.VideoCapture | None = None

    try:
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera device_index={device_index!r}")

        while len(accepted) < target_count:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            corners = find_chessboard_corners(gray, cols, rows)
            preview = np.asarray(frame).copy()
            if corners is not None:
                cv2.drawChessboardCorners(preview, (cols, rows), corners, True)

            if not no_display:
                cv2.imshow(_CAMERACALIBRATE_WINDOW, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if corners is not None:
                    accepted.append(np.asarray(frame).copy())
            elif key == ord("q"):
                break

        if len(accepted) < target_count:
            raise RuntimeError(
                f"Capture ended with {len(accepted)} of {target_count} frames "
                "(quit early, missing detections on SPACE, or read failures)."
            )

        return accepted

    finally:
        if cap is not None:
            cap.release()
        if not no_display:
            try:
                cv2.destroyWindow(_CAMERACALIBRATE_WINDOW)
            except cv2.error:
                pass
            cv2.waitKey(1)


def find_chessboard_corners(gray: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    """Detect inner chessboard corners and refine them with sub-pixel accuracy.

    ``cols`` and ``rows`` are the counts of **inner** corners along each axis, matching
    ``cv2.findChessboardCorners(..., patternSize=(cols, rows))``.

    Returns:
        Float array of shape ``(cols * rows, 1, 2)`` on success, else ``None``.
    """
    g = np.asarray(gray)
    pattern_size = (cols, rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(g, pattern_size, flags)
    if not ok or corners is None:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(g, corners, (11, 11), (-1, -1), criteria)
    return refined


def calibrate_from_frames(
    frames: list[np.ndarray],
    cols: int,
    rows: int,
    square_size_m: float,
) -> dict[str, object]:
    """Calibrate intrinsics from grayscale or BGR frames containing a chessboard.

    Each frame where ``find_chessboard_corners`` succeeds contributes one view.
    All frames must share the same resolution.

    Returns:
        ``{"K", "D", "rms", "image_size", "n_used"}`` with ``K`` (3x3) and ``D`` (1-d),
        ``rms`` reprojection RMSE from OpenCV, ``image_size`` ``(width, height)``, and
        ``n_used`` the number of frames that yielded detections.
    """
    if not frames:
        raise ValueError("frames must be non-empty")

    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    objp *= float(square_size_m)

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []

    first = np.asarray(frames[0])
    h0, w0 = first.shape[:2]

    for frame in frames:
        f = np.asarray(frame)
        if f.shape[:2] != (h0, w0):
            raise ValueError("All frames must have the same shape.")
        if f.ndim == 3:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        else:
            gray = f
        corners = find_chessboard_corners(gray, cols, rows)
        if corners is None:
            continue
        objpoints.append(objp)
        imgpoints.append(corners.astype(np.float32))

    if not objpoints:
        raise ValueError("Chessboard not found in any frame.")

    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        (w0, h0),
        None,
        None,
    )
    K = np.asarray(camera_matrix, dtype=np.float64)
    D = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)

    return {
        "K": K,
        "D": D,
        "rms": float(rms),
        "image_size": (int(w0), int(h0)),
        "n_used": len(objpoints),
    }


def run_calibration(
    *,
    source: Source | str,
    device_index: int,
    images: Path | None,
    cols: int,
    rows: int,
    square_size_m: float,
    out: Path,
    frame_id: str,
    camera_name: str,
    target_count: int,
    no_display: bool,
) -> dict[str, object]:
    """Run calibration from the requested frame source and write CameraInfo YAML."""
    source_value = Source(source)
    if cols < 1:
        raise ValueError("cols must be >= 1")
    if rows < 1:
        raise ValueError("rows must be >= 1")
    if square_size_m <= 0:
        raise ValueError("square_size_m must be > 0")

    if source_value is Source.folder:
        if images is None:
            raise ValueError("--images is required when --source folder")
        frames = load_frames_from_folder(str(images))
    else:
        frames = capture_frames_from_webcam(
            device_index,
            target_count,
            cols,
            rows,
            no_display=no_display,
        )

    result = calibrate_from_frames(frames, cols, rows, square_size_m)
    image_width, image_height = result["image_size"]
    out.parent.mkdir(parents=True, exist_ok=True)
    write_camera_info_yaml(
        str(out),
        image_width=int(image_width),
        image_height=int(image_height),
        camera_name=camera_name,
        frame_id=frame_id,
        K=np.asarray(result["K"], dtype=np.float64),
        D=np.asarray(result["D"], dtype=np.float64),
    )
    return result


@app.command()
def calibrate(
    source: Source = typer.Option(..., "--source", help="Frame source: webcam or folder"),
    device_index: int = typer.Option(0, "--device-index", help="Webcam device index"),
    images: Path | None = typer.Option(
        None, "--images", help="Directory of calibration images for --source folder"
    ),
    cols: int = typer.Option(..., "--cols", help="Inner chessboard corner columns"),
    rows: int = typer.Option(..., "--rows", help="Inner chessboard corner rows"),
    square_size_m: float = typer.Option(
        ..., "--square-size-m", help="Chessboard square size in meters"
    ),
    out: Path = typer.Option(..., "--out", help="Output ROS CameraInfo YAML path"),
    frame_id: str = typer.Option("camera_optical", "--frame-id", help="Camera optical frame id"),
    camera_name: str = typer.Option("webcam", "--camera-name", help="Camera name in YAML"),
    target_count: int = typer.Option(20, "--target-count", help="Accepted webcam frame count"),
    no_display: bool = typer.Option(
        False, "--no-display", help="Disable OpenCV preview windows"
    ),
) -> None:
    """Calibrate camera intrinsics and write ROS CameraInfo YAML."""
    try:
        result = run_calibration(
            source=source,
            device_index=device_index,
            images=images,
            cols=cols,
            rows=rows,
            square_size_m=square_size_m,
            out=out,
            frame_id=frame_id,
            camera_name=camera_name,
            target_count=target_count,
            no_display=no_display,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"RMS: {float(result['rms']):.6f} px ({int(result['n_used'])} frame(s) used)")
    typer.echo(f"Wrote camera info YAML to {out}")


def main(args: list[str] | None = None) -> None:
    """CLI entry point."""
    app(args=args)
