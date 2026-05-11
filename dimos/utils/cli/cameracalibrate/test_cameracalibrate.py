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

import os
import tempfile

import numpy as np

from dimos.perception.common.utils import load_camera_info
from dimos.utils.cli.cameracalibrate.cameracalibrate import main, write_camera_info_yaml


def test_main_stub_runs() -> None:
    main()


def test_write_camera_info_yaml_round_trip_matches_k_d_size_and_model() -> None:
    K = np.array([[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]])
    D = np.array([-0.1, 0.05, 0.0, 0.0, 0.0])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        path = f.name
    try:
        write_camera_info_yaml(
            path,
            image_width=640,
            image_height=480,
            camera_name="test_cam",
            frame_id="camera_optical_frame",
            K=K,
            D=D,
            distortion_model="plumb_bob",
        )
        info = load_camera_info(path, frame_id="camera_link")
        assert info.width == 640
        assert info.height == 480
        assert info.distortion_model == "plumb_bob"
        assert np.allclose(np.asarray(info.K, dtype=np.float64).reshape(3, 3), K)
        assert np.allclose(np.asarray(info.D, dtype=np.float64).ravel(), D.ravel())
    finally:
        os.unlink(path)


def test_write_camera_info_yaml_custom_r_p_and_distortion_model() -> None:
    K = np.array([[400.0, 1.0, 160.0], [0.0, 401.0, 120.0], [0.0, 0.0, 1.0]])
    D = np.array([-0.05, 0.02, 0.001, -0.0005])
    R = np.eye(3)
    P = np.array([[400.0, 0.0, 160.0, 0.01], [0.0, 401.0, 120.0, 0.02], [0.0, 0.0, 1.0, 0.0]])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        path = f.name
    try:
        write_camera_info_yaml(
            path,
            image_width=320,
            image_height=240,
            camera_name="narrow",
            frame_id="cam0",
            K=K,
            D=D,
            R=R,
            P=P,
            distortion_model="rational_polynomial",
        )
        info = load_camera_info(path)
        assert info.width == 320
        assert info.height == 240
        assert info.distortion_model == "rational_polynomial"
        assert np.allclose(np.asarray(info.K, dtype=np.float64).reshape(3, 3), K)
        assert np.allclose(np.asarray(info.D, dtype=np.float64).ravel(), D.ravel())
        assert np.allclose(np.asarray(info.R, dtype=np.float64).reshape(3, 3), R)
        assert np.allclose(np.asarray(info.P, dtype=np.float64).reshape(3, 4), P)
    finally:
        os.unlink(path)
