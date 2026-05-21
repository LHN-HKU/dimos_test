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

from typing import Any

import numpy as np

from dimos.agents_deprecated.memory.spatial_vector_db import SpatialVectorDB
from dimos.types.robot_location import RobotLocation


class _EmbeddingProvider:
    def get_text_embedding(self, text: str) -> np.ndarray:
        return np.array([float(len(text)), 1.0, 0.0], dtype=np.float32)


class _LocationCollection:
    def __init__(self) -> None:
        self.add_kwargs: dict[str, Any] | None = None
        self.query_kwargs: dict[str, Any] | None = None

    def add(self, **kwargs: Any) -> None:
        self.add_kwargs = kwargs

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.query_kwargs = kwargs
        return {
            "ids": [["loc_test"]],
            "metadatas": [
                [
                    {
                        "location_name": "trash bin",
                        "pos_x": 1.0,
                        "pos_y": 2.0,
                        "pos_z": 0.0,
                        "rot_x": 0.0,
                        "rot_y": 0.0,
                        "rot_z": 0.5,
                        "location_id": "loc_test",
                    }
                ]
            ],
            "distances": [[0.12]],
        }


def test_tagged_location_uses_explicit_local_embeddings() -> None:
    db = SpatialVectorDB.__new__(SpatialVectorDB)
    db.embedding_provider = _EmbeddingProvider()
    db.location_collection = _LocationCollection()

    location = RobotLocation(
        name="trash bin",
        position=(1.0, 2.0, 0.0),
        rotation=(0.0, 0.0, 0.5),
        location_id="loc_test",
    )

    db.tag_location(location)
    found, distance = db.query_tagged_location("trash bin")

    assert found is not None
    assert found.name == "trash bin"
    assert distance == 0.12

    assert db.location_collection.add_kwargs is not None
    assert "embeddings" in db.location_collection.add_kwargs
    assert db.location_collection.add_kwargs["documents"] == ["trash bin"]

    assert db.location_collection.query_kwargs is not None
    assert "query_embeddings" in db.location_collection.query_kwargs
    assert "query_texts" not in db.location_collection.query_kwargs
