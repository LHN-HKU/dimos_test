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

"""Shared JPEG-aware LCM decoder for DimSim blueprints.

DimSim publishes JPEG-compressed images over LCM. The Rerun bridge
subscribes to all LCM topics and needs this custom decoder to handle
JPEG images alongside standard LCM messages.
"""

from typing import Any

from dimos.msgs.sensor_msgs.Image import Image
from dimos.protocol.pubsub.impl.lcmpubsub import LCM


class SimJpegLCM(LCM):  # type: ignore[misc]
    """LCM that JPEG-decodes DimSim image topics.

    Falls back to standard decode if the message isn't actually JPEG
    (e.g. when the bridge has already decoded it to bgr8).
    """

    _JPEG_SUFFIXES = ("/color_image",)

    def decode(self, msg: bytes, topic: Any) -> Any:  # type: ignore[override]
        topic_str = getattr(topic, "topic", "") or ""
        bare_topic = topic_str.split("#")[0]
        if any(bare_topic.endswith(s) for s in self._JPEG_SUFFIXES):
            try:
                return Image.lcm_jpeg_decode(msg)
            except ValueError:
                return super().decode(msg, topic)
        return super().decode(msg, topic)


__all__ = ["SimJpegLCM"]
