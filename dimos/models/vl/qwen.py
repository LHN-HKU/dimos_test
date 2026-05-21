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

from functools import cached_property
import os
from typing import Any

import httpx
import numpy as np

from dimos.models.vl.base import VlModel, VlModelConfig
from dimos.msgs.sensor_msgs.Image import Image


class QwenVlModelConfig(VlModelConfig):
    """Configuration for Qwen VL model."""

    model_name: str = "qwen2.5-vl-72b-instruct"
    api_key: str | None = None
    base_url: str | None = None


class QwenVlModel(VlModel):
    config: QwenVlModelConfig

    @cached_property
    def _client(self) -> httpx.Client:
        api_key = self.config.api_key or os.getenv("ALIBABA_API_KEY")
        if not api_key:
            raise ValueError(
                "Alibaba API key must be provided or set in ALIBABA_API_KEY environment variable"
            )

        base_url = (
            self.config.base_url
            or os.getenv("DASH_SCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        return httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120.0,
            trust_env=False,
        )

    def _chat_completion(
        self,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
        }
        if response_format:
            payload["response_format"] = response_format

        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def query(self, image: Image | np.ndarray, query: str) -> str:  # type: ignore[override]
        if isinstance(image, np.ndarray):
            import warnings

            warnings.warn(
                "QwenVlModel.query should receive standard dimos Image type, not a numpy array",
                DeprecationWarning,
                stacklevel=2,
            )

            image = Image.from_numpy(image)

        # Apply auto_resize if configured
        image, _ = self._prepare_image(image)

        img_base64 = image.to_base64()

        return self._chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                        },
                        {"type": "text", "text": query},
                    ],
                }
            ],
        )

    def query_batch(
        self,
        images: list[Image],
        query: str,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Query VLM with multiple images using a single API call."""
        if not images:
            return []

        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self._prepare_image(img)[0].to_base64()}"
                },
            }
            for img in images
        ]
        content.append({"type": "text", "text": query})

        response_text = self._chat_completion(
            messages=[{"role": "user", "content": content}],
            response_format=response_format,
        )
        # Return one response per image (same response since API analyzes all images together)
        return [response_text] * len(images)

    def stop(self) -> None:
        """Release the HTTP client."""
        if "_client" in self.__dict__:
            self._client.close()
            del self.__dict__["_client"]
