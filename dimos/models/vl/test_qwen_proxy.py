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

import httpx
import pytest

import dimos.models.vl.qwen as qwen_module


def test_qwen_vl_client_ignores_shell_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALIBABA_API_KEY", "test-key")
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:7897/")
    monkeypatch.setenv("HTTPS_PROXY", "socks://127.0.0.1:7897/")

    model = qwen_module.QwenVlModel()
    client = model._client

    assert isinstance(client, httpx.Client)
    assert client._trust_env is False

    model.stop()
    assert client.is_closed is True


def test_qwen_vl_uses_dashscope_http_without_openai_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "trash bin response"}}]},
        )

    monkeypatch.setenv("ALIBABA_API_KEY", "test-key")
    model = qwen_module.QwenVlModel()
    model.__dict__["_client"] = httpx.Client(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        headers={"Authorization": "Bearer test-key"},
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )

    assert not hasattr(qwen_module, "OpenAI")
    assert model._chat_completion([{"role": "user", "content": "hello"}]) == "trash bin response"
    assert requests[0].url == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    model.stop()
