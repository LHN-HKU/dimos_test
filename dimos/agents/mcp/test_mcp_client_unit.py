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

import json
from queue import Empty, Queue
from unittest.mock import MagicMock, patch

import httpx
from langchain_core.messages import HumanMessage
from langchain_core.messages.base import BaseMessage
import pytest

from dimos.agents.mcp.mcp_client import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MODEL,
    McpClient,
    _missing_api_key_for_model,
    _resolve_chat_model,
)
from dimos.utils.sequential_ids import SequentialIds


def _mock_post(url: str, **kwargs: object) -> MagicMock:
    """Return a fake httpx response based on the JSON-RPC method."""
    body = kwargs.get("json") or (kwargs.get("content") and json.loads(kwargs["content"]))
    assert isinstance(body, dict)
    method = body["method"]
    req_id = body["id"]

    result: object
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "dimensional", "version": "1.0.0"},
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "add",
                    "description": "Add two numbers",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                        },
                        "required": ["x", "y"],
                    },
                },
                {
                    "name": "greet",
                    "description": "Say hello",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                        },
                    },
                },
            ]
        }
    elif method == "tools/call":
        name = body["params"]["name"]
        args = body["params"].get("arguments", {})
        if name == "add":
            text = str(args.get("x", 0) + args.get("y", 0))
        elif name == "greet":
            text = f"Hello, {args.get('name', 'world')}!"
        else:
            text = "Skill not found"
        result = {"content": [{"type": "text", "text": text}]}
    else:
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown: {method}"},
        }
        return resp

    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"jsonrpc": "2.0", "id": req_id, "result": result}
    return resp


@pytest.fixture
def mcp_client() -> McpClient:
    """Build an McpClient wired to the mock MCP post handler."""
    mock_http = MagicMock()
    mock_http.post.side_effect = _mock_post

    with patch("dimos.agents.mcp.mcp_client.httpx.Client", return_value=mock_http):
        client = McpClient.__new__(McpClient)

    client._http_client = mock_http
    client._seq_ids = SequentialIds()
    client.config = MagicMock()
    client.config.mcp_server_url = "http://localhost:9990/mcp"
    return client


def test_qwen_default_requires_alibaba_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALIBABA_API_KEY", raising=False)

    assert _missing_api_key_for_model(DEFAULT_QWEN_MODEL) == "ALIBABA_API_KEY"


def test_qwen_model_resolves_to_proxy_free_openai_compatible_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALIBABA_API_KEY", "test-key")
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:7897/")
    monkeypatch.setenv("HTTPS_PROXY", "socks://127.0.0.1:7897/")

    with patch("langchain_openai.ChatOpenAI") as chat_openai:
        model = _resolve_chat_model(
            DEFAULT_QWEN_MODEL,
            qwen_base_url=DEFAULT_QWEN_BASE_URL,
            deepseek_base_url=DEFAULT_DEEPSEEK_BASE_URL,
        )

    assert model is chat_openai.return_value
    chat_openai.assert_called_once()
    kwargs = chat_openai.call_args.kwargs
    assert kwargs["model"] == "qwen-plus"
    assert kwargs["api_key"] == "test-key"
    assert kwargs["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert isinstance(kwargs["http_client"], httpx.Client)
    assert kwargs["http_client"]._trust_env is False
    assert kwargs["timeout"] == 60.0
    assert kwargs["max_retries"] == 1
    kwargs["http_client"].close()


def test_image_artifact_gets_qwen_vision_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from dimos.agents.mcp import mcp_client as mcp_client_module
    from dimos.models.vl import qwen as qwen_module

    class FakeQwenVlModel:
        def _chat_completion(self, messages):
            content = messages[0]["content"]
            assert content[0]["type"] == "image_url"
            assert "robot camera" in content[1]["text"]
            return "A trash bin is visible ahead."

    messages: list[HumanMessage] = []
    fake_client = type("FakeClient", (), {"add_message": messages.append})()
    image_item = {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,AAAA"},
    }

    monkeypatch.setattr(qwen_module, "QwenVlModel", FakeQwenVlModel)

    mcp_client_module._append_image_to_history(fake_client, "observe", "uuid-1", image_item)

    assert len(messages) == 1
    text = messages[0].content[0]["text"]
    assert "Qwen vision analysis" in text
    assert "A trash bin is visible ahead." in text

def test_deepseek_model_still_resolves_to_openai_compatible_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    with patch("langchain_openai.ChatOpenAI") as chat_openai:
        model = _resolve_chat_model(
            DEFAULT_DEEPSEEK_MODEL,
            qwen_base_url=DEFAULT_QWEN_BASE_URL,
            deepseek_base_url=DEFAULT_DEEPSEEK_BASE_URL,
        )

    assert model is chat_openai.return_value
    kwargs = chat_openai.call_args.kwargs
    assert kwargs["model"] == "deepseek-v4-pro"
    assert kwargs["api_key"] == "test-key"
    assert kwargs["base_url"] == "https://api.deepseek.com"
    assert kwargs["timeout"] == 60.0
    assert kwargs["max_retries"] == 1
    kwargs["http_client"].close()


def test_openai_model_still_requires_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert _missing_api_key_for_model("gpt-4o") == "OPENAI_API_KEY"


def test_fetch_tools_from_mcp_server(mcp_client: McpClient) -> None:
    tools = mcp_client._fetch_tools()

    assert len(tools) == 2
    assert tools[0].name == "add"
    assert tools[1].name == "greet"


def test_tool_invocation_via_mcp(mcp_client: McpClient) -> None:
    tools = mcp_client._fetch_tools()
    add_tool = next(t for t in tools if t.name == "add")
    greet_tool = next(t for t in tools if t.name == "greet")

    assert add_tool.func(x=2, y=3) == "5"
    assert greet_tool.func(name="Alice") == "Hello, Alice!"


def test_mcp_request_error_propagation(mcp_client: McpClient) -> None:
    def error_post(url: str, **kwargs: object) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Unknown: bad/method"},
        }
        return resp

    mcp_client._http_client.post.side_effect = error_post

    try:
        mcp_client._mcp_request("bad/method")
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        assert "Unknown: bad/method" in str(e)


def test_tool_stream_notification_becomes_human_message(mcp_client: McpClient) -> None:
    """A `notifications/message` delivered over LCM becomes a HumanMessage."""
    mcp_client._message_queue = Queue()

    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "info",
            "logger": "follow_person",
            "data": "Person follow stopped: lost track.",
        },
    }
    mcp_client._on_tool_stream_message(notification)

    msg: BaseMessage = mcp_client._message_queue.get_nowait()
    assert isinstance(msg, HumanMessage)
    assert "[tool:follow_person]" in str(msg.content)
    assert "Person follow stopped: lost track." in str(msg.content)


def test_tool_stream_ignores_unrelated_frames(mcp_client: McpClient) -> None:
    """Unknown methods and empty bodies are dropped on the floor."""

    mcp_client._message_queue = Queue()

    mcp_client._on_tool_stream_message({"jsonrpc": "2.0", "method": "notifications/other"})
    mcp_client._on_tool_stream_message(
        {"jsonrpc": "2.0", "method": "notifications/message", "params": {"data": ""}}
    )
    mcp_client._on_tool_stream_message(
        {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"message": ""}}
    )

    with pytest.raises(Empty):
        mcp_client._message_queue.get_nowait()


def test_tool_stream_progress_frame_becomes_human_message(mcp_client: McpClient) -> None:
    """A `notifications/progress` frame is routed as a HumanMessage."""

    mcp_client._message_queue = Queue()

    progress_frame = {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {
            "progressToken": "pt-abc",
            "progress": 1,
            "message": "Found a person",
            "_meta": {"tool_name": "follow_person"},
        },
    }
    mcp_client._on_tool_stream_message(progress_frame)

    msg: BaseMessage = mcp_client._message_queue.get_nowait()
    assert isinstance(msg, HumanMessage)
    assert str(msg.content) == "[tool:follow_person] Found a person"


def test_mcp_tool_call_sends_progress_token(mcp_client: McpClient) -> None:
    """Every `tools/call` request carries a `_meta.progressToken`."""
    captured: dict[str, object] = {}

    def fake_request(method: str, params: dict[str, object] | None = None) -> dict[str, object]:
        captured["method"] = method
        captured["params"] = params
        return {"content": [{"type": "text", "text": "ok"}]}

    mcp_client._mcp_request = fake_request
    mcp_client._mcp_tool_call("add", {"x": 1, "y": 2})

    assert captured["method"] == "tools/call"
    params = captured["params"]
    assert isinstance(params, dict)
    assert params["name"] == "add"
    assert params["arguments"] == {"x": 1, "y": 2}
    meta = params["_meta"]
    assert isinstance(meta, dict)
    token = meta["progressToken"]
    assert isinstance(token, str) and len(token) > 0
