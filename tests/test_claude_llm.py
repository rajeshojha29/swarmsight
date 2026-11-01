"""ClaudeLLM request shape and error handling, checked against a fake client (no network)."""

from types import SimpleNamespace

import pytest

from swarmsight.llm import ClaudeLLM, LLMError, MockLLM, get_llm
from swarmsight.schemas import AgentDecision


class FakeMessages:
    def __init__(self, response):
        self.response, self.calls = response, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def make(response):
    llm = ClaudeLLM.__new__(ClaudeLLM)
    llm.model, llm.name = "claude-opus-5-5", "claude:claude-opus-5-5"
    messages = FakeMessages(response)
    llm._client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return llm, messages


def text_response(text, stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)])


def test_structured_request_shape():
    llm, messages = make(text_response('{"action": "idle", "stance": 0.1}'))
    assert llm.structured("sys", "prompt", AgentDecision, task="act") == AgentDecision(action="idle", stance=0.1)
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"]["effort"] == "low"
    schema = call["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False and "action" in schema["properties"]
    assert call["fallbacks"] == "default"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "thinking" not in call and "temperature" not in call


@pytest.mark.parametrize("response", [text_response("", "refusal"), text_response('{"act', "max_tokens"),
                                      text_response('{"action": "dance", "stance": 0}')])
def test_structured_bad_responses_raise(response):
    llm, _ = make(response)
    with pytest.raises(LLMError):
        llm.structured("sys", "prompt", AgentDecision, task="act")


def test_text_joins_text_blocks():
    blocks = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="Hello "),
              SimpleNamespace(type="text", text="there")]
    llm, _ = make(SimpleNamespace(stop_reason="end_turn", content=blocks))
    assert llm.text("sys", [{"role": "user", "content": "hi"}], task="chat") == "Hello there"


def test_get_llm_auto_falls_back_to_mock(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("SWARMSIGHT_LLM", raising=False)
    assert isinstance(get_llm(), MockLLM)
    assert isinstance(get_llm("mock"), MockLLM)


def test_real_sdk_builds_valid_request():
    """Run the real Anthropic SDK against an in-process transport and inspect the HTTP request."""
    import importlib
    import json

    import anthropic

    # anthropic 1.x+ moved its HTTP layer from httpx to the httpx2 fork.
    sdk_major = int(anthropic.__version__.split(".")[0])
    httpx = importlib.import_module("httpx2" if sdk_major >= 1 else "httpx")

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["headers"], seen["body"] = str(request.url), request.headers, json.loads(request.content)
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": json.dumps({"action": "post", "content": "hi", "stance": 0.4,
                                                              "reflection": "", "target_post_id": None})}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5}})

    llm = ClaudeLLM.__new__(ClaudeLLM)
    llm.model, llm.name = "claude-opus-5-5", "claude:claude-opus-5-5"
    llm._client = anthropic.Anthropic(
        api_key="test", http_client=anthropic.DefaultHttpxClient(transport=httpx.MockTransport(handler)))

    decision = llm.structured("system prompt", "user prompt", AgentDecision, task="act")
    assert decision.action == "post" and decision.stance == 0.4
    body = seen["body"]
    assert seen["url"].split("?")[0].endswith("/v1/messages")
    assert "server-side-fallback-2026-07-01" in seen["headers"]["anthropic-beta"]
    assert body["fallbacks"] == "default"
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
