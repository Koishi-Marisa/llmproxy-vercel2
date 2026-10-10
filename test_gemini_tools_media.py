"""Test multimodal input and function calling on the Gemini route.

Runs against a mock Google upstream (mock_gemini.py) plus a tiny local file
server that stands in for a remote image host. Both are started automatically on
first use, so there is no network dependency.

    python test_gemini_tools_media.py http://127.0.0.1:8135
"""
import base64
import json
import os
import socket
import subprocess
import sys
import time

import httpx
from fastapi import FastAPI
from fastapi.responses import Response

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8135").rstrip("/")
HDR = {"Authorization": "Bearer AIza_fake_key", "Content-Type": "application/json"}
LAST_REQUEST = "mock_gemini_last_request.json"
PY = sys.executable

MOCK_PORT = 8136
REMOTE_PORT = 8137

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        FAIL.append((name, exc))
        print(f"  FAIL  {name}: {exc}")


def body(**kw):
    b = {"model": "gemini-mock", "messages": [{"role": "user", "content": "hi"}]}
    b.update(kw)
    return b


def sent_upstream():
    """Read the request the mock upstream last received."""
    for _ in range(20):
        try:
            with open(LAST_REQUEST, encoding="utf-8") as fh:
                return json.load(fh)["payload"]
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.05)
    raise AssertionError("mock upstream never recorded a request")


def post(payload, base=None):
    return httpx.post(f"{(base or BASE)}/gemini/chat/completions", headers=HDR,
                      json=payload, timeout=60)


def reply_text(response):
    assert response.status_code == 200, f"HTTP {response.status_code}: {response.text[:400]}"
    return response.json()["choices"][0]["message"]["content"]


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_up(url, attempts=60):
    for _ in range(attempts):
        try:
            httpx.get(url, timeout=3)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(0.3)
    return False


PROCS = []


def start_helpers():
    """Start the mock upstream and a local 'remote image host' if not running."""
    env = dict(os.environ, GEMINI_API_BASE=f"http://127.0.0.1:{MOCK_PORT}", NO_PROXY="127.0.0.1,localhost")
    if port_free(MOCK_PORT):
        PROCS.append(subprocess.Popen(
            [PY, "-m", "uvicorn", "mock_gemini:app", "--host", "127.0.0.1",
             "--port", str(MOCK_PORT), "--log-level", "warning"], env=env))
        assert wait_up(f"http://127.0.0.1:{MOCK_PORT}/openapi.json"), "mock upstream failed to start"

    if port_free(REMOTE_PORT):
        PROCS.append(subprocess.Popen(
            [PY, "-m", "uvicorn", "image_host:app", "--host", "127.0.0.1",
             "--port", str(REMOTE_PORT), "--log-level", "warning"], env=env))
        assert wait_up(f"http://127.0.0.1:{REMOTE_PORT}/openapi.json"), "image host failed to start"


# A 1x1 transparent PNG, inlined as a data URI.
PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB/AL+"
           "1wAAAABJRU5ErkJggg==")
DATA_URI = f"data:image/png;base64,{PNG_B64}"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": ["string", "null"], "enum": ["c", "f"], "default": "c"},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


def main():
    print(f"Target: {BASE}\nUpstream mock: http://127.0.0.1:{MOCK_PORT}\n")
    print("--- multimodal ---")

    def t_data_uri_image():
        r = post(body(model="echo-image", messages=[{"role": "user", "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": DATA_URI, "detail": "auto"}},
        ]}]))
        text = reply_text(r)
        assert text.startswith("PARTS=2_MEDIA=1_image/png"), text
        payload = sent_upstream()
        parts = payload["contents"][0]["parts"]
        assert parts[0] == {"text": "What is in this image?"}, parts
        assert parts[1]["inline_data"]["mime_type"] == "image/png", parts
        # Must be genuine PNG bytes, re-encoded from the data URI.
        assert base64.b64decode(parts[1]["inline_data"]["data"]).startswith(b"\x89PNG"), parts

    check("image as base64 data URI -> inline_data", t_data_uri_image)

    def t_remote_image():
        url = f"http://127.0.0.1:{REMOTE_PORT}/pixel.png"
        r = post(body(model="echo-image", messages=[{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": url},
        ]}]))
        text = reply_text(r)
        assert text.startswith("PARTS=2_MEDIA=1_image/png"), text
        parts = sent_upstream()["contents"][0]["parts"]
        assert base64.b64decode(parts[1]["inline_data"]["data"]).startswith(b"\x89PNG"), parts

    check("remote image URL -> fetched and inlined", t_remote_image)

    def t_audio():
        r = post(body(model="echo-image", messages=[{"role": "user", "content": [
            {"type": "text", "text": "transcribe"},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(b"RIFFfake").decode(), "format": "wav"}},
        ]}]))
        text = reply_text(r)
        assert text.startswith("PARTS=2_MEDIA=1_audio/wav"), text

    check("input_audio -> inline_data audio/wav", t_audio)

    def t_audio_garbage_rejected():
        """`b64decode(..., validate=False)` would silently return b'' here."""
        r = post(body(model="echo-image", messages=[{"role": "user", "content": [
            {"type": "text", "text": "transcribe"},
            {"type": "input_audio", "input_audio": {"data": "!!!!not base64!!!!", "format": "wav"}},
        ]}]))
        assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text[:200]}"
        assert "base64" in r.text, r.text[:200]

    check("garbage input_audio -> 400, not an empty part", t_audio_garbage_rejected)

    def t_data_uri_garbage_rejected():
        r = post(body(model="echo-image", messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!!"}},
        ]}]))
        assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text[:200]}"

    check("garbage base64 data URI -> 400", t_data_uri_garbage_rejected)

    def t_multi_turn_media():
        r = post(body(model="echo-image", messages=[
            {"role": "user", "content": [{"type": "text", "text": "first"}]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [
                {"type": "text", "text": "second"},
                {"type": "image_url", "image_url": {"url": DATA_URI}},
            ]},
        ]))
        assert reply_text(r).startswith("PARTS=4_MEDIA=1"), "media not carried through history"
        payload = sent_upstream()
        roles = [c["role"] for c in payload["contents"]]
        assert roles == ["user", "model", "user"], roles

    check("media in a multi-turn history", t_multi_turn_media)

    def t_non_image_url_rejected():
        r = post(body(model="echo-image", messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": f"http://127.0.0.1:{REMOTE_PORT}/not-an-image"},
        ]}]))
        assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text[:200]}"
        assert "did not return an image" in r.text, r.text[:200]

    check("non-image content-type -> 400", t_non_image_url_rejected)

    def t_bad_media_is_400():
        r = post(body(model="echo-image", messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": "not-a-url"},
        ]}]))
        assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text[:200]}"
        assert "data: URI" in r.text or "http(s)" in r.text, r.text[:200]

    check("malformed image url -> 400 (not 500)", t_bad_media_is_400)

    def t_unfetchable_image():
        r = post(body(model="echo-image", messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": "http://127.0.0.1:9/nope.png"},
        ]}]))
        assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text[:200]}"
        assert "could not fetch" in r.text, r.text[:200]

    check("unfetchable image -> 400", t_unfetchable_image)

    print("\n--- function calling: request side ---")

    def t_tools_declared():
        r = post(body(model="echo-tools", tools=[WEATHER_TOOL], tool_choice="auto"))
        text = reply_text(r)
        assert "DECL=get_weather_MODE=AUTO" in text, text
        payload = sent_upstream()
        decl = payload["tools"][0]["functionDeclarations"][0]
        assert decl["name"] == "get_weather", decl
        assert decl["description"] == "Get the current weather for a city", decl
        schema = decl["parameters"]
        for bad in ("$schema", "additionalProperties", "default"):
            assert bad not in json.dumps(schema), f"{bad} leaked into schema: {schema}"
        assert schema["properties"]["unit"]["nullable"] is True, schema
        assert "title" not in schema["properties"]["city"], schema

    check("tools -> functionDeclarations, schema pruned", t_tools_declared)

    def t_tool_choice_modes():
        post(body(model="echo-tools", tools=[WEATHER_TOOL], tool_choice="required"))
        mode = sent_upstream()["toolConfig"]["functionCallingConfig"]["mode"]
        assert mode == "ANY", mode

        post(body(model="echo-tools", tools=[WEATHER_TOOL],
                  tool_choice={"type": "function", "function": {"name": "get_weather"}}))
        cfg = sent_upstream()["toolConfig"]["functionCallingConfig"]
        assert cfg["mode"] == "ANY" and cfg["allowedFunctionNames"] == ["get_weather"], cfg

        post(body(model="echo-tools", tools=[WEATHER_TOOL], tool_choice="none"))
        payload = sent_upstream()
        assert "tools" not in payload, "tool_choice=none must not send tools"

    check("tool_choice auto/required/named/none", t_tool_choice_modes)

    def t_tool_result_roundtrip():
        messages = [
            {"role": "user", "content": "weather in paris?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_abc", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]},
            {"role": "tool", "tool_call_id": "call_abc", "name": "get_weather",
             "content": '{"temp_c": 21}'},
        ]
        reply_text(post(body(model="echo-tools", messages=messages, tools=[WEATHER_TOOL])))
        payload = sent_upstream()
        parts = payload["contents"][1]["parts"]
        assert parts[0]["functionCall"]["name"] == "get_weather", parts
        assert parts[0]["functionCall"]["args"] == {"city": "Paris"}, parts
        result = payload["contents"][2]
        assert result["role"] == "user", result
        assert result["parts"][0]["functionResponse"]["name"] == "get_weather", result
        assert result["parts"][0]["functionResponse"]["response"]["temp_c"] == 21, result

    check("assistant tool_calls + tool result round trip", t_tool_result_roundtrip)

    def t_parallel_tool_results_merged():
        messages = [
            {"role": "user", "content": "weather and time?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}]},
            {"role": "tool", "name": "get_weather", "content": "21"},
            {"role": "tool", "name": "get_time", "content": "noon"},
        ]
        reply_text(post(body(model="echo-tools", messages=messages, tools=[WEATHER_TOOL])))
        contents = sent_upstream()["contents"]
        roles = [c["role"] for c in contents]
        assert roles == ["user", "model", "user"], f"tool results must merge into one turn: {roles}"
        assert len(contents[2]["parts"]) == 2, contents[2]

    check("parallel tool results merged into one user turn", t_parallel_tool_results_merged)

    def t_plain_text_arg():
        messages = [
            {"role": "user", "content": "x"},
            {"role": "tool", "name": "get_weather", "content": "just text, not json"},
        ]
        reply_text(post(body(model="echo-tools", messages=messages)))
        contents = sent_upstream()["contents"]
        # Both turns merge into one "user" content, whose last part is the
        # function response. (A bare tool message first would be contents[0].)
        response = contents[-1]["parts"][-1]["functionResponse"]["response"]
        assert isinstance(response, dict), response
        assert response.get("result") == "just text, not json", response

    check("non-JSON tool output wrapped in an object", t_plain_text_arg)

    print("\n--- function calling: response side ---")

    def t_function_call_response():
        r = post(body(model="trigger-tools", tools=[WEATHER_TOOL], tool_choice="auto"))
        assert r.status_code == 200, r.text[:300]
        j = r.json()
        choice = j["choices"][0]
        assert choice["finish_reason"] == "tool_calls", choice
        call = choice["message"]["tool_calls"][0]
        assert call["type"] == "function", call
        assert call["function"]["name"] == "get_weather", call
        args = json.loads(call["function"]["arguments"])
        assert args == {"city": "Paris", "unit": "c"}, args
        assert call["id"].startswith("call_"), call
        assert choice["message"]["content"] == "", choice["message"]

    check("functionCall -> tool_calls + finish_reason", t_function_call_response)

    def t_text_plus_tool_call():
        r = post(body(model="trigger-tools-text", tools=[WEATHER_TOOL]))
        message = r.json()["choices"][0]["message"]
        assert message["content"] == "Let me check.", message
        assert message["tool_calls"][0]["function"]["name"] == "get_weather", message

    check("text alongside a tool call", t_text_plus_tool_call)

    def t_two_tool_calls():
        r = post(body(model="trigger-two-tools"))
        calls = r.json()["choices"][0]["message"]["tool_calls"]
        assert len(calls) == 2, calls
        assert {c["function"]["name"] for c in calls} == {"get_weather", "get_time"}, calls
        assert calls[0]["id"] != calls[1]["id"], "tool call ids must be unique"

    check("two parallel functionCalls -> two tool_calls", t_two_tool_calls)

    def t_usage_reported():
        usage = post(body(model="gemini-mock")).json()["usage"]
        assert usage == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}, usage

    check("usageMetadata -> OpenAI usage", t_usage_reported)

    print("\n--- function calling: streaming ---")

    def t_stream_tool_call_deltas():
        """Cumulative args must be emitted as increment-only deltas."""
        text, calls, done, usage_chunk = "", {}, False, None
        payload = body(model="trigger-tools", tools=[WEATHER_TOOL], stream=True,
                       stream_options={"include_usage": True})
        with httpx.stream("POST", f"{BASE}/gemini/chat/completions", headers=HDR,
                          json=payload, timeout=60) as r:
            assert r.status_code == 200, r.status_code
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    done = True
                    continue
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage_chunk = chunk["usage"]
                for c in chunk.get("choices", []):
                    delta = c.get("delta") or {}
                    text += delta.get("content") or ""
                    for call in delta.get("tool_calls") or []:
                        slot = calls.setdefault(call["index"], {"name": "", "args": ""})
                        slot["name"] = (call["function"] or {}).get("name") or slot["name"]
                        slot["args"] += (call["function"] or {}).get("arguments") or ""
        assert done, "no [DONE]"
        assert text == "Checking", f"text deltas wrong: {text!r}"
        assert list(calls) == [0], calls
        assert calls[0]["name"] == "get_weather", calls
        # "Par" then "is" -> "Paris"; the repeated identical chunk must add nothing.
        assert calls[0]["args"] == '{"city": "Paris"}', f"args reconstructed wrong: {calls[0]['args']!r}"
        assert usage_chunk == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}, usage_chunk

    check("streamed functionCall -> incremental tool_call deltas + usage", t_stream_tool_call_deltas)

    def t_stream_finish_reason_tool_calls():
        finish = None
        with httpx.stream("POST", f"{BASE}/gemini/chat/completions", headers=HDR,
                          json=body(model="trigger-tools", tools=[WEATHER_TOOL], stream=True),
                          timeout=60) as r:
            for line in r.iter_lines():
                if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                    for c in json.loads(line[6:]).get("choices", []):
                        if c.get("finish_reason"):
                            finish = c["finish_reason"]
        assert finish == "tool_calls", finish

    check("streamed finish_reason=tool_calls", t_stream_finish_reason_tool_calls)

    print("\n" + "=" * 64)
    print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
    for name, exc in FAIL:
        print(f"  - {name}: {exc}")
    print("=" * 64)
    return 1 if FAIL else 0


if __name__ == "__main__":
    start_helpers()
    try:
        code = main()
    finally:
        for p in PROCS:
            p.terminate()
    sys.exit(code)
