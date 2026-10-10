"""Live Gemini verification through the deployment, using a real API key.

Exercises the genuine article rather than the mock: a generated image the model
can actually describe, a real function-call round trip, and streaming.

    set GEMINI_API_KEY=...        # or pass as argv[2]
    python verify_gemini_key.py https://ai.xiaowan.dpdns.org [key]
"""
import base64
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib

BASE = (sys.argv[1] if len(sys.argv) > 1 else "https://ai.xiaowan.dpdns.org").rstrip("/")
KEY = (sys.argv[2] if len(sys.argv) > 2 else os.getenv("GEMINI_API_KEY", "")).strip()
MODEL = os.getenv("GEMINI_TEST_MODEL", "gemini-2.5-flash")

passed = []
failed = []

# Cloudflare in front of the deployment answers error 1010 to urllib's default
# User-Agent, so every request must identify itself like a real client would.
UA = "llmproxy-verify/1.0 (OpenAI-compatible client)"


def check(name, fn):
    try:
        detail = fn()
        passed.append(name)
        print(f"  PASS  {name}" + (f" -- {detail}" if detail else ""), flush=True)
    except AssertionError as exc:
        failed.append(f"{name}: {exc}")
        print(f"  FAIL  {name}: {exc}", flush=True)
    except Exception as exc:  # noqa: BLE001
        failed.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}", flush=True)


def post(path, payload, key=True, raw_stream=False):
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "User-Agent": UA}
    if key:
        headers["Authorization"] = f"Bearer {KEY}"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            body = r.read()
            return r.status, body.decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def text_of(obj):
    try:
        return obj["choices"][0]["message"].get("content") or ""
    except Exception:  # noqa: BLE001
        return ""


def png_two_colors(width=64, height=64):
    """Hand-built PNG: left half red, right half blue. No image library needed."""
    rows = b""
    for _y in range(height):
        row = b"\x00"  # filter type 0
        row += bytes([220, 30, 30]) * (width // 2)
        row += bytes([30, 30, 220]) * (width - width // 2)
        rows += row

    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


print(f"Base URL : {BASE}")
print(f"Model    : {MODEL}")
print(f"Key      : {KEY[:8]}...{KEY[-4:]} ({len(KEY)} chars)" if KEY else "Key      : MISSING")
print()

# ---------------------------------------------------------------- build marker
def t_build_marker():
    """The strict-base64 build rejects garbage audio locally with a 400."""
    status, text = post("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": "!!!!not base64!!!!", "format": "wav"}}]}],
    })
    assert status == 400, f"expected 400, got {status}: {text[:200]}"
    assert "base64" in text, f"400 but not about base64: {text[:200]}"


if KEY:
    check("deployment carries the strict-base64 fix", t_build_marker)
else:
    print("  SKIP  deployment marker (no key)")


# ---------------------------------------------------------------- plain chat
def t_plain_chat():
    status, text = post("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly one word: PONG"}],
    })
    assert status == 200, f"{status}: {text[:300]}"
    obj = json.loads(text)
    content = text_of(obj)
    assert content, f"empty content: {text[:300]}"
    assert obj.get("choices", [{}])[0].get("finish_reason"), "no finish_reason"
    assert "usage" in obj and obj["usage"].get("total_tokens"), f"no usage: {text[:200]}"
    return repr(content.strip()[:60])


def t_openai_shape():
    status, text = post("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hi."}],
    })
    assert status == 200, f"{status}: {text[:200]}"
    obj = json.loads(text)
    assert obj.get("object") == "chat.completion", f"object={obj.get('object')!r}"
    assert obj.get("id", "").startswith("chatcmpl-"), f"id={obj.get('id')!r}"
    assert isinstance(obj.get("created"), int), "created is not an int"
    assert obj.get("model"), "no model echoed"
    return f"object={obj['object']} model={obj['model']}"


def t_extra_fields():
    """Fields the old pydantic model would have 422'd, plus an unknown one."""
    status, text = post("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say OK."}],
        "temperature": 0.2, "top_p": 0.9, "seed": 7, "stop": ["zzz"],
        "stream_options": {"include_usage": True}, "reasoning_effort": "low",
        "some_future_openai_field": {"nested": True},
    })
    assert status != 422, f"rejected locally with 422: {text[:300]}"
    assert status == 200, f"expected 200, got {status}: {text[:300]}"
    return "accepted (no 422)"


def t_system_prompt():
    status, text = post("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You must answer every question with the single word BANANA."},
            {"role": "developer", "content": "Never break character."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    })
    assert status == 200, f"{status}: {text[:300]}"
    content = text_of(json.loads(text)).lower()
    assert "banana" in content, f"system prompt ignored, got {content[:120]!r}"
    return f"obeyed: {content.strip()[:40]!r}"


# ---------------------------------------------------------------- multimodal
def t_image_data_uri():
    png = png_two_colors()
    uri = "data:image/png;base64," + base64.b64encode(png).decode()
    status, text = post("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "This image has a left half and a right half. "
                                     "Name the color of each, in the form LEFT=<color>, RIGHT=<color>."},
            {"type": "image_url", "image_url": {"url": uri}},
        ]}],
    })
    assert status == 200, f"{status}: {text[:300]}"
    content = text_of(json.loads(text)).lower()
    assert "red" in content, f"left half is red but model said {content[:200]!r}"
    assert "blue" in content, f"right half is blue but model said {content[:200]!r}"
    return f"model saw both halves: {content.strip()[:80]!r}"


def t_image_url_shape():
    """The other common OpenAI shape: a bare string inside image_url."""
    png = png_two_colors()
    uri = "data:image/png;base64," + base64.b64encode(png).decode()
    status, text = post("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": uri},
            {"type": "text", "text": "One word: what color dominates the right half?"},
        ]}],
    })
    assert status == 200, f"{status}: {text[:300]}"
    content = text_of(json.loads(text)).lower()
    assert "blue" in content, f"got {content[:120]!r}"
    return f"{content.strip()[:40]!r}"


def t_garbage_base64_image():
    status, text = post("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!!"}}]}],
    })
    assert status == 400, f"expected 400, got {status}: {text[:200]}"


def t_unfetchable_image():
    status, text = post("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://127.0.0.1:9/nope.png"}}]}],
    })
    assert status == 400, f"expected 400, got {status}: {text[:200]}"


# ---------------------------------------------------------------- function calling
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "WeatherArgs",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["c", "f"], "default": "c"},
                "days": {"type": ["integer", "null"], "exclusiveMinimum": 0},
            },
            "required": ["city"],
        },
    },
}


def t_tool_call():
    status, text = post("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}],
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
    })
    assert status == 200, f"{status}: {text[:400]}"
    obj = json.loads(text)
    choice = obj["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    assert calls, f"no tool_calls: {text[:400]}"
    call = calls[0]
    assert call["function"]["name"] == "get_weather", f"name={call['function']['name']!r}"
    assert call.get("id"), "tool_call has no id"
    assert call.get("type") == "function", f"type={call.get('type')!r}"
    args = json.loads(call["function"]["arguments"])
    assert "city" in args, f"arguments missing city: {args}"
    assert choice.get("finish_reason") == "tool_calls", f"finish_reason={choice.get('finish_reason')!r}"
    return f"name={call['function']['name']} args={args} finish={choice['finish_reason']}"


def t_tool_round_trip():
    status, text = post("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is the weather in Paris? Use the tool."},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_abc123", "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris", "unit": "c"}'}}]},
            {"role": "tool", "tool_call_id": "call_abc123", "name": "get_weather",
             "content": '{"temperature_c": 17, "conditions": "rainy"}'},
        ],
        "tools": [WEATHER_TOOL],
    })
    assert status == 200, f"{status}: {text[:400]}"
    content = text_of(json.loads(text)).lower()
    assert content, f"empty answer after tool result: {text[:300]}"
    assert "17" in content or "rain" in content, f"tool result not used: {content[:200]!r}"
    return f"used the result: {content.strip()[:80]!r}"


def t_tool_choice_forced():
    status, text = post("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Tell me a joke."}],
        "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    })
    assert status == 200, f"{status}: {text[:400]}"
    calls = json.loads(text)["choices"][0]["message"].get("tool_calls") or []
    assert calls and calls[0]["function"]["name"] == "get_weather", \
        f"forced choice ignored: {text[:300]}"
    return "forced call honoured"


def t_tool_choice_none():
    status, text = post("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say the word HELLO and nothing else."}],
        "tools": [WEATHER_TOOL],
        "tool_choice": "none",
    })
    assert status == 200, f"{status}: {text[:400]}"
    obj = json.loads(text)
    assert not (obj["choices"][0]["message"].get("tool_calls")), "called a tool despite tool_choice=none"
    return repr(text_of(obj).strip()[:40])


def t_parameterless_tool():
    status, text = post("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "What time is it? Use the tool."}],
        "tools": [{"type": "function", "function": {"name": "get_time",
                                                    "description": "Get the current time."}}],
    })
    assert status == 200, f"{status}: {text[:400]}"
    calls = json.loads(text)["choices"][0]["message"].get("tool_calls") or []
    assert calls and calls[0]["function"]["name"] == "get_time", f"no call: {text[:300]}"
    return "no-parameters tool accepted"


def t_parallel_tools():
    status, text = post("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content":
                      "I need BOTH the weather in Paris AND the weather in Tokyo. Call the tool twice."}],
        "tools": [WEATHER_TOOL],
    })
    assert status == 200, f"{status}: {text[:400]}"
    calls = json.loads(text)["choices"][0]["message"].get("tool_calls") or []
    assert len(calls) >= 2, f"expected 2 parallel calls, got {len(calls)}: {text[:300]}"
    ids = [c.get("id") for c in calls]
    assert len(set(ids)) == len(ids), f"duplicate tool_call ids: {ids}"
    cities = [json.loads(c["function"]["arguments"]).get("city", "").lower() for c in calls]
    assert any("paris" in c for c in cities) and any("tokyo" in c for c in cities), f"cities={cities}"
    return f"{len(calls)} parallel calls, cities={cities}"


# ---------------------------------------------------------------- streaming
def stream(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(path if path.startswith("http") else BASE + path, data=data,
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": UA,
                                          "Authorization": f"Bearer {KEY}"}, method="POST")
    chunks = []
    with urllib.request.urlopen(req, timeout=180) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:"):
                chunks.append(line[5:].strip())
    return chunks


def t_stream_text():
    chunks = stream("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Count from 1 to 5, space separated, nothing else."}],
        "stream": True,
    })
    assert chunks, "no SSE frames at all"
    assert chunks[-1] == "[DONE]", f"last frame is {chunks[-1]!r}, not [DONE]"
    text = ""
    for c in chunks:
        if c == "[DONE]":
            continue
        obj = json.loads(c)
        assert obj.get("object") == "chat.completion.chunk", f"bad object: {c[:120]}"
        for ch in obj.get("choices") or []:
            text += (ch.get("delta") or {}).get("content") or ""
    assert any(d.isdigit() for d in text), f"no digits streamed: {text[:150]!r}"
    return f"{len(chunks)} frames, text={text.strip()[:60]!r}"


def t_stream_usage():
    chunks = stream("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say OK."}],
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    frames = [json.loads(c) for c in chunks if c != "[DONE]"]
    with_usage = [f for f in frames if f.get("usage")]
    assert with_usage, "include_usage requested but no frame carried usage"
    usage = with_usage[-1]["usage"]
    assert usage.get("total_tokens"), f"usage has no total_tokens: {usage}"
    return f"usage={usage.get('prompt_tokens')}+{usage.get('completion_tokens')}={usage.get('total_tokens')}"


def t_stream_tool_calls():
    chunks = stream("/gemini/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Weather in Paris? Call get_weather."}],
        "tools": [WEATHER_TOOL],
        "stream": True,
    })
    assert chunks[-1] == "[DONE]", f"last frame {chunks[-1]!r}"
    slots = {}
    finish = None
    for c in chunks:
        if c == "[DONE]":
            continue
        obj = json.loads(c)
        for ch in obj.get("choices") or []:
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                slot = slots.setdefault(tc.get("index", 0), {"name": None, "id": None, "args": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if tc.get("id"):
                    slot["id"] = tc["id"]
                slot["args"] += fn.get("arguments") or ""
    assert slots, f"no streamed tool_call deltas (finish={finish})"
    slot = slots[0]
    assert slot["name"] == "get_weather", f"name={slot['name']!r}"
    assert slot["id"], "no id announced"
    args = json.loads(slot["args"])
    assert "city" in args, f"reassembled args lack city: {slot['args']!r}"
    assert finish == "tool_calls", f"finish_reason={finish!r}"
    return f"name={slot['name']} args={args} finish={finish}"


# ---------------------------------------------------------------- errors
def t_no_credentials():
    status, text = post("/gemini/chat/completions",
                        {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}, key=False)
    assert status == 401, f"expected 401, got {status}: {text[:200]}"


def t_bad_credentials():
    global KEY
    good = KEY
    KEY = "definitely-not-a-real-key"
    try:
        status, text = post("/gemini/chat/completions",
                            {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
    finally:
        KEY = good
    assert status in (400, 401, 403), f"expected upstream rejection, got {status}: {text[:200]}"
    return f"upstream {status}"


def t_bad_model():
    status, text = post("/gemini/chat/completions",
                        {"model": "no-such-model-xyz-99", "messages": [{"role": "user", "content": "hi"}]})
    assert status in (400, 404), f"expected 400/404, got {status}: {text[:200]}"
    return f"{status}"


print("--- connectivity / build ---")
if KEY:
    check("plain chat returns a visible answer", t_plain_chat)

print("\n--- OpenAI compatibility ---")
if KEY:
    check("response carries OpenAI object/id/created/model", t_openai_shape)
    check("extra + unknown OpenAI fields accepted (no 422)", t_extra_fields)
    check("system + developer turns become a system instruction", t_system_prompt)

print("\n--- multimodal input ---")
if KEY:
    check("generated red/blue image is actually seen", t_image_data_uri)
    check("image_url as a bare string works", t_image_url_shape)
    check("garbage base64 data URI -> 400", t_garbage_base64_image)
    check("unfetchable remote image -> 400", t_unfetchable_image)

print("\n--- function calling ---")
if KEY:
    check("tool call returned with id/name/args", t_tool_call)
    check("tool result fed back is used", t_tool_round_trip)
    check("forced tool_choice", t_tool_choice_forced)
    check("tool_choice=none suppresses calls", t_tool_choice_none)
    check("parameterless tool accepted", t_parameterless_tool)
    check("parallel tool calls get unique ids", t_parallel_tools)

print("\n--- streaming ---")
if KEY:
    check("text stream reassembles, ends with [DONE]", t_stream_text)
    check("stream_options.include_usage", t_stream_usage)
    check("streamed tool_call deltas reassemble", t_stream_tool_calls)

print("\n--- errors ---")
check("no credentials -> 401", t_no_credentials)
if KEY:
    check("invalid key -> upstream rejection", t_bad_credentials)
    check("unknown model -> upstream error", t_bad_model)

print()
print("=" * 64)
print(f"PASSED {len(passed)}   FAILED {len(failed)}")
for f in failed:
    print(f"  - {f}")
print("=" * 64)
sys.exit(1 if failed else 0)
