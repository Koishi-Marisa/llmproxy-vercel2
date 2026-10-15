"""Test the Gemini route end-to-end against a mocked Google endpoint.

This network cannot reach generativelanguage.googleapis.com, so the upstream is
faked. That is enough to prove the proxy builds a valid request, converts the
response, and handles errors. Run with the local uvicorn server on port 8125.

    python test_gemini_mock.py http://127.0.0.1:8125
"""
import json
import sys

import httpx

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8125").rstrip("/")
HDR = {"Authorization": "Bearer AIza_fake_key", "Content-Type": "application/json"}
LAST_REQUEST = "mock_gemini_last_request.json"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

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
    b = {"model": "gemini-1.5-flash", "messages": [{"role": "user", "content": "hi"}]}
    b.update(kw)
    return b


print(f"Target: {BASE}\n")


def t_omitted_optional():
    """max_tokens absent must not send maxOutputTokens: null to Google."""
    r = httpx.post(f"{BASE}/gemini/chat/completions", headers=HDR, json=body(), timeout=60)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"
    j = r.json()
    assert j["choices"][0]["message"]["content"] == "MOCK_REPLY", j
    assert j["object"] == "chat.completion", j


check("omitted max_tokens -> 200, OpenAI-shaped reply", t_omitted_optional)


def t_content_parts():
    r = httpx.post(f"{BASE}/gemini/chat/completions", headers=HDR, json=body(
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    ), timeout=60)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"


check("content as parts list -> 200 (was 422)", t_content_parts)


def t_system_prompt():
    r = httpx.post(f"{BASE}/gemini/chat/completions", headers=HDR, json=body(
        messages=[{"role": "system", "content": "be terse"},
                  {"role": "user", "content": "hi"}]
    ), timeout=60)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"


check("system prompt accepted", t_system_prompt)


def t_extra_field():
    r = httpx.post(f"{BASE}/gemini/chat/completions", headers=HDR,
                   json=body(seed=42, response_format={"type": "text"}), timeout=60)
    assert r.status_code != 422, f"our validation rejected extras: {r.text[:300]}"


check("extra OpenAI fields -> no 422", t_extra_field)


def t_stream():
    chunks, text, done = 0, "", False
    with httpx.stream("POST", f"{BASE}/gemini/chat/completions", headers=HDR,
                      json=body(stream=True), timeout=60) as r:
        assert r.status_code == 200, f"HTTP {r.status_code}"
        assert "text/event-stream" in r.headers.get("content-type", ""), \
            r.headers.get("content-type")
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                done = True
                continue
            chunks += 1
            obj = json.loads(data)
            for c in obj.get("choices", []):
                text += (c.get("delta") or {}).get("content") or ""
    assert chunks > 0, "no chunks"
    assert text == "MOCK_STREAM_REPLY", f"text={text!r}"
    assert done, "no [DONE]"


check("streaming -> SSE chunks reassemble correctly", t_stream)


def t_upstream_error():
    r = httpx.post(f"{BASE}/gemini/chat/completions", headers=HDR,
                   json=body(model="trigger-error"), timeout=60)
    assert r.status_code == 400, f"expected upstream 400, got {r.status_code}"
    assert "API key not valid" in r.text, r.text[:200]


check("upstream error passed through", t_upstream_error)


def t_blocked_prompt():
    """Google signals a blocked prompt as 200 with no candidates, so the raw
    body (promptFeedback.blockReason) must reach the caller instead of an
    empty assistant message."""
    r = httpx.post(f"{BASE}/gemini/chat/completions", headers=HDR,
                   json=body(model="trigger-blocked"), timeout=60)
    assert r.status_code == 200, f"HTTP {r.status_code}"
    j = r.json()
    assert j.get("promptFeedback", {}).get("blockReason") == "SAFETY", \
        f"block reason not surfaced: {j}"


check("blocked prompt surfaces raw Gemini body", t_blocked_prompt)


def t_upstream_payload_clean():
    """The exact bytes sent to Google must contain no JSON nulls.

    Regression guard: generationConfig.maxOutputTokens was previously sent as
    null whenever max_tokens was omitted, and Google rejects null outright.
    """
    httpx.post(f"{BASE}/gemini/chat/completions", headers=HDR, json=body(), timeout=60)
    with open(LAST_REQUEST, encoding="utf-8") as fh:
        sent = json.load(fh)
    payload = sent["payload"]
    assert "maxOutputTokens" not in payload["generationConfig"], \
        f"null maxOutputTokens sent upstream: {payload['generationConfig']}"
    raw = json.dumps(payload)
    assert "null" not in raw, f"null present in upstream payload: {raw}"
    assert sent["api_key"] == "AIza_fake_key", f"key not forwarded: {sent['api_key']}"
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}], payload["contents"]


check("upstream payload has no nulls, key forwarded", t_upstream_payload_clean)


def t_max_tokens_forwarded():
    httpx.post(f"{BASE}/gemini/chat/completions", headers=HDR,
               json=body(max_tokens=256), timeout=60)
    with open(LAST_REQUEST, encoding="utf-8") as fh:
        sent = json.load(fh)
    assert sent["payload"]["generationConfig"].get("maxOutputTokens") == 256, \
        sent["payload"]["generationConfig"]


check("max_tokens -> maxOutputTokens", t_max_tokens_forwarded)


def t_system_instruction():
    httpx.post(f"{BASE}/gemini/chat/completions", headers=HDR, json=body(
        messages=[{"role": "system", "content": "be terse"},
                  {"role": "user", "content": "hi"}]
    ), timeout=60)
    with open(LAST_REQUEST, encoding="utf-8") as fh:
        sent = json.load(fh)
    payload = sent["payload"]
    assert payload.get("systemInstruction") == {"parts": [{"text": "be terse"}]}, \
        f"system prompt lost: {payload.get('systemInstruction')}"
    assert len(payload["contents"]) == 1, "system message must not appear in contents"
    assert payload["contents"][0]["role"] == "user", payload["contents"]


check("system prompt -> systemInstruction", t_system_instruction)


def t_no_credentials():
    r = httpx.post(f"{BASE}/gemini/chat/completions", json=body(), timeout=60)
    assert r.status_code == 401, f"expected 401, got {r.status_code}"


check("no credentials -> 401", t_no_credentials)


print("\n--- grounding with Google Search (native passthrough) ---")

NATIVE = "gemini-3.5-flash-lite"
GROUND_TOOLS = {"tools": [{"google_search": {}}]}
GROUND_BODY = {"contents": [{"parts": [{"text": "今天有什么重要的科技新闻？"}]}]}


def native(payload, headers=None, model=NATIVE, suffix="generateContent", query=""):
    h = dict(headers if headers is not None else HDR)
    return httpx.post(f"{BASE}/gemini/v1beta/models/{model}:{suffix}{query}",
                      headers=h, json=payload, timeout=60)


def t_native_grounding():
    payload = dict(GROUND_BODY, **GROUND_TOOLS)
    r = native(payload)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"
    j = r.json()
    meta = j["candidates"][0]["groundingMetadata"]
    assert meta["webSearchQueries"], f"no webSearchQueries: {meta}"
    assert meta["groundingChunks"][0]["web"]["uri"], meta
    # The google_search tool must have reached Google untouched.
    with open(LAST_REQUEST, encoding="utf-8") as fh:
        sent = json.load(fh)
    assert sent["action"] == "generateContent", sent["action"]
    assert sent["payload"]["tools"] == [{"google_search": {}}], sent["payload"]["tools"]
    assert sent["api_key"] == "AIza_fake_key", sent["api_key"]


check("native generateContent keeps google_search -> groundingMetadata", t_native_grounding)


def t_native_without_tools():
    r = native(GROUND_BODY)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"
    candidate = r.json()["candidates"][0]
    assert "groundingMetadata" not in candidate, f"grounding appeared without tools: {candidate}"


check("native generateContent without tools -> no groundingMetadata", t_native_without_tools)


def t_native_x_goog_api_key():
    r = native(dict(GROUND_BODY, **GROUND_TOOLS),
               headers={"x-goog-api-key": "AIza_native_key", "Content-Type": "application/json"})
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"
    with open(LAST_REQUEST, encoding="utf-8") as fh:
        sent = json.load(fh)
    assert sent["api_key"] == "AIza_native_key", sent["api_key"]


check("native route accepts x-goog-api-key", t_native_x_goog_api_key)


def t_native_query_key():
    r = native(dict(GROUND_BODY, **GROUND_TOOLS),
               headers={"Content-Type": "application/json"}, query="?key=AIza_query_key")
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"
    with open(LAST_REQUEST, encoding="utf-8") as fh:
        sent = json.load(fh)
    assert sent["api_key"] == "AIza_query_key", sent["api_key"]


check("native route accepts ?key=", t_native_query_key)


def t_native_no_credentials():
    r = native(GROUND_BODY, headers={"Content-Type": "application/json"})
    assert r.status_code == 401, f"expected 401, got {r.status_code}: {r.text[:200]}"


check("native route without credentials -> 401", t_native_no_credentials)


def t_native_unsupported_model():
    """gemini-2.x cannot ground; the request must fail rather than answer dry."""
    r = native(dict(GROUND_BODY, **GROUND_TOOLS), model="gemini-2.5-flash")
    assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text[:200]}"
    assert "Gemini 3" in r.text, r.text[:200]


check("native grounding on gemini-2.5 -> 400 (not silent)", t_native_unsupported_model)


def t_native_streaming():
    body_text, frames = "", []
    with httpx.stream("POST", f"{BASE}/gemini/v1beta/models/{NATIVE}:streamGenerateContent",
                      headers=HDR, json=dict(GROUND_BODY, **GROUND_TOOLS), timeout=60) as r:
        assert r.status_code == 200, f"HTTP {r.status_code}"
        for line in r.iter_lines():
            if line.strip():
                frames.append(json.loads(line))
    for frame in frames:
        for candidate in frame.get("candidates", []):
            for part in (candidate.get("content") or {}).get("parts") or []:
                body_text += part.get("text") or ""
    assert body_text == "MOCK_STREAM_REPLY", f"raw frames not relayed: {body_text!r}"
    assert frames[-1].get("usageMetadata"), "usage frame missing from passthrough"


check("native streamGenerateContent relayed raw", t_native_streaming)


print("\n--- grounding with Google Search (OpenAI-compatible route) ---")


def t_openai_grounding():
    r = httpx.post(f"{BASE}/gemini/chat/completions", headers=HDR, timeout=60,
                   json={'model': NATIVE, 'messages': [{'role': 'user', 'content': 'hi'}],
                         **GROUND_TOOLS})
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"
    message = r.json()["choices"][0]["message"]
    assert message["grounding_metadata"]["webSearchQueries"], message
    assert "example.com/news/ai-chip" in message["content"], f"sources not in content: {message['content']!r}"
    assert r.json()["choices"][0]["finish_reason"] == "stop", r.json()["choices"][0]
    with open(LAST_REQUEST, encoding="utf-8") as fh:
        sent = json.load(fh)
    assert sent["payload"]["tools"] == [{"google_search": {}}], sent["payload"]["tools"]


check("openai route: {\"google_search\":{}} -> sources + grounding_metadata", t_openai_grounding)


def t_openai_grounding_tagged_form():
    """`{"type":"google_search","google_search":{}}` must work, not 500."""
    r = httpx.post(f"{BASE}/gemini/chat/completions", headers=HDR, timeout=60,
                   json={'model': NATIVE, 'messages': [{'role': 'user', 'content': 'hi'}],
                         'tools': [{'type': 'google_search', 'google_search': {}}]})
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"
    assert r.json()["choices"][0]["message"]["grounding_metadata"]["webSearchQueries"]


check("openai route: tagged google_search form -> 200", t_openai_grounding_tagged_form)


def t_openai_grounding_mixed_with_function():
    r = httpx.post(f"{BASE}/gemini/chat/completions", headers=HDR, timeout=60,
                   json={'model': NATIVE, 'messages': [{'role': 'user', 'content': 'hi'}],
                         'tools': [WEATHER_TOOL, {'google_search': {}}]})
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"
    with open(LAST_REQUEST, encoding="utf-8") as fh:
        tools = json.load(fh)["payload"]["tools"]
    assert len(tools) == 2, tools
    assert tools[0]["functionDeclarations"][0]["name"] == "get_weather", tools
    assert tools[1] == {"google_search": {}}, tools


check("openai route: function tool + google_search coexist", t_openai_grounding_mixed_with_function)


def t_openai_grounding_unsupported_model():
    r = httpx.post(f"{BASE}/gemini/chat/completions", headers=HDR, timeout=60,
                   json={'model': 'gemini-1.5-flash', 'messages': [{'role': 'user', 'content': 'hi'}],
                         **GROUND_TOOLS})
    assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text[:200]}"
    assert "Gemini 3" in r.text, r.text[:200]


check("openai route: grounding on gemini-1.5 -> 400 (not silent)", t_openai_grounding_unsupported_model)


def t_openai_untranslatable_tool():
    """An unknown tool must be an explicit error, never silently dropped."""
    r = httpx.post(f"{BASE}/gemini/chat/completions", headers=HDR, timeout=60,
                   json={'model': NATIVE, 'messages': [{'role': 'user', 'content': 'hi'}],
                         'tools': [{'type': 'web_search'}]})
    assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text[:200]}"
    assert "unsupported tool" in r.text, r.text[:200]


check("openai route: untranslatable tool -> 400 (not dropped)", t_openai_untranslatable_tool)


def t_openai_grounding_stream():
    text, meta, done = "", None, False
    header = {**HDR, 'Content-Type': 'application/json'}
    with httpx.stream("POST", f"{BASE}/gemini/chat/completions", headers=header, timeout=60,
                      json={'model': NATIVE, 'messages': [{'role': 'user', 'content': 'hi'}],
                            'stream': True, **GROUND_TOOLS}) as r:
        assert r.status_code == 200, r.status_code
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                done = True
                continue
            chunk = json.loads(data)
            for c in chunk.get("choices", []):
                delta = c.get("delta") or {}
                text += delta.get("content") or ""
                meta = delta.get("grounding_metadata") or meta
    assert done, "no [DONE]"
    assert "MOCK_STREAM_REPLY" in text, f"stream text wrong: {text!r}"
    assert "example.com/news/ai-chip" in text, f"sources not streamed: {text!r}"
    assert meta and meta.get("webSearchQueries"), f"no streamed grounding_metadata: {meta}"


check("openai route: streaming grounding -> sources + metadata", t_openai_grounding_stream)


print("\n--- model listing ---")


def t_models_list():
    for path in ("/gemini/models", "/gemini/v1beta/models"):
        r = httpx.get(f"{BASE}{path}", headers=HDR, timeout=60)
        assert r.status_code == 200, f"{path}: HTTP {r.status_code}: {r.text[:200]}"
        assert r.json()["models"], f"{path}: no models"


check("GET /gemini/models and /gemini/v1beta/models -> 200", t_models_list)


print("\n" + "=" * 62)
print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
for name, exc in FAIL:
    print(f"  - {name}: {exc}")
print("=" * 62)
sys.exit(1 if FAIL else 0)
