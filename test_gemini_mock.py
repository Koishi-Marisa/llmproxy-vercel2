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


print("\n" + "=" * 62)
print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
for name, exc in FAIL:
    print(f"  - {name}: {exc}")
print("=" * 62)
sys.exit(1 if FAIL else 0)
