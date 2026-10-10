#!/usr/bin/env python3
"""Exercise every chat-capable Groq model through the deployed proxy.

    .venv/Scripts/python.exe test_all_models.py https://ai.xiaowan.dpdns.org
"""
import json
import os
import sys
import time

import httpx

BASE = (sys.argv[1] if len(sys.argv) > 1 else "https://ai.xiaowan.dpdns.org").rstrip("/")
API_KEY = os.environ.get("GROQ_API_KEY", "")
if not API_KEY:
    print("GROQ_API_KEY not set")
    sys.exit(2)

HDR = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# The Windows console here is GBK, but allam-2-7b answers in Arabic and
# gpt-oss models can emit any script. Print as UTF-8 and escape the rest.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:  # noqa: BLE001
    pass

# Chat-capable models from GET /groq/models. The rest of the list is
# specialised: whisper-* = transcription, orpheus-* = speech synthesis,
# llama-prompt-guard-* = content classification.
CHAT_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3.8-27b",
    "allam-2-7b",
]

PROMPT = "What is the capital of France? Answer with just the city name."


def visible(msg):
    return (msg.get("content") or "").strip() or (msg.get("reasoning") or "").strip()


def safe(s):
    """Printable ASCII preview, so no console encoding can break the run."""
    return s.encode("ascii", "backslashreplace").decode("ascii")


def run_nonstream(model):
    t0 = time.time()
    r = httpx.post(f"{BASE}/groq/chat/completions", headers=HDR, timeout=180, json={
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 512,
        "temperature": 0.2,
    })
    dt = time.time() - t0
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}: {r.text[:160]}", dt, None
    j = r.json()
    if not j.get("choices"):
        return False, f"no choices: {str(j)[:160]}", dt, None
    msg = j["choices"][0]["message"]
    text = visible(msg)
    if not text:
        return False, f"empty content (finish={j['choices'][0].get('finish_reason')})", dt, None
    usage = j.get("usage") or {}
    note = (f"tokens={usage.get('prompt_tokens')}+{usage.get('completion_tokens')}"
            f" reasoning={bool(msg.get('reasoning'))}")
    return True, safe(text[:70].replace("\n", " ")), dt, note


def run_stream(model):
    t0 = time.time()
    chunks, text = 0, ""
    done = False
    with httpx.stream("POST", f"{BASE}/groq/chat/completions", headers=HDR, timeout=180, json={
        "model": model,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "max_tokens": 512,
        "stream": True,
    }) as r:
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}", time.time() - t0
        ctype = r.headers.get("content-type", "")
        if "text/event-stream" not in ctype:
            return False, f"bad content-type: {ctype}", time.time() - t0
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                done = True
                continue
            chunks += 1
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            for c in obj.get("choices", []):
                d = c.get("delta") or {}
                text += (d.get("content") or "") + (d.get("reasoning") or "")
    dt = time.time() - t0
    if not chunks:
        return False, "no chunks", dt
    if not text.strip():
        return False, f"{chunks} chunks but empty text", dt
    return True, f"{chunks} chunks, [DONE]={done}", dt


def main():
    print(f"Endpoint: {BASE}\n")
    ok_count = 0
    total = 0

    print("=" * 74)
    print("NON-STREAMING")
    print("=" * 74)
    for m in CHAT_MODELS:
        total += 1
        ok, info, dt, note = run_nonstream(m)
        ok_count += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {m:<32} {dt:6.2f}s  {info}")
        if ok and note:
            print(f"       {'':<32}         {note}")

    print()
    print("=" * 74)
    print("STREAMING (SSE)")
    print("=" * 74)
    for m in CHAT_MODELS:
        total += 1
        ok, info, dt = run_stream(m)
        ok_count += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {m:<32} {dt:6.2f}s  {info}")

    print()
    print("=" * 74)
    print(f"RESULT: {ok_count}/{total} passed")
    print("=" * 74)
    return 0 if ok_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
