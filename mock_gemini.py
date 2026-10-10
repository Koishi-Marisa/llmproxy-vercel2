"""Mock Google Generative Language API, for testing the Gemini route offline.

Serves the two endpoints the proxy calls:
    POST /v1beta/models/{model}:generateContent
    POST /v1beta/models/{model}:streamGenerateContent

Model names drive the behaviour, so one mock exercises the happy path, upstream
errors, blocked prompts, multimodal input and function calling:

    gemini-mock       plain text reply, reports usageMetadata
    echo-image        states how many inline/file parts it received
    echo-tools        reports the declarations and toolConfig it received
    trigger-tools     returns a functionCall for the first declared function
    trigger-error     400 with INVALID_ARGUMENT
    trigger-blocked   200 with no candidates (promptFeedback.blockReason)
    any model + :streamGenerateContent -> streamed response

The last request body is written to mock_gemini_last_request.json so a test can
assert on exactly what the proxy forwarded upstream.

    python mock_gemini.py 8126
"""
import json
import sys

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
LAST_REQUEST_PATH = "mock_gemini_last_request.json"


def total_parts(payload):
    return sum(len(c.get("parts") or []) for c in payload.get("contents") or []
               if isinstance(c, dict))


def media_kinds(payload):
    kinds = []
    for content in payload.get("contents") or []:
        for part in content.get("parts") or []:
            if "inline_data" in part:
                kinds.append(part["inline_data"].get("mime_type"))
            elif "file_data" in part:
                kinds.append(part["file_data"].get("file_uri"))
    return kinds


def first_declaration(payload):
    for block in payload.get("tools") or []:
        for decl in block.get("functionDeclarations") or []:
            return decl
    return None


def _frame(obj):
    """One streaming frame; Google separates them with newlines and the proxy
    parses the stream line by line."""
    return json.dumps(obj, ensure_ascii=False) + "\n"


@app.post("/v1beta/models/{model_action}")
async def generate(model_action: str, request: Request):
    model = model_action.split(":")[0]
    action = model_action.split(":")[1] if ":" in model_action else "generateContent"
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        payload = None
    payload = payload or {}

    with open(LAST_REQUEST_PATH, "w", encoding="utf-8") as fh:
        json.dump({"model": model, "action": action,
                   "api_key": request.headers.get("x-goog-api-key"),
                   "payload": payload}, fh, ensure_ascii=False, indent=2)

    if model == "trigger-error":
        return JSONResponse(
            status_code=400,
            content={"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.",
                               "status": "INVALID_ARGUMENT"}},
        )

    if model == "trigger-blocked":
        return JSONResponse(status_code=200, content={
            "promptFeedback": {"blockReason": "SAFETY"},
            "usageMetadata": {"promptTokenCount": 7},
        })

    usage = {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8}

    # ---- decide what the "model" replies -----------------------------------
    # Each branch only builds the reply *parts*. Serialising them for a
    # streaming or a non-streaming response happens afterwards, so every model
    # can be exercised over BOTH transports. (Returning early from a branch
    # here would silently turn a streaming request into a buffered reply.)
    name = (first_declaration(payload) or {}).get("name") or "unknown_fn"

    if model == "echo-image":
        kinds = media_kinds(payload)
        # Report part count and media count separately: a data-URI image adds
        # TWO parts (the text prompt plus the inline image).
        reply = f"PARTS={total_parts(payload)}_MEDIA={len(kinds)}_" + ",".join(str(k) for k in kinds)
        if payload.get("systemInstruction"):
            reply += "_SYS"
        parts = [{"text": reply}]
    elif model == "echo-tools":
        mode = ((payload.get("toolConfig") or {}).get("functionCallingConfig") or {}).get("mode")
        decl = first_declaration(payload)
        parts = [{"text": f"DECL={decl and decl.get('name')}_MODE={mode}_PARTS={total_parts(payload)}"}]
    elif model == "trigger-tools":
        # Echo back the required args so the test can check arg pass-through.
        parts = [{"functionCall": {"name": name, "args": {"city": "Paris", "unit": "c"}}}]
    elif model == "trigger-tools-text":
        parts = [{"text": "Let me check."},
                 {"functionCall": {"name": name, "args": {"city": "Paris"}}}]
    elif model == "trigger-two-tools":
        parts = [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}},
                 {"functionCall": {"name": "get_time", "args": {"city": "Paris"}}}]
    else:
        parts = [{"text": "MOCK_REPLY"}]

    if action == "streamGenerateContent":

        def gen():
            if model == "trigger-tools":
                # Gemini streams cumulative args, so send growing prefixes and
                # repeat the same args, plus one text part first.
                yield _frame({"candidates": [{"content": {"role": "model",
                                                          "parts": [{"text": "Checking"}]}}]})
                for args in [{"city": "Par"}, {"city": "Paris"}, {"city": "Paris"}]:
                    yield _frame({"candidates": [{"content": {"role": "model", "parts": [
                        {"functionCall": {"name": name, "args": args}}]}}]})
            elif model == "trigger-tools-text":
                yield _frame({"candidates": [{"content": {"role": "model",
                                                          "parts": [{"text": "Let me check."}]}}]})
                yield _frame({"candidates": [{"content": {"role": "model", "parts": [
                    {"functionCall": {"name": name, "args": {"city": "Paris"}}}]}}]})
            elif model == "trigger-two-tools":
                for part in parts:
                    yield _frame({"candidates": [{"content": {"role": "model", "parts": [part]}}]})
            else:
                for piece in ["MOCK_", "STREAM_", "REPLY"]:
                    yield _frame({"candidates": [{"content": {"role": "model",
                                                              "parts": [{"text": piece}]}}]})
            yield _frame({"usageMetadata": usage})

        return StreamingResponse(gen(), media_type="application/json")

    return JSONResponse(status_code=200, content={
        "candidates": [{
            "content": {"role": "model", "parts": parts},
            "finishReason": "STOP",
        }],
        "usageMetadata": usage,
    })


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8126
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
