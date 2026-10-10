"""Mock Google Generative Language API, for testing the Gemini route offline.

Serves the two endpoints the proxy calls:
    POST /v1beta/models/{model}:generateContent
    POST /v1beta/models/{model}:streamGenerateContent

Behaviour is keyed off the model name, so a single mock can exercise the happy
path, an upstream error, and a blocked prompt (200 with no candidates).

Also writes the last received request body to mock_gemini_last_request.json so a
test can assert on exactly what the proxy forwarded upstream.

    python mock_gemini.py 8126
"""
import json
import sys

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
LAST_REQUEST_PATH = "mock_gemini_last_request.json"


@app.post("/v1beta/models/{model_action}")
async def generate(model_action: str, request: Request):
    model = model_action.split(":")[0]
    action = model_action.split(":")[1] if ":" in model_action else "generateContent"
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        payload = None

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

    if action == "streamGenerateContent":
        def gen():
            for piece in ["MOCK_", "STREAM_", "REPLY"]:
                chunk = {"candidates": [{"content": {"role": "model",
                                                     "parts": [{"text": piece}]}}]}
                yield json.dumps(chunk, ensure_ascii=False) + "\n"

        return StreamingResponse(gen(), media_type="application/json")

    return JSONResponse(status_code=200, content={
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "MOCK_REPLY"}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
    })


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8126
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
