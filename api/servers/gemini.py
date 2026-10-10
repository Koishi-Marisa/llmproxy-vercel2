#!/usr/bin/env python
''' Convert Gemini API to OpenAI API format

Gemini API docs:
- https://ai.google.dev/gemini-api/docs/text-generation?lang=rest
'''
from loguru import logger
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Header, Query
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import typing
from typing import Any, Dict, List, Optional
from .base import Message, parse_api_key
import os
import time
import json

router = APIRouter()


GEMINI_API_BASE = os.getenv(
    "GEMINI_API_BASE", "https://generativelanguage.googleapis.com").rstrip("/")
GEMINI_ENDPOINT = f"{GEMINI_API_BASE}/v1beta/models/{{}}:generateContent"
GEMINI_STREAM_ENDPOINT = f"{GEMINI_API_BASE}/v1beta/models/{{}}:streamGenerateContent"


class OpenAIProxyArgs(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 1
    n: int = 1
    max_tokens: Optional[int] = None
    presence_penalty: float = 0
    frequency_penalty: float = 0

    class Config:
        extra = "allow"


class MessageConverter:
    def __init__(self, messages: List[Dict[str, Any]]):
        self.messages = messages

    @staticmethod
    def _text(content: Any) -> str:
        """Flatten OpenAI message content into a plain string.

        The SDK allows both a bare string and a list of content parts.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "".join(parts)
        return str(content)

    def convert(self) -> List[Dict[str, Any]]:
        converted_messages = []
        for message in self.messages:
            if message.get("role") == "system":
                continue  # Gemini has no system role; use systemInstruction.
            role = "user" if message.get("role") == "user" else "model"
            converted_messages.append({
                "role": role,
                "parts": [{"text": self._text(message.get("content"))}]
            })
        return converted_messages


def convert_gemini_to_openai_response(gemini_response: dict, model: str) -> dict:
    """Convert Gemini API response to OpenAI-compatible format."""
    return {
        "id": gemini_response.get("candidates", [{}])[0].get("content", {}).get("role", ""),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "usage": {
            "prompt_tokens": 0,  # Gemini doesn't provide token counts
            "completion_tokens": 0,
            "total_tokens": 0
        },
        "choices": [{
            "message": {
                "role": "assistant",
                "content": gemini_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            },
            "finish_reason": "stop",
            "index": 0
        }]
    }


async def stream_gemini_response(model: str, payload: dict, api_key: str):
    """Relay Gemini's streaming JSON array as OpenAI-style SSE chunks."""
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            GEMINI_STREAM_ENDPOINT.format(model),
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            }
        ) as response:
            if response.status_code != 200:
                error_body = (await response.aread()).decode("utf-8", "replace")
                logger.error(f"[gemini] upstream stream error {response.status_code}: {error_body[:400]}")
                yield f"data: {json.dumps({'error': {'message': error_body or 'Gemini upstream error', 'code': response.status_code}})}\n\n"
                yield "data: [DONE]\n\n"
                return

            async for line in response.aiter_lines():
                line = line.strip()
                # Responses arrive either as a JSON array or as one object per
                # line. Parsing beats the old regex, which broke on escaped
                # quotes and on any text split across two chunks.
                if not line or line in ("[", "]", ","):
                    continue
                line = line.rstrip(",")
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                for candidate in obj.get("candidates", []):
                    for part in (candidate.get("content") or {}).get("parts", []):
                        text_content = part.get("text")
                        if not text_content:
                            continue
                        openai_format = {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": text_content
                                },
                                "finish_reason": None
                            }]
                        }

                        yield f"data: {json.dumps(openai_format, ensure_ascii=False)}\n\n"

    final_chunk = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/chat/completions")
async def proxy_chat_completions(
    args: OpenAIProxyArgs,
    authorization: str = Header(None),
    x_api_key: str = Header(None, alias="x-api-key"),
):
    # `authorization.split(" ")[1]` used to raise IndexError -> HTTP 500 for
    # any header without a space (e.g. a bare token or "Basic ...").
    try:
        api_key = parse_api_key(authorization, x_api_key)
    except ValueError as exc:
        raise HTTPException(
            status_code=401,
            detail=f"{exc}. Send 'Authorization: Bearer <your gemini api key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    model = args.model

    # Transform args into Gemini API format.
    # Google rejects explicit JSON nulls ("Cannot set generationConfig.temperature
    # to null"), so unset knobs must be omitted rather than sent as None.
    generation_config = {
        "temperature": args.temperature,
        "maxOutputTokens": args.max_tokens,
        "topP": args.top_p,
        "topK": 10,
    }
    generation_config = {k: v for k, v in generation_config.items() if v is not None}

    gemini_payload = {
        "contents": MessageConverter(args.messages).convert(),
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ],
        "generationConfig": generation_config,
    }

    # Gemini has no system role; a system prompt belongs in systemInstruction.
    # Dropping it silently would quietly change the model's behaviour.
    system_text = "".join(
        MessageConverter._text(m.get("content"))
        for m in args.messages
        if m.get("role") == "system"
    ).strip()
    if system_text:
        gemini_payload["systemInstruction"] = {"parts": [{"text": system_text}]}

    if args.stream:
        return StreamingResponse(stream_gemini_response(model, gemini_payload, api_key), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GEMINI_ENDPOINT.format(model),
                json=gemini_payload,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key
                }
            )

            try:
                body = response.json()
            except ValueError:
                body = None

            # A blocked prompt returns 200 with an empty candidate list, which
            # would otherwise surface as an empty assistant message.
            blocked = (
                response.status_code == 200
                and not (body or {}).get("candidates")
            )
            if response.status_code != 200 or blocked or body is None:
                detail = body or {
                    "error": {"message": "Gemini returned a non-JSON response",
                              "code": response.status_code}
                }
                return JSONResponse(content=detail, status_code=response.status_code)

            openai_compatible_response = convert_gemini_to_openai_response(body, args.model)

            return JSONResponse(openai_compatible_response)
