#!/usr/bin/env python
''' Convert Gemini API to OpenAI API format

Gemini API docs:
- https://ai.google.dev/gemini-api/docs/text-generation?lang=rest
- https://ai.google.dev/gemini-api/docs/vision?lang=rest
- https://ai.google.dev/gemini-api/docs/function-calling?lang=rest

This route accepts an OpenAI-shaped request and translates it to Gemini's native
REST shape, then translates the reply back. Supported beyond plain text:

* multimodal input - `image_url` (base64 data URI or http(s) URL), `input_audio`
  and a generic `video_url`, all as Gemini `inline_data` parts
* function calling - `tools` -> `functionDeclarations`, `tool_choice` ->
  `toolConfig`, assistant `tool_calls` -> `functionCall` parts, `role: "tool"`
  -> `functionResponse` parts, and back again for both streaming and not
'''
from loguru import logger
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Header, Query
from fastapi.responses import JSONResponse, StreamingResponse
import base64
import binascii
import httpx
import typing
from typing import Any, Dict, List, Optional, Tuple
from .base import Message, parse_api_key
import os
import time
import json

router = APIRouter()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


GEMINI_API_BASE = os.getenv(
    "GEMINI_API_BASE", "https://generativelanguage.googleapis.com").rstrip("/")
GEMINI_ENDPOINT = f"{GEMINI_API_BASE}/v1beta/models/{{}}:generateContent"
GEMINI_STREAM_ENDPOINT = f"{GEMINI_API_BASE}/v1beta/models/{{}}:streamGenerateContent"

# Google has an inline request size limit (~20MB); stay well clear of it.
MAX_INLINE_BYTES = _env_int("GEMINI_MAX_INLINE_BYTES", 18 * 1024 * 1024)
# Remote images are fetched server-side and re-encoded as inline_data, because
# only some clients inline them and Google cannot fetch an arbitrary URL.
FETCH_REMOTE_MEDIA = os.getenv("GEMINI_FETCH_REMOTE_MEDIA", "1").lower() not in ("0", "false", "no")
MEDIA_FETCH_TIMEOUT = httpx.Timeout(
    connect=_env_int("GEMINI_MEDIA_CONNECT_TIMEOUT", 10),
    read=_env_int("GEMINI_MEDIA_READ_TIMEOUT", 30),
    write=30,
    pool=10,
)

# JSON Schema keywords Gemini's Schema object does not accept. Sending any of
# them makes the whole request fail, so the tool schema is pruned to the subset
# Google documents (type, format, description, nullable, enum, items,
# properties, required).
_UNSUPPORTED_SCHEMA_KEYS = {
    "$schema", "$id", "$ref", "$comment", "$defs", "definitions",
    "additionalProperties", "unevaluatedProperties", "patternProperties",
    "exclusiveMinimum", "exclusiveMaximum", "const", "examples", "default",
    "title", "allOf", "anyOf", "oneOf", "not", "if", "then", "else",
    "dependencies", "dependentSchemas", "dependentRequired", "readOnly",
    "writeOnly", "contentEncoding", "contentMediaType", "deprecated",
}

_SCHEMA_TYPES = {"string", "number", "integer", "boolean", "array", "object"}


class OpenAIProxyArgs(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1
    n: int = 1
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    stop: Optional[Any] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None

    class Config:
        extra = "allow"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _json_obj(value: Any) -> Dict[str, Any]:
    """Coerce a value into a JSON object.

    Gemini's `functionResponse.response` is a Struct, so a bare string would be
    rejected. Passing `{"result": ...}` keeps arbitrary tool output usable.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"result": parsed}
        return {"result": value}
    if isinstance(value, list):
        return {"result": value}
    return {"result": str(value)}


def _reasoning_or_text(message: dict) -> str:
    """Some OpenAI-compatible clients put the answer in a reasoning field."""
    return (message.get("content") or message.get("reasoning") or "").strip()


def _decode_base64(data: str, what: str) -> bytes:
    """Strictly decode base64, tolerating surrounding whitespace.

    `base64.b64decode(..., validate=False)` silently DROPS characters outside
    the alphabet, so a value like "!!!!" decodes to b"" instead of failing --
    which would quietly forward an empty media part upstream. Whitespace is
    stripped first (line-wrapped base64 is legitimate), everything else must be
    real base64 that decodes to at least one byte.
    """
    compact = "".join(str(data).split())
    if not compact:
        raise ValueError(f"{what} is empty")
    try:
        blob = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{what} is not valid base64") from exc
    if not blob:
        raise ValueError(f"{what} decoded to zero bytes")
    return blob


def parse_data_uri(url: str) -> Optional[Tuple[str, bytes]]:
    """Split a `data:<mime>;base64,<payload>` URI into (mime, bytes)."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    header, _, payload = url[5:].partition(",")
    if not payload:
        return None
    if ";base64" not in header.lower():
        # Percent-encoded data URIs are legal but rare for media; decode them too.
        try:
            from urllib.parse import unquote_to_bytes
            return (header.split(";")[0] or "application/octet-stream",
                    unquote_to_bytes(payload))
        except Exception:  # noqa: BLE001
            return None
    mime = header.split(";")[0].strip() or "application/octet-stream"
    try:
        return mime, _decode_base64(payload, "data URI payload")
    except ValueError:
        return None


async def fetch_media(url: str, client: httpx.AsyncClient) -> Tuple[str, bytes]:
    """Download a remote image. Raises ValueError with a client-safe message."""
    if not FETCH_REMOTE_MEDIA:
        raise ValueError("remote media is disabled on this proxy")
    try:
        response = await client.get(url, timeout=MEDIA_FETCH_TIMEOUT, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise ValueError(f"could not fetch {url}: {type(exc).__name__}") from exc
    if response.status_code != 200:
        raise ValueError(f"could not fetch {url}: upstream HTTP {response.status_code}")
    body = response.content
    if not body:
        raise ValueError(f"could not fetch {url}: empty body")
    if len(body) > MAX_INLINE_BYTES:
        raise ValueError(f"{url} is too large to inline ({len(body)} bytes)")
    mime = (response.headers.get("content-type") or "").split(";")[0].strip()
    if not mime or not mime.startswith("image/"):
        raise ValueError(f"{url} did not return an image (content-type: {mime or 'unknown'})")
    return mime, body


def _image_field(value: Any) -> Optional[Tuple[str, str]]:
    """Normalise an OpenAI image/content field to (url_or_mime, detail)."""
    if isinstance(value, str):
        return value, "base64"
    if isinstance(value, dict):
        # OpenAI: {"url": ..., "detail": ...}; Chat Completions media blocks
        # sometimes use {"data": ..., "mime_type": ...}.
        for key in ("url", "data", "image_url", "source"):
            inner = value.get(key)
            if isinstance(inner, str):
                return inner, value.get("detail") or "base64"
            if isinstance(inner, dict):
                return _image_field(inner)
    return None


# ---------------------------------------------------------------------------
# Tool schema conversion (OpenAI -> Gemini)
# ---------------------------------------------------------------------------


def _clean_schema(node: Any, depth: int = 0) -> Any:
    """Prune a JSON Schema down to what Gemini's Schema object accepts."""
    if depth > 12 or not isinstance(node, dict):
        return node
    cleaned: Dict[str, Any] = {}
    for key, value in node.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS:
            continue
        if key == "type":
            if isinstance(value, list):
                # ["string", "null"] -> "string" + nullable
                real = [v for v in value if isinstance(v, str) and v != "null"]
                if "null" in value:
                    cleaned["nullable"] = True
                if real:
                    cleaned["type"] = real[0]
                continue
            if isinstance(value, str) and value not in _SCHEMA_TYPES:
                # Gemini only knows a fixed set; default to string rather than
                # failing the whole request on an exotic type.
                cleaned["type"] = "string"
                continue
            cleaned["type"] = value
        elif key == "properties" and isinstance(value, dict):
            cleaned["properties"] = {k: _clean_schema(v, depth + 1) for k, v in value.items()}
        elif key == "items":
            cleaned["items"] = _clean_schema(value, depth + 1)
        elif value is False and key in ("nullable",):
            continue
        else:
            cleaned[key] = value
    if "type" not in cleaned and "properties" in cleaned:
        cleaned["type"] = "object"
    return cleaned


def _function_declaration(tool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """OpenAI tool dict -> one Gemini functionDeclaration."""
    if not isinstance(tool, dict):
        return None
    # OpenAI nests under "function"; some clients send the function inline.
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    if tool.get("type") not in (None, "function") and not isinstance(tool.get("function"), dict):
        return None
    name = fn.get("name")
    if not name:
        return None
    declaration: Dict[str, Any] = {"name": name}
    if fn.get("description"):
        declaration["description"] = fn["description"]
    params = fn.get("parameters")
    # Gemini rejects an empty argument list in some model versions, so a
    # parameterless function gets a permissive object schema instead.
    if isinstance(params, dict) and params:
        declaration["parameters"] = _clean_schema(params)
    else:
        declaration["parameters"] = {"type": "object", "properties": {}}
    return declaration


def build_tools(tools: Optional[List[Dict[str, Any]]],
                tool_choice: Any) -> Optional[Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    """Return (tools, toolConfig) for Gemini, or None when tools are unused."""
    if not tools:
        return None
    declarations = [d for d in (_function_declaration(t) for t in tools) if d]
    if not declarations:
        return None

    if isinstance(tool_choice, str) and tool_choice.lower() == "none":
        return None

    tool_config: Optional[Dict[str, Any]] = None
    if isinstance(tool_choice, str):
        mode = {"auto": "AUTO", "required": "ANY", "any": "ANY"}.get(tool_choice.lower())
        if mode:
            tool_config = {"functionCallingConfig": {"mode": mode}}
    elif isinstance(tool_choice, dict):
        chosen = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else tool_choice
        name = chosen.get("name") if isinstance(chosen, dict) else None
        if name:
            tool_config = {"functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": [name],
            }}
    return [{"functionDeclarations": declarations}], tool_config


# ---------------------------------------------------------------------------
# Message conversion (OpenAI -> Gemini)
# ---------------------------------------------------------------------------


async def _content_parts(content: Any, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """Turn OpenAI message content into Gemini `parts`."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []

    parts: List[Dict[str, Any]] = []
    if not isinstance(content, list):
        return [{"text": str(content)}]

    for raw in content:
        if isinstance(raw, str):
            if raw:
                parts.append({"text": raw})
            continue
        if not isinstance(raw, dict):
            continue
        ptype = (raw.get("type") or "").lower()

        if ptype == "text" or (ptype == "" and isinstance(raw.get("text"), str)):
            text = raw.get("text")
            if isinstance(text, str) and text:
                parts.append({"text": text})
            continue

        if ptype in ("image_url", "image", "input_image"):
            field = _image_field(raw.get("image_url") or raw.get("image") or raw)
            if not field:
                raise ValueError("image block is missing a url")
            url, _detail = field
            parsed = parse_data_uri(url)
            if parsed:
                mime, data = parsed
            elif url.startswith(("http://", "https://")):
                mime, data = await fetch_media(url, client)
            else:
                raise ValueError("image_url must be a data: URI or an http(s) URL")
            if len(data) > MAX_INLINE_BYTES:
                raise ValueError(f"image is too large to inline ({len(data)} bytes)")
            parts.append({"inline_data": {
                "mime_type": mime,
                "data": base64.b64encode(data).decode("ascii"),
            }})
            continue

        if ptype in ("input_audio", "audio"):
            block = raw.get("input_audio") or raw
            data = block.get("data") if isinstance(block, dict) else None
            fmt = (block.get("format") if isinstance(block, dict) else None) or "wav"
            if not data:
                raise ValueError("input_audio block is missing data")
            parsed = parse_data_uri(data)
            if parsed:
                mime, blob = parsed
            else:
                mime = {
                    "wav": "audio/wav", "mp3": "audio/mp3",
                    "ogg": "audio/ogg", "flac": "audio/flac",
                }.get(str(fmt).lower(), "audio/wav")
                # Bare base64 (not a data URI): decode strictly, or an empty
                # media part would be forwarded upstream without complaint.
                blob = _decode_base64(data, "input_audio data")
            parts.append({"inline_data": {
                "mime_type": mime,
                "data": base64.b64encode(blob).decode("ascii"),
            }})
            continue

        if ptype in ("video_url", "video"):
            field = _image_field(raw.get("video_url") or raw)
            if not field:
                raise ValueError("video block is missing a url")
            url, _detail = field
            parsed = parse_data_uri(url)
            if parsed:
                mime, data = parsed
            elif url.startswith(("http://", "https://")):
                # Gemini also accepts a public file URI, which avoids inlining.
                parts.append({"file_data": {"mime_type": "video/mp4", "file_uri": url}})
                continue
            else:
                raise ValueError("video_url must be a data: URI or an http(s) URL")
            parts.append({"inline_data": {
                "mime_type": mime,
                "data": base64.b64encode(data).decode("ascii"),
            }})
            continue

        # Unknown block type: fall back to its text if it has any rather than
        # failing the whole request.
        if isinstance(raw.get("text"), str) and raw["text"]:
            parts.append({"text": raw["text"]})

    return parts


async def convert_messages(messages: List[Dict[str, Any]],
                          client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """OpenAI `messages` -> Gemini `contents`.

    Handles assistant tool_calls, role:"tool" results and multimodal content.
    Consecutive same-role turns are merged, because OpenAI's message list can
    interleave several "tool" results where Gemini expects them in one turn.
    """
    contents: List[Dict[str, Any]] = []

    def append(role: str, parts: List[Dict[str, Any]]) -> None:
        if not parts:
            return
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts})

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system" or role == "developer":
            continue  # goes to systemInstruction

        if role == "tool" or message.get("tool_call_id"):
            name = message.get("name") or message.get("tool_call_id") or "tool"
            output = _json_obj(message.get("content"))
            # Gemini pairs a response with the call by name; echo both the
            # raw output and a `result` key so either convention works.
            response_payload = dict(output)
            if "result" not in response_payload:
                response_payload["result"] = message.get("content")
            append("user", [{"functionResponse": {
                "name": name,
                "response": _json_obj(response_payload),
            }}])
            continue

        tool_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            parts: List[Dict[str, Any]] = []
            text = message.get("content")
            if isinstance(text, str) and text:
                parts.append({"text": text})
            elif isinstance(text, list):
                parts.extend(await _content_parts(text, client))
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else call
                name = fn.get("name") or "function"
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args) if args.strip() else {}
                    except (json.JSONDecodeError, ValueError):
                        parsed = {"_raw": args}
                elif isinstance(args, dict):
                    parsed = args
                else:
                    parsed = {}
                part: Dict[str, Any] = {"functionCall": {"name": name, "args": parsed}}
                # Preserve the id (as `id`) so a round trip can be matched.
                if call.get("id"):
                    part["functionCall"]["id"] = call["id"]
                parts.append(part)
            append("model", parts)
            continue

        parts = await _content_parts(message.get("content"), client)
        if message.get("name") and role == "function":
            # Legacy function-role message.
            append("user", [{"functionResponse": {
                "name": message["name"],
                "response": _json_obj(message.get("content")),
            }}])
            continue
        append("user" if role == "user" else "model", parts)

    return contents


def collect_system_instruction(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Gemini has no system role; collect system/developer turns into one."""
    texts: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") in ("system", "developer"):
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        texts.append(part["text"])
    joined = "".join(texts).strip()
    if not joined:
        return None
    return {"parts": [{"text": joined}]}


# ---------------------------------------------------------------------------
# Response conversion (Gemini -> OpenAI)
# ---------------------------------------------------------------------------


def _openai_tool_call(name: str, args: Any, index: int = 0,
                      call_id: Optional[str] = None) -> Dict[str, Any]:
    """Build one OpenAI tool_call from a Gemini functionCall."""
    if isinstance(args, str):
        arguments = args
    else:
        arguments = json.dumps(args if args is not None else {}, ensure_ascii=False)
    return {
        "id": call_id or f"call_{int(time.time() * 1000)}_{index}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _usage_from(metadata: Optional[Dict[str, Any]]) -> Dict[str, int]:
    metadata = metadata or {}
    prompt = metadata.get("promptTokenCount") or 0
    completion = metadata.get("candidatesTokenCount") or 0
    total = metadata.get("totalTokenCount") or (prompt + completion)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _parts_of(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    return ((candidate.get("content") or {}).get("parts") or [])


def parse_response_parts(candidate: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Split one Gemini candidate into (text, tool_calls)."""
    text_chunks: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for part in _parts_of(candidate):
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str):
            text_chunks.append(part["text"])
        call = part.get("functionCall")
        if isinstance(call, dict):
            tool_calls.append(_openai_tool_call(
                call.get("name") or "function",
                call.get("args"),
                len(tool_calls),
                call.get("id"),
            ))
    return "".join(text_chunks), tool_calls


def convert_gemini_to_openai_response(gemini_response: dict, model: str) -> dict:
    """Convert a Gemini `generateContent` reply to an OpenAI completion."""
    candidates = gemini_response.get("candidates") or [{}]
    candidate = candidates[0] if isinstance(candidates[0], dict) else {}
    text, tool_calls = parse_response_parts(candidate)

    finish_reason = "stop"
    if tool_calls:
        finish_reason = "tool_calls"
    raw_finish = (candidate.get("finishReason") or "").upper()
    if raw_finish in ("MAX_TOKENS", "MAX_OUTPUT_TOKENS"):
        finish_reason = "length"
    elif raw_finish in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"):
        finish_reason = "content_filter"

    # OpenAI always includes `content`; it is null/empty when the model only
    # asked for a tool call.
    message: Dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "usage": _usage_from(gemini_response.get("usageMetadata")),
        "choices": [{
            "message": message,
            "finish_reason": finish_reason,
            "index": 0,
        }],
    }


def _chunk(model: str, delta: Dict[str, Any], finish_reason: Optional[str] = None,
           usage: Optional[Dict[str, int]] = None) -> str:
    payload: Dict[str, Any] = {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _strip_closers(text: str) -> str:
    """Make a serialised arguments snapshot 'open', so snapshots form a prefix chain.

    Gemini streams functionCall args cumulatively, so a later snapshot extends an
    earlier one -- but the raw strings do not show it, because JSON closes every
    value and object it emits:

        {"city": "Par"}    ->  {"city": "Paris"}
        {"city": "Paris"}  ->  {"city": "Paris", "unit": "c"}

    In the first pair the value's closing quote differs, in the second the
    object's closing brace does. Dropping trailing `}`/`]` and a trailing `"`
    leaves the still-open form, which DOES extend: `{"city": "Par"` -> `{"city":
    "Paris"`. Deltas are the difference between successive open forms, and the
    dropped closers are re-emitted once the stream ends, so a client that simply
    concatenates deltas still rebuilds the final JSON exactly.
    """
    text = text.rstrip("}]")
    if text.endswith('"'):
        text = text[:-1]
    return text


async def stream_gemini_response(model: str, payload: dict, api_key: str,
                                 include_usage: bool = False):
    """Relay Gemini's streaming response as OpenAI-style SSE chunks.

    Gemini streaming emits cumulative functionCall args (and repeats the call),
    so only newly-appeared argument text is forwarded as a delta.
    """
    # Gemini does not put an index on a streamed functionCall part: it re-sends
    # the call with longer (cumulative) args. Calls are therefore identified by
    # their POSITION among the functionCall parts of a chunk, which is how
    # Gemini orders parallel calls; an explicit `index`, if ever present, wins.
    slots: List[Dict[str, Any]] = []
    last_usage: Dict[str, int] = _usage_from(None)

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
                yield "data: " + json.dumps(
                    {"error": {"message": error_body or "Gemini upstream error",
                               "code": response.status_code}}) + "\n\n"
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

                if obj.get("usageMetadata"):
                    last_usage = _usage_from(obj["usageMetadata"])

                for candidate in obj.get("candidates", []):
                    position = 0
                    for part in _parts_of(candidate):
                        if not isinstance(part, dict):
                            continue
                        text_content = part.get("text")
                        if text_content:
                            yield _chunk(model, {"content": text_content})

                        call = part.get("functionCall")
                        if not isinstance(call, dict):
                            continue
                        index = call.get("index")
                        if not isinstance(index, int):
                            index = position
                        position += 1
                        arguments = call.get("args")
                        arguments = arguments if isinstance(arguments, str) else json.dumps(
                            arguments if arguments is not None else {}, ensure_ascii=False)
                        name = call.get("name") or "function"

                        # First sighting of this call: open a slot, announce the
                        # name and id, but send no arguments yet.
                        if index >= len(slots):
                            slots.extend({"name": None, "id": None, "args": "", "sent": ""}
                                         for _ in range(index - len(slots) + 1))
                        slot = slots[index]
                        if slot["name"] is None:
                            slot["name"] = name
                            slot["id"] = call.get("id") or f"call_{int(time.time() * 1000)}_{index}"
                            yield _chunk(model, {"tool_calls": [{
                                "index": index,
                                "id": slot["id"],
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }]})

                        # Only newly-appeared text becomes a delta, so a client
                        # that concatenates deltas rebuilds the final arguments.
                        slot["args"] = arguments
                        wanted = _strip_closers(arguments)
                        sent = slot["sent"]
                        if wanted == sent:
                            continue  # unchanged repeat
                        if wanted.startswith(sent):
                            delta_args = wanted[len(sent):]
                        else:
                            # Not cumulative (or it shrank): restart this call's
                            # argument text from the new snapshot.
                            delta_args = wanted
                        slot["sent"] = wanted
                        if not delta_args:
                            continue
                        yield _chunk(model, {"tool_calls": [{
                            "index": index,
                            "function": {"arguments": delta_args},
                        }]})

    # Close every argument object that was left open: the closers are held back
    # while the args are still growing, and emitted once the stream ends.
    for index, slot in enumerate(slots):
        final = slot["args"]
        sent = slot["sent"]
        if final and final != sent and final.startswith(sent):
            yield _chunk(model, {"tool_calls": [{
                "index": index,
                "function": {"arguments": final[len(sent):]},
            }]})
            slot["sent"] = final

    finish_reason = "tool_calls" if slots else "stop"
    yield _chunk(model, {}, finish_reason, usage=last_usage if include_usage else None)
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
        "maxOutputTokens": args.max_tokens if args.max_tokens is not None
        else args.max_completion_tokens,
        "topP": args.top_p,
        "topK": 10,
    }
    if args.stop:
        stops = args.stop if isinstance(args.stop, list) else [args.stop]
        stops = [s for s in stops if isinstance(s, str) and s]
        if stops:
            generation_config["stopSequences"] = stops[:5]
    generation_config = {k: v for k, v in generation_config.items() if v is not None}

    # Remote images are fetched with a bounded timeout; the conversion itself
    # can fail on bad base64 or an unfetchable URL, which is a client error.
    try:
        async with httpx.AsyncClient(timeout=MEDIA_FETCH_TIMEOUT) as media_client:
            contents = await convert_messages(args.messages, media_client)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    gemini_payload: Dict[str, Any] = {
        "contents": contents,
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ],
        "generationConfig": generation_config,
    }

    system_instruction = collect_system_instruction(args.messages)
    if system_instruction:
        gemini_payload["systemInstruction"] = system_instruction

    built_tools = build_tools(args.tools, args.tool_choice)
    if built_tools:
        tools, tool_config = built_tools
        gemini_payload["tools"] = tools
        if tool_config:
            gemini_payload["toolConfig"] = tool_config

    if args.stream:
        include_usage = bool((args.stream_options or {}).get("include_usage"))
        return StreamingResponse(
            stream_gemini_response(model, gemini_payload, api_key, include_usage),
            media_type="text/event-stream",
        )
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
