import json
import os
import httpx
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Request models
#
# IMPORTANT: these models are deliberately *permissive*. Real-world clients
# (OpenAI SDK, Cherry Studio, LobeChat, ChatGPT-Next-Web, Cursor, ...) send
# fields such as `seed`, `stop`, `tools`, `tool_choice`, `response_format`,
# `stream_options`, `reasoning_effort` or provider-specific options. The old
# models silently rejected them with HTTP 422, which broke those clients even
# though the upstream platform supports the field. Anything we don't declare
# is collected into `model_extra` and forwarded untouched (see
# `build_payload`).
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """Kept for backwards compatibility: `api/servers/gemini.py` imports this."""
    role: str
    content: Any


class OpenAIProxyArgs(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False
    # Not a real upstream field: makes Groq/OpenAI emit the final usage chunk.
    stream_options: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    logit_bias: Optional[Dict[str, Any]] = None
    parallel_tool_calls: Optional[bool] = None
    user: Optional[str] = None
    reasoning_effort: Optional[str] = None
    modalities: Optional[List[str]] = None
    # OpenAI SDK sends this when using the Responses-style preview; harmless
    # for most providers, but present so it is not rejected.
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


# Fields that are transport/routing concerns for *this* proxy and must never be
# forwarded verbatim into the request body because we rebuild them ourselves.
_RESERVED_FIELDS = {"raw_request"}

# HTTP-level knobs. Read from the environment so a slow platform can be tuned
# on Vercel without a code change.
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


STREAM_TIMEOUT = httpx.Timeout(
    connect=_env_float("PROXY_CONNECT_TIMEOUT", 15.0),
    read=_env_float("PROXY_STREAM_READ_TIMEOUT", 300.0),
    write=_env_float("PROXY_WRITE_TIMEOUT", 60.0),
    pool=_env_float("PROXY_POOL_TIMEOUT", 15.0),
)
REQUEST_TIMEOUT = httpx.Timeout(
    connect=_env_float("PROXY_CONNECT_TIMEOUT", 15.0),
    read=_env_float("PROXY_READ_TIMEOUT", 300.0),
    write=_env_float("PROXY_WRITE_TIMEOUT", 60.0),
    pool=_env_float("PROXY_POOL_TIMEOUT", 15.0),
)

# Upstream headers we must not copy onto our own response.
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-encoding",
    "content-length",
}


def build_payload(args: OpenAIProxyArgs) -> Dict[str, Any]:
    """Build the upstream JSON body.

    Declared fields are included only when currently set; undeclared fields
    (pydantic `model_extra`) are forwarded as-is so provider-specific options
    keep working.
    """
    payload = args.dict(exclude_none=True)
    extra = getattr(args, "__pydantic_extra__", None)
    if extra:
        for key, value in extra.items():
            if key in _RESERVED_FIELDS or value is None:
                continue
            # Never let a stray body field override our own routing.
            if key not in payload:
                payload[key] = value
    return payload


def parse_api_key(authorization: Optional[str], x_api_key: Optional[str]) -> str:
    """Extract the upstream API key from the incoming request headers.

    Accepts `Authorization: Bearer <key>`, a bare `Authorization: <key>` and
    `x-api-key: <key>`. Raises ValueError for anything malformed so the caller
    can answer 401 instead of leaking an IndexError as a 500.
    """
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    if not authorization or not authorization.strip():
        raise ValueError("missing credential")
    raw = authorization.strip()
    parts = raw.split(None, 1)
    if len(parts) == 2:
        scheme, token = parts[0], parts[1].strip()
        if scheme.lower() != "bearer" or not token:
            raise ValueError("expected 'Authorization: Bearer <api_key>'")
        return token
    if " " not in raw and raw:
        # Bare key, no scheme.
        return raw
    raise ValueError("expected 'Authorization: Bearer <api_key>'")


def upstream_headers(api_key: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        # Some upstreams (and Cloudflare in front of them) reject requests
        # without a User-Agent.
        "User-Agent": "llmproxy-vercel/1.0",
    }
    if extra:
        headers.update(extra)
    return headers


def redact(text: str, secret: Optional[str]) -> str:
    """Remove a credential from an error string before returning it to a client."""
    if not text:
        return text
    out = text
    if secret:
        out = out.replace(secret, "***")
    return out


def _bearer_secret(headers: Dict[str, str]) -> Optional[str]:
    value = headers.get("Authorization") or headers.get("authorization") or ""
    if value.lower().startswith("bearer "):
        token = value[7:].strip()
        return token or None
    return value.strip() or None


def sse_error(message: str, error_type: str = "upstream_error") -> str:
    """Build a well-formed SSE error event (JSON-encoded, so safe for any text)."""
    return "data: " + json.dumps({"error": {"message": message, "type": error_type}}) + "\n\n"


async def stream_upstream(
    endpoint: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: httpx.Timeout = STREAM_TIMEOUT,
):
    """Relay an SSE stream from `endpoint` to the client.

    The upstream is opened with `client.stream(...)` so the body is *not*
    buffered. Upstream errors are surfaced to logs and relayed as an SSE error
    event, because the HTTP status has already been sent as 200 by then.
    """
    secret = _bearer_secret(headers)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        try:
            async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    detail = redact(body.decode("utf-8", "replace"), secret)
                    print(f"[llmproxy] upstream stream error {response.status_code}: {detail[:500]}", flush=True)
                    yield sse_error(detail[:500] or f"upstream returned HTTP {response.status_code}")
                    return
                async for chunk in response.aiter_raw():
                    if chunk:
                        yield chunk
        except httpx.HTTPError as exc:
            detail = redact(str(exc), secret)
            print(f"[llmproxy] upstream stream transport error: {detail}", flush=True)
            yield sse_error(f"upstream transport error: {detail[:300]}")
        except Exception as exc:  # noqa: BLE001 - never crash the SSE generator
            detail = redact(str(exc), secret)
            print(f"[llmproxy] upstream stream unexpected error: {detail}", flush=True)
            yield sse_error(f"proxy error: {detail[:300]}", "proxy_error")
