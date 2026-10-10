import httpx
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .base import (
    REQUEST_TIMEOUT,
    OpenAIProxyArgs,
    build_payload,
    parse_api_key,
    redact,
    stream_upstream,
    upstream_headers,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Upstream chat-completions endpoints.
#
# These MUST be complete URLs including `/chat/completions`. A base URL such as
# "https://api.groq.com/openai/v1" makes the proxy POST to the API root, which
# the platform answers with an empty `404` -- indistinguishable to the client
# from "route not found". Groq regressed exactly this way; keep the suffixes.
# ---------------------------------------------------------------------------
PLATFORM_API_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "cerebras": "https://api.cerebras.ai/v1/chat/completions",
    "nvidia": "https://integrate.api.nvidia.com/v1/chat/completions",
    "sambanova": "https://api.sambanova.ai/v1/chat/completions",
}

# Extra path spellings accepted for compatibility, mapped to a canonical
# platform. OpenAI-compatible clients commonly append `/v1` to a configured
# base_url, so `/groq/v1/chat/completions` must resolve to the Groq upstream.
PLATFORM_ALIASES = {
    "deepseek": "openai",
    "moonshot": "openai",
    "v1": "groq",
}

# `/v1/chat/completions` (and `/openai/v1/...`) with no platform segment: a
# client pointed at the bare proxy root. Default to Groq, which is what this
# deployment exists for.
DEFAULT_PLATFORM = "groq"


def resolve_platform(platform: str) -> str:
    name = platform.strip("/").lower()
    if name in PLATFORM_API_URLS:
        return name
    if name in PLATFORM_ALIASES:
        return PLATFORM_ALIASES[name]
    raise HTTPException(
        status_code=404,
        detail=(
            f"Platform '{platform}' not supported. "
            f"Supported: {', '.join(sorted(PLATFORM_API_URLS))}"
        ),
    )


@router.post("/{platform}/chat/completions")
@router.post("/{platform}/v1/chat/completions")
async def proxy_chat_completions(
    platform: str,
    args: OpenAIProxyArgs,
    authorization: str = Header(None),
    x_api_key: str = Header(None, alias="x-api-key"),
):
    canonical = resolve_platform(platform)
    api_url = PLATFORM_API_URLS[canonical]

    try:
        api_key = parse_api_key(authorization, x_api_key)
    except ValueError as exc:
        raise HTTPException(
            status_code=401,
            detail=f"{exc}. Send 'Authorization: Bearer <your {canonical} api key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    headers = upstream_headers(api_key)
    payload = build_payload(args)

    if args.stream:
        return StreamingResponse(
            stream_upstream(api_url, payload, headers),
            media_type="text/event-stream",
            headers={
                "X-Content-Type-Options": "nosniff",
                "X-Experimental-Stream-Data": "true",
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
            },
        )

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        try:
            response = await client.post(api_url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"Upstream request failed: {redact(str(exc), api_key)}"
            )

    # Pass the upstream status and body straight through so clients see the
    # real Groq error (401 invalid key, 429 rate limit, 422 bad model, ...)
    # instead of a generic 500.
    content_type = response.headers.get("content-type", "application/json")
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=content_type,
    )


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def proxy_chat_completions_default(
    args: OpenAIProxyArgs,
    authorization: str = Header(None),
    x_api_key: str = Header(None, alias="x-api-key"),
):
    """Bare-root compatibility: `/v1/chat/completions` defaults to Groq."""
    return await proxy_chat_completions(
        platform=DEFAULT_PLATFORM,
        args=args,
        authorization=authorization,
        x_api_key=x_api_key,
    )


@router.get("/{platform}/models")
async def list_models(
    platform: str,
    authorization: str = Header(None),
    x_api_key: str = Header(None, alias="x-api-key"),
):
    """Convenience pass-through so clients can enumerate models through us."""
    canonical = resolve_platform(platform)
    chat_url = PLATFORM_API_URLS[canonical]
    models_url = chat_url[: -len("/chat/completions")] + "/models"

    try:
        api_key = parse_api_key(authorization, x_api_key)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc),
                            headers={"WWW-Authenticate": "Bearer"})

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        try:
            response = await client.get(models_url, headers=upstream_headers(api_key))
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"Upstream request failed: {redact(str(exc), api_key)}"
            )

    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )
