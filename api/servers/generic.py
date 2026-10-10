from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import httpx
from typing import Dict
from .base import stream_openai_response, OpenAIProxyArgs

router = APIRouter()

PLATFORM_API_URLS: Dict[str, str] = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "cerebras": "https://api.cerebras.ai/v1/chat/completions",
    "nvidia": "https://integrate.api.nvidia.com/v1/chat/completions",
    "sambanova": "https://api.sambanova.ai/v1/chat/completions",
}


@router.post("/{platform}/chat/completions")
async def proxy_chat_completions(platform: str, args: OpenAIProxyArgs, authorization: str = Header(...)):
    if platform not in PLATFORM_API_URLS:
        raise HTTPException(
            status_code=404, detail=f"Platform '{platform}' not supported")

    api_url = PLATFORM_API_URLS[platform]
    api_key = authorization.split(" ")[1]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = args.dict(exclude_none=True)

    if args.stream:
        return StreamingResponse(
            stream_openai_response(api_url, payload, headers),
            media_type="text/event-stream",
            headers={"X-Content-Type-Options": "nosniff",
                     "X-Experimental-Stream-Data": "true"}
        )
    else:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                return JSONResponse(response.json())
            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=e.response.status_code, detail=str(e.response.text))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
