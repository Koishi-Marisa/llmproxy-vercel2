import asyncio
import pytest
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import random
from loguru import logger
from config import PRODUCTION_API_ENDPOINT, DEVELOPMENT_API_ENDPOINT
load_dotenv()


def api_endpoint():
    env = os.environ.get('ENV', 'development')
    if env == 'production':
        return PRODUCTION_API_ENDPOINT
    elif env == 'development':
        return DEVELOPMENT_API_ENDPOINT
    else:
        raise ValueError(f"Invalid environment: {env}")


BASE_URL = api_endpoint()
logger.info(f"BASE_URL: {BASE_URL}")


async def make_request(supplier: str, api_key: str, model: str):
    BASE_URL = api_endpoint() + f"/{supplier}"
    query = "用汉字从一数到十，如一，二，三，四，五，..."

    client = AsyncOpenAI(base_url=BASE_URL, api_key=api_key)

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            stream=True,
        )

        content = ""
        async for chunk in stream:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                content += delta_content
                print(f"Received chunk: {delta_content}")  # Debug print

        print(f"Full content: {content}")  # Debug print

        if not content:
            raise ValueError("Received empty content from API")

        for word in ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]:
            assert word in content, f"Expected '{word}' in content, but it's missing. Content: {content}"

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


@pytest.mark.asyncio
async def test_gemini_streaming():
    await make_request(
        supplier="gemini",
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-1.5-flash"
    )
