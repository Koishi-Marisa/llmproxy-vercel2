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
    query = "Count from 1 to 5"

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

        for i in range(1, 6):
            assert str(
                i) in content, f"Expected {i} in content, but it's missing. Content: {content}"

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


@pytest.mark.asyncio
async def test_openai_streaming():
    await make_request(
        supplier="openai",
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-3.5-turbo"
    )


@pytest.mark.asyncio
async def test_groq_streaming():
    await make_request(
        supplier="groq",
        api_key=os.environ["GROQ_API_KEY"],
        model="llama3-70b-8192"
    )


@pytest.mark.asyncio
async def test_gemini_streaming():
    await make_request(
        supplier="gemini",
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-1.5-flash"
    )


@pytest.mark.asyncio
async def test_cerebras_streaming():
    await make_request(
        supplier="cerebras",
        api_key=os.environ["CEREBRAS_API_KEY"],
        model="llama3.1-8b"
    )


@pytest.mark.asyncio
async def test_nvidia_streaming():
    await make_request(
        supplier="nvidia",
        api_key=os.environ["NVIDIA_API_KEY"],
        model="meta/llama-3.2-3b-instruct"
    )


@pytest.mark.asyncio
async def test_mistral():
    await make_request(
        supplier="mistral",
        api_key=os.environ["MISTRAL_API_KEY"],
        model="mistral-large-latest",
    )


@pytest.mark.asyncio
async def test_sambanova():
    await make_request(
        supplier="sambanova",
        api_key=os.environ["SAMBANOVA_API_KEY"],
        model="Meta-Llama-3.1-405B-Instruct",
    )
