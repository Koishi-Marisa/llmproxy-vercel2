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


async def make_request(api_key: str,
                       model: str,
                       supplier: str,
                       query: str = "The first president of the United States, give me his full name and only his full name"):
    client = AsyncOpenAI(base_url=BASE_URL + f"/{supplier}", api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant。"},
            {"role": "user", "content": query}
        ],
        temperature=0.7,
        top_p=1,
        max_tokens=20
    )
    print(type(response), response)
    content = response.choices[0].message.content
    assert "George Washington" in content, f"Expected 'George Washington' in content, but got {content}"
    return content


@pytest.mark.asyncio
async def test_groq():
    await make_request(
        supplier="groq",
        api_key=os.environ["GROQ_API_KEY"],
        model="llama3-70b-8192"
    )


@pytest.mark.asyncio
async def test_openai():
    await make_request(
        supplier="openai",
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o-mini"
    )


@pytest.mark.asyncio
async def test_gemini():
    await make_request(
        supplier="gemini",
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-1.5-flash"
    )


@pytest.mark.asyncio
async def test_cerebras():
    await make_request(
        supplier="cerebras",
        api_key=os.environ["CEREBRAS_API_KEY"],
        model="llama3.1-8b"
    )


@pytest.mark.asyncio
async def test_nvidia():
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
