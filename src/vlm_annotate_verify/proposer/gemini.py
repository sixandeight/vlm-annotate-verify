"""Async Gemini client wrapping google-genai with retry plus 429 backoff."""
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

MODEL_FLASH = "gemini-2.5-flash"
MODEL_PRO = "gemini-2.5-pro"


class GeminiError(Exception):
    pass


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    max_retries: int = 3
    base_delay_s: float = 1.0
    rate_limit_floor_s: float = 30.0


def make_config(api_key: str | None = None) -> GeminiConfig:
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise GeminiError("GEMINI_API_KEY not set")
    return GeminiConfig(api_key=key)


async def call_gemini_video(
    config: GeminiConfig,
    model: str,
    video_path: Path,
    prompt: str,
    response_schema: dict | None = None,
) -> str:
    """Send video plus prompt to Gemini, return raw text response.

    Retries up to config.max_retries on transient errors with exponential backoff.
    On 429, the next backoff is at least config.rate_limit_floor_s seconds.
    """
    client = genai.Client(api_key=config.api_key)
    video_bytes = video_path.read_bytes()
    parts = [
        types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
        types.Part.from_text(text=prompt),
    ]
    gen_config = types.GenerateContentConfig(
        response_mime_type="application/json",
    )
    if response_schema is not None:
        gen_config.response_schema = response_schema

    delay = config.base_delay_s
    last_err: Exception | None = None
    for _ in range(config.max_retries):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=parts,
                config=gen_config,
            )
            return response.text
        except ClientError as e:
            if getattr(e, "code", None) == 429:
                delay = max(delay, config.rate_limit_floor_s)
            last_err = e
        except APIError as e:
            last_err = e
        await asyncio.sleep(delay)
        delay *= 4
    raise GeminiError(
        f"Gemini call failed after {config.max_retries} retries: {last_err}"
    )
