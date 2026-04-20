import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vlm_annotate_verify.proposer.gemini import (
    GeminiConfig, GeminiError, MODEL_FLASH, MODEL_PRO,
    call_gemini_video, make_config,
)


@pytest.fixture
def fake_video(tmp_path) -> Path:
    p = tmp_path / "ep.mp4"
    p.write_bytes(b"\x00\x00\x00 ftyp")  # minimal mp4-ish bytes
    return p


def _client_returning(text: str) -> MagicMock:
    """Build a mock genai.Client whose models.generate_content returns `text`."""
    client = MagicMock()
    response = MagicMock()
    response.text = text
    client.models.generate_content.return_value = response
    return client


def test_make_config_uses_env_var(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg = make_config()
    assert cfg.api_key == "test-key"


def test_make_config_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        make_config()


def test_make_config_explicit_key_overrides_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    cfg = make_config(api_key="explicit")
    assert cfg.api_key == "explicit"


def test_call_gemini_video_returns_response_text(fake_video):
    client = _client_returning('{"task": "x"}')
    cfg = GeminiConfig(api_key="test")
    with patch("vlm_annotate_verify.proposer.gemini.genai.Client", return_value=client):
        out = asyncio.run(
            call_gemini_video(cfg, MODEL_FLASH, fake_video, "do the thing")
        )
    assert out == '{"task": "x"}'
    args, kwargs = client.models.generate_content.call_args
    assert kwargs["model"] == MODEL_FLASH


def test_call_gemini_video_uses_pro_model(fake_video):
    client = _client_returning("ok")
    cfg = GeminiConfig(api_key="test")
    with patch("vlm_annotate_verify.proposer.gemini.genai.Client", return_value=client):
        asyncio.run(call_gemini_video(cfg, MODEL_PRO, fake_video, "p"))
    args, kwargs = client.models.generate_content.call_args
    assert kwargs["model"] == MODEL_PRO


def test_call_gemini_video_retries_on_transient_error(fake_video, monkeypatch):
    from google.genai.errors import APIError

    client = MagicMock()
    boom = APIError(code=500, response_json={}, response=MagicMock())
    success = MagicMock()
    success.text = "ok"
    client.models.generate_content.side_effect = [boom, boom, success]

    sleep_calls: list[float] = []

    async def fake_sleep(s):
        sleep_calls.append(s)

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    cfg = GeminiConfig(api_key="test", max_retries=3, base_delay_s=1.0)
    with patch("vlm_annotate_verify.proposer.gemini.genai.Client", return_value=client):
        out = asyncio.run(
            call_gemini_video(cfg, MODEL_FLASH, fake_video, "x")
        )
    assert out == "ok"
    assert client.models.generate_content.call_count == 3
    # Exponential backoff: 1s then 4s
    assert sleep_calls[:2] == [1.0, 4.0]


def test_call_gemini_video_429_uses_rate_limit_floor(fake_video, monkeypatch):
    from google.genai.errors import ClientError

    client = MagicMock()
    rate = ClientError(code=429, response_json={}, response=MagicMock())
    success = MagicMock()
    success.text = "ok"
    client.models.generate_content.side_effect = [rate, success]

    sleep_calls: list[float] = []

    async def fake_sleep(s):
        sleep_calls.append(s)

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    cfg = GeminiConfig(
        api_key="test", max_retries=3, base_delay_s=1.0, rate_limit_floor_s=30.0,
    )
    with patch("vlm_annotate_verify.proposer.gemini.genai.Client", return_value=client):
        asyncio.run(call_gemini_video(cfg, MODEL_FLASH, fake_video, "x"))
    # First (and only) sleep should be at least the rate-limit floor
    assert sleep_calls[0] >= 30.0


def test_call_gemini_video_raises_after_max_retries(fake_video, monkeypatch):
    from google.genai.errors import APIError

    client = MagicMock()
    boom = APIError(code=500, response_json={}, response=MagicMock())
    client.models.generate_content.side_effect = [boom, boom, boom]

    async def fake_sleep(_):
        pass

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    cfg = GeminiConfig(api_key="test", max_retries=3, base_delay_s=0.01)
    with patch("vlm_annotate_verify.proposer.gemini.genai.Client", return_value=client):
        with pytest.raises(GeminiError, match="failed after"):
            asyncio.run(call_gemini_video(cfg, MODEL_FLASH, fake_video, "x"))
    assert client.models.generate_content.call_count == 3
