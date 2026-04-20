import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from vlm_annotate_verify.proposer.gemini import GeminiConfig, MODEL_FLASH
from vlm_annotate_verify.proposer.passes import (
    PassError, pass1_boundaries, pass2_labels,
)
from vlm_annotate_verify.schemas import Boundary


@pytest.fixture
def fake_video(tmp_path) -> Path:
    p = tmp_path / "ep.mp4"
    p.write_bytes(b"\x00")
    return p


def _async_value(value):
    async def _inner(*_args, **_kwargs):
        return value
    return _inner


def test_pass1_returns_task_and_boundaries(fake_video):
    raw = json.dumps({
        "task": "place cube on shelf",
        "boundaries": [
            {"t_s": 5.2, "label": "grasp",    "confidence": 0.91},
            {"t_s": 8.4, "label": "transfer", "confidence": 0.78},
        ],
    })
    cfg = GeminiConfig(api_key="x")
    with patch(
        "vlm_annotate_verify.proposer.passes.call_gemini_video",
        new=_async_value(raw),
    ):
        task, boundaries = asyncio.run(pass1_boundaries(cfg, fake_video, MODEL_FLASH))
    assert task == "place cube on shelf"
    assert len(boundaries) == 2
    assert boundaries[0] == Boundary(t_s=5.2, label="grasp", confidence=0.91)


def test_pass1_malformed_json_raises(fake_video):
    cfg = GeminiConfig(api_key="x")
    with patch(
        "vlm_annotate_verify.proposer.passes.call_gemini_video",
        new=_async_value("{not json"),
    ):
        with pytest.raises(PassError, match="malformed"):
            asyncio.run(pass1_boundaries(cfg, fake_video, MODEL_FLASH))


def test_pass1_missing_field_raises(fake_video):
    cfg = GeminiConfig(api_key="x")
    raw = json.dumps({"boundaries": []})
    with patch(
        "vlm_annotate_verify.proposer.passes.call_gemini_video",
        new=_async_value(raw),
    ):
        with pytest.raises(PassError, match="malformed"):
            asyncio.run(pass1_boundaries(cfg, fake_video, MODEL_FLASH))


def test_pass2_returns_segments(fake_video):
    boundaries = [Boundary(t_s=5.2, label="grasp", confidence=0.91)]
    raw = json.dumps({
        "segments": [
            {
                "idx": 0,
                "start_s": 0.0, "end_s": 5.2,
                "quality": 4, "quality_conf": 0.82,
                "success": True, "success_conf": 0.95,
                "mistakes": [
                    {"type": "slip", "t_s": 3.2, "note": "wobble", "confidence": 0.65},
                ],
                "reasoning": "smooth grasp",
            },
        ],
    })
    cfg = GeminiConfig(api_key="x")
    with patch(
        "vlm_annotate_verify.proposer.passes.call_gemini_video",
        new=_async_value(raw),
    ):
        segments = asyncio.run(
            pass2_labels(cfg, fake_video, boundaries, 12.0, MODEL_FLASH)
        )
    assert len(segments) == 1
    assert segments[0].quality == 4
    assert segments[0].mistakes[0].type == "slip"
    assert segments[0].reasoning == "smooth grasp"


def test_pass2_segment_without_mistakes_or_reasoning_ok(fake_video):
    raw = json.dumps({
        "segments": [
            {
                "idx": 0,
                "start_s": 0.0, "end_s": 1.0,
                "quality": 3, "quality_conf": 0.5,
                "success": True, "success_conf": 0.5,
            },
        ],
    })
    cfg = GeminiConfig(api_key="x")
    with patch(
        "vlm_annotate_verify.proposer.passes.call_gemini_video",
        new=_async_value(raw),
    ):
        segments = asyncio.run(
            pass2_labels(cfg, fake_video, [], 1.0, MODEL_FLASH)
        )
    assert segments[0].mistakes == []
    assert segments[0].reasoning == ""


def test_pass2_malformed_raises(fake_video):
    cfg = GeminiConfig(api_key="x")
    with patch(
        "vlm_annotate_verify.proposer.passes.call_gemini_video",
        new=_async_value("not json at all"),
    ):
        with pytest.raises(PassError):
            asyncio.run(pass2_labels(cfg, fake_video, [], 1.0, MODEL_FLASH))
