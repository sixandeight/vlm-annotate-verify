"""End-to-end smoke: mp4 fixtures -> proposer (mocked gemini) -> verified -> export."""
import asyncio
import csv
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from vlm_annotate_verify.config import load_config
from vlm_annotate_verify.exporter.pi_data_sharing import export
from vlm_annotate_verify.proposer.batch import BatchPaths, run_batch
from vlm_annotate_verify.proposer.gemini import GeminiConfig, MODEL_FLASH
from vlm_annotate_verify.schemas import Boundary, Segment, Verified, Review, utc_now_iso


FIXTURE_DATASET = Path(__file__).parent / "fixtures" / "mini_dataset"


def _copy_dataset(tmp_path: Path) -> Path:
    out = tmp_path / "mini_dataset"
    shutil.copytree(FIXTURE_DATASET, out)
    return out


def _async_value(value):
    async def _inner(*_args, **_kwargs):
        return value
    return _inner


def _mock_pass1_by_ep(video_path, *_args, **_kwargs):
    # Returns (task, boundaries) keyed off the ep filename stem so different
    # episodes get plausibly different splits.
    ep = Path(video_path).stem
    return {
        "ep_000": ("place cube on shelf",
                   [Boundary(t_s=2.0, label="grasp", confidence=0.91),
                    Boundary(t_s=4.0, label="transfer", confidence=0.88)]),
        "ep_001": ("pick up marker",
                   [Boundary(t_s=2.0, label="grasp", confidence=0.93)]),
        "ep_002": ("stack blocks",
                   [Boundary(t_s=3.0, label="grasp", confidence=0.80),
                    Boundary(t_s=5.0, label="transfer", confidence=0.85)]),
    }[ep]


def _mock_pass2_by_ep(video_path, *_args, **_kwargs):
    ep = Path(video_path).stem
    quality_map = {"ep_000": 4, "ep_001": 5, "ep_002": 3}
    return [Segment(
        idx=0, start_s=0.0, end_s=2.0,
        quality=quality_map[ep], quality_conf=0.85,
        success=True, success_conf=0.95,
        mistakes=[], reasoning="ok",
    )]


def _fake_pass1(config, video_path, model):
    return _mock_pass1_by_ep(video_path)


async def _fake_pass1_async(config, video_path, model):
    return _mock_pass1_by_ep(video_path)


async def _fake_pass2_async(config, video_path, boundaries, duration_s, model):
    return _mock_pass2_by_ep(video_path)


def _auto_accept_all_to_verified(root: Path) -> None:
    """Simulate Phase B by turning every proposal into a verified record."""
    proposals = (root / "proposals.jsonl").read_text().splitlines()
    lines = []
    for p in proposals:
        d = json.loads(p)
        if d.get("status") == "failed":
            continue
        verified = {
            "ep_id": d["ep_id"],
            "task": d["task"],
            "boundaries": d["boundaries"],
            "segments": d["segments"],
            "review": {
                "reviewer_id": "nathan",
                "review_seconds": 3.0,
                "actions": ["accept"],
                "reprompt_used": False,
            },
            "verified_at": utc_now_iso(),
        }
        lines.append(json.dumps(verified))
    (root / "verified.jsonl").write_text("\n".join(lines) + "\n")


def test_end_to_end_propose_verify_export(tmp_path):
    root = _copy_dataset(tmp_path)
    paths = BatchPaths(dataset_root=root)
    cfg = GeminiConfig(api_key="test-key")

    with patch("vlm_annotate_verify.proposer.batch.pass1_boundaries",
               new=_fake_pass1_async):
        with patch("vlm_annotate_verify.proposer.batch.pass2_labels",
                   new=_fake_pass2_async):
            asyncio.run(run_batch(paths, cfg, model=MODEL_FLASH, concurrency=3))

    # Proposals file has one row per episode
    prop_lines = (root / "proposals.jsonl").read_text().strip().splitlines()
    assert len(prop_lines) == 3
    ep_ids = {json.loads(line)["ep_id"] for line in prop_lines}
    assert ep_ids == {"ep_000", "ep_001", "ep_002"}

    # Frames cached on disk
    for ep in ep_ids:
        frames = list((root / "frames" / ep).glob("*.jpg"))
        assert len(frames) == 16

    # Simulate Phase B auto-accept (skips the TUI; verified.jsonl mirrors proposals)
    _auto_accept_all_to_verified(root)

    # Export writes CSV + JSON
    config = load_config(root)
    csv_path, json_path = export(root, config)
    assert csv_path.exists()
    assert json_path.exists()

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    tasks = {r["task"] for r in rows}
    assert tasks == {"place cube on shelf", "pick up marker", "stack blocks"}

    data = json.loads(json_path.read_text())
    assert len(data) == 3


def test_resume_skips_already_proposed(tmp_path):
    root = _copy_dataset(tmp_path)
    paths = BatchPaths(dataset_root=root)
    cfg = GeminiConfig(api_key="test-key")

    # First run: process only ep_000 by limiting the episodes directory
    # Simpler approach: pre-populate proposals.jsonl with ep_000 marker
    (root / "proposals.jsonl").write_text('{"ep_id": "ep_000"}\n')

    call_counter = {"pass1": 0}

    async def counting_pass1(config, video_path, model):
        call_counter["pass1"] += 1
        return _mock_pass1_by_ep(video_path)

    with patch("vlm_annotate_verify.proposer.batch.pass1_boundaries",
               new=counting_pass1):
        with patch("vlm_annotate_verify.proposer.batch.pass2_labels",
                   new=_fake_pass2_async):
            asyncio.run(run_batch(paths, cfg, model=MODEL_FLASH))

    # Only ep_001 and ep_002 should have been processed
    assert call_counter["pass1"] == 2
    lines = (root / "proposals.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3  # original ep_000 marker + 2 new
