import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from vlm_annotate_verify.proposer.batch import (
    BatchPaths, DEFAULT_CONCURRENCY, already_proposed, failed_eps,
    list_episodes, propose_one, run_batch,
)
from vlm_annotate_verify.proposer.gemini import GeminiConfig, MODEL_FLASH
from vlm_annotate_verify.proposer.passes import PassError
from vlm_annotate_verify.schemas import Boundary, Segment


def _async_value(value):
    async def _inner(*_args, **_kwargs):
        return value
    return _inner


def _async_raise(exc):
    async def _inner(*_args, **_kwargs):
        raise exc
    return _inner


def _make_dataset(tmp_path: Path, ep_ids: list[str]) -> Path:
    root = tmp_path / "ds"
    eps = root / "episodes"
    eps.mkdir(parents=True)
    for ep_id in ep_ids:
        (eps / f"{ep_id}.mp4").write_bytes(b"\x00")
    return root


def test_list_episodes_returns_sorted_mp4s(tmp_path):
    root = _make_dataset(tmp_path, ["ep_002", "ep_000", "ep_001"])
    paths = list_episodes(root / "episodes")
    assert [p.stem for p in paths] == ["ep_000", "ep_001", "ep_002"]


def test_already_proposed_empty_when_file_missing(tmp_path):
    assert already_proposed(tmp_path / "missing.jsonl") == set()


def test_already_proposed_reads_ep_ids(tmp_path):
    p = tmp_path / "proposals.jsonl"
    p.write_text(
        '{"ep_id": "ep_000"}\n'
        '{"ep_id": "ep_001"}\n'
    )
    assert already_proposed(p) == {"ep_000", "ep_001"}


def test_failed_eps_filters_status_failed(tmp_path):
    p = tmp_path / "proposals.jsonl"
    p.write_text(
        '{"ep_id": "ep_000"}\n'
        '{"ep_id": "ep_001", "status": "failed", "error": "x"}\n'
        '{"ep_id": "ep_002", "status": "failed", "error": "y"}\n'
    )
    assert failed_eps(p) == {"ep_001", "ep_002"}


def test_propose_one_success_returns_proposal_line(tmp_path):
    root = _make_dataset(tmp_path, ["ep_000"])
    paths = BatchPaths(dataset_root=root)
    cfg = GeminiConfig(api_key="x")
    boundaries = [Boundary(t_s=2.0, label="grasp", confidence=0.9)]
    segments = [Segment(idx=0, start_s=0.0, end_s=2.0,
                        quality=4, quality_conf=0.8,
                        success=True, success_conf=0.9)]

    with patch("vlm_annotate_verify.proposer.batch.extract_frames",
               return_value=[paths.frames_dir / "ep_000" / "01.jpg"]):
        with patch("vlm_annotate_verify.proposer.batch.get_video_duration",
                   return_value=2.0):
            with patch("vlm_annotate_verify.proposer.batch.pass1_boundaries",
                       new=_async_value(("place cube", boundaries))):
                with patch("vlm_annotate_verify.proposer.batch.pass2_labels",
                           new=_async_value(segments)):
                    line = asyncio.run(propose_one(
                        cfg, paths, root / "episodes" / "ep_000.mp4", MODEL_FLASH,
                    ))
    parsed = json.loads(line)
    assert parsed["ep_id"] == "ep_000"
    assert parsed["task"] == "place cube"
    assert parsed["model"] == MODEL_FLASH
    assert "status" not in parsed  # success row, no status field


def test_propose_one_failure_returns_failure_line(tmp_path):
    root = _make_dataset(tmp_path, ["ep_042"])
    paths = BatchPaths(dataset_root=root)
    cfg = GeminiConfig(api_key="x")
    with patch("vlm_annotate_verify.proposer.batch.extract_frames",
               return_value=[]):
        with patch("vlm_annotate_verify.proposer.batch.get_video_duration",
                   return_value=2.0):
            with patch("vlm_annotate_verify.proposer.batch.pass1_boundaries",
                       new=_async_raise(PassError("malformed"))):
                line = asyncio.run(propose_one(
                    cfg, paths, root / "episodes" / "ep_042.mp4", MODEL_FLASH,
                ))
    parsed = json.loads(line)
    assert parsed["ep_id"] == "ep_042"
    assert parsed["status"] == "failed"
    assert "malformed" in parsed["error"]


def test_run_batch_writes_lines_for_each_ep(tmp_path):
    root = _make_dataset(tmp_path, ["ep_000", "ep_001"])
    paths = BatchPaths(dataset_root=root)
    cfg = GeminiConfig(api_key="x")
    boundaries = [Boundary(t_s=2.0, label="grasp", confidence=0.9)]
    segments = [Segment(idx=0, start_s=0.0, end_s=2.0,
                        quality=5, quality_conf=1.0,
                        success=True, success_conf=1.0)]
    with patch("vlm_annotate_verify.proposer.batch.extract_frames", return_value=[]):
        with patch("vlm_annotate_verify.proposer.batch.get_video_duration",
                   return_value=2.0):
            with patch("vlm_annotate_verify.proposer.batch.pass1_boundaries",
                       new=_async_value(("t", boundaries))):
                with patch("vlm_annotate_verify.proposer.batch.pass2_labels",
                           new=_async_value(segments)):
                    asyncio.run(run_batch(paths, cfg, model=MODEL_FLASH, concurrency=2))
    lines = paths.proposals_path.read_text().strip().splitlines()
    assert len(lines) == 2
    ids = {json.loads(line)["ep_id"] for line in lines}
    assert ids == {"ep_000", "ep_001"}


def test_run_batch_skips_already_proposed(tmp_path):
    root = _make_dataset(tmp_path, ["ep_000", "ep_001"])
    paths = BatchPaths(dataset_root=root)
    paths.proposals_path.write_text('{"ep_id": "ep_000"}\n')
    cfg = GeminiConfig(api_key="x")
    boundaries = [Boundary(t_s=1.0, label="g", confidence=1.0)]
    segments = [Segment(idx=0, start_s=0.0, end_s=1.0,
                        quality=3, quality_conf=0.5,
                        success=True, success_conf=0.5)]
    with patch("vlm_annotate_verify.proposer.batch.extract_frames", return_value=[]):
        with patch("vlm_annotate_verify.proposer.batch.get_video_duration",
                   return_value=1.0):
            with patch("vlm_annotate_verify.proposer.batch.pass1_boundaries",
                       new=_async_value(("t", boundaries))):
                with patch("vlm_annotate_verify.proposer.batch.pass2_labels",
                           new=_async_value(segments)):
                    asyncio.run(run_batch(paths, cfg, model=MODEL_FLASH))
    lines = paths.proposals_path.read_text().strip().splitlines()
    # Original ep_000 line + new ep_001 line
    assert len(lines) == 2
    ids = [json.loads(line)["ep_id"] for line in lines]
    assert "ep_000" in ids and "ep_001" in ids


def test_run_batch_force_wipes_existing(tmp_path):
    root = _make_dataset(tmp_path, ["ep_000"])
    paths = BatchPaths(dataset_root=root)
    paths.proposals_path.write_text('{"ep_id": "stale"}\n')
    cfg = GeminiConfig(api_key="x")
    boundaries = [Boundary(t_s=1.0, label="g", confidence=1.0)]
    segments = [Segment(idx=0, start_s=0.0, end_s=1.0,
                        quality=3, quality_conf=0.5,
                        success=True, success_conf=0.5)]
    with patch("vlm_annotate_verify.proposer.batch.extract_frames", return_value=[]):
        with patch("vlm_annotate_verify.proposer.batch.get_video_duration",
                   return_value=1.0):
            with patch("vlm_annotate_verify.proposer.batch.pass1_boundaries",
                       new=_async_value(("t", boundaries))):
                with patch("vlm_annotate_verify.proposer.batch.pass2_labels",
                           new=_async_value(segments)):
                    asyncio.run(run_batch(paths, cfg, force=True))
    lines = paths.proposals_path.read_text().strip().splitlines()
    ids = [json.loads(line)["ep_id"] for line in lines]
    assert ids == ["ep_000"]
    assert "stale" not in ids


def test_run_batch_retry_failed_only_redoes_failures(tmp_path):
    root = _make_dataset(tmp_path, ["ep_000", "ep_001"])
    paths = BatchPaths(dataset_root=root)
    paths.proposals_path.write_text(
        '{"ep_id": "ep_000"}\n'
        '{"ep_id": "ep_001", "status": "failed", "error": "old"}\n'
    )
    cfg = GeminiConfig(api_key="x")
    boundaries = [Boundary(t_s=1.0, label="g", confidence=1.0)]
    segments = [Segment(idx=0, start_s=0.0, end_s=1.0,
                        quality=3, quality_conf=0.5,
                        success=True, success_conf=0.5)]
    with patch("vlm_annotate_verify.proposer.batch.extract_frames", return_value=[]):
        with patch("vlm_annotate_verify.proposer.batch.get_video_duration",
                   return_value=1.0):
            with patch("vlm_annotate_verify.proposer.batch.pass1_boundaries",
                       new=_async_value(("t", boundaries))):
                with patch("vlm_annotate_verify.proposer.batch.pass2_labels",
                           new=_async_value(segments)):
                    asyncio.run(run_batch(paths, cfg, retry_failed=True))
    lines = paths.proposals_path.read_text().strip().splitlines()
    parsed = [json.loads(line) for line in lines]
    ids = [p["ep_id"] for p in parsed]
    # ep_000 success preserved, ep_001 has a fresh success row, no failure rows remain
    assert "ep_000" in ids
    assert "ep_001" in ids
    assert all(p.get("status") != "failed" for p in parsed)


def test_default_concurrency_is_positive():
    assert DEFAULT_CONCURRENCY > 0
