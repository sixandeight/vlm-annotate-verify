import csv
import json
from pathlib import Path

import pytest

from vlm_annotate_verify.config import Config
from vlm_annotate_verify.exporter.pi_data_sharing import (
    METADATA_COLUMNS, ExportError,
    episode_duration, export, load_verified,
    overall_quality, overall_success,
    write_annotation_json, write_metadata_csv,
)
from vlm_annotate_verify.schemas import (
    Boundary, Mistake, Review, Segment, Verified,
)


CONFIG = Config(
    operator_id="nathan",
    station_id="home_lab",
    robot_id="aloha_01",
    is_eval_episode=False,
    dataset_name="mini_dataset",
)


def _verified(ep_id="ep_000", segs=None, task="place cube on shelf") -> Verified:
    default = [Segment(
        idx=0, start_s=0.0, end_s=5.0,
        quality=4, quality_conf=1.0,
        success=True, success_conf=1.0,
        mistakes=[], reasoning="",
    )]
    return Verified(
        ep_id=ep_id,
        task=task,
        boundaries=[Boundary(t_s=5.0, label="grasp", confidence=1.0)],
        segments=default if segs is None else segs,
        review=Review(
            reviewer_id="nathan", review_seconds=3.5,
            actions=["accept"], reprompt_used=False,
        ),
        verified_at="2026-04-20T10:00:00Z",
    )


def test_overall_quality_is_min_across_segments():
    segs = [
        Segment(idx=0, start_s=0, end_s=1, quality=5, quality_conf=1.0,
                success=True, success_conf=1.0),
        Segment(idx=1, start_s=1, end_s=2, quality=3, quality_conf=1.0,
                success=True, success_conf=1.0),
    ]
    v = _verified(segs=segs)
    assert overall_quality(v) == 3


def test_overall_success_is_all_across_segments():
    segs = [
        Segment(idx=0, start_s=0, end_s=1, quality=4, quality_conf=1.0,
                success=True, success_conf=1.0),
        Segment(idx=1, start_s=1, end_s=2, quality=4, quality_conf=1.0,
                success=False, success_conf=1.0),
    ]
    v = _verified(segs=segs)
    assert overall_success(v) is False


def test_episode_duration_uses_last_segment_end():
    segs = [
        Segment(idx=0, start_s=0, end_s=3, quality=4, quality_conf=1.0,
                success=True, success_conf=1.0),
        Segment(idx=1, start_s=3, end_s=8, quality=4, quality_conf=1.0,
                success=True, success_conf=1.0),
    ]
    assert episode_duration(_verified(segs=segs)) == 8.0


def test_episode_duration_no_segments_is_zero():
    assert episode_duration(_verified(segs=[])) == 0.0


def test_metadata_columns_contain_required_fields():
    for col in ("episode_id", "operator_id", "station_id", "robot_id",
                "is_eval_episode", "dataset_name", "task",
                "quality_overall", "success_overall", "duration_s"):
        assert col in METADATA_COLUMNS


def test_write_metadata_csv_writes_header_and_row(tmp_path):
    out = tmp_path / "custom_metadata.csv"
    write_metadata_csv([_verified()], CONFIG, out)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["episode_id"] == "ep_000"
    assert rows[0]["operator_id"] == "nathan"
    assert rows[0]["task"] == "place cube on shelf"
    assert rows[0]["quality_overall"] == "4"
    assert rows[0]["success_overall"] in ("True", "1", "true")


def test_write_annotation_json_writes_structured_payload(tmp_path):
    out = tmp_path / "custom_annotation.json"
    write_annotation_json([_verified()], out)
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert data[0]["episode_id"] == "ep_000"
    assert data[0]["task"] == "place cube on shelf"
    assert data[0]["segments"][0]["quality"] == 4
    assert "boundaries" in data[0]


def test_load_verified_reads_jsonl(tmp_path):
    path = tmp_path / "verified.jsonl"
    path.write_text("\n".join([
        _verified("ep_000").to_json(),
        _verified("ep_001").to_json(),
    ]) + "\n")
    loaded = load_verified(path)
    assert [v.ep_id for v in loaded] == ["ep_000", "ep_001"]


def test_load_verified_missing_file_raises(tmp_path):
    with pytest.raises(ExportError, match="missing"):
        load_verified(tmp_path / "nope.jsonl")


def test_export_end_to_end(tmp_path):
    verified_path = tmp_path / "verified.jsonl"
    verified_path.write_text(_verified().to_json() + "\n")
    csv_path, json_path = export(tmp_path, CONFIG)
    assert csv_path.exists()
    assert json_path.exists()
    assert csv_path.parent.name == "export"
    data = json.loads(json_path.read_text())
    assert data[0]["episode_id"] == "ep_000"
