"""Transform verified.jsonl into pi-data-sharing CSV and JSON files."""
import csv
import json
from dataclasses import asdict
from pathlib import Path

from vlm_annotate_verify.config import Config
from vlm_annotate_verify.schemas import Verified


class ExportError(Exception):
    pass


METADATA_COLUMNS = [
    "episode_id", "operator_id", "station_id", "robot_id",
    "is_eval_episode", "dataset_name", "task", "quality_overall",
    "success_overall", "duration_s",
]


def overall_quality(verified: Verified) -> int:
    return min(s.quality for s in verified.segments)


def overall_success(verified: Verified) -> bool:
    return all(s.success for s in verified.segments)


def episode_duration(verified: Verified) -> float:
    return max(s.end_s for s in verified.segments) if verified.segments else 0.0


def write_metadata_csv(
    verified_records: list[Verified],
    config: Config,
    out_path: Path,
) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for v in verified_records:
            writer.writerow({
                "episode_id": v.ep_id,
                "operator_id": config.operator_id,
                "station_id": config.station_id,
                "robot_id": config.robot_id,
                "is_eval_episode": config.is_eval_episode,
                "dataset_name": config.dataset_name,
                "task": v.task,
                "quality_overall": overall_quality(v),
                "success_overall": overall_success(v),
                "duration_s": episode_duration(v),
            })


def write_annotation_json(
    verified_records: list[Verified],
    out_path: Path,
) -> None:
    payload = []
    for v in verified_records:
        payload.append({
            "episode_id": v.ep_id,
            "task": v.task,
            "boundaries": [asdict(b) for b in v.boundaries],
            "segments": [
                {
                    "idx": s.idx,
                    "start_s": s.start_s, "end_s": s.end_s,
                    "quality": s.quality, "success": s.success,
                    "mistakes": [asdict(m) for m in s.mistakes],
                    "reasoning": s.reasoning,
                }
                for s in v.segments
            ],
        })
    out_path.write_text(json.dumps(payload, indent=2))


def load_verified(verified_path: Path) -> list[Verified]:
    if not verified_path.exists():
        raise ExportError(f"verified.jsonl missing at {verified_path}")
    out: list[Verified] = []
    for line in verified_path.read_text().splitlines():
        if not line.strip():
            continue
        out.append(Verified.from_json(line))
    return out


def export(
    dataset_root: Path,
    config: Config,
) -> tuple[Path, Path]:
    verified = load_verified(dataset_root / "verified.jsonl")
    export_dir = dataset_root / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    csv_path = export_dir / "custom_metadata.csv"
    json_path = export_dir / "custom_annotation.json"
    write_metadata_csv(verified, config, csv_path)
    write_annotation_json(verified, json_path)
    return csv_path, json_path
