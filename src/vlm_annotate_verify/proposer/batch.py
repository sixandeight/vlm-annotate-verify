"""Batch orchestrator: parallel Gemini calls plus JSONL queue writer."""
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from vlm_annotate_verify.proposer.frames import (
    extract_frames, get_video_duration, FrameExtractionError,
)
from vlm_annotate_verify.proposer.gemini import (
    GeminiConfig, GeminiError, MODEL_FLASH,
)
from vlm_annotate_verify.proposer.passes import (
    pass1_boundaries, pass2_labels, PassError,
)
from vlm_annotate_verify.schemas import (
    FailureRow, Proposal, utc_now_iso,
)


DEFAULT_CONCURRENCY = 8


@dataclass(frozen=True)
class BatchPaths:
    dataset_root: Path

    @property
    def episodes_dir(self) -> Path:
        return self.dataset_root / "episodes"

    @property
    def frames_dir(self) -> Path:
        return self.dataset_root / "frames"

    @property
    def proposals_path(self) -> Path:
        return self.dataset_root / "proposals.jsonl"


def list_episodes(episodes_dir: Path) -> list[Path]:
    return sorted(episodes_dir.glob("*.mp4"))


def already_proposed(proposals_path: Path) -> set[str]:
    if not proposals_path.exists():
        return set()
    done: set[str] = set()
    for line in proposals_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["ep_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def failed_eps(proposals_path: Path) -> set[str]:
    if not proposals_path.exists():
        return set()
    failed: set[str] = set()
    for line in proposals_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row.get("status") == "failed":
                failed.add(row["ep_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return failed


async def propose_one(
    config: GeminiConfig,
    paths: BatchPaths,
    video_path: Path,
    model: str,
) -> str:
    """Process a single episode end-to-end. Return the JSONL line to append."""
    ep_id = video_path.stem
    try:
        ep_frames_dir = paths.frames_dir / ep_id
        frame_paths = extract_frames(video_path, ep_frames_dir)
        duration_s = get_video_duration(video_path)
        task, boundaries = await pass1_boundaries(config, video_path, model)
        segments = await pass2_labels(
            config, video_path, boundaries, duration_s, model,
        )
        proposal = Proposal(
            ep_id=ep_id,
            video_path=str(video_path.relative_to(paths.dataset_root)),
            duration_s=duration_s,
            frame_paths=[
                str(p.relative_to(paths.dataset_root)) for p in frame_paths
            ],
            task=task,
            model=model,
            boundaries=boundaries,
            segments=segments,
            created_at=utc_now_iso(),
        )
        return proposal.to_json()
    except (FrameExtractionError, PassError, GeminiError) as e:
        return FailureRow(
            ep_id=ep_id, error=str(e), attempted_at=utc_now_iso(),
        ).to_json()


async def run_batch(
    paths: BatchPaths,
    config: GeminiConfig,
    model: str = MODEL_FLASH,
    concurrency: int = DEFAULT_CONCURRENCY,
    force: bool = False,
    retry_failed: bool = False,
) -> None:
    paths.frames_dir.mkdir(parents=True, exist_ok=True)
    if force and paths.proposals_path.exists():
        paths.proposals_path.unlink()
    if retry_failed and paths.proposals_path.exists():
        kept = [
            line for line in paths.proposals_path.read_text().splitlines()
            if line.strip() and json.loads(line).get("status") != "failed"
        ]
        paths.proposals_path.write_text(
            "\n".join(kept) + ("\n" if kept else "")
        )

    done = already_proposed(paths.proposals_path)
    eps = [
        v for v in list_episodes(paths.episodes_dir) if v.stem not in done
    ]
    if not eps:
        return

    semaphore = asyncio.Semaphore(concurrency)
    out_lock = asyncio.Lock()

    async def worker(video_path: Path) -> None:
        async with semaphore:
            line = await propose_one(config, paths, video_path, model)
        async with out_lock:
            with paths.proposals_path.open("a") as f:
                f.write(line + "\n")

    await asyncio.gather(*(worker(v) for v in eps))
