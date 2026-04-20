"""Pass 1 (boundaries plus task) and Pass 2 (per-segment labels) orchestration."""
import json
from pathlib import Path

from vlm_annotate_verify.proposer.gemini import (
    GeminiConfig, MODEL_FLASH, call_gemini_video,
)
from vlm_annotate_verify.proposer.prompts import (
    PASS_1_PROMPT, render_pass2_prompt,
)
from vlm_annotate_verify.schemas import Boundary, Mistake, Segment


class PassError(Exception):
    pass


async def pass1_boundaries(
    config: GeminiConfig,
    video_path: Path,
    model: str = MODEL_FLASH,
) -> tuple[str, list[Boundary]]:
    """Returns (task_description, boundaries)."""
    raw = await call_gemini_video(config, model, video_path, PASS_1_PROMPT)
    try:
        data = json.loads(raw)
        task = data["task"]
        boundaries = [Boundary(**b) for b in data["boundaries"]]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise PassError(
            f"pass1 returned malformed JSON: {e}\nraw: {raw[:500]}"
        ) from e
    return task, boundaries


async def pass2_labels(
    config: GeminiConfig,
    video_path: Path,
    boundaries: list[Boundary],
    duration_s: float,
    model: str = MODEL_FLASH,
) -> list[Segment]:
    boundaries_json = json.dumps(
        [{"t_s": b.t_s, "label": b.label} for b in boundaries]
    )
    prompt = render_pass2_prompt(boundaries_json)
    raw = await call_gemini_video(config, model, video_path, prompt)
    try:
        data = json.loads(raw)
        segments = []
        for s in data["segments"]:
            mistakes = [Mistake(**m) for m in s.get("mistakes", [])]
            segments.append(Segment(
                idx=s["idx"],
                start_s=s["start_s"], end_s=s["end_s"],
                quality=s["quality"], quality_conf=s["quality_conf"],
                success=s["success"], success_conf=s["success_conf"],
                mistakes=mistakes, reasoning=s.get("reasoning", ""),
            ))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise PassError(
            f"pass2 returned malformed JSON: {e}\nraw: {raw[:500]}"
        ) from e
    return segments
