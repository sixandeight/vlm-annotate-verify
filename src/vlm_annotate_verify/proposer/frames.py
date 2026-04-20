"""Extract N evenly-spaced frames per episode using ffmpeg. Cache to disk."""
import subprocess
from pathlib import Path

DEFAULT_NUM_FRAMES = 16


class FrameExtractionError(Exception):
    pass


def get_video_duration(video_path: Path) -> float:
    """Return duration in seconds via ffprobe."""
    if not video_path.exists():
        raise FrameExtractionError(f"video not found: {video_path}")
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise FrameExtractionError(f"ffprobe failed: {result.stderr.strip()}")
    return float(result.stdout.strip())


def extract_frames(
    video_path: Path,
    out_dir: Path,
    num_frames: int = DEFAULT_NUM_FRAMES,
) -> list[Path]:
    """Extract num_frames evenly spaced from video. Skip any that already exist."""
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = [out_dir / f"{i:02d}.jpg" for i in range(1, num_frames + 1)]
    if all(p.exists() and p.stat().st_size > 0 for p in expected):
        return expected
    duration = get_video_duration(video_path)
    if duration <= 0:
        raise FrameExtractionError(
            f"invalid video duration ({duration}) for {video_path}"
        )
    timestamps = [duration * (i + 0.5) / num_frames for i in range(num_frames)]
    for i, t in enumerate(timestamps, start=1):
        out_path = out_dir / f"{i:02d}.jpg"
        if out_path.exists() and out_path.stat().st_size > 0:
            continue
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(t),
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "3",
                str(out_path),
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise FrameExtractionError(
                f"ffmpeg failed for {video_path} at t={t}: "
                f"{result.stderr.decode(errors='replace')}"
            )
    return expected
