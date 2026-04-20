from pathlib import Path
from unittest.mock import patch

import pytest

from vlm_annotate_verify.cli import main


def _mkds(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "episodes").mkdir(parents=True)
    (root / "roboannotate.toml").write_text(
        'operator_id = "n"\nstation_id = "s"\nrobot_id = "r"\n'
        'is_eval_episode = false\ndataset_name = "d"\n'
    )
    return root


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for sub in ("propose", "review", "export", "status"):
        assert sub in out


def test_no_args_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


def test_propose_missing_gemini_key_errors(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    root = _mkds(tmp_path)
    rc = main(["propose", str(root)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "GEMINI_API_KEY" in err


def test_propose_happy_path_calls_run_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    root = _mkds(tmp_path)

    captured = {}

    async def fake_run_batch(paths, config, *, model, concurrency,
                              force, retry_failed):
        captured["root"] = paths.dataset_root
        captured["concurrency"] = concurrency
        captured["force"] = force
        captured["retry_failed"] = retry_failed

    with patch("vlm_annotate_verify.cli.run_batch", new=fake_run_batch):
        rc = main(["propose", str(root), "--concurrency", "4", "--force"])
    assert rc == 0
    assert captured["root"] == root.resolve()
    assert captured["concurrency"] == 4
    assert captured["force"] is True
    assert captured["retry_failed"] is False


def test_propose_retry_failed_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    root = _mkds(tmp_path)

    captured = {}

    async def fake_run_batch(*args, retry_failed, **kwargs):
        captured["retry_failed"] = retry_failed

    with patch("vlm_annotate_verify.cli.run_batch", new=fake_run_batch):
        rc = main(["propose", str(root), "--retry-failed"])
    assert rc == 0
    assert captured["retry_failed"] is True


def test_review_missing_config_errors(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    root = tmp_path / "empty"
    root.mkdir()
    rc = main(["review", str(root)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "config" in err.lower()


def test_review_calls_app_run(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    root = _mkds(tmp_path)

    run_calls = {"count": 0, "from_ep": None}

    class _FakeApp:
        def __init__(self, *, dataset_root, gemini_config, reviewer_id, from_ep):
            run_calls["from_ep"] = from_ep
        def run(self):
            run_calls["count"] += 1

    with patch("vlm_annotate_verify.cli.ReviewerApp", _FakeApp):
        rc = main(["review", str(root), "--from", "ep_042"])
    assert rc == 0
    assert run_calls["count"] == 1
    assert run_calls["from_ep"] == "ep_042"


def test_export_missing_verified_errors(tmp_path, capsys):
    root = _mkds(tmp_path)
    rc = main(["export", str(root)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "verified" in err.lower() or "missing" in err.lower()


def test_status_reports_counts(tmp_path, capsys):
    root = _mkds(tmp_path)
    (root / "episodes" / "ep_000.mp4").write_bytes(b"")
    (root / "episodes" / "ep_001.mp4").write_bytes(b"")
    (root / "proposals.jsonl").write_text(
        '{"ep_id": "ep_000"}\n'
        '{"ep_id": "ep_001", "status": "failed", "error": "x"}\n'
    )
    (root / "verified.jsonl").write_text(
        '{"ep_id": "ep_000", "task": "t", "boundaries": [], "segments": [],'
        ' "review": {"reviewer_id": "n", "review_seconds": 1.0, "actions": [], "reprompt_used": false},'
        ' "verified_at": "2026-04-20T10:00:00Z"}\n'
    )
    rc = main(["status", str(root)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "episodes:  2" in out
    assert "proposed:  2" in out or "proposed:  1" in out  # failure rows count as proposed
    assert "failed:  1" in out
    assert "verified:  1" in out
