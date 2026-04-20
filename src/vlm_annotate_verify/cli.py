"""CLI entry: propose, review, export, status subcommands."""
import argparse
import asyncio
import sys
from pathlib import Path

from vlm_annotate_verify.config import load_config, ConfigError
from vlm_annotate_verify.exporter.pi_data_sharing import export, ExportError
from vlm_annotate_verify.proposer.batch import (
    BatchPaths, DEFAULT_CONCURRENCY, already_proposed, failed_eps,
    list_episodes, run_batch,
)
from vlm_annotate_verify.proposer.gemini import GeminiError, MODEL_FLASH, make_config
from vlm_annotate_verify.reviewer.app import ReviewerApp, already_verified_ids


def _propose(args: argparse.Namespace) -> int:
    paths = BatchPaths(dataset_root=Path(args.dataset_root).resolve())
    try:
        config = make_config()
    except GeminiError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    asyncio.run(run_batch(
        paths, config,
        model=MODEL_FLASH,
        concurrency=args.concurrency,
        force=args.force,
        retry_failed=args.retry_failed,
    ))
    return 0


def _review(args: argparse.Namespace) -> int:
    root = Path(args.dataset_root).resolve()
    try:
        cfg = load_config(root)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    try:
        gemini_config = make_config()
    except GeminiError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    app = ReviewerApp(
        dataset_root=root,
        gemini_config=gemini_config,
        reviewer_id=cfg.operator_id,
        from_ep=args.from_ep,
    )
    app.run()
    return 0


def _export(args: argparse.Namespace) -> int:
    root = Path(args.dataset_root).resolve()
    try:
        cfg = load_config(root)
        csv_path, json_path = export(root, cfg)
    except (ConfigError, ExportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    return 0


def _status(args: argparse.Namespace) -> int:
    paths = BatchPaths(dataset_root=Path(args.dataset_root).resolve())
    eps = list_episodes(paths.episodes_dir)
    proposed = already_proposed(paths.proposals_path)
    failed = failed_eps(paths.proposals_path)
    verified = already_verified_ids(paths.dataset_root / "verified.jsonl")
    print(f"episodes:  {len(eps)}")
    print(f"proposed:  {len(proposed)}")
    print(f"  failed:  {len(failed)}")
    print(f"verified:  {len(verified)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vlm-annotate-verify")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("propose", help="run gemini phase a on dataset")
    p.add_argument("dataset_root")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.set_defaults(func=_propose)

    r = sub.add_parser("review", help="open the TUI reviewer")
    r.add_argument("dataset_root")
    r.add_argument("--from", dest="from_ep", default=None)
    r.set_defaults(func=_review)

    e = sub.add_parser("export", help="write pi-data-sharing files")
    e.add_argument("dataset_root")
    e.set_defaults(func=_export)

    s = sub.add_parser("status", help="show queue counts")
    s.add_argument("dataset_root")
    s.set_defaults(func=_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
