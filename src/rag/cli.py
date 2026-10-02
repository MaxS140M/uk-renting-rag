"""Shared command-line options for choosing a retrieval setup, used by the scripts."""

from __future__ import annotations

import argparse
import dataclasses

from rag.config import PROJECT_ROOT, RETRIEVAL_MODES, RAGConfig
from rag.experiment import load_experiments


def add_retrieval_args(parser: argparse.ArgumentParser) -> None:
    """Add --config, --mode, --rerank/--no-rerank, --k and --candidates to a script."""
    group = parser.add_argument_group("retrieval")
    group.add_argument(
        "--config",
        help="a named experiment from eval/configs.yaml, e.g. hybrid_rerank_bge (the deployed "
        "configuration); the other options then adjust it. Default: the Phase 2 baseline",
    )
    group.add_argument("--mode", choices=RETRIEVAL_MODES, help="retrieval mode")
    group.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="rerank candidates with a cross-encoder",
    )
    group.add_argument("--k", type=int, help="number of passages passed to the LLM")
    group.add_argument("--candidates", type=int, help="candidate pool size before fusion/rerank")


def config_from_args(args: argparse.Namespace, base: RAGConfig | None = None) -> RAGConfig:
    """Apply any retrieval options given on the command line to a config."""
    overrides = {
        "retrieval_mode": args.mode,
        "use_reranker": args.rerank,
        "final_k": args.k,
        "candidate_k": args.candidates,
    }
    if getattr(args, "config", None):
        experiments = {e.name: e for e in load_experiments(PROJECT_ROOT / "eval" / "configs.yaml")}
        if args.config not in experiments:
            raise SystemExit(f"Unknown --config '{args.config}'. Options: {', '.join(experiments)}")
        base = experiments[args.config].config
    config = base or RAGConfig()
    changes = {key: value for key, value in overrides.items() if value is not None}
    if "final_k" in changes and "candidate_k" not in changes:
        changes["candidate_k"] = max(config.candidate_k, changes["final_k"])
    return dataclasses.replace(config, **changes)
