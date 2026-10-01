"""Shared command-line options for choosing a retrieval setup, used by the scripts."""

from __future__ import annotations

import argparse
import dataclasses

from rag.config import RETRIEVAL_MODES, RAGConfig


def add_retrieval_args(parser: argparse.ArgumentParser) -> None:
    """Add --mode, --rerank/--no-rerank, --k and --candidates to a script's arguments."""
    group = parser.add_argument_group("retrieval")
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
    config = base or RAGConfig()
    changes = {key: value for key, value in overrides.items() if value is not None}
    if "final_k" in changes and "candidate_k" not in changes:
        changes["candidate_k"] = max(config.candidate_k, changes["final_k"])
    return dataclasses.replace(config, **changes)
