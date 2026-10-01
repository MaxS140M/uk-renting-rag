"""Chunk every document in data/raw/ and write the result to data/chunks.jsonl.

Usage:
    python scripts/chunk_corpus.py
    python scripts/chunk_corpus.py --max-tokens 256 --overlap 32 --out data/chunks_256.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from rag.chunking import chunk_documents
from rag.config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNKS_FILE, RAW_DIR


def load_documents(raw_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(raw_dir.glob("*.json"))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-tokens", type=int, default=CHUNK_MAX_TOKENS)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP_TOKENS)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=CHUNKS_FILE)
    args = parser.parse_args()

    docs = load_documents(args.raw_dir)
    if not docs:
        print(f"No documents in {args.raw_dir}. Run scripts/download_docs.py first.")
        return 1

    chunks = chunk_documents(docs, max_tokens=args.max_tokens, overlap_tokens=args.overlap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    lengths = [c.n_tokens for c in chunks]
    print(f"Documents: {len(docs)}")
    print(f"Chunks:    {len(chunks)}  (max_tokens={args.max_tokens}, overlap={args.overlap})")
    print(
        f"Tokens per chunk: min {min(lengths)}, median {statistics.median(lengths):.0f}, "
        f"max {max(lengths)}"
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
