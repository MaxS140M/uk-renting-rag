"""Embed every chunk in data/chunks.jsonl and save a FAISS index to data/index/.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --model sentence-transformers/all-mpnet-base-v2 \
        --out data/index_mpnet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rag.config import CHUNKS_FILE, EMBEDDING_MAX_SEQ_LENGTH, EMBEDDING_MODEL, INDEX_DIR
from rag.indexing import SentenceTransformerEmbedder, build_index, save_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", type=Path, default=CHUNKS_FILE)
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--max-seq-length", type=int, default=EMBEDDING_MAX_SEQ_LENGTH)
    parser.add_argument("--out", type=Path, default=INDEX_DIR)
    args = parser.parse_args()

    if not args.chunks.exists():
        print(f"{args.chunks} not found. Run scripts/chunk_corpus.py first.")
        return 1
    lines = args.chunks.read_text(encoding="utf-8").splitlines()
    chunks = [json.loads(line) for line in lines if line]

    print(f"Loading {args.model} (max_seq_length={args.max_seq_length})")
    embedder = SentenceTransformerEmbedder(args.model, args.max_seq_length)
    tokenizer = embedder.model.tokenizer
    # +2 for the [CLS] and [SEP] tokens the model adds around every input.
    too_long = sum(
        len(tokenizer.encode(c["text"], add_special_tokens=False)) + 2 > args.max_seq_length
        for c in chunks
    )
    if too_long:
        print(f"Warning: {too_long} chunks exceed max_seq_length and will be truncated")

    start = time.perf_counter()
    vectors = embedder.encode([c["text"] for c in chunks], show_progress_bar=True)
    elapsed = time.perf_counter() - start

    index = build_index(vectors)
    meta = save_index(
        args.out,
        index,
        chunks,
        embedding_model=args.model,
        embedding_max_seq_length=args.max_seq_length,
        source_chunks_file=args.chunks.name,
    )
    print(f"Embedded {meta['n_vectors']} chunks ({meta['dimension']} dims) in {elapsed:.1f}s")
    print(f"Saved index to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
