"""Download the models and build the search index for the deployed configuration.

Run once at Docker build time, so the image contains everything it needs and the
container starts without downloading anything:

    python scripts/prepare_deployment.py              # RAG_CONFIG env var, or the default
"""

from __future__ import annotations

import os
import sys
import time

from rag.chunking import get_token_counter
from rag.config import EMBEDDING_MODEL, PROJECT_ROOT, RAW_DIR
from rag.eval_utils import load_corpus
from rag.experiment import ensure_index, load_experiments
from rag.factory import build_retriever

DEFAULT_CONFIG = "hybrid_rerank_bge"


def main() -> int:
    name = os.environ.get("RAG_CONFIG") or DEFAULT_CONFIG
    experiments = {e.name: e for e in load_experiments(PROJECT_ROOT / "eval" / "configs.yaml")}
    if name not in experiments:
        print(f"Unknown RAG_CONFIG '{name}'. Options: {', '.join(experiments)}")
        return 1
    config = experiments[name].config
    start = time.perf_counter()

    # Chunk sizes are measured with this tokenizer, so it is needed to build the index.
    get_token_counter(EMBEDDING_MODEL)
    docs = load_corpus(RAW_DIR)
    print(f"Building the index for {name} from {len(docs)} documents")
    ensure_index(config, list(docs.values()))

    # Loading the retriever downloads the embedding model and, if used, the reranker; a
    # test query checks the whole retrieval stack works before the image is finished.
    results = build_retriever(config).retrieve(
        "How long does my landlord have to protect my deposit?", 3
    )
    if not results:
        print("Test retrieval returned nothing")
        return 1
    print(
        f"Ready in {time.perf_counter() - start:.0f}s. Test retrieval top result: "
        f"{results[0].title} > {results[0].section}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
