"""Assemble the files for a Hugging Face Docker Space in build/space/ (see DEPLOY.md).

A Space is configured by a YAML header at the top of its README.md. That header would look
out of place on the GitHub README, so the Space gets its own short README, plus only the
files the Docker image needs.

Usage:
    python scripts/prepare_space.py
"""

from __future__ import annotations

import shutil
import sys

from rag.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "build" / "space"
GITHUB_URL = "https://github.com/MaxS140M/uk-renting-rag"
INCLUDE = [
    "Dockerfile",
    ".dockerignore",
    "requirements.txt",
    "LICENSE",
    "src",
    "app",
    "scripts/prepare_deployment.py",
    "eval/configs.yaml",
    "data/raw",
]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")

SPACE_README = f"""---
title: UK Renting Guidance Assistant
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: true
short_description: Cited answers about renting in England from GOV.UK
---

# UK Renting Guidance Assistant

Ask a question about renting a home in England and get an answer drawn only from official
GOV.UK guidance, with links to the sources. Hybrid search (dense embeddings + BM25) with a
cross-encoder reranker finds the passages; Claude writes a cited answer, and says so when
the guidance does not cover the question.

**Demo project, not legal advice.** Code, evaluation and documentation:
[{GITHUB_URL}]({GITHUB_URL}).

Contains public sector information licensed under the Open Government Licence v3.0.
"""


def main() -> int:
    missing = [p for p in INCLUDE if not (PROJECT_ROOT / p).exists()]
    if missing:
        print(f"Missing files: {missing}")
        return 1
    if OUT.exists():
        shutil.rmtree(OUT)
    for item in INCLUDE:
        source, target = PROJECT_ROOT / item, OUT / item
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=IGNORE)
        else:
            shutil.copy2(source, target)
    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8")
    secrets = [p for p in OUT.rglob("*") if p.name == ".env" or p.name.startswith(".env.")]
    if secrets:
        print(f"Refusing to continue: secret files found {secrets}")
        return 1
    files = [p for p in OUT.rglob("*") if p.is_file()]
    size_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"Prepared {len(files)} files ({size_mb:.1f} MB) in {OUT}")
    print("Next: follow 'Push to the Space' in DEPLOY.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
