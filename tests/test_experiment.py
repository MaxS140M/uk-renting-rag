"""Tests for loading experiment configurations."""

from pathlib import Path

import pytest

from rag.experiment import changed_settings, index_dir_for, load_experiments

ROOT = Path(__file__).resolve().parent.parent


def write(tmp_path, text):
    path = tmp_path / "configs.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_project_configs_change_one_variable_at_a_time():
    experiments = load_experiments(ROOT / "eval" / "configs.yaml")
    by_name = {e.name: e for e in experiments}
    assert by_name["dense_baseline"].compare_to is None
    for e in experiments:
        if e.compare_to:
            changed = set(changed_settings(e, by_name[e.compare_to]))
            # Chunk size and overlap are one variable (overlap scales with the chunk size).
            changed -= {"chunk_overlap_tokens"} if "chunk_max_tokens" in changed else set()
            assert len(changed) == 1, f"{e.name} changes {changed}"


def test_settings_are_applied_and_index_dir_is_derived(tmp_path):
    path = write(
        tmp_path,
        "experiments:\n"
        "  - name: small_chunks\n"
        "    settings: {retrieval_mode: hybrid, chunk_max_tokens: 256, chunk_overlap_tokens: 32}\n",
    )
    (experiment,) = load_experiments(path, root=tmp_path)
    assert experiment.config.retrieval_mode == "hybrid"
    assert experiment.config.chunk_max_tokens == 256
    assert experiment.config.index_dir == tmp_path / "index_c256_o32_all-minilm-l6-v2"


def test_configs_sharing_chunking_and_model_share_an_index(tmp_path):
    path = write(
        tmp_path,
        "experiments:\n"
        "  - {name: a, settings: {retrieval_mode: dense}}\n"
        "  - {name: b, settings: {retrieval_mode: hybrid, use_reranker: true}}\n",
    )
    a, b = load_experiments(path, root=tmp_path)
    assert a.config.index_dir == b.config.index_dir


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("experiments: []\n", "non-empty"),
        ("experiments:\n  - {name: Bad-Name}\n", "lower_snake_case"),
        ("experiments:\n  - {name: a}\n  - {name: a}\n", "defined twice"),
        ("experiments:\n  - {name: a, settings: {index_dir: x}}\n", "unknown settings"),
        ("experiments:\n  - {name: a, settings: {colour: red}}\n", "unknown settings"),
        ("experiments:\n  - {name: a, compare_to: b}\n", "must come earlier"),
        ("experiments:\n  - {name: a, settings: {retrieval_mode: magic}}\n", "retrieval_mode"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        load_experiments(write(tmp_path, text), root=tmp_path)


def test_index_dir_names_are_filesystem_safe(tmp_path):
    from rag.config import RAGConfig

    config = RAGConfig(embedding_model="BAAI/bge-base-en-v1.5")
    assert index_dir_for(config, tmp_path).name == "index_c400_o50_bge-base-en-v1.5"
