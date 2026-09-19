"""Offline tests for the document-QA dataset contract, the pinned corpus reader, the corpus metrics and
baselines, BYOD loaders, JSONL export, artifact-manifest rejections and adapt() argument validation. Nothing
here imports torch, transformers or pyarrow; the corpus columns come from an injected fetcher."""

from __future__ import annotations

import json

import pytest

from layoutlm_document_qa_pipeline import (
    ARTIFACT_FORMAT,
    CORPUS_FILES,
    ENCODER_LAYERS,
    MODEL_ID,
    MODEL_REVISION,
    QUESTION_TEMPLATES,
    SAMPLE_SPLIT,
    WEIGHT_SHA256,
    LayoutLMDocumentQAPipeline,
    build_sample_dataset,
    check_split_disjoint,
    column_digest,
    dataset_digest,
    docqa_metrics,
    fetch_corpus,
    fetch_sample_dataset,
    keyword_lookup_answer,
    keyword_lookup_baseline,
    last_number_baseline,
    load_byod_dataset,
    page_from_ground_truth,
    questions_for_page,
    read_corpus,
    split_dataset,
    validate_dataset,
    write_dataset_jsonl,
)
from layoutlm_document_qa_pipeline import pipeline as pl
from layoutlm_document_qa_pipeline import samples as sm


def _quad(x0, y0, x1, y1):
    return {"x1": x0, "y1": y0, "x2": x1, "y2": y0, "x3": x1, "y3": y1, "x4": x0, "y4": y1}


def _line(category, group_id, y, tokens):
    """One CORD `valid_line`: `tokens` is a list of (text, is_key) laid out left to right at row `y`; a
    text with spaces becomes one word entry per space-separated piece, as CORD annotates it."""
    words = []
    x = 20
    for text, is_key in tokens:
        for piece in text.split():
            words.append(
                {
                    "quad": _quad(x, y, x + 12 * len(piece), y + 16),
                    "is_key": is_key,
                    "row_id": y,
                    "text": piece,
                }
            )
            x += 12 * len(piece) + 8
    return {"words": words, "category": category, "group_id": group_id, "sub_group_id": 0}


def _receipt(index, *, second_item=False, split_total=False, empty_word=False):
    lines = [
        _line("menu.nm", 1, 40, [(f"ES TEH {index}", 0)]),
        _line("menu.price", 1, 40, [("10,000", 0)]),
        _line("sub_total.subtotal_price", 5, 120, [("SUBTOTAL", 1), ("10,000", 0)]),
        _line("sub_total.tax_price", 5, 140, [("PB1", 1), ("1,000", 0)]),
        _line("total.total_price", 9, 180, [("TOTAL", 1), ("11,000", 0)]),
        _line("total.cashprice", 9, 200, [("CASH", 1), ("20,000", 0)]),
    ]
    if second_item:
        lines.insert(2, _line("menu.nm", 2, 60, [("KOPI SUSU", 0)]))
    if split_total:
        lines.append(_line("total.total_price", 10, 220, [("TOTAL", 1), ("11,000", 0)]))
    if empty_word:
        lines[0]["words"].append({"quad": _quad(0, 0, 0, 0), "is_key": 0, "row_id": 0, "text": "  "})
    return json.dumps(
        {
            "gt_parse": {},
            "meta": {
                "version": "2.0.0",
                "split": "test",
                "image_id": index,
                "image_size": {"width": 400, "height": 300},
            },
            "valid_line": lines,
            "roi": {},
            "repeating_symbol": [],
            "dontcare": [],
        }
    )


def _rows(n, prefix=0):
    return [_receipt(prefix + i) for i in range(n)]


def _fake_pipeline(count_windows=None):
    def runner(question, words, grid_boxes):
        return {"start": len(words) - 1, "end": len(words) - 1, "score": 0.5, "n_windows": 1}

    return LayoutLMDocumentQAPipeline(runner, "cpu", "float32", "injected", count_windows)


def _records(n_pages=4, prefix="r"):
    out = []
    for page_index in range(n_pages):
        page = page_from_ground_truth(_receipt(page_index), f"{prefix}{page_index}")
        for question in questions_for_page(page):
            out.append({"id": f"{prefix}{page_index}-{len(out):03d}", **question})
    return out


# --- corpus reader ----------------------------------------------------------------------------------


def test_pinned_corpus_constants():
    assert sm.CORPUS_REPO == "naver-clova-ix/cord-v2" and len(sm.CORPUS_REVISION) == 40
    assert set(CORPUS_FILES) == {"test", "validation"}
    for spec in CORPUS_FILES.values():
        assert spec["path"].startswith("data/") and spec["path"].endswith(".parquet")
        assert len(spec["sha256"]) == 64 and len(spec["column_sha256"]) == 64 and spec["rows"] == 100
    assert sum(SAMPLE_SPLIT.values()) == 199 and set(SAMPLE_SPLIT) == {"train", "validation", "test"}
    assert set(QUESTION_TEMPLATES) >= {"total.total_price", "menu.nm"}


def test_fetch_corpus_verifies_digests_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    rows = {"test": _rows(3), "validation": _rows(2, prefix=10)}
    for split, values in rows.items():
        monkeypatch.setitem(CORPUS_FILES[split], "rows", len(values))
        monkeypatch.setitem(CORPUS_FILES[split], "column_sha256", column_digest(values))
    calls = []

    def fetcher(split, spec):
        calls.append(split)
        return rows[split]

    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == rows
    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == rows  # served from the cache
    assert sorted(calls) == ["test", "validation"]
    with pytest.raises(ValueError, match="pinned"):
        fetch_corpus(cache_dir=tmp_path / "other", fetcher=lambda split, spec: ["tampered"] * spec["rows"])
    monkeypatch.setitem(CORPUS_FILES["test"], "rows", 99)
    with pytest.raises(ValueError, match="pinned 99"):
        fetch_corpus(cache_dir=tmp_path / "third", fetcher=fetcher)


def test_page_from_ground_truth_keeps_boxes_inside_the_page_and_marks_value_spans(forbid_model_imports):
    page = page_from_ground_truth(_receipt(0, empty_word=True), "p0")
    assert page["image_size"] == [400, 300] and len(page["words"]) == len(page["boxes"]) == 12
    assert page["words"][:3] == ["ES", "TEH", "0"] and page["words"][-2:] == ["CASH", "20,000"]
    assert all(0 <= b[0] < b[2] <= 400 and 0 <= b[1] < b[3] <= 300 for b in page["boxes"])
    total = next(line for line in page["lines"] if line["category"] == "total.total_price")
    assert (total["start"], total["end"], total["value_start"], total["value_end"]) == (8, 9, 9, 9)
    split = page_from_ground_truth(
        json.dumps(
            {
                "meta": {"image_size": {"width": 400, "height": 100}},
                "valid_line": [_line("total.total_price", 1, 10, [("1", 0), ("TOTAL", 1), ("2", 0)])],
            }
        ),
        "p1",
    )
    assert split["lines"][0]["value_start"] is None


def test_questions_for_page_follow_the_template_rules(forbid_model_imports):
    plain = questions_for_page(page_from_ground_truth(_receipt(0), "p0"))
    assert [q["field"] for q in plain] == [
        "total.total_price",
        "sub_total.subtotal_price",
        "sub_total.tax_price",
        "total.cashprice",
        "menu.nm",
    ]
    total = plain[0]
    assert total["question"] == "What is the total amount?" and total["answers"] == ["11,000"]
    assert total["words"][total["answer_start"] : total["answer_end"] + 1] == ["11,000"]
    first = plain[-1]
    assert first["answers"] == ["ES TEH 0"] and (first["answer_start"], first["answer_end"]) == (0, 2)
    two_items = questions_for_page(page_from_ground_truth(_receipt(0, second_item=True), "p1"))
    assert next(q for q in two_items if q["field"] == "menu.nm")["answers"] == ["ES TEH 0"]  # topmost item
    two_totals = questions_for_page(page_from_ground_truth(_receipt(0, split_total=True), "p2"))
    assert "total.total_price" not in {q["field"] for q in two_totals}  # ambiguous category is skipped


def test_build_sample_dataset_is_seeded_page_disjoint_and_deduplicated(forbid_model_imports):
    rows = {"test": _rows(6), "validation": _rows(4, prefix=10) + [_receipt(0)]}  # one duplicate receipt
    pages = read_corpus(rows)
    assert [p["id"] for p in pages["validation"]][:2] == ["validation-000", "validation-001"]
    sizes = {"train": 6, "validation": 2, "test": 2}
    splits = build_sample_dataset(pages, seed=1, sizes=sizes)
    assert {k: len({r["page_id"] for r in v}) for k, v in splits.items()} == sizes
    assert splits["train"][0]["id"] == "train-0000" and check_split_disjoint(splits)
    assert build_sample_dataset(pages, seed=1, sizes=sizes) == splits
    assert build_sample_dataset(pages, seed=2, sizes=sizes) != splits
    with pytest.raises(ValueError, match="split sizes need 11"):
        build_sample_dataset(pages, sizes={"train": 7, "validation": 2, "test": 2})


def test_check_split_disjoint_catches_shared_pages_and_shared_words(forbid_model_imports):
    records = _records(3)
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint({"train": records[:2], "test": records[1:3]})
    other = [{**r, "page_id": "elsewhere"} for r in records[:1]]
    with pytest.raises(ValueError, match="words"):
        check_split_disjoint({"train": records[:1], "test": other})
    assert check_split_disjoint({"train": records[:5], "test": records[10:]}) == {
        "train": 5,
        "test": len(records) - 10,
    }


def test_fetch_sample_dataset_end_to_end_with_injected_fetcher(tmp_path, monkeypatch, forbid_model_imports):
    rows = {"test": _rows(5), "validation": _rows(5, prefix=10)}
    for split, values in rows.items():
        monkeypatch.setitem(CORPUS_FILES[split], "rows", len(values))
        monkeypatch.setitem(CORPUS_FILES[split], "column_sha256", column_digest(values))
    splits = fetch_sample_dataset(
        cache_dir=tmp_path,
        fetcher=lambda split, spec: rows[split],
        sizes={"train": 6, "validation": 2, "test": 2},
    )
    assert validate_dataset(splits["train"])["unique_pages"] == 6


# --- dataset validation -------------------------------------------------------------------------------


def test_validate_dataset_reports_and_rejects(forbid_model_imports):
    records = _records(4)
    report = validate_dataset(records)
    assert report["n_records"] == 20 and report["unique_pages"] == 4
    assert report["digest"] == dataset_digest(report["records"]) and report["model_id"] == MODEL_ID
    assert report["answer_words"] == {"min": 1, "max": 3} and report["words_per_page"] == {
        "min": 12,
        "max": 12,
    }
    good = records
    for bad, message in (
        (good[:7], "8..20000"),
        ([{**good[0], "id": "bad id"}, *good[1:]], "id must match"),
        ([{**good[0], "id": good[1]["id"]}, *good[1:]], "duplicate id"),
        ([{**good[0], "question": ""}, *good[1:]], "non-whitespace"),
        ([{**good[0], "boxes": good[0]["boxes"][:-1]}, *good[1:]], "one pixel xyxy box per word"),
        ([{**good[0], "image_size": [400.0, 300]}, *good[1:]], "image_size"),
        ([{**good[0], "answer_end": 99}, *good[1:]], "not inside"),
        ([{**good[0], "answer_start": "0"}, *good[1:]], "must be ints"),
        ([{**good[0], "answers": ["other"]}, *good[1:]], "span text"),
        ([{k: v for k, v in good[0].items() if k != "words"}, *good[1:]], "missing 'words'"),
        (["not a mapping", *good[1:]], "must be a mapping"),
        ({"a": 1}, "must be a list"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_dataset(bad)
    without_answers = [{k: v for k, v in r.items() if k != "answers"} for r in good]
    assert validate_dataset(without_answers)["records"][0]["answers"] == good[0]["answers"]


def test_split_dataset_keeps_pages_together_and_is_seeded(forbid_model_imports):
    records = _records(6)
    splits = split_dataset(records, val_fraction=0.15, test_fraction=0.2, seed=3)
    assert sum(len(v) for v in splits.values()) == len(records) and check_split_disjoint(splits)
    assert split_dataset(records, val_fraction=0.15, test_fraction=0.2, seed=3) == splits
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)
    with pytest.raises(ValueError, match="at least"):
        split_dataset(records, val_fraction=0.0, test_fraction=0.9)


# --- metrics and baselines ----------------------------------------------------------------------------


def test_docqa_metrics_and_baselines(forbid_model_imports):
    records = _records(4)
    golds = [r["answers"] for r in records]
    perfect = docqa_metrics([g[0] for g in golds], golds)
    assert perfect["anls"] == 1.0 and perfect["exact_match"] == 1.0 and perfect["empty_rate"] == 0.0
    empty = docqa_metrics([""] * len(records), golds)
    assert empty["anls"] == 0.0 and empty["empty_rate"] == 1.0
    with pytest.raises(ValueError, match="gold lists"):
        docqa_metrics(["a"], [["a"], ["b"]])
    last = last_number_baseline(records)
    assert last["n"] == len(records) and "digit" in last["baseline"]
    cash = [r for r in records if r["field"] == "total.cashprice"]
    assert last_number_baseline(cash)["exact_match"] == 1.0  # CASH 20,000 is the last line
    page = records[0]["words"]
    assert keyword_lookup_answer("What is the total amount?", page) == "11,000"
    assert keyword_lookup_answer("What is the subtotal?", page) == "10,000"
    assert keyword_lookup_answer("What is the tax amount?", page) == "20,000"  # no keyword → last number
    lookup = keyword_lookup_baseline(records)
    assert 0.0 < lookup["exact_match"] < 1.0 and lookup["anls"] >= lookup["exact_match"]


# --- BYOD loaders and JSONL ---------------------------------------------------------------------------


def test_byod_json_jsonl_round_trip_and_rejections(tmp_path, forbid_model_imports):
    records = [{k: v for k, v in r.items() if k != "field"} for r in _records(2)]
    path = write_dataset_jsonl(records, tmp_path / "data.jsonl")
    assert load_byod_dataset(path) == records
    (tmp_path / "data.json").write_text(json.dumps(records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "data.json") == records
    (tmp_path / "obj.json").write_text('{"records": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="array of records"):
        load_byod_dataset(tmp_path / "obj.json")
    (tmp_path / "data.csv").write_text("id,question\n", encoding="utf-8")
    with pytest.raises(ValueError, match=".json or .jsonl"):
        load_byod_dataset(tmp_path / "data.csv")
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "missing.json")


# --- adaptation and artifacts without a model ---------------------------------------------------------


def test_adapt_and_artifacts_need_a_loaded_model(tmp_path, forbid_model_imports):
    pipe = _fake_pipeline(count_windows=lambda question, words: 1 + (len(words) > 30))
    records = _records(4)
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(records, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(records, lr=1.0)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(records, batch_size=0)
    with pytest.raises(ValueError, match="trainable_encoder_layers"):
        pipe.adapt(records, trainable_encoder_layers=ENCODER_LAYERS + 1)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(records)
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)
    # evaluate() and check_fit() only need the answer path and the window counter, so they work injected
    metrics = pipe.evaluate(records)
    assert metrics["n"] == len(records) and metrics["verdict"] == "measured-small-sample"
    assert metrics["adapted"] is False and 0.0 <= metrics["anls"] <= 1.0
    long_page = {
        **records[0],
        "id": "long",
        "words": records[0]["words"] * 4,
        "boxes": records[0]["boxes"] * 4,
    }
    fit = pipe.check_fit([*records, long_page])
    assert fit["n_fitting"] == len(records) and fit["dropped"] == ["long"]
    with pytest.raises(ValueError, match="from_pretrained"):
        _fake_pipeline().check_fit(records)


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path, forbid_model_imports):
    pipe = _fake_pipeline()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["qa_outputs.weight"],
        "adapter": {"trainable_encoder_layers": 1},
    }
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    bad_base = {**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(bad_base))
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)


def test_load_artifact_refuses_unsupported_versions_extra_files_and_traversal(tmp_path, forbid_model_imports):
    pipe = _fake_pipeline()
    good = {
        "format": ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": [],
        "adapter": {"trainable_encoder_layers": 1},
    }

    def write(manifest):
        (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))

    write({**good, "format_version": "0.9"})
    with pytest.raises(ValueError, match="format_version"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": good["files"] * 2})
    with pytest.raises(ValueError, match="exactly one file"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "other.safetensors"}]})
    with pytest.raises(ValueError, match="must name exactly"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "../" + pl.ARTIFACT_WEIGHTS_NAME}]})
    with pytest.raises(ValueError, match="must name exactly|inside the artifact directory"):
        pipe.load_artifact(tmp_path)
    write({**good, "base_model": {**good["base_model"], "weight_file": "other.bin"}})
    with pytest.raises(ValueError, match="different base weight file"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {}})
    with pytest.raises(ValueError, match="trainable_encoder_layers"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {"trainable_encoder_layers": ENCODER_LAYERS + 1}})
    with pytest.raises(ValueError, match="trainable_encoder_layers"):
        pipe.load_artifact(tmp_path)
    write(good)  # every manifest check passes; the weights file is still missing, and no model was imported
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
