"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): a referenced
evaluation, a one-epoch adaptation of the last encoder block on a dozen receipt questions, the artifact
round trip, the loader's scope check, the transactional guarantee and — where CUDA is visible — the same
path on the accelerator. Skipped when the weights are absent."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest
import torch

from layoutlm_document_qa_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    WEIGHT_FILE,
    LayoutLMDocumentQAPipeline,
    page_from_ground_truth,
    questions_for_page,
)

pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)


def _quad(x0, y0, x1, y1):
    return {"x1": x0, "y1": y0, "x2": x1, "y2": y0, "x3": x1, "y3": y1, "x4": x0, "y4": y1}


def _line(category, group_id, y, tokens):
    words, x = [], 20
    for text, is_key in tokens:
        for piece in text.split():
            words.append({"quad": _quad(x, y, x + 12 * len(piece), y + 16), "is_key": is_key, "text": piece})
            x += 12 * len(piece) + 8
    return {"words": words, "category": category, "group_id": group_id}


def _receipt(index):
    item, total = ["ES TEH", "KOPI SUSU", "NASI GORENG", "AIR MINERAL"][index % 4], 11_000 + 500 * index
    lines = [
        _line("menu.nm", 1, 40, [(item, 0)]),
        _line("menu.price", 1, 40, [(f"{total - 1000:,}", 0)]),
        _line("sub_total.subtotal_price", 5, 120, [("SUBTOTAL", 1), (f"{total - 1000:,}", 0)]),
        _line("sub_total.tax_price", 5, 140, [("PB1", 1), ("1,000", 0)]),
        _line("total.total_price", 9, 180, [("TOTAL", 1), (f"{total:,}", 0)]),
        _line("total.cashprice", 9, 200, [("CASH", 1), (f"{total + 9000:,}", 0)]),
    ]
    return {"meta": {"image_size": {"width": 400, "height": 300}}, "valid_line": lines}


RECORDS = []
for _index in range(4):
    for _question in questions_for_page(page_from_ground_truth(_receipt(_index), f"p{_index}")):
        RECORDS.append({"id": f"p{_index}-{len(RECORDS):02d}", **_question})
assert len(RECORDS) == 20


def _answers(pipe, records):
    return [
        pipe.answer(r["question"], words=r["words"], boxes=r["boxes"], image_size=r["image_size"])["answer"]
        for r in records
    ]


@pytest.fixture(scope="module")
def pipe():
    return LayoutLMDocumentQAPipeline.from_pretrained(device="cpu")


def test_evaluate_scores_gold_spans_and_check_fit_counts_windows(pipe):
    metrics = pipe.evaluate(RECORDS[:5])
    assert metrics["n"] == 5 and 0.0 <= metrics["anls"] <= 1.0 and metrics["adapted"] is False
    fit = pipe.check_fit(RECORDS)
    assert fit["n_fitting"] == 20 and fit["dropped"] == []
    long_page = {
        **RECORDS[0],
        "id": "long",
        "words": RECORDS[0]["words"] * 60,
        "boxes": RECORDS[0]["boxes"] * 60,
    }
    assert pipe.check_fit([long_page])["dropped"] == ["long"]


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, tmp_path):
    result = pipe.adapt(RECORDS[:12], RECORDS[12:], epochs=1, trainable_encoder_layers=1, batch_size=4)
    assert result["n_trainable"] == 7_089_410 and result["history"][0]["note"] == "frozen model"
    assert all(
        name.startswith("layoutlm.encoder.layer.11.") or name.startswith("qa_outputs.")
        for name in result["trainable_names"]
    )
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"]) == 18
    reloaded = LayoutLMDocumentQAPipeline.from_artifact(artifact, device="cpu")
    assert _answers(pipe, RECORDS[:4]) == _answers(reloaded, RECORDS[:4])
    assert reloaded.adapter["best_epoch"] == result["best_epoch"]
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, tmp_path):
    result = pipe.adapt(RECORDS[:12], None, epochs=2, trainable_encoder_layers=1, batch_size=4)
    assert result["best_epoch"] == 2 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 3
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = LayoutLMDocumentQAPipeline.from_artifact(artifact, device="cpu")
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])
    assert reloaded.adapter["best_epoch"] == 2 and reloaded.adapter["trainable_encoder_layers"] == 1


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(pipe, tmp_path):
    from safetensors.torch import load_file, save_file

    pipe.adapt(RECORDS[:12], None, epochs=1, trainable_encoder_layers=1, batch_size=4)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        LayoutLMDocumentQAPipeline.from_artifact(fewer, device="cpu")
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["zz.extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    files = [
        {**manifest["files"][0], "bytes": (extra / "adapter.safetensors").stat().st_size, "sha256": digest}
    ]
    (extra / "manifest.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        LayoutLMDocumentQAPipeline.from_artifact(extra, device="cpu")
    other_layers = tmp_path / "other_layers"
    shutil.copytree(artifact, other_layers)
    adapter = {**manifest["adapter"], "trainable_encoder_layers": 2}
    (other_layers / "manifest.json").write_text(json.dumps({**manifest, "adapter": adapter}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        LayoutLMDocumentQAPipeline.from_artifact(other_layers, device="cpu")


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe):
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(RECORDS[:12], None, epochs=2, trainable_encoder_layers=1, batch_size=4, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before) and pipe.adapter is None
    assert not any(p.requires_grad for p in pipe._model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_answer_adapt_and_reload_run_on_a_cuda_device(tmp_path):
    """Every tensor the runner and the trainer build must land on the model's device."""
    cuda = LayoutLMDocumentQAPipeline.from_pretrained(device="cuda:0")
    assert cuda.device == "cuda:0"
    first = cuda.answer(
        RECORDS[0]["question"],
        words=RECORDS[0]["words"],
        boxes=RECORDS[0]["boxes"],
        image_size=RECORDS[0]["image_size"],
    )
    assert first["device"] == "cuda:0" and 0.0 <= first["score"] <= 1.0
    result = cuda.adapt(RECORDS[:12], RECORDS[12:], epochs=1, trainable_encoder_layers=1, batch_size=4)
    assert result["best_epoch"] in (0, 1) and result["history"][1]["train_loss"] > 0.0
    artifact = cuda.save_artifact(tmp_path / "cuda")
    reloaded = LayoutLMDocumentQAPipeline.from_artifact(artifact, device="cuda:0")
    assert _answers(cuda, RECORDS[:4]) == _answers(reloaded, RECORDS[:4])
