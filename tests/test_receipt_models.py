"""LayoutLM extractor, alignment and bounded training with a tiny RANDOM model (interface evidence only)."""

from pathlib import Path

import pytest

from _receipt_fixtures import document, tiny_snapshot

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import receipt_models as models  # noqa: E402
import receipt_training as training  # noqa: E402
from receipt_common import FIELDS, ContractError, IntegrityError  # noqa: E402


@pytest.fixture(scope="module")
def extractor(tmp_path_factory):
    base = tmp_path_factory.mktemp("tiny")
    snapshot, manifest = tiny_snapshot(base)
    return models.LayoutLMExtractor.load(snapshot, manifest, device="cpu")


def test_snapshot_verification_refuses_altered_weights(tmp_path: Path):
    snapshot, manifest = tiny_snapshot(tmp_path)
    (snapshot / "model.safetensors").write_bytes(b"x")
    with pytest.raises(IntegrityError):
        models.LayoutLMExtractor.load(snapshot, manifest, device="cpu")


def test_predictions_are_contiguous_ocr_spans_with_matching_text(extractor):
    doc = document([["SUBTOTAL", "70.000"], ["TAX", "7.000"], ["TOTAL", "77.000"]])
    out = extractor.predict(doc, "cord_mixed_v1")
    assert set(out) == set(FIELDS)
    for result in out.values():
        ids = result["token_ids"]
        assert ids == list(range(ids[0], ids[-1] + 1))
        assert result["raw_text"] == " ".join(doc["tokens"][i]["text"] for i in ids)
        assert 0 <= result["span_score"] <= 1 and result["n_windows"] == 1
    again = extractor.predict(doc, "cord_mixed_v1")
    assert again == out


def test_long_documents_are_windowed_not_truncated(extractor):
    doc = document([["word"] * 10] * 70)  # 700 words -> several 512-token windows
    out = extractor.predict(doc, "cord_mixed_v1")
    assert out["total_amount"]["n_windows"] > 1


def test_input_limit_is_a_structured_failure(extractor):
    out = extractor.predict(document([["w"] * 5] * 401), "cord_mixed_v1")
    assert {v["parse_status"] for v in out.values()} == {"failed"}
    assert out["total_amount"]["reasons"] == ["input_limit_exceeded"]


def test_adapter_tensor_allowlist(extractor):
    state = extractor.trainable_state()
    assert len(state) == 4 * 16 + 2 and all(
        k.startswith(
            (
                "layoutlm.encoder.layer.8",
                "layoutlm.encoder.layer.9",
                "layoutlm.encoder.layer.10",
                "layoutlm.encoder.layer.11",
                "qa_outputs.",
            )
        )
        for k in state
    )
    with pytest.raises(IntegrityError):
        extractor.apply_adapter({k: v for k, v in list(state.items())[:-1]})
    bad = dict(state)
    key = next(iter(bad))
    bad[key] = torch.zeros(3)
    with pytest.raises(IntegrityError):
        extractor.apply_adapter(bad)


REF = {"state": "present_usable", "normalized_amount": "77000"}


def test_alignment_unique_nested_ambiguous_and_unrecoverable():
    doc = document([["TOTAL", "Rp", "77.000"], ["CASH", "100.000"]])
    box = doc["tokens"][2]["box"]
    aligned = training.align_field(doc, {**REF, "value_boxes": [box]}, "cord_mixed_v1")
    assert aligned == {
        "status": "aligned",
        "start": 2,
        "end": 2,
    }  # nested "Rp 77.000" collapses to the number
    elsewhere = training.align_field(doc, {**REF, "value_boxes": [[0, 200, 5, 205]]}, "cord_mixed_v1")
    assert elsewhere["status"] == "target_not_recoverable_from_ocr"  # right value, wrong region
    twice = document([["77.000", "77.000"]])
    region = [0, 0, 600, 100]
    assert (
        training.align_field(twice, {**REF, "value_boxes": [region]}, "cord_mixed_v1")["status"]
        == "ambiguous_alignment"
    )
    misread = document([["TOTAL", "77.00O"]])
    assert (
        training.align_field(misread, {**REF, "value_boxes": [region]}, "cord_mixed_v1")["status"]
        == "target_not_recoverable_from_ocr"
    )
    assert (
        training.align_field(doc, {"state": "not_annotated"}, "cord_mixed_v1")["status"]
        == "no_usable_reference"
    )


def test_epoch_selection_excludes_epoch_zero_and_breaks_ties_early():
    history = [
        {"epoch": 0, "validation_total_em": 0.9},
        {"epoch": 1, "validation_total_em": 0.4},
        {"epoch": 2, "validation_total_em": 0.6},
        {"epoch": 3, "validation_total_em": 0.6},
        {"epoch": 4, "validation_total_em": 0.5},
    ]
    assert training.select_epoch(history) == 2
    with pytest.raises(ContractError):
        training.select_epoch([{"epoch": 1, "validation_total_em": None}])


def _train_set(n=6):
    docs, refs = {}, {}
    for k in range(n):
        value = f"{10 + k}.000"
        doc = document([["SUBTOTAL", value], ["TOTAL", value]])
        docs[f"r{k}"] = doc
        refs[f"r{k}"] = {
            "fields": {
                "total_amount": {
                    "state": "present_usable",
                    "normalized_amount": f"{10 + k}000",
                    "value_boxes": [doc["tokens"][3]["box"]],
                },
                "subtotal_amount": {
                    "state": "present_usable",
                    "normalized_amount": f"{10 + k}000",
                    "value_boxes": [doc["tokens"][1]["box"]],
                },
                "tax_amount": {"state": "not_annotated"},
                "service_charge": {"state": "not_annotated"},
            }
        }
    return docs, refs


def test_minimum_supervision_is_enforced(extractor):
    docs, refs = _train_set(3)
    with pytest.raises(ContractError, match="at least 8"):
        training.build_examples(docs, refs, extractor, "cord_mixed_v1")


def test_bounded_training_changes_only_permitted_tensors(tmp_path: Path):
    snapshot, manifest = tiny_snapshot(tmp_path)
    ex = models.LayoutLMExtractor.load(snapshot, manifest, device="cpu", system_id="layoutlm_adapted")
    docs, refs = _train_set(6)
    built = training.build_examples(docs, refs, ex, "cord_mixed_v1")
    assert built["summary"]["examples"] == 12
    scores = iter([0.5, 0.25, 0.5, 0.25])
    recipe = {**training.RECIPE, "learning_rate": 1e-3}
    before = ex.trainable_state()
    result = training.train(
        ex,
        docs,
        built["examples"],
        lambda epoch: {"validation_total_em": next(scores)},
        recipe=recipe,
        log=lambda _m: None,
    )
    assert [h["epoch"] for h in result["history"]] == [1, 2, 3, 4] and result["selected_epoch"] == 1
    assert result["optimizer_steps"] == 8 and result["changed_tensors"] > 0
    assert result["n_trainable"] < result["n_total"]
    assert ex.frozen_digest() == result["frozen_digest"]
    after = ex.trainable_state()
    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert not any(p.requires_grad for p in ex.model.parameters())
