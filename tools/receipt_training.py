"""Training-only OCR-to-reference alignment and bounded LayoutLM fine-tuning (spec section 10).

Supervision uses the recognised OCR tokens; reference text never replaces them. Epoch selection is
capstone-owned: epochs 1..4 only, highest validation_model total exact match, earliest epoch on ties.
The host pipeline's ``adapt()`` (validation ANLS, may keep epoch 0) is neither called nor changed.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from receipt_common import FIELDS, QUESTIONS, ContractError
from receipt_fields import amount, box_union, check_document, intersection_over_area
from receipt_models import MAX_ANSWER_TOKENS, LayoutLMExtractor, normalize_box, trainable_names

RECIPE = {
    "id": "receipt-layoutlm-v1",
    "initialization": "fresh verified upstream base",
    "trainable": "last 4 encoder blocks + qa_outputs",
    "optimizer": "AdamW",
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "epochs": 4,
    "batch_size": 8,
    "precision": "float32",
    "gradient_clip_norm": 1.0,
    "scheduler": None,
    "seed": 42,
    "loss": "start/end span cross-entropy on aligned OCR tokens",
    "max_examples_per_receipt": 4,
    "selection": "highest validation_model total-amount normalized EM over epochs 1-4; earliest tie",
    "min_examples": 8,
    "min_receipts": 2,
    "alignment_min_ioa": 0.5,
    "max_span_words": 4,
}


def align_field(
    document: Mapping[str, Any],
    reference: Mapping[str, Any],
    policy: str,
    min_ioa: float = 0.5,
    max_words: int = 4,
) -> dict[str, Any]:
    """Find the unique OCR span that equals the reference amount inside its labelled value region.

    Candidates are contiguous spans (same OCR line, at most ``max_words`` words) whose normalised
    amount equals the reference and whose box has intersection-over-area >= ``min_ioa`` with a
    labelled value box. Nested candidates collapse to the shortest (a currency marker is optional
    evidence); two or more non-nested survivors are ambiguous. Nothing is repaired or synthesised.
    """
    if reference.get("state") != "present_usable":
        return {"status": "no_usable_reference"}
    tokens = document["tokens"]
    target = reference["normalized_amount"]
    regions = reference.get("value_boxes") or []
    found = []
    for start in range(len(tokens)):
        for end in range(start, min(len(tokens), start + max_words)):
            if tokens[end]["line"] != tokens[start]["line"]:
                break
            text = " ".join(str(tokens[i]["text"]) for i in range(start, end + 1))
            parsed = amount(text, policy)
            if parsed["parse_status"] != "ok" or parsed["normalized_amount"] != target:
                continue
            box = box_union([tokens[i]["box"] for i in range(start, end + 1)])
            if any(intersection_over_area(box, region) >= min_ioa for region in regions):
                found.append((start, end))
    if not found:
        return {"status": "target_not_recoverable_from_ocr"}
    minimal = [s for s in found if not any(o != s and s[0] <= o[0] and o[1] <= s[1] for o in found)]
    if len(minimal) != 1:
        return {"status": "ambiguous_alignment", "candidates": minimal}
    start, end = minimal[0]
    return {"status": "aligned", "start": start, "end": end}


def build_examples(
    documents: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    extractor: LayoutLMExtractor,
    policy: str,
) -> dict[str, Any]:
    """Aligned positive-span examples from TRAIN receipts, plus every skip by receipt and field."""
    examples, skips = [], []
    eligibility = Counter()
    for rid in sorted(documents):
        document, ref = documents[rid], references[rid]
        failure = check_document(document)
        per_receipt = 0
        for field in FIELDS:
            reference = ref["fields"][field]
            if reference["state"] != "present_usable":
                eligibility[f"{field}:{reference['state']}"] += 1
                continue
            if failure:
                skips.append({"receipt_id": rid, "field": field, "reason": failure})
                continue
            result = align_field(document, reference, policy)
            if result["status"] != "aligned":
                skips.append({"receipt_id": rid, "field": field, "reason": result["status"]})
                continue
            words = [str(t["text"]) for t in document["tokens"]]
            if extractor.count_windows(QUESTIONS[field], words) != 1:
                skips.append({"receipt_id": rid, "field": field, "reason": "training_window_limit"})
                continue
            encoded = extractor.encode(QUESTIONS[field], words, windows=False, padding=False)
            positions = [
                p
                for p, (sid, wid) in enumerate(zip(encoded.sequence_ids(0), encoded.word_ids(0), strict=True))
                if sid == 1 and wid is not None and result["start"] <= wid <= result["end"]
            ]
            if not positions or positions[-1] - positions[0] + 1 > MAX_ANSWER_TOKENS:
                skips.append({"receipt_id": rid, "field": field, "reason": "answer_token_limit"})
                continue
            if per_receipt >= RECIPE["max_examples_per_receipt"]:
                skips.append({"receipt_id": rid, "field": field, "reason": "per_receipt_ceiling"})
                continue
            per_receipt += 1
            examples.append(
                {
                    "receipt_id": rid,
                    "field": field,
                    "question": QUESTIONS[field],
                    "start_word": result["start"],
                    "end_word": result["end"],
                    "start_token": positions[0],
                    "end_token": positions[-1],
                }
            )
            eligibility[f"{field}:aligned"] += 1
    receipts = {e["receipt_id"] for e in examples}
    summary = {
        "examples": len(examples),
        "receipts_with_examples": len(receipts),
        "skips_by_reason": dict(sorted(Counter(s["reason"] for s in skips).items())),
        "by_field": {
            f: {
                "usable_references": sum(
                    1 for r in references.values() if r["fields"][f]["state"] == "present_usable"
                ),
                "aligned": sum(1 for e in examples if e["field"] == f),
            }
            for f in FIELDS
        },
        "eligibility_counts": dict(sorted(eligibility.items())),
    }
    if len(examples) < RECIPE["min_examples"] or len(receipts) < RECIPE["min_receipts"]:
        raise ContractError(
            f"Only {len(examples)} aligned training examples from {len(receipts)} receipts; "
            "the E2E training stage requires at least 8 from multiple receipts"
        )
    return {"examples": examples, "skips": skips, "summary": summary}


def select_epoch(history: Sequence[Mapping[str, Any]]) -> int:
    """Highest validation_model total EM among trained epochs (>=1); earliest epoch on ties."""
    trained = [h for h in history if h["epoch"] >= 1]
    if not trained or any(h.get("validation_total_em") is None for h in trained):
        raise ContractError("Epoch selection needs a defined validation total EM for every trained epoch")
    best = max(h["validation_total_em"] for h in trained)
    return min(h["epoch"] for h in trained if h["validation_total_em"] == best)


def seed_everything(seed: int) -> dict[str, int]:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {"python": seed, "numpy": seed, "torch": seed}


def train(
    extractor: LayoutLMExtractor,
    documents: Mapping[str, Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
    evaluate_epoch: Callable[[int], dict[str, Any]],
    recipe: Mapping[str, Any] = RECIPE,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Bounded fine-tuning; returns history, selected epoch and the selected trainable tensors.

    ``evaluate_epoch(epoch)`` scores validation_model with the model in its current state and
    returns at least ``validation_total_em``. The caller owns references; this function never sees them.
    """
    import torch

    seeds = seed_everything(recipe["seed"])
    model, device = extractor.model, extractor.device
    names = trainable_names(model)
    wanted = set(names)
    frozen_before = extractor.frozen_digest()
    initial = extractor.trainable_state()
    for name, param in model.named_parameters():
        param.requires_grad_(name in wanted)
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    optimiser = torch.optim.AdamW(params, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    generator = torch.Generator().manual_seed(recipe["seed"])
    history, states, steps = [], {}, 0
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(1, recipe["epochs"] + 1):
            model.train()
            order = torch.randperm(len(examples), generator=generator).tolist()
            losses = []
            for offset in range(0, len(order), recipe["batch_size"]):
                batch = [examples[i] for i in order[offset : offset + recipe["batch_size"]]]
                encoded = extractor.tokenizer(
                    [e["question"].split() for e in batch],
                    [[str(t["text"]) for t in documents[e["receipt_id"]]["tokens"]] for e in batch],
                    is_split_into_words=True,
                    padding=True,
                    truncation="only_second",
                    max_length=512,
                    return_token_type_ids=True,
                    return_tensors="pt",
                )
                bbox = []
                for i, e in enumerate(batch):
                    document = documents[e["receipt_id"]]
                    w, h = document["image_size"]
                    grid = [normalize_box(t["box"], w, h) for t in document["tokens"]]
                    bbox.append(extractor.window_bboxes(encoded, i, grid))
                out = model(
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                    token_type_ids=encoded["token_type_ids"].to(device),
                    bbox=torch.tensor(bbox).to(device),
                    start_positions=torch.tensor([e["start_token"] for e in batch], device=device),
                    end_positions=torch.tensor([e["end_token"] for e in batch], device=device),
                )
                if not torch.isfinite(out.loss):
                    raise ContractError(f"Non-finite training loss at epoch {epoch}")
                optimiser.zero_grad(set_to_none=True)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(params, recipe["gradient_clip_norm"])
                optimiser.step()
                steps += 1
                losses.append(float(out.loss.detach()))
            model.eval()
            with torch.no_grad():
                validation = evaluate_epoch(epoch)
            entry = {
                "epoch": epoch,
                "train_loss": sum(losses) / len(losses),
                "optimizer_steps": steps,
                **validation,
            }
            history.append(entry)
            states[epoch] = extractor.trainable_state()
            log(
                f"epoch {epoch}: train loss {entry['train_loss']:.4f}, validation_model total EM "
                f"{entry['validation_total_em']}"
            )
    finally:
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
    selected = select_epoch(history)
    extractor.apply_adapter(states[selected])
    frozen_after = extractor.frozen_digest()
    if frozen_after != frozen_before:
        raise ContractError("A frozen tensor changed during training")
    changed = [n for n in names if not torch.equal(states[selected][n], initial[n])]
    if steps <= 0 or not changed:
        raise ContractError("Training produced no optimizer step or no permitted tensor change")
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
    return {
        "history": history,
        "selected_epoch": selected,
        "tensors": states[selected],
        "trainable_names": names,
        "n_trainable": n_trainable,
        "n_total": n_total,
        "optimizer_steps": steps,
        "changed_tensors": len(changed),
        "frozen_digest": frozen_before,
        "seeds": seeds,
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated_bytes": peak,
        "gpu_peak_reserved_bytes": reserved,
        "recipe": dict(recipe),
    }
