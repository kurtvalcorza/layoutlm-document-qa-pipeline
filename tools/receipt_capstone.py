"""Stage runner for the Small-Business Receipt Intelligence capstone.

Each stage runs in its own process (the notebook calls ``python capstone.py --root R --stage S``) and
commits its outputs atomically through ``StageStore``; a stage refuses to start unless every upstream
success receipt, output digest, source digest and configuration digest verifies. Learner-facing copies
of stage outputs are written to ``<root>/outputs``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import receipt_artifact as artifact
import receipt_byod as byod
import receipt_data as data
import receipt_fields as fields
import receipt_metrics as metrics
import receipt_ocr as ocr
import receipt_policy as policy
from receipt_common import (
    FIELDS,
    QUESTIONS,
    ContractError,
    IntegrityError,
    StageStore,
    digest,
    file_hash,
    read_json,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)

HERE = Path(__file__).resolve().parent  # sibling modules import because the script directory leads sys.path
NOTEBOOK_REVISION = "0.1.0-candidate"
PREDICTION_SCHEMA = "org.dimer.receipt-intelligence.prediction.v1"
SYSTEMS = ("rules_baseline", "layoutlm_frozen", "layoutlm_adapted")
EVAL_ROLES = ("validation_model", "validation_policy", "test")
CARRIED_SOURCES = (
    "capstone.py",
    "receipt_common.py",
    "receipt_fields.py",
    "receipt_data.py",
    "receipt_ocr.py",
    "receipt_models.py",
    "receipt_training.py",
    "receipt_metrics.py",
    "receipt_policy.py",
    "receipt_artifact.py",
    "receipt_byod.py",
)
CANONICAL_GRAPH = {
    "prepare": [],
    "references": ["prepare"],
    "ocr_runtime": [],
    "model": [],
    "ocr": ["prepare", "ocr_runtime"],
    "keywords": ["references"],
    "rules": ["ocr", "keywords"],
    "frozen": ["ocr", "model"],
    "train": ["ocr", "references", "model", "frozen"],
    "select_policy": ["rules", "frozen", "train", "references"],
    "freeze": ["select_policy"],
    "evaluate": ["freeze"],
    "diagnose": ["evaluate"],
    "export": ["evaluate"],
    "replay": ["export"],
    "report": ["replay", "diagnose"],
}
BYOD_INFERENCE_GRAPH = {
    "prepare": [],
    "ocr_runtime": [],
    "model": [],
    "ocr": ["prepare", "ocr_runtime"],
    "byod_infer": ["ocr", "model"],
}
STAGE_OUTPUTS: dict[str, list[str]] = {}


# ---- context ------------------------------------------------------------------------------------


class Run:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = read_json(root / "run_config.json")
        mode = self.config.get("mode", "canonical")
        self.graph = BYOD_INFERENCE_GRAPH if mode == "byod_inference" else CANONICAL_GRAPH
        code = {name: file_hash(HERE / name) for name in CARRIED_SOURCES if (HERE / name).exists()}
        self.code = code
        self.store = StageStore(root / "stages", digest(code), self.config, self.graph)
        self.outputs = root / "outputs"
        self.outputs.mkdir(parents=True, exist_ok=True)
        (self.outputs / "figures").mkdir(exist_ok=True)
        self.image_store = root / "work" / "images"
        self.cache = Path(self.config.get("cache_dir") or (root / "cache"))
        self.policy_id = self.config.get("number_format_policy", fields.DEFAULT_POLICY)
        self.currency = self.config.get("currency", "unspecified")
        self.manifests = {
            name: read_json(HERE / f"{name}_manifest.json") for name in ("data", "split", "model", "ocr")
        }

    def stage_dir(self, name: str) -> Path:
        self.store.verify(name)
        return self.store.root / name

    def public(self, name: str, source: Path) -> None:
        target = self.outputs / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    def references(self, purpose: str, roles: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        reader = data.ReferenceReader(
            self.stage_dir("references") / "references.jsonl", self.outputs / "reference_access_log.jsonl"
        )
        return reader.get(roles, purpose)

    def cohort(self) -> list[dict[str, Any]]:
        return read_jsonl(self.stage_dir("prepare") / "cohort.jsonl")

    def documents(self, roles: tuple[str, ...] | None = None) -> dict[str, dict[str, Any]]:
        cohort = {r["receipt_id"]: r["role"] for r in self.cohort()}
        out = {}
        for row in read_jsonl(self.stage_dir("ocr") / "ocr.jsonl"):
            if roles is None or cohort[row["receipt_id"]] in roles:
                out[row["receipt_id"]] = ocr.as_document(row)
        return out

    def roles(self) -> dict[str, str]:
        return {r["receipt_id"]: r["role"] for r in self.cohort()}

    def ocr_engine(self):
        """Return (engine, identity) for the installed, re-verified OCR runtime."""
        identity = read_json(self.stage_dir("ocr_runtime") / "ocr_runtime.json")
        if identity.get("test_double"):
            table = read_json(Path(self.config["ocr_test_double"]))
            engine, live = ocr.test_double_engine(table, self.manifests["ocr"])
        else:
            executable, tessdata = Path(identity["executable_path"]), Path(identity["tessdata_path"])
            live = ocr.verify_engine(
                executable, tessdata, self.manifests["ocr"], development=not identity["qualifying_runtime"]
            )
            engine = ocr.tesseract_engine(executable, tessdata, live["settings"])
        if (
            live["executable_sha256"] != identity["executable_sha256"]
            or live["language_sha256"] != identity["language_sha256"]
        ):
            raise IntegrityError("OCR engine or language data changed since installation")
        return engine, identity

    def model_snapshot(self) -> tuple[Path, dict[str, Any]]:
        info = read_json(self.stage_dir("model") / "model_identity.json")
        return Path(info["snapshot_path"]), info["manifest"]

    def load_extractor(self, system_id: str):
        from receipt_models import LayoutLMExtractor

        snapshot, manifest = self.model_snapshot()
        return LayoutLMExtractor.load(
            snapshot, manifest, system_id=system_id, device=self.config.get("device") or None
        )


def log(message: str) -> None:
    print(message, flush=True)


def resources(stage: str, started: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"stage": stage, "seconds": round(time.perf_counter() - started, 2)}
    try:
        import resource

        record["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except ImportError:
        try:
            import psutil

            record["rss_bytes_at_end"] = psutil.Process().memory_info().rss
        except ImportError:
            pass
    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available():
            record["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            record["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            record["gpu"] = torch.cuda.get_device_name(0)
    record.update(extra or {})
    return record


def prediction_record(
    run: Run,
    receipt_id: str,
    system_id: str,
    result: dict[str, Any],
    image_sha256: str,
    input_source: str = "actual_ocr",
) -> dict[str, Any]:
    return {
        "schema": PREDICTION_SCHEMA,
        "receipt_id": receipt_id,
        "image_sha256": image_sha256,
        "system_id": system_id,
        "input_source": input_source,
        "fields": result,
        # No input-side reconciliation contract exists in v1: subtotal + tax + service = total is not assumed.
        "arithmetic_check": {"state": "not_evaluable", "reason": "no complete reconciliation contract in v1"},
        "run_id": run.config["run_id"],
    }


def predictions_of(path: Path) -> dict[str, dict[str, Any]]:
    return {r["receipt_id"]: r for r in read_jsonl(path)}


def hash_order(ids, tag: str) -> list[str]:
    return sorted(ids, key=lambda i: hashlib.sha256(f"receipt-capstone-v1|{tag}|{i}".encode()).hexdigest())


def fmt_amount(value: Any) -> str:
    return "" if value is None else str(value)


# ---- figures ------------------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110}
    )
    return plt


def overlay(image, tokens, highlight: dict[str, list[int]], path: Path, title: str) -> None:
    from PIL import ImageDraw

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    colours = {
        "total_amount": (213, 94, 0),
        "subtotal_amount": (0, 114, 178),
        "tax_amount": (0, 158, 115),
        "service_charge": (204, 121, 167),
    }
    marked = {i: f for f, ids in highlight.items() for i in ids}
    for token in tokens:
        colour = colours.get(marked.get(token["id"]), (120, 120, 120))
        draw.rectangle(token["box"], outline=colour, width=3 if token["id"] in marked else 1)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 5 * canvas.height / max(canvas.width, 1)))
    ax.imshow(canvas)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)
    handles = [
        plt.Line2D([], [], color=[c / 255 for c in colour], lw=3, label=f)
        for f, colour in colours.items()
        if f in highlight and highlight[f]
    ]
    if handles:
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=2,
            fontsize=8,
            frameon=False,
        )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---- stages: data and OCR -----------------------------------------------------------------------


def fixture_roles(cohort: list[dict[str, Any]], fixture: dict[str, Any]) -> dict[str, str]:
    """Reduced CPU fixture only: deterministic re-roling of a small subset. Never canonical."""
    order = hash_order([r["receipt_id"] for r in cohort], "fixture")
    roles, cursor = {}, 0
    for role in ("train", "validation_model", "validation_policy", "test"):
        for rid in order[cursor : cursor + int(fixture["counts"][role])]:
            roles[rid] = role
        cursor += int(fixture["counts"][role])
    return roles


def stage_prepare(run: Run, out: Path) -> None:
    pending = run.root / "work" / "pending_references"
    if pending.exists():
        shutil.rmtree(pending)
    if run.config.get("source", "cord") == "byod":
        summary = byod.prepare_byod(
            Path(run.config["byod_archive"]),
            run.config["byod_mode"],
            out,
            pending,
            run.image_store,
            run.root / "work",
        )
    else:
        fixture = run.config.get("fixture")
        summary = data.prepare_cord(
            out,
            pending,
            run.image_store,
            run.cache / "cord",
            run.manifests,
            shard_filter=set(fixture["shards"]) if fixture else None,
        )
        if fixture:
            cohort = read_jsonl(out / "cohort.jsonl")
            roles = fixture_roles(cohort, fixture)
            cohort = [
                {**r, "role": roles[r["receipt_id"]], "fixture_source_role": r["role"]}
                for r in cohort
                if r["receipt_id"] in roles
            ]
            write_jsonl(out / "cohort.jsonl", cohort)
            refs = [
                {**r, "role": roles[r["receipt_id"]]}
                for r in read_jsonl(pending / "references.jsonl")
                if r["receipt_id"] in roles
            ]
            write_jsonl(pending / "references.jsonl", refs)
            card = read_json(out / "data_card.json")
            card.update(
                reduced_fixture=True,
                fixture=fixture,
                frozen_role_counts=dict(Counter(roles.values())),
                cohort_digest=digest(cohort),
                label="REDUCED CPU FIXTURE - mechanics only, not capstone evidence",
            )
            write_json(out / "data_card.json", card)
    cohort = read_jsonl(out / "cohort.jsonl")
    card = read_json(out / "data_card.json")
    nominal = card.get("nominal_counts", {})
    exclusions = Counter()
    split_roles = {r["receipt_id"]: r["role"] for r in run.manifests["split"]["rows"]}
    for item in card.get("exclusions", []):
        exclusions[split_roles.get(item["receipt_id"], "unknown")] += 1
    counts = Counter(r["role"] for r in cohort)
    rows = []
    for role in ("train", "validation_model", "validation_policy", "test", "inference"):
        if counts[role] or role in ("train", "validation_model", "validation_policy", "test"):
            source = {
                "train": "official train",
                "validation_model": "official validation (first 50 by hash)",
                "validation_policy": "official validation (remaining 50)",
                "test": "official test",
                "inference": "BYOD inference",
            }[role]
            rows.append(
                {
                    "role": role,
                    "source": source,
                    "excluded": exclusions[role],
                    "frozen_cohort": counts[role],
                    "permitted_use": {
                        "train": "gradient updates, keyword development, alignment",
                        "validation_model": "epoch selection only",
                        "validation_policy": "review-threshold selection and activity",
                        "test": "one frozen evaluation",
                        "inference": "unscored",
                    }[role],
                }
            )
    write_csv(
        out / "data_summary.csv", rows, ["role", "source", "excluded", "frozen_cohort", "permitted_use"]
    )
    run.public("data_summary.csv", out / "data_summary.csv")
    run.public("data_card.json", out / "data_card.json")
    log(
        f"Frozen cohort: {dict(counts)}; nominal source counts {nominal}; "
        f"summary {summary.get('role_counts')}"
    )


def stage_references(run: Run, out: Path) -> None:
    pending = run.root / "work" / "pending_references"
    for name in ("references.jsonl", "reference_audit.json"):
        shutil.move(str(pending / name), str(out / name))
    shutil.rmtree(pending, ignore_errors=True)
    refs = read_jsonl(out / "references.jsonl")
    table = []
    for role in ("train", "validation_model", "validation_policy", "test"):
        for field in FIELDS:
            states = Counter(r["fields"][field]["state"] for r in refs if r["role"] == role)
            table.append(
                {
                    "role": role,
                    "field": field,
                    **{
                        s: states.get(s, 0)
                        for s in (
                            "present_usable",
                            "not_annotated",
                            "ambiguous_reference",
                            "invalid_reference",
                        )
                    },
                }
            )
    write_csv(
        out / "reference_states.csv",
        table,
        ["role", "field", "present_usable", "not_annotated", "ambiguous_reference", "invalid_reference"],
    )
    examples = []
    for field in FIELDS:  # training receipts only
        for ref in hash_order(
            [
                r["receipt_id"]
                for r in refs
                if r["role"] == "train" and r["fields"][field]["state"] == "present_usable"
            ],
            "example",
        )[:2]:
            item = next(r for r in refs if r["receipt_id"] == ref)["fields"][field]
            examples.append(
                {
                    "receipt_id": ref,
                    "field": field,
                    "raw_text": item["raw_text"],
                    "normalized_amount": item["normalized_amount"],
                    "state": item["state"],
                }
            )
    for reason, count in sorted(read_json(out / "reference_audit.json").get("reason_counts", {}).items()):
        examples.append(
            {
                "receipt_id": "(count)",
                "field": "",
                "raw_text": reason,
                "normalized_amount": "",
                "state": f"{count} field references",
            }
        )
    write_csv(
        out / "reference_examples.csv",
        examples,
        ["receipt_id", "field", "raw_text", "normalized_amount", "state"],
    )
    for name in ("reference_states.csv", "reference_examples.csv", "reference_audit.json"):
        run.public(name, out / name)


def stage_ocr_runtime(run: Run, out: Path) -> None:
    override = run.config.get("ocr_override")
    shared = run.config.get("ocr_runtime_from")
    if run.config.get("ocr_test_double"):
        _, identity = ocr.test_double_engine(
            read_json(Path(run.config["ocr_test_double"])), run.manifests["ocr"]
        )
        write_json(out / "ocr_runtime.json", identity)
        log("OCR: labelled TEST DOUBLE (recorded tokens); not Tesseract evidence")
        return
    if override:
        executable, tessdata = Path(override["executable"]), Path(override["tessdata"])
        identity = ocr.verify_engine(executable, tessdata, run.manifests["ocr"], development=True)
        identity["note"] = "development OCR build on a non-qualifying platform"
    elif shared:
        source = read_json(Path(shared))
        executable, tessdata = Path(source["executable_path"]), Path(source["tessdata_path"])
        identity = ocr.verify_engine(executable, tessdata, run.manifests["ocr"])
        if identity["executable_sha256"] != source["executable_sha256"]:
            raise IntegrityError("Shared OCR runtime changed")
        identity.update({k: source[k] for k in ("install_method",) if k in source})
    else:
        runtime = run.root / "runtime"
        runtime.mkdir(exist_ok=True)
        identity = ocr.install_tesseract(runtime, run.manifests["ocr"], run.cache)
        executable = runtime / "ocr-env" / run.manifests["ocr"]["executable"]
        tessdata = runtime / "tessdata"
    identity.update(executable_path=str(executable), tessdata_path=str(tessdata))
    write_json(out / "ocr_runtime.json", identity)
    run.public("ocr_runtime.json", out / "ocr_runtime.json")
    log(f"OCR engine: {identity['engine_version']} ({identity['leptonica']}); languages eng+ind verified")


def stage_model(run: Run, out: Path) -> None:
    from receipt_models import stage_snapshot, verify_snapshot

    override = run.config.get("model_override")
    if override:
        manifest = read_json(Path(override["manifest"]))
        snapshot = verify_snapshot(Path(override["snapshot"]), manifest)
    else:
        manifest = run.manifests["model"]
        snapshot = stage_snapshot(run.cache / "model", manifest)
    info = {
        "snapshot_path": str(snapshot),
        "manifest": manifest,
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "fixture_model": bool(override),
        "note": "tiny randomly initialised test model" if override and override.get("random") else None,
    }
    write_json(out / "model_identity.json", info)
    log(f"Base checkpoint verified: {manifest['modelId']}@{manifest['revision'][:12]}")


def stage_ocr(run: Run, out: Path) -> None:
    engine, identity = run.ocr_engine()
    cohort = run.cohort()
    started = time.perf_counter()

    def load(record):
        return data.cohort_image(run.image_store, record)

    rows = ocr.run_cohort(
        cohort,
        load,
        identity,
        engine,
        run.cache / "ocr",
        workers=run.config.get("ocr_workers"),
        progress=lambda n, total: log(f"  OCR {n}/{total} receipts"),
    )
    write_jsonl(out / "ocr.jsonl", rows)
    roles = {r["receipt_id"]: r["role"] for r in cohort}
    summary = []
    for role in sorted(set(roles.values())):
        items = [r for r in rows if roles[r["receipt_id"]] == role]
        statuses = Counter(r["ocr_status"] for r in items)
        words = sorted(len(r["tokens"]) for r in items)
        summary.append(
            {
                "role": role,
                "receipts": len(items),
                **{
                    s: statuses.get(s, 0)
                    for s in ("ok", "ocr_empty", "ocr_timeout", "ocr_error", "image_error")
                },
                "over_2000_words": sum(1 for w in words if w > 2000),
                "median_words": words[len(words) // 2] if words else 0,
                "max_words": words[-1] if words else 0,
                "cache_hits": sum(1 for r in items if r.get("cache_hit")),
            }
        )
    write_csv(out / "ocr_summary.csv", summary, list(summary[0]))
    run.public("ocr_summary.csv", out / "ocr_summary.csv")
    # Inspect one TRAINING receipt: tokens, boxes and confidences (no evaluation receipt is shown).
    train_ids = hash_order(
        [
            r["receipt_id"]
            for r in cohort
            if r["role"] == "train" and next(o for o in rows if o["receipt_id"] == r["receipt_id"])["tokens"]
        ],
        "inspect",
    )
    if train_ids:
        rid = train_ids[0]
        record = next(r for r in cohort if r["receipt_id"] == rid)
        tokens = next(o for o in rows if o["receipt_id"] == rid)["tokens"]
        image, _ = data.cohort_image(run.image_store, record)
        amounts = [t["id"] for t in tokens if fields.is_amount_token(t["text"], run.policy_id)]
        overlay(
            image,
            tokens,
            {"total_amount": amounts},
            run.outputs / "figures" / "ocr_inspection.png",
            f"Training receipt {rid}: all OCR words (grey), parseable amount tokens (orange)",
        )
        write_csv(
            run.outputs / "ocr_tokens_example.csv",
            [
                {
                    "token_id": t["id"],
                    "text": t["text"],
                    "box_xyxy": " ".join(str(v) for v in t["box"]),
                    "line": "/".join(str(v) for v in t["line"]),
                    "confidence": t["confidence"],
                    "parseable_amount": t["id"] in amounts,
                }
                for t in tokens
            ],
            ["token_id", "text", "box_xyxy", "line", "confidence", "parseable_amount"],
        )
    write_json(out / "ocr_stage.json", resources("ocr", started, {"receipts": len(rows)}))
    log(f"OCR finished for {len(rows)} receipts in {time.perf_counter() - started:.0f} s")


def stage_keywords(run: Run, out: Path) -> None:
    source = run.config.get("keyword_rules_from")
    if source:
        rules = read_json(Path(source))
        rules = {**rules, "transferred_from": "canonical CORD training keyword dictionary"}
    else:
        counts: dict[str, Counter] = {}
        for ref in run.references("keyword_development", ("train",)).values():
            for item in ref["key_phrases"]:
                counts.setdefault(item["phrase"], Counter())[item["category"]] += 1
        rules = fields.derive_keywords({k: dict(v) for k, v in counts.items()})
    write_json(out / "keyword_rules.json", rules)
    table = [
        {
            "field": f,
            "phrase": " ".join(k["phrase"]),
            "training_support": k["support"],
            "purity": round(k["purity"], 3),
        }
        for f in FIELDS
        for k in rules["fields"][f]
    ]
    write_csv(out / "keyword_rules.csv", table, ["field", "phrase", "training_support", "purity"])
    run.public("keyword_rules.csv", out / "keyword_rules.csv")
    log(f"Keyword dictionary: {len(table)} phrases from training annotations only")


def _validation_table(run: Run, name: str, system: str, preds: dict[str, dict]) -> None:
    refs = run.references("model_selection", ("validation_model",))
    ids = sorted(refs)
    result = metrics.system_metrics(preds, refs, ids)
    rows = [
        {
            "system": system,
            "field": f,
            "usable_references": result["fields"][f]["usable_references"],
            "correct": result["fields"][f]["correct"],
            "em": result["fields"][f]["em"],
            "parse_coverage": result["fields"][f]["parse_coverage"],
        }
        for f in FIELDS
    ]
    path = run.outputs / name
    existing = list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []
    existing = [r for r in existing if r["system"] != system] + rows
    write_csv(path, existing, ["system", "field", "usable_references", "correct", "em", "parse_coverage"])


def stage_rules(run: Run, out: Path) -> None:
    rules = read_json(run.stage_dir("keywords") / "keyword_rules.json")
    cohort = {r["receipt_id"]: r for r in run.cohort()}
    docs = run.documents(EVAL_ROLES + ("inference",))
    records = [
        prediction_record(
            run,
            rid,
            "rules_baseline",
            fields.rule_extract(doc, rules, run.policy_id, run.currency),
            cohort[rid]["image_sha256"],
        )
        for rid, doc in sorted(docs.items())
    ]
    write_jsonl(out / "predictions_rules.jsonl", records)
    if any(cohort[r]["role"] == "validation_model" for r in docs):
        _validation_table(
            run,
            "validation_comparison.csv",
            "rules_baseline",
            predictions_of(out / "predictions_rules.jsonl"),
        )
    log(f"Rule baseline predictions: {len(records)} receipts x 4 fields")


def stage_frozen(run: Run, out: Path) -> None:
    extractor = run.load_extractor("layoutlm_frozen")
    cohort = {r["receipt_id"]: r for r in run.cohort()}
    docs = run.documents(EVAL_ROLES + ("inference",))
    started, records = time.perf_counter(), []
    for n, (rid, doc) in enumerate(sorted(docs.items()), 1):
        records.append(
            prediction_record(
                run,
                rid,
                "layoutlm_frozen",
                extractor.predict(doc, run.policy_id, run.currency),
                cohort[rid]["image_sha256"],
            )
        )
        if n % 50 == 0:
            log(f"  frozen LayoutLM {n}/{len(docs)}")
    write_jsonl(out / "predictions_frozen.jsonl", records)
    write_json(out / "frozen_stage.json", resources("frozen", started, {"model": extractor.identity}))
    if any(cohort[r]["role"] == "validation_model" for r in docs):
        _validation_table(
            run,
            "validation_comparison.csv",
            "layoutlm_frozen",
            predictions_of(out / "predictions_frozen.jsonl"),
        )
    log(f"Frozen LayoutLM predictions: {len(records)} receipts in {time.perf_counter() - started:.0f} s")


# ---- stages: adaptation, policy, freeze, evaluation ------------------------------------------------


def stage_train(run: Run, out: Path) -> None:
    import receipt_training as training
    from safetensors.torch import save_file

    extractor = run.load_extractor("layoutlm_adapted")  # fresh verified base, no prior adapter
    cohort = {r["receipt_id"]: r for r in run.cohort()}
    train_docs = run.documents(("train",))
    train_refs = run.references("training_supervision", ("train",))
    built = training.build_examples(train_docs, train_refs, extractor, run.policy_id)
    write_json(
        out / "alignment.json",
        {"summary": built["summary"], "skips": built["skips"], "recipe": training.RECIPE},
    )
    val_docs = run.documents(("validation_model",))
    val_refs = run.references("model_selection", ("validation_model",))
    val_ids = sorted(val_docs)
    if not any(val_refs[i]["fields"]["total_amount"]["state"] == "present_usable" for i in val_ids):
        raise ContractError("validation_model has no usable total reference; the selection rule cannot run")
    epoch_predictions: dict[int, dict[str, Any]] = {}

    def evaluate_epoch(epoch: int) -> dict[str, Any]:
        preds = {
            rid: {"fields": extractor.predict(val_docs[rid], run.policy_id, run.currency)} for rid in val_ids
        }
        epoch_predictions[epoch] = preds
        result = metrics.system_metrics(preds, val_refs, val_ids)
        return {
            "validation_total_em": result["total_em"],
            "validation_total_correct": result["total_correct"],
            "validation_total_usable": result["total_usable"],
            "validation_field_em": {f: result["fields"][f]["em"] for f in FIELDS},
        }

    log(
        f"Training on {built['summary']['examples']} aligned OCR examples from "
        f"{built['summary']['receipts_with_examples']} receipts"
    )
    trained = training.train(extractor, train_docs, built["examples"], evaluate_epoch, log=log)
    save_file(
        {k: v.contiguous() for k, v in sorted(trained["tensors"].items())},
        str(out / "adapter.safetensors"),
        metadata={"format": "pt"},
    )
    history = {
        "history": trained["history"],
        "selected_epoch": trained["selected_epoch"],
        "selection_rule": training.RECIPE["selection"],
        "trainable_tensors": trained["trainable_names"],
        "n_trainable": trained["n_trainable"],
        "n_total": trained["n_total"],
        "optimizer_steps": trained["optimizer_steps"],
        "changed_tensors": trained["changed_tensors"],
        "frozen_tensor_digest": trained["frozen_digest"],
        "seeds": trained["seeds"],
        "seconds": trained["seconds"],
        "gpu_peak_allocated_bytes": trained["gpu_peak_allocated_bytes"],
        "gpu_peak_reserved_bytes": trained["gpu_peak_reserved_bytes"],
        "recipe": trained["recipe"],
        "alignment": built["summary"],
        "adapter_sha256": file_hash(out / "adapter.safetensors"),
        "base_identity": extractor.identity,
        "note": "the adapted candidate is selected from epochs 1-4 even if it loses to the frozen model",
    }
    write_json(out / "training_history.json", history)
    run.public("training_history.json", out / "training_history.json")
    # Predictions of the selected adapted model on every evaluation role (scoring happens later).
    docs = run.documents(EVAL_ROLES + ("inference",))
    records = [
        prediction_record(
            run,
            rid,
            "layoutlm_adapted",
            extractor.predict(doc, run.policy_id, run.currency),
            cohort[rid]["image_sha256"],
        )
        for rid, doc in sorted(docs.items())
    ]
    write_jsonl(out / "predictions_adapted.jsonl", records)
    _validation_table(
        run,
        "validation_comparison.csv",
        "layoutlm_adapted",
        predictions_of(out / "predictions_adapted.jsonl"),
    )
    skips = Counter(s["reason"] for s in built["skips"])
    rows = [{"field": f, **built["summary"]["by_field"][f]} for f in FIELDS]
    rows += [
        {"field": f"skipped: {k}", "usable_references": "", "aligned": v} for k, v in sorted(skips.items())
    ]
    write_csv(run.outputs / "alignment_summary.csv", rows, ["field", "usable_references", "aligned"])
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    epochs = [h["epoch"] for h in trained["history"]]
    ax[0].plot(epochs, [h["train_loss"] for h in trained["history"]], marker="o", color="#0072B2")
    ax[0].set(
        xlabel="Epoch", ylabel="Mean training loss", title="Training loss (train receipts)", xticks=epochs
    )
    ax[1].plot(epochs, [h["validation_total_em"] for h in trained["history"]], marker="o", color="#D55E00")
    ax[1].axvline(trained["selected_epoch"], color="grey", ls="--", lw=1)
    ax[1].set(
        xlabel="Epoch",
        ylabel="Total exact match",
        title="validation_model total EM (selection)",
        xticks=epochs,
        ylim=(0, 1),
    )
    fig.tight_layout()
    fig.savefig(run.outputs / "figures" / "training_curve.png")
    plt.close(fig)
    write_json(out / "train_stage.json", resources("train", time.perf_counter() - trained["seconds"]))


def system_predictions(run: Run) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "rules_baseline": predictions_of(run.stage_dir("rules") / "predictions_rules.jsonl"),
        "layoutlm_frozen": predictions_of(run.stage_dir("frozen") / "predictions_frozen.jsonl"),
        "layoutlm_adapted": predictions_of(run.stage_dir("train") / "predictions_adapted.jsonl"),
    }


def _policy_figure(
    run: Run, sweeps: dict[str, list], chosen: dict[str, dict], path: Path, title: str
) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    colours = {"rules_baseline": "#999999", "layoutlm_frozen": "#0072B2", "layoutlm_adapted": "#D55E00"}
    for system, rows in sweeps.items():
        pts = [(r["coverage"], r["accuracy"]) for r in rows if r["accuracy"] is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=".", color=colours[system], label=system)
        pick = chosen[system]
        if pick["threshold"] is not None:
            ax.scatter(
                [pick["coverage"]],
                [pick["empirical_accuracy"]],
                s=80,
                facecolors="none",
                edgecolors=colours[system],
            )
    target = next(iter(chosen.values()))["targets"]
    ax.axhline(target["min_accuracy"], color="black", lw=0.8, ls=":")
    ax.axvline(target["min_coverage"], color="black", lw=0.8, ls=":")
    ax.set(
        xlabel="Coverage of scoreable validation_policy receipts",
        ylabel="Accuracy among unflagged totals",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
        title=title,
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def stage_select_policy(run: Run, out: Path, targets: dict[str, float] | None = None) -> dict[str, Any]:
    preds = system_predictions(run)
    refs = run.references("policy_selection", ("validation_policy",))
    ids = sorted(refs)
    chosen, sweeps = {}, {}
    for system in SYSTEMS:
        result = policy.select(preds[system], refs, ids, system, targets or policy.DEFAULT_TARGETS)
        chosen[system], sweeps[system] = result["policy"], result["sweep"]
    write_json(
        out / "review_policy.json",
        {
            "schema": "org.dimer.receipt-review-policy.v1",
            "systems": chosen,
            "states": ["needs_review", "total_unflagged"],
            "meaning": "routing of total amounts only; neither state is verification",
        },
    )
    rows = [
        {
            "system": s,
            "kind": r["kind"],
            "threshold": r["threshold"],
            "accepted": r["accepted"],
            "coverage": r["coverage"],
            "accuracy": r["accuracy"],
            "errors": r["errors"],
        }
        for s in SYSTEMS
        for r in sweeps[s]
    ]
    write_csv(
        out / "policy_sweep.csv",
        rows,
        ["system", "kind", "threshold", "accepted", "coverage", "accuracy", "errors"],
    )
    summary = [
        {
            "system": s,
            "policy_feasible": chosen[s]["policy_feasible"],
            "threshold": chosen[s]["threshold"],
            "selected_state": chosen[s]["selected_state"],
            "accepted": chosen[s]["accepted"],
            "selection_support": chosen[s]["selection_support"],
            "coverage": chosen[s]["coverage"],
            "empirical_accuracy": chosen[s]["empirical_accuracy"],
            "accuracy_interval": chosen[s]["empirical_accuracy_interval_wilson"],
        }
        for s in SYSTEMS
    ]
    write_csv(out / "policy_selection.csv", summary, list(summary[0]))
    for name in ("review_policy.json", "policy_sweep.csv", "policy_selection.csv"):
        run.public(name, out / name)
    _policy_figure(
        run,
        sweeps,
        chosen,
        run.outputs / "figures" / "policy_validation.png",
        "Review-policy development (validation_policy only)",
    )
    return chosen


def config_hashes(run: Run) -> dict[str, str]:
    return {
        "questions": digest(QUESTIONS),
        "number_format_policy": digest(number_policy_record(run)),
        "keyword_rules": digest(read_json(run.stage_dir("keywords") / "keyword_rules.json")),
        "preprocessing": digest(ocr.PREPROCESSING),
        "field_schema": digest(field_schema()),
        "review_policy": digest(read_json(run.stage_dir("select_policy") / "review_policy.json")),
    }


def number_policy_record(run: Run) -> dict[str, Any]:
    return {
        "policy_id": run.policy_id,
        "currency": run.currency,
        "grammar": "nonnegative integers; consistent 3-digit grouping with ',' or '.'; optional 2-digit "
        "decimal suffix after the other separator (cord_mixed_v1); single separator + 3 digits is "
        "grouping under this declared grammar",
        "never": ["O->0 substitution", "digit invention", "sign stripping", "summing components"],
    }


def field_schema() -> dict[str, Any]:
    return {
        "schema": PREDICTION_SCHEMA,
        "fields": list(FIELDS),
        "parse_status": list(fields.PARSE_STATES),
        "review_states": ["needs_review", "total_unflagged"],
        "evidence_status": "machine_extracted_unverified",
        "distinctions": "not_annotated, no_candidate, parse_failed and the value zero are different states",
    }


def stage_freeze(run: Run, out: Path) -> None:
    card = read_json(run.stage_dir("prepare") / "data_card.json")
    history = read_json(run.stage_dir("train") / "training_history.json")
    review = read_json(run.stage_dir("select_policy") / "review_policy.json")
    record = {
        "schema": "org.dimer.receipt-selection-record.v1",
        "notebook_revision": NOTEBOOK_REVISION,
        "run_id": run.config["run_id"],
        "cohort_digest": card["cohort_digest"],
        "split_manifest_digest": card.get("split_manifest_digest"),
        "reduced_fixture": card.get("reduced_fixture"),
        "data_identity": {k: card.get(k) for k in ("dataset_id", "revision", "license")},
        "model_identity": history["base_identity"],
        "ocr_identity": {
            k: v
            for k, v in read_json(run.stage_dir("ocr_runtime") / "ocr_runtime.json").items()
            if k not in ("executable_path", "tessdata_path")
        },
        "config_hashes": config_hashes(run),
        "training_recipe": history["recipe"],
        "selected_epoch": history["selected_epoch"],
        "adapter_sha256": history["adapter_sha256"],
        "review_policies": {
            s: {
                k: p[k]
                for k in (
                    "threshold",
                    "policy_feasible",
                    "selection_support",
                    "accepted",
                    "empirical_accuracy",
                    "policy_id",
                    "targets",
                )
            }
            for s, p in review["systems"].items()
        },
        "metric_definitions": metrics.METRIC_DEFINITIONS,
        "failure_semantics": "OCR/model/parse/ambiguity failures remain in denominators as incorrect",
        "primary_contrast": "layoutlm_adapted minus layoutlm_frozen total EM on test",
        "diagnostic_plan": [
            "reference_text_diagnostic (B and C, same weights)",
            "OCR recoverability (evaluator only)",
            "deterministic failure panel",
        ],
        "frozen_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "test_scored_before_freeze": False,
    }
    record["record_digest"] = digest(record)
    write_json(out / "selection_record.json", record)
    run.public("selection_record.json", out / "selection_record.json")
    log(f"Selection record frozen: {record['record_digest'][:16]}")


def verified_selection(run: Run) -> dict[str, Any]:
    record = read_json(run.stage_dir("freeze") / "selection_record.json")
    body = {k: v for k, v in record.items() if k != "record_digest"}
    if digest(body) != record["record_digest"]:
        raise IntegrityError("Selection record digest mismatch")
    if record["config_hashes"] != config_hashes(run):
        raise IntegrityError("Configuration changed after the selection record was frozen")
    if file_hash(run.stage_dir("train") / "adapter.safetensors") != record["adapter_sha256"]:
        raise IntegrityError("Adapter changed after the selection record was frozen")
    return record


def _comparison_figure(rows: list[dict[str, Any]], path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    colours = ["#999999", "#0072B2", "#D55E00"]
    for i, row in enumerate(rows):
        if row["total_em"] is None:
            continue
        low, high = row["total_em_interval"] or [row["total_em"], row["total_em"]]
        ax.bar(i, row["total_em"], color=colours[i % 3], width=0.6)
        ax.errorbar(
            i,
            row["total_em"],
            yerr=[[max(0.0, row["total_em"] - low)], [max(0.0, high - row["total_em"])]],
            color="black",
            capsize=4,
        )
        ax.text(
            i, min(1.0, high) + 0.03, f"{row['total_correct']}/{row['total_usable']}", ha="center", fontsize=8
        )
    ax.set_xticks(range(len(rows)), [r["system"] for r in rows])
    ax.set(
        ylabel="Total exact match (test)",
        ylim=(0, 1.12),
        title="Held-out test: total amount exact match (Wilson 95% interval)",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def stage_evaluate(run: Run, out: Path) -> None:
    record = verified_selection(run)
    preds = system_predictions(run)
    refs = run.references("final_evaluation", ("test",))
    ids = sorted(refs)
    result = {s: metrics.system_metrics(preds[s], refs, ids) for s in SYSTEMS}
    contrasts = {
        "adapted_minus_frozen": ("layoutlm_adapted", "layoutlm_frozen"),
        "frozen_minus_rules": ("layoutlm_frozen", "rules_baseline"),
        "adapted_minus_rules": ("layoutlm_adapted", "rules_baseline"),
    }
    boot = {
        name: metrics.paired_bootstrap(preds[a], preds[b], refs, ids) for name, (a, b) in contrasts.items()
    }
    review = read_json(run.stage_dir("select_policy") / "review_policy.json")["systems"]
    reviews = {s: policy.review_metrics(preds[s], refs, ids, review[s]) for s in SYSTEMS}
    card = read_json(run.stage_dir("prepare") / "data_card.json")
    write_json(
        out / "field_metrics.json",
        {
            "role": "test",
            "selection_record_digest": record["record_digest"],
            "systems": result,
            "paired_bootstrap": boot,
            "nominal_test": card.get("nominal_counts", {}).get("test"),
            "frozen_test": len(ids),
        },
    )
    write_json(
        out / "review_metrics.json",
        {s: {k: v for k, v in r.items() if k != "routes"} for s, r in reviews.items()},
    )
    comparison = [
        {
            "system": s,
            "frozen_receipts": result[s]["frozen_receipts"],
            "total_usable": result[s]["total_usable"],
            "total_correct": result[s]["total_correct"],
            "total_em": result[s]["total_em"],
            "total_em_interval": result[s]["total_em_interval_wilson"],
            "field_macro_em": result[s]["field_macro_em"],
            "all_annotated_fields_match": result[s]["all_annotated_fields_match"],
            **{f"{f}_em": result[s]["fields"][f]["em"] for f in FIELDS[1:]},
            "scored_coverage": reviews[s]["scored_coverage"],
            "selective_error": reviews[s]["selective_error"],
            "unflagged_scored": reviews[s]["unflagged_scored"],
            "policy_feasible": reviews[s]["policy_feasible"],
        }
        for s in SYSTEMS
    ]
    write_csv(out / "system_comparison.csv", comparison, list(comparison[0]))
    field_rows = [
        {
            "system": s,
            "field": f,
            "usable_references": result[s]["fields"][f]["usable_references"],
            "correct": result[s]["fields"][f]["correct"],
            "em": result[s]["fields"][f]["em"],
            "text_em": result[s]["fields"][f]["text_em"],
            "anls": result[s]["fields"][f]["anls"],
            "candidate_coverage": result[s]["fields"][f]["candidate_coverage"],
            "parse_coverage": result[s]["fields"][f]["parse_coverage"],
            "unscored_on_not_annotated": result[s]["fields"][f]["unscored_predictions_on_not_annotated"],
        }
        for s in SYSTEMS
        for f in FIELDS
    ]
    write_csv(out / "field_metrics.csv", field_rows, list(field_rows[0]))
    boot_rows = [
        {
            "contrast": k,
            "n": v["n"],
            "difference": v["difference"],
            "interval_low": (v["interval"] or [None, None])[0],
            "interval_high": (v["interval"] or [None, None])[1],
            "resamples": v["resamples"],
        }
        for k, v in boot.items()
    ]
    write_csv(out / "paired_bootstrap.csv", boot_rows, list(boot_rows[0]))
    failures = []
    for s in SYSTEMS:
        for f in FIELDS:
            states = Counter(preds[s][i]["fields"][f]["parse_status"] for i in ids)
            reasons = Counter(r for i in ids for r in preds[s][i]["fields"][f].get("reasons", []))
            failures.append(
                {
                    "system": s,
                    "field": f,
                    **{k: states.get(k, 0) for k in fields.PARSE_STATES},
                    "reasons": "; ".join(f"{k}={v}" for k, v in sorted(reasons.items())),
                }
            )
    write_csv(out / "failure_summary.csv", failures, ["system", "field", *fields.PARSE_STATES, "reasons"])
    review_rows = [
        {"system": s, **{k: v for k, v in reviews[s].items() if k not in ("routes", "policy_id")}}
        for s in SYSTEMS
    ]
    write_csv(out / "review_metrics.csv", review_rows, list(review_rows[0]))
    write_json(out / "routes.json", {s: reviews[s]["routes"] for s in SYSTEMS})
    for name in (
        "field_metrics.json",
        "review_metrics.json",
        "system_comparison.csv",
        "field_metrics.csv",
        "paired_bootstrap.csv",
        "failure_summary.csv",
        "review_metrics.csv",
    ):
        run.public(name, out / name)
    _comparison_figure(comparison, run.outputs / "figures" / "test_comparison.png")
    log("Held-out test evaluated once with the frozen selection record " + record["record_digest"][:16])


# ---- stages: diagnostics, export, replay, report ------------------------------------------------

PANEL_CATEGORIES = ("success", "ocr_loss", "extraction_error", "numeric_ambiguity", "failure_or_no_candidate")


def stage_diagnose(run: Run, out: Path) -> None:
    import receipt_models as models

    verified_selection(run)
    refs = run.references("reference_text_diagnostic", ("test",))
    ids = sorted(refs)
    cohort = {r["receipt_id"]: r for r in run.cohort()}
    docs = run.documents(("test",))
    rules = read_json(run.stage_dir("keywords") / "keyword_rules.json")
    # Reference-text inputs: words/boxes only; categories, is_key and parsed values are stripped.
    ref_docs = {}
    for rid in ids:
        tokens = [
            {"id": t["id"], "text": t["text"], "box": t["box"], "line": t["line"], "confidence": None}
            for t in refs[rid]["diagnostic_tokens"]
        ]
        ref_docs[rid] = {
            "receipt_id": rid,
            "image_size": cohort[rid]["image_size"],
            "tokens": tokens,
            "ocr_status": "ok" if tokens else "ocr_empty",
            "input_source": "reference_text_diagnostic",
        }
    diag_preds: dict[str, dict[str, dict]] = {}
    extractor = run.load_extractor("layoutlm_frozen")
    diag_preds["layoutlm_frozen"] = {
        rid: {"fields": extractor.predict(d, run.policy_id, run.currency)} for rid, d in ref_docs.items()
    }
    extractor.apply_adapter(artifact.load_tensors(run.stage_dir("train")))
    extractor.system_id = "layoutlm_adapted"
    diag_preds["layoutlm_adapted"] = {
        rid: {"fields": extractor.predict(d, run.policy_id, run.currency)} for rid, d in ref_docs.items()
    }
    diag_preds["rules_baseline"] = {
        rid: {"fields": fields.rule_extract(d, rules, run.policy_id, run.currency)}
        for rid, d in ref_docs.items()
    }
    primary = system_predictions(run)
    rows = []
    for system in SYSTEMS:
        a = metrics.system_metrics({i: primary[system][i] for i in ids}, refs, ids)
        b = metrics.system_metrics(diag_preds[system], refs, ids)
        for f in FIELDS:
            rows.append(
                {
                    "system": system,
                    "field": f,
                    "usable_references": a["fields"][f]["usable_references"],
                    "actual_ocr_em": a["fields"][f]["em"],
                    "reference_text_diagnostic_em": b["fields"][f]["em"],
                }
            )
    write_csv(
        out / "reference_text_diagnostic.csv",
        rows,
        ["system", "field", "usable_references", "actual_ocr_em", "reference_text_diagnostic_em"],
    )
    write_jsonl(
        out / "reference_text_predictions.jsonl",
        [
            {
                "receipt_id": rid,
                "system_id": s,
                "input_source": "reference_text_diagnostic",
                "fields": diag_preds[s][rid]["fields"],
            }
            for s in SYSTEMS
            for rid in ids
        ],
    )
    # OCR recoverability (evaluator-only; never a predictor feature or a primary-score filter).
    recover = {
        rid: {f: metrics.recoverability(docs[rid], refs[rid]["fields"][f], run.policy_id) for f in FIELDS}
        for rid in ids
    }
    rec_rows = []
    for f in FIELDS:
        states = Counter(recover[rid][f] for rid in ids)
        scoreable = [rid for rid in ids if recover[rid][f] != "not_scoreable"]
        recoverable = [rid for rid in scoreable if recover[rid][f].startswith("recoverable")]
        for system in SYSTEMS:
            hits = [
                rid
                for rid in recoverable
                if metrics.correct(primary[system][rid]["fields"][f], refs[rid]["fields"][f])
            ]
            rec_rows.append(
                {
                    "field": f,
                    "system": system,
                    "usable_references": len(scoreable),
                    "ocr_recoverable": len(recoverable),
                    "recoverable_ambiguous": states["recoverable_ambiguous"],
                    "not_recoverable": states["not_recoverable"],
                    "recoverable_proportion": metrics.rate(len(recoverable), len(scoreable)),
                    "em_given_recoverable_secondary": metrics.rate(len(hits), len(recoverable)),
                }
            )
    write_csv(out / "ocr_recoverability.csv", rec_rows, list(rec_rows[0]))
    # Deterministic failure panel for the adapted system, selected after scores were fixed.
    adapted = primary["layoutlm_adapted"]
    chosen, used = [], set()
    for category in PANEL_CATEGORIES:
        pick = None
        for rid in hash_order(ids, "panel"):
            ref, pred = refs[rid]["fields"]["total_amount"], adapted[rid]["fields"]["total_amount"]
            if rid in used or ref["state"] != "present_usable":
                continue
            ok = metrics.correct(pred, ref)
            rec = recover[rid]["total_amount"].startswith("recoverable")
            status = pred["parse_status"]
            label = (
                "success"
                if ok
                else "failure_or_no_candidate"
                if status in ("failed", "no_candidate")
                else "numeric_ambiguity"
                if status in ("ambiguous", "unsupported", "parse_failed")
                else "extraction_error"
                if rec
                else "ocr_loss"
            )
            if label == category:
                pick = rid
                break
        chosen.append({"category": category, "receipt_id": pick or "(no test example in this category)"})
        if pick:
            used.add(pick)
    panel = []
    for item in chosen:
        rid = item["receipt_id"]
        if rid not in refs:
            panel.append(
                {
                    **item,
                    "reference": "",
                    "prediction_raw": "",
                    "prediction_amount": "",
                    "parse_status": "",
                    "recoverability": "",
                }
            )
            continue
        pred, ref = adapted[rid]["fields"]["total_amount"], refs[rid]["fields"]["total_amount"]
        panel.append(
            {
                **item,
                "reference": ref["normalized_amount"],
                "prediction_raw": pred["raw_text"],
                "prediction_amount": pred["normalized_amount"],
                "parse_status": pred["parse_status"],
                "recoverability": recover[rid]["total_amount"],
            }
        )
        image, _ = data.cohort_image(run.image_store, cohort[rid])
        overlay(
            image,
            docs[rid]["tokens"],
            {"total_amount": pred["token_ids"]},
            run.outputs / "figures" / f"panel_{item['category']}.png",
            f"[{item['category']}] {rid}: predicted {pred['raw_text']!r} -> {pred['normalized_amount']}; "
            f"reference {ref['normalized_amount']}",
        )
    write_csv(
        out / "failure_panel.csv",
        panel,
        [
            "category",
            "receipt_id",
            "reference",
            "prediction_raw",
            "prediction_amount",
            "parse_status",
            "recoverability",
        ],
    )
    for name in ("reference_text_diagnostic.csv", "ocr_recoverability.csv", "failure_panel.csv"):
        run.public(name, out / name)
    del models


def replay_ids(run: Run) -> list[str]:
    return hash_order([r["receipt_id"] for r in run.cohort() if r["role"] == "test"], "replay")[:3]


def stage_export(run: Run, out: Path) -> None:
    record = verified_selection(run)
    history = read_json(run.stage_dir("train") / "training_history.json")
    review = read_json(run.stage_dir("select_policy") / "review_policy.json")
    tensors = artifact.load_tensors(run.stage_dir("train"))
    parts = {
        "model_config": {
            "architecture": "LayoutLMForQuestionAnswering",
            "max_seq_len": 512,
            "doc_stride": 128,
            "max_answer_tokens": 15,
            "max_words": 2000,
            "box_grid": 1000,
            "trainable_encoder_layers": 4,
        },
        "field_schema": field_schema(),
        "questions": dict(QUESTIONS),
        "preprocessing": ocr.PREPROCESSING,
        "ocr_manifest": run.manifests["ocr"],
        "number_format_policy": number_policy_record(run),
        "keyword_rules": read_json(run.stage_dir("keywords") / "keyword_rules.json"),
        "review_policy": review,
        "runtime_manifest": {
            "ocr_identity": record["ocr_identity"],
            "notebook_revision": NOTEBOOK_REVISION,
            "python": sys.version.split()[0],
            "run_id": run.config["run_id"],
        },
    }
    bundle = out / "receipt_intelligence_artifact"
    manifest = artifact.export_bundle(
        bundle,
        tensors,
        parts,
        history["base_identity"],
        {
            "recipe": history["recipe"]["id"],
            "selected_epoch": history["selected_epoch"],
            "adapter_sha256": history["adapter_sha256"],
            "selection_record_digest": record["record_digest"],
        },
    )
    # Records every downstream process could consume (evaluation roles only; no references).
    preds = system_predictions(run)
    routes = read_json(run.stage_dir("evaluate") / "routes.json")
    roles = run.roles()
    lines, csv_rows = [], []
    for system in SYSTEMS:
        pol = review["systems"][system]
        for rid in sorted(preds[system]):
            rec = dict(preds[system][rid])
            rec["total_review"] = routes[system].get(rid) or policy.route(
                rec["fields"]["total_amount"], pol["threshold"], pol["policy_id"]
            )
            rec["role"] = roles[rid]
            lines.append(rec)
            row = {
                "receipt_id": rid,
                "role": roles[rid],
                "system_id": system,
                "input_source": rec["input_source"],
                "total_review_state": rec["total_review"]["state"],
                "total_review_reasons": ";".join(rec["total_review"]["reasons"]),
            }
            for f in FIELDS:
                item = rec["fields"][f]
                row.update(
                    {
                        f"{f}_raw_text": item["raw_text"],
                        f"{f}_normalized": fmt_amount(item["normalized_amount"]),
                        f"{f}_parse_status": item["parse_status"],
                        f"{f}_score": item["span_score"],
                    }
                )
            csv_rows.append(row)
    write_jsonl(out / "predictions.jsonl", lines)
    columns = [
        "receipt_id",
        "role",
        "system_id",
        "input_source",
        "total_review_state",
        "total_review_reasons",
    ]
    columns += [f"{f}_{k}" for f in FIELDS for k in ("raw_text", "normalized", "parse_status", "score")]
    write_csv(out / "records.csv", csv_rows, columns)
    # Pre-export expectations for the fresh-process replay (inputs + outputs, no references).
    ocr_rows = {r["receipt_id"]: r for r in read_jsonl(run.stage_dir("ocr") / "ocr.jsonl")}
    expected = {
        rid: {
            "tokens": ocr_rows[rid]["tokens"],
            "ocr_status": ocr_rows[rid]["ocr_status"],
            "rules": preds["rules_baseline"][rid]["fields"],
            "adapted": preds["layoutlm_adapted"][rid]["fields"],
            "route": routes["layoutlm_adapted"][rid],
        }
        for rid in replay_ids(run)
    }
    write_json(out / "replay_expected.json", expected)
    inventory = [
        {
            "file": name,
            "bytes": (bundle / name).stat().st_size,
            "sha256": (
                manifest["files"].get(name, {}).get("sha256")
                if name != "manifest.json"
                else file_hash(bundle / name)
            ),
        }
        for name in artifact.BUNDLE_FILES
    ]
    write_csv(out / "artifact_inventory.csv", inventory, ["file", "bytes", "sha256"])
    for name in ("records.csv", "artifact_inventory.csv"):
        run.public(name, out / name)
    log(
        f"Exported {len(manifest['tensors'])} trained tensors "
        f"({(bundle / 'adapter.safetensors').stat().st_size:,} bytes)"
    )


def _close(a: Any, b: Any, atol: float = 1e-6, rtol: float = 1e-5) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))


def compare_fields(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    problems = []
    for f in FIELDS:
        e, o = expected[f], observed[f]
        for key in ("raw_text", "normalized_amount", "parse_status", "token_ids"):
            if e.get(key) != o.get(key):
                problems.append(f"{f}.{key}")
        if not _close(o.get("span_score"), e.get("span_score")):
            problems.append(f"{f}.span_score")
    return problems


def stage_replay(run: Run, out: Path) -> None:
    """Fresh process: verify bundle -> fresh base -> rebuild config -> image-to-record -> parity."""
    from receipt_models import LayoutLMExtractor, trainable_names

    record = verified_selection(run)
    bundle = run.stage_dir("export") / "receipt_intelligence_artifact"
    expected_base = read_json(run.stage_dir("train") / "training_history.json")["base_identity"]
    pins = {
        "review_policy.json": record["config_hashes"]["review_policy"],
        "keyword_rules.json": record["config_hashes"]["keyword_rules"],
        "questions.json": record["config_hashes"]["questions"],
        "number_format_policy.json": record["config_hashes"]["number_format_policy"],
        "preprocessing.json": record["config_hashes"]["preprocessing"],
    }
    snapshot, manifest = run.model_snapshot()
    extractor = LayoutLMExtractor.load(
        snapshot, manifest, system_id="layoutlm_adapted", device=run.config.get("device") or None
    )
    bundle_manifest = artifact.verify_bundle(
        bundle,
        expected_base=expected_base,
        allowed_tensors=set(trainable_names(extractor.model)),
        expected=pins,
    )
    if bundle_manifest["base_model"] != extractor.identity:
        raise IntegrityError("Fresh base identity differs from the bundle")
    extractor.apply_adapter(artifact.load_tensors(bundle))
    questions = read_json(bundle / "questions.json")
    rules = read_json(bundle / "keyword_rules.json")
    grammar = read_json(bundle / "number_format_policy.json")
    pol = read_json(bundle / "review_policy.json")["systems"]["layoutlm_adapted"]
    ocr_lock = read_json(bundle / "ocr_manifest.json")
    engine, identity = run.ocr_engine()
    if ocr.settings_from(ocr_lock) != identity["settings"]:
        raise IntegrityError("OCR settings differ from the bundle")
    expected = read_json(run.stage_dir("export") / "replay_expected.json")
    cohort = {r["receipt_id"]: r for r in run.cohort()}
    report, rows = [], []
    for rid in sorted(expected):
        image, _ = data.cohort_image(run.image_store, cohort[rid])
        result = engine(image)
        document = {
            "receipt_id": rid,
            "image_size": list(image.size),
            "tokens": result["tokens"],
            "ocr_status": result["ocr_status"],
            "input_source": "actual_ocr",
        }
        adapted = extractor.predict(document, grammar["policy_id"], grammar["currency"], questions)
        rule_out = fields.rule_extract(document, rules, grammar["policy_id"], grammar["currency"])
        route = policy.route(adapted["total_amount"], pol["threshold"], pol["policy_id"])
        want = expected[rid]
        problems = []
        if [(t["text"], t["box"]) for t in result["tokens"]] != [
            (t["text"], t["box"]) for t in want["tokens"]
        ]:
            problems.append("ocr_tokens")
        problems += ["adapted." + p for p in compare_fields(want["adapted"], adapted)]
        problems += ["rules." + p for p in compare_fields(want["rules"], rule_out)]
        if route["state"] != want["route"]["state"]:
            problems.append("review_decision")
        report.append({"receipt_id": rid, "parity": not problems, "problems": problems})
        rows.append(
            {
                "receipt_id": rid,
                "total_raw_text": adapted["total_amount"]["raw_text"],
                "total_normalized": fmt_amount(adapted["total_amount"]["normalized_amount"]),
                "total_parse_status": adapted["total_amount"]["parse_status"],
                "total_score": adapted["total_amount"]["span_score"],
                "total_review_state": route["state"],
                "subtotal_normalized": fmt_amount(adapted["subtotal_amount"]["normalized_amount"]),
                "tax_normalized": fmt_amount(adapted["tax_amount"]["normalized_amount"]),
                "service_normalized": fmt_amount(adapted["service_charge"]["normalized_amount"]),
                "parity": not problems,
            }
        )
    verdict = {
        "pid": os.getpid(),
        "receipts": len(report),
        "all_parity": all(r["parity"] for r in report),
        "details": report,
        "tolerance": {"atol": 1e-6, "rtol": 1e-5},
        "labels_supplied": False,
        "note": "held-out demonstration replay, not a second independent test",
    }
    write_json(out / "replay_report.json", verdict)
    write_csv(out / "replay_records.csv", rows, list(rows[0]) if rows else ["receipt_id"])
    for name in ("replay_report.json", "replay_records.csv"):
        run.public(name, out / name)
    if not verdict["all_parity"]:
        raise IntegrityError(f"Fresh-process replay parity failed: {report}")
    log(f"Fresh-process replay (pid {os.getpid()}): {len(report)} receipts, parity PASS")


def stage_report(run: Run, out: Path) -> None:
    record = verified_selection(run)
    metrics_json = read_json(run.stage_dir("evaluate") / "field_metrics.json")
    review_json = read_json(run.stage_dir("evaluate") / "review_metrics.json")

    def num(value: Any) -> str:
        return "undefined" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)

    comparison = [
        {
            "system": sname,
            "total_em": num(m["total_em"]),
            "total_correct": m["total_correct"],
            "total_usable": m["total_usable"],
            "field_macro_em": num(m["field_macro_em"]),
            "scored_coverage": num(review_json[sname]["scored_coverage"]),
            "selective_error": num(review_json[sname]["selective_error"]),
        }
        for sname, m in metrics_json["systems"].items()
    ]
    boot = [
        {
            "contrast": k,
            "difference": num(v["difference"]),
            "interval_low": num((v["interval"] or [None])[0]),
            "interval_high": num((v["interval"] or [None, None])[1]),
            "n": v["n"],
        }
        for k, v in metrics_json["paired_bootstrap"].items()
    ]
    history = read_json(run.stage_dir("train") / "training_history.json")
    replay = read_json(run.stage_dir("replay") / "replay_report.json")
    card = read_json(run.stage_dir("prepare") / "data_card.json")
    resources_rows = []
    for stage in [g for g in run.graph if g != "report"]:
        receipt = read_json(run.stage_dir(stage) / "stage_receipt.json")
        resources_rows.append({"stage": stage, "seconds": round(receipt["elapsed_seconds"], 1)})
    extra = {}
    log_path = run.outputs / "resources.jsonl"
    if log_path.exists():
        for row in read_jsonl(log_path):
            extra[row["stage"]] = row
    for row in resources_rows:
        info = extra.get(row["stage"], {})
        row.update(
            peak_rss_gib=round(info["peak_rss_bytes"] / 1024**3, 2) if info.get("peak_rss_bytes") else "",
            gpu_peak_allocated_gib=round(info["gpu_peak_allocated_bytes"] / 1024**3, 2)
            if info.get("gpu_peak_allocated_bytes")
            else "",
        )
    write_csv(
        out / "runtime_summary.csv",
        resources_rows,
        ["stage", "seconds", "peak_rss_gib", "gpu_peak_allocated_gib"],
    )
    lines = [
        f"# Receipt intelligence capstone - run summary ({NOTEBOOK_REVISION})",
        "",
        f"Run `{run.config['run_id']}`; selection record `{record['record_digest'][:16]}`; data "
        f"{card.get('dataset_id')}@{str(card.get('revision'))[:12]}"
        + (" **(REDUCED CPU FIXTURE: mechanics only)**" if card.get("reduced_fixture") else ""),
        "",
        "| System | Test total EM | Correct / usable | Field-macro EM | Scored coverage | Selective error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in comparison:
        lines.append(
            f"| {row['system']} | {row['total_em']} | "
            f"{row['total_correct']}/{row['total_usable']} "
            f"| {row['field_macro_em']} | {row['scored_coverage']} | "
            f"{row['selective_error']} |"
        )
    lines += ["", "Paired receipt bootstrap (2,000 resamples) of total-EM differences:", ""]
    lines += [
        f"- {b['contrast']}: {b['difference']} "
        f"(95% interval {b['interval_low']} to {b['interval_high']}, n={b['n']})"
        for b in boot
    ]
    lines += [
        "",
        f"Selected epoch {history['selected_epoch']} of 4 (validation_model total EM); "
        f"{history['alignment']['examples']} aligned OCR training examples; fresh-process replay parity: "
        f"{'PASS' if replay['all_parity'] else 'FAIL'}.",
        "",
        "Scope: Indonesian CORD v2 receipts. No Philippine, unseen-merchant, production or accounting-grade "
        "claim follows. Review states are routing, not verification.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    html_rows = "".join(
        f"<tr><td>{r['system']}</td><td>{r['total_em']}</td><td>{r['total_correct']}/{r['total_usable']}</td></tr>"
        for r in comparison
    )
    (out / "summary.html").write_text(
        f"<html><body><h1>Receipt intelligence capstone {NOTEBOOK_REVISION}</h1><table border=1>"
        f"<tr><th>System</th><th>Test total EM</th><th>Correct/usable</th></tr>{html_rows}</table>"
        "<p>Routing states are not verification. Indonesian CORD v2 sample only.</p></body></html>\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "org.dimer.receipt-run-manifest.v1",
        "notebook_revision": NOTEBOOK_REVISION,
        "run_id": run.config["run_id"],
        "selection_record_digest": record["record_digest"],
        "carried_source_sha256": run.code,
        "stages": {
            s: digest(read_json(run.stage_dir(s) / "stage_receipt.json")) for s in run.graph if s != "report"
        },
        "reduced_fixture": card.get("reduced_fixture"),
        "python": sys.version.split()[0],
        "evidence_state": "hosted-executed only if this run happened on the documented Colab T4 runtime",
    }
    write_json(out / "run_manifest.json", manifest)
    # Evidence ZIP: derived evidence, manifests and the small trained bundle.
    # No receipt images, figures or OCR transcripts are included.
    members = {
        "summary.md": out / "summary.md",
        "summary.html": out / "summary.html",
        "run_manifest.json": out / "run_manifest.json",
        "runtime_summary.csv": out / "runtime_summary.csv",
        "selection_record.json": run.stage_dir("freeze") / "selection_record.json",
        "training_history.json": run.stage_dir("train") / "training_history.json",
        "review_policy.json": run.stage_dir("select_policy") / "review_policy.json",
        "policy_sweep.csv": run.stage_dir("select_policy") / "policy_sweep.csv",
        "review_metrics.json": run.stage_dir("evaluate") / "review_metrics.json",
        "field_metrics.json": run.stage_dir("evaluate") / "field_metrics.json",
        "system_comparison.csv": run.stage_dir("evaluate") / "system_comparison.csv",
        "failure_summary.csv": run.stage_dir("evaluate") / "failure_summary.csv",
        "paired_bootstrap.csv": run.stage_dir("evaluate") / "paired_bootstrap.csv",
        "reference_text_diagnostic.csv": run.stage_dir("diagnose") / "reference_text_diagnostic.csv",
        "ocr_recoverability.csv": run.stage_dir("diagnose") / "ocr_recoverability.csv",
        "predictions.jsonl": run.stage_dir("export") / "predictions.jsonl",
        "records.csv": run.stage_dir("export") / "records.csv",
        "replay_report.json": run.stage_dir("replay") / "replay_report.json",
        "data_card.json": run.stage_dir("prepare") / "data_card.json",
    }
    bundle = run.stage_dir("export") / "receipt_intelligence_artifact"
    for name in artifact.BUNDLE_FILES:
        members[f"receipt_intelligence_artifact/{name}"] = bundle / name
    archive = out / "results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle_zip:
        for name, path in sorted(members.items()):
            bundle_zip.write(path, name)
    checks = {"members": len(members), "bytes": archive.stat().st_size}
    with zipfile.ZipFile(archive) as bundle_zip:
        if bundle_zip.testzip() is not None:
            raise IntegrityError("results.zip CRC failure")
        names = bundle_zip.namelist()
        forbidden = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg", ".parquet"))]
        if forbidden or sorted(names) != sorted(members):
            raise IntegrityError(f"results.zip membership is wrong: {forbidden}")
        for name, path in members.items():
            if hashlib.sha256(bundle_zip.read(name)).hexdigest() != file_hash(path):
                raise IntegrityError(f"results.zip member differs: {name}")
    checks.update(
        crc="pass",
        member_digests="pass",
        excluded="receipt images, figures, OCR transcripts, weights of the base",
    )
    write_json(out / "archive_verification.json", checks)
    for name in (
        "summary.md",
        "runtime_summary.csv",
        "run_manifest.json",
        "results.zip",
        "archive_verification.json",
    ):
        run.public(name, out / name)


def run_activity(run: Run, target: float) -> dict[str, Any]:
    """Change-one-thing activity: a different accuracy target on cached validation_policy predictions.

    Writes only to outputs/activity/<target>/; the canonical policy, model and test results are untouched.
    """
    canonical = read_json(run.stage_dir("select_policy") / "review_policy.json")["systems"]
    preds = system_predictions(run)
    refs = run.references("policy_selection", ("validation_policy",))
    ids = sorted(refs)
    folder = run.outputs / "activity" / f"target_{target:.2f}"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    rows, chosen, sweeps = [], {}, {}
    for system in SYSTEMS:
        result = policy.activity(preds[system], refs, ids, system, target)
        chosen[system], sweeps[system] = result["policy"], result["sweep"]
        for label, pol in (
            ("canonical_0.95", canonical[system]),
            (f"activity_{target:.2f}", result["policy"]),
        ):
            rows.append(
                {
                    "system": system,
                    "setting": label,
                    "policy_feasible": pol["policy_feasible"],
                    "threshold": pol["threshold"],
                    "accepted": pol["accepted"],
                    "coverage": pol["coverage"],
                    "empirical_accuracy": pol["empirical_accuracy"],
                    "selection_support": pol["selection_support"],
                }
            )
    write_csv(folder / "activity_comparison.csv", rows, list(rows[0]))
    write_json(
        folder / "activity.json",
        {
            "target": target,
            "role": "validation_policy",
            "policies": chosen,
            "canonical_unchanged": True,
            "test_used": False,
        },
    )
    _policy_figure(
        run,
        sweeps,
        chosen,
        folder / "activity.png",
        f"Activity: accuracy target {target:.2f} (validation_policy)",
    )
    return {"folder": str(folder)}


def stage_byod_infer(run: Run, out: Path) -> None:
    """BYOD inference with the canonical bundle; routing is transferred, never treated as validated."""
    from receipt_models import LayoutLMExtractor, trainable_names

    bundle = Path(run.config["bundle_from"])
    snapshot, manifest = run.model_snapshot()
    extractor = LayoutLMExtractor.load(
        snapshot, manifest, system_id="layoutlm_adapted", device=run.config.get("device") or None
    )
    artifact.verify_bundle(
        bundle, expected_base=extractor.identity, allowed_tensors=set(trainable_names(extractor.model))
    )
    extractor.apply_adapter(artifact.load_tensors(bundle))
    rules = read_json(bundle / "keyword_rules.json")
    pol = read_json(bundle / "review_policy.json")["systems"]["layoutlm_adapted"]
    cohort = {r["receipt_id"]: r for r in run.cohort()}
    rows, lines = [], []
    for rid, doc in sorted(run.documents().items()):
        adapted = extractor.predict(doc, run.policy_id, run.currency)
        transferred = policy.route(adapted["total_amount"], pol["threshold"], pol["policy_id"])
        review = {
            "state": "needs_review",
            "reasons": ["byod_policy_not_validated"],
            "policy_id": "byod-default-refer",
            "transferred_cord_decision": transferred["state"],
        }
        lines.append(
            {
                **prediction_record(run, rid, "layoutlm_adapted", adapted, cohort[rid]["image_sha256"]),
                "total_review": review,
                "rules_baseline": fields.rule_extract(doc, rules, run.policy_id, run.currency),
            }
        )
        rows.append(
            {
                "receipt_id": rid,
                "ocr_status": doc["ocr_status"],
                "words": len(doc["tokens"]),
                **{f"{f}_raw_text": adapted[f]["raw_text"] for f in FIELDS},
                **{f"{f}_normalized": fmt_amount(adapted[f]["normalized_amount"]) for f in FIELDS},
                "total_review_state": review["state"],
                "transferred_cord_decision": transferred["state"],
            }
        )
    write_jsonl(out / "byod_predictions.jsonl", lines)
    write_csv(out / "byod_records.csv", rows, list(rows[0]))
    run.public("byod_records.csv", out / "byod_records.csv")
    log(f"BYOD inference: {len(rows)} receipts; no accuracy is reported without checked references")


STAGES = {
    "prepare": stage_prepare,
    "references": stage_references,
    "ocr_runtime": stage_ocr_runtime,
    "model": stage_model,
    "ocr": stage_ocr,
    "keywords": stage_keywords,
    "rules": stage_rules,
    "frozen": stage_frozen,
    "train": stage_train,
    "select_policy": stage_select_policy,
    "freeze": stage_freeze,
    "evaluate": stage_evaluate,
    "diagnose": stage_diagnose,
    "export": stage_export,
    "replay": stage_replay,
    "report": stage_report,
    "byod_infer": stage_byod_infer,
}


def run_stage(root: Path, stage: str) -> None:
    run = Run(root)
    if stage not in run.graph:
        raise ContractError(f"Stage {stage} is not part of this run's graph")
    started = time.perf_counter()
    with run.store.stage(stage) as out:
        STAGES[stage](run, out)
    record = resources(stage, started)
    with (run.outputs / "resources.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def inspect_byod(archive: Path, mode: str) -> dict[str, Any]:
    """Validate a BYOD ZIP's manifest before a run directory is configured (no OCR, no model)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        names = artifact.safe_extract(
            archive, Path(tmp) / "x", allowed_suffixes=(".json", ".jsonl", ".png", ".jpg", ".jpeg")
        )
        if "manifest.json" not in names:
            raise ContractError("The BYOD ZIP needs manifest.json at its root")
        return byod.validate_manifest(read_json(Path(tmp) / "x" / "manifest.json"), mode)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--stage")
    parser.add_argument("--activity-target", type=float)
    parser.add_argument("--byod-inspect", type=Path)
    parser.add_argument("--byod-mode", default="inference")
    args = parser.parse_args(argv)
    if args.byod_inspect is not None:
        print(json.dumps(inspect_byod(args.byod_inspect, args.byod_mode)))
        return
    if args.activity_target is not None:
        print(json.dumps(run_activity(Run(args.root), args.activity_target)))
        return
    run_stage(args.root, args.stage)


if __name__ == "__main__":
    main()
