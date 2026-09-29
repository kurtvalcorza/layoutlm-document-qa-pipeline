"""Capstone-owned LayoutLM document-QA extractor (Systems B and C).

Reproduces the host pipeline's inference semantics (question words + OCR words, 512-token windows
with stride 128, document-token p_mask, spans of at most 15 tokens, score = softmax(start) x
softmax(end) inside the winning window) without importing or changing the host package. The span
score is a ranking signal, not a calibrated probability, and the decoder has no learned no-answer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from receipt_common import FIELDS, QUESTIONS, IntegrityError, download_pinned, verify_file
from receipt_fields import check_document, field_result

MAX_SEQ_LEN = 512
DOC_STRIDE = 128
MAX_ANSWER_TOKENS = 15
MAX_WORDS = 2000
BOX_GRID = 1000
ENCODER_LAYERS = 12
TRAINABLE_LAYERS = 4
MODEL_HOSTS = ("huggingface.co",)
SNAPSHOT_FILES = (
    "config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def stage_snapshot(destination: Path, manifest: Mapping[str, Any]) -> Path:
    """Download (or re-verify) every manifest file at the pinned revision."""
    for entry in manifest["files"]:
        url = f"https://huggingface.co/{manifest['modelId']}/resolve/{manifest['revision']}/{entry['path']}"
        download_pinned(url, destination / entry["path"], entry, MODEL_HOSTS, max_bytes=700 * 1024**2)
    return verify_snapshot(destination, manifest)


def verify_snapshot(root: Path, manifest: Mapping[str, Any]) -> Path:
    listed = {e["path"] for e in manifest["files"]}
    if not set(SNAPSHOT_FILES) <= listed:
        raise IntegrityError("Model manifest lacks required snapshot files")
    for entry in manifest["files"]:
        verify_file(Path(root) / entry["path"], entry)
    return Path(root)


def normalize_box(box: Sequence[float], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = (float(v) for v in box)
    return [
        int(BOX_GRID * (x0 / width)),
        int(BOX_GRID * (y0 / height)),
        int(BOX_GRID * (x1 / width)),
        int(BOX_GRID * (y1 / height)),
    ]


def trainable_names(model: Any, layers: int = TRAINABLE_LAYERS) -> list[str]:
    first = ENCODER_LAYERS - layers
    prefixes = tuple(f"layoutlm.encoder.layer.{k}." for k in range(first, ENCODER_LAYERS)) + ("qa_outputs.",)
    return [n for n, _ in model.named_parameters() if n.startswith(prefixes)]


def tensor_digest(state: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().to("cpu").contiguous()
        h.update(name.encode() + b"\0" + str(tuple(tensor.shape)).encode() + str(tensor.dtype).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


class LayoutLMExtractor:
    """Fixed-question extractor over one OCR document. ``system_id`` labels its outputs."""

    def __init__(self, model: Any, tokenizer: Any, device: str, system_id: str, identity: dict[str, Any]):
        if not tokenizer.is_fast:
            raise IntegrityError("A fast tokenizer is required for word_ids/sequence_ids")
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.system_id, self.identity = system_id, identity
        self.sep_id = tokenizer.sep_token_id

    @classmethod
    def load(
        cls,
        snapshot: Path,
        manifest: Mapping[str, Any],
        *,
        device: str | None = None,
        system_id: str = "layoutlm_frozen",
        verify: bool = True,
    ) -> LayoutLMExtractor:
        import torch
        from transformers import AutoTokenizer, LayoutLMForQuestionAnswering

        if verify:
            verify_snapshot(snapshot, manifest)
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        common = {"local_files_only": True, "trust_remote_code": False}
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), **common)
        model = LayoutLMForQuestionAnswering.from_pretrained(str(snapshot), dtype=torch.float32, **common)
        model = model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        weights = next(e for e in manifest["files"] if e["path"] == "model.safetensors")
        identity = {
            "model_id": manifest["modelId"],
            "revision": manifest["revision"],
            "weight_sha256": weights["sha256"],
            "tokenizer_sha256": {
                e["path"]: e["sha256"]
                for e in manifest["files"]
                if e["path"].startswith(("tokenizer", "vocab", "merges"))
            },
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        return cls(model, tokenizer, device, system_id, identity)

    # ---- encoding --------------------------------------------------------------------------------
    def encode(self, question: str, words: list[str], *, windows: bool = True, padding: Any = "max_length"):
        kwargs = {
            "max_length": MAX_SEQ_LEN,
            "truncation": "only_second",
            "return_token_type_ids": True,
            "padding": padding,
            "return_tensors": "pt",
        }
        if windows:
            kwargs.update(stride=DOC_STRIDE, return_overflowing_tokens=True)
        return self.tokenizer(text=question.split(), text_pair=words, is_split_into_words=True, **kwargs)

    def count_windows(self, question: str, words: list[str]) -> int:
        return int(self.encode(question, words)["input_ids"].shape[0])

    def window_bboxes(self, encoding: Any, window: int, grid: Sequence[Sequence[int]]) -> list[list[int]]:
        out = []
        for input_id, sequence_id, word_id in zip(
            encoding["input_ids"][window].tolist(),
            encoding.sequence_ids(window),
            encoding.word_ids(window),
            strict=True,
        ):
            if sequence_id == 1:
                out.append(list(grid[word_id]))
            elif input_id == self.sep_id:
                out.append([BOX_GRID] * 4)
            else:
                out.append([0] * 4)
        return out

    # ---- inference -------------------------------------------------------------------------------
    def spans(self, document: Mapping[str, Any], questions: Mapping[str, str]) -> dict[str, dict[str, Any]]:
        """Best word span per field over every window; windows of all fields run in one batch."""
        import torch

        words = [str(t["text"]) for t in document["tokens"]]
        width, height = document["image_size"]
        grid = [normalize_box(t["box"], width, height) for t in document["tokens"]]
        encoded, rows = {}, []
        for field, question in questions.items():
            enc = self.encode(question, words)
            encoded[field] = enc
            rows += [(field, w) for w in range(enc["input_ids"].shape[0])]
        batch = {
            k: torch.cat([encoded[f][k][w : w + 1] for f, w in rows]).to(self.device)
            for k in ("input_ids", "attention_mask", "token_type_ids")
        }
        batch["bbox"] = torch.tensor([self.window_bboxes(encoded[f], w, grid) for f, w in rows]).to(
            self.device
        )
        with torch.inference_mode():
            outputs = self.model(**batch)
        best = {
            f: {
                "start": None,
                "end": None,
                "score": 0.0,
                "n_windows": int(encoded[f]["input_ids"].shape[0]),
                "window": None,
            }
            for f in questions
        }
        for index, (field, window) in enumerate(rows):
            sequence_ids = encoded[field].sequence_ids(window)
            word_ids = encoded[field].word_ids(window)
            allowed = torch.tensor([sid == 1 for sid in sequence_ids], device=self.device)
            start = outputs.start_logits[index].float().masked_fill(~allowed, float("-inf")).softmax(-1)
            end = outputs.end_logits[index].float().masked_fill(~allowed, float("-inf")).softmax(-1)
            candidates = start[:, None] * end[None, :]
            candidates = torch.triu(candidates) - torch.triu(candidates, diagonal=MAX_ANSWER_TOKENS)
            flat = int(candidates.argmax())
            s_index, e_index = divmod(flat, candidates.shape[1])
            score = float(candidates[s_index, e_index])
            if (
                score > best[field]["score"]
                and word_ids[s_index] is not None
                and word_ids[e_index] is not None
            ):
                best[field].update(start=word_ids[s_index], end=word_ids[e_index], score=score, window=window)
        return best

    def predict(
        self,
        document: Mapping[str, Any],
        policy: str,
        currency: str = "unspecified",
        questions: Mapping[str, str] = QUESTIONS,
    ) -> dict[str, Any]:
        """Input-only prediction: OCR document + fixed questions -> four field results."""
        failure = check_document(document, MAX_WORDS)
        if failure:
            return {
                f: field_result(
                    [], None, score=None, policy=policy, currency=currency, status="failed", reasons=[failure]
                )
                for f in FIELDS
            }
        spans = self.spans(document, {f: questions[f] for f in FIELDS})
        out = {}
        for field in FIELDS:
            span = spans[field]
            if span["start"] is None:
                out[field] = field_result(
                    [],
                    None,
                    score=None,
                    policy=policy,
                    currency=currency,
                    status="no_candidate",
                    reasons=["no_valid_span"],
                )
            else:
                out[field] = field_result(
                    document["tokens"],
                    list(range(span["start"], span["end"] + 1)),
                    score=span["score"],
                    policy=policy,
                    currency=currency,
                )
            out[field]["n_windows"] = span["n_windows"]
        return out

    # ---- adapter tensors -------------------------------------------------------------------------
    def trainable_state(self) -> dict[str, Any]:
        names = set(trainable_names(self.model))
        return {k: v.detach().to("cpu").clone() for k, v in self.model.state_dict().items() if k in names}

    def frozen_digest(self) -> str:
        names = set(trainable_names(self.model))
        return tensor_digest({k: v for k, v in self.model.state_dict().items() if k not in names})

    def apply_adapter(self, tensors: Mapping[str, Any]) -> None:
        state = self.model.state_dict()
        allowed = set(trainable_names(self.model))
        if set(tensors) != allowed:
            raise IntegrityError("Adapter tensor set differs from the declared trainable subset")
        for key, value in tensors.items():
            if tuple(value.shape) != tuple(state[key].shape):
                raise IntegrityError(f"Adapter tensor {key} has the wrong shape")
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in tensors.items()})
        self.model.load_state_dict(merged, strict=True)
        self.model.eval()


def model_config_record(extractor: LayoutLMExtractor) -> dict[str, Any]:
    config = json.loads(extractor.model.config.to_json_string())
    return {
        "architecture": "LayoutLMForQuestionAnswering",
        "config": config,
        "max_seq_len": MAX_SEQ_LEN,
        "doc_stride": DOC_STRIDE,
        "max_answer_tokens": MAX_ANSWER_TOKENS,
        "max_words": MAX_WORDS,
        "box_grid": BOX_GRID,
        "trainable_encoder_layers": TRAINABLE_LAYERS,
    }
