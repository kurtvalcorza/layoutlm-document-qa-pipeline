"""Extractive document question answering with the pinned ``impira/layoutlm-document-qa`` checkpoint.

The class loads the tokenizer and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the LayoutLM architecture comes from the pinned ``transformers`` release,
the weights are SafeTensors, and no model-repository code is executed.

LayoutLM (v1) reads **words and their boxes**, not pixels: OCR is an input provider outside the model.
The pipeline therefore takes ``words``/``boxes`` from the caller (bring-your-own OCR) and offers
``ocr_words_with_tesseract`` as an optional adapter that is imported only when called.

The adaptation contract (``check_fit``, ``evaluate``, ``adapt``, ``save_artifact``, ``from_artifact``)
fine-tunes the last encoder blocks plus the span head on a validated ``{id, question, words, boxes,
image_size, answer_start, answer_end}`` dataset with validation-ANLS epoch selection and exports the trained
tensors as a safetensors adapter bound to the pinned base weights. The inference contract above is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "impira/layoutlm-document-qa"
MODEL_REVISION = "beed3c4d02d86017ebca5bd0fdf210046b907aa6"
MODEL_LICENSE = "mit"
MODEL_KEY = "layoutlm-document-qa"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHT_FILE = "model.safetensors"
WEIGHT_SHA256 = (
    "e4bbad3e4a1b5ae50c787b7afd6049a0bfa99fd823b50436e444e092ae2347b9"  # manifest digest of WEIGHT_FILE
)
PARAMETER_COUNT = 127_792_898
ENCODER_LAYERS = 12  # config.json num_hidden_layers
DEFAULT_TRAINABLE_ENCODER_LAYERS = 4  # the last four encoder blocks plus the span head (28,353,026 params)
MAX_EVAL_RECORDS = 2_000
MAX_RECORDS_FIT = 20_000  # check_fit accepts a whole dataset before it is split
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.layoutlm-document-qa.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"

# Encoding ceilings. The checkpoint's max_position_embeddings is 514 (RoBERTa layout: 512 usable
# tokens); longer documents are split into overlapping windows of MAX_SEQ_LEN with DOC_STRIDE overlap
# and the best-scoring span across windows is returned (the transformers document-question-answering
# pipeline's convention, as is MAX_ANSWER_TOKENS).
MAX_SEQ_LEN = 512
DOC_STRIDE = 128
MAX_ANSWER_TOKENS = 15
MAX_WORDS = 2000
MAX_QUESTION_CHARS = 256
# Box ceilings. Boxes are pixel xyxy in the page's coordinate frame and are normalised to the
# 0..1000 grid LayoutLM expects (max_2d_position_embeddings 1024).
MAX_IMAGE_SIDE = 10000
MIN_IMAGE_SIDE = 1
BOX_GRID = 1000
# ANLS (DocVQA's official metric): a normalised Levenshtein similarity below this threshold scores 0.
ANLS_THRESHOLD = 0.5
_PUNCT_RE = re.compile(r"[^\w\s]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def normalize_answer(text: str) -> str:
    """DocVQA-style normalisation: lower-case, punctuation removed, whitespace collapsed."""
    return " ".join(_PUNCT_RE.sub(" ", text.lower()).split())


def _levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def anls(prediction: str, golds: Sequence[str], *, threshold: float = ANLS_THRESHOLD) -> float:
    """Average Normalised Levenshtein Similarity for one question (Biten et al., ICDAR 2019).

    ``1 - lev(pred, gold) / max(len(pred), len(gold))`` over normalised strings, maximised over the
    accepted ``golds``; a similarity below ``threshold`` scores 0 so a near-miss is not rewarded.
    """
    if not golds:
        raise ValueError("golds must contain at least one accepted answer")
    pred = normalize_answer(prediction)
    best = 0.0
    for gold in golds:
        ref = normalize_answer(gold)
        longest = max(len(pred), len(ref))
        similarity = 1.0 if longest == 0 else 1.0 - _levenshtein(pred, ref) / longest
        best = max(best, similarity)
    return best if best >= threshold else 0.0


def exact_match(prediction: str, golds: Sequence[str]) -> bool:
    """Whether the normalised prediction equals any normalised accepted answer."""
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(gold) for gold in golds)


def normalize_box(box: Sequence[float], width: int, height: int) -> list[int]:
    """Pixel xyxy -> LayoutLM's 0..1000 grid (the transformers pipeline's normalize_bbox)."""
    x0, y0, x1, y1 = (float(v) for v in box)
    return [
        int(BOX_GRID * (x0 / width)),
        int(BOX_GRID * (y0 / height)),
        int(BOX_GRID * (x1 / width)),
        int(BOX_GRID * (y1 / height)),
    ]


def ocr_words_with_tesseract(image: Image.Image, *, lang: str = "eng") -> tuple[list[str], list[list[float]]]:
    """Optional OCR adapter: words and pixel xyxy boxes from Tesseract via ``pytesseract``.

    Neither ``pytesseract`` nor the Tesseract binary is part of this package's pinned runtime; the
    import happens here so the rest of the pipeline stays usable with any OCR the caller prefers.
    """
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise RuntimeError("pytesseract is not installed; supply words and boxes from your own OCR") from exc
    data = pytesseract.image_to_data(image.convert("RGB"), lang=lang, output_type=pytesseract.Output.DICT)
    words: list[str] = []
    boxes: list[list[float]] = []
    for text, left, top, w, h in zip(
        data["text"], data["left"], data["top"], data["width"], data["height"], strict=True
    ):
        token = str(text).strip()
        if not token:
            continue
        words.append(token)
        boxes.append([float(left), float(top), float(left + w), float(top + h)])
    return words, boxes


INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        "one question string plus the page's OCR words (list of str) with one pixel xyxy box per word "
        "and the page size (width, height) the boxes are expressed in; pixels are not read by the model"
    ),
    "words": [1, MAX_WORDS],
    "question_chars": [1, MAX_QUESTION_CHARS],
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "encoding": (
        f"<s> question </s></s> words </s> with RoBERTa byte-level BPE; boxes normalised to 0..{BOX_GRID}; "
        f"windows of {MAX_SEQ_LEN} tokens with {DOC_STRIDE}-token overlap when the words do not fit"
    ),
    "output": (
        f"the best word span of at most {MAX_ANSWER_TOKENS} tokens with its start/end word indices and the "
        "span score (softmax start x softmax end within the window)"
    ),
}


def _check_page(image_size: Any) -> tuple[int, int]:
    if (
        not isinstance(image_size, Sequence)
        or isinstance(image_size, str)
        or len(image_size) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in image_size)
    ):
        raise TypeError("image_size must be a (width, height) pair of ints")
    width, height = int(image_size[0]), int(image_size[1])
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return width, height


def _check_document(
    words: Any, boxes: Any, image_size: Any
) -> tuple[list[str], list[list[int]], tuple[int, int]]:
    """Raise TypeError/ValueError naming the first violated ceiling; return words, grid boxes, size."""
    width, height = _check_page(image_size)
    if isinstance(words, str) or not isinstance(words, Sequence):
        raise TypeError("words must be a sequence of str")
    if not 1 <= len(words) <= MAX_WORDS:
        raise ValueError(f"words has {len(words)} entries; expected 1..MAX_WORDS={MAX_WORDS}")
    if not all(isinstance(word, str) and word.strip() for word in words):
        raise ValueError("every word must be a non-empty str")
    if isinstance(boxes, str) or not isinstance(boxes, Sequence) or len(boxes) != len(words):
        raise ValueError(f"boxes must have one pixel xyxy box per word ({len(words)})")
    grid: list[list[int]] = []
    for index, box in enumerate(boxes):
        if isinstance(box, str) or not isinstance(box, Sequence) or len(box) != 4:
            raise ValueError(f"boxes[{index}] must be [x0, y0, x1, y1]")
        try:
            x0, y0, x1, y1 = (float(v) for v in box)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"boxes[{index}] must hold numbers") from exc
        if not (0 <= x0 <= x1 <= width and 0 <= y0 <= y1 <= height):
            raise ValueError(f"boxes[{index}] {list(box)} is not inside the {width}x{height} page")
        grid.append(normalize_box((x0, y0, x1, y1), width, height))
    return [str(word) for word in words], grid, (width, height)


def _check_question(question: Any) -> str:
    if not isinstance(question, str):
        raise TypeError("question must be a str")
    checked = " ".join(question.split())
    if not checked:
        raise ValueError("question must contain at least one non-whitespace character")
    if len(checked) > MAX_QUESTION_CHARS:
        raise ValueError(f"question has {len(checked)} chars > MAX_QUESTION_CHARS {MAX_QUESTION_CHARS}")
    return checked


def validate_inputs(
    words: Sequence[str],
    boxes: Sequence[Sequence[float]],
    questions: Sequence[str],
    *,
    image_size: Sequence[int],
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Every question is checked exactly as ``answer`` would check it; rejection is reported by raising,
    and a caller that wants the finding recorded catches the exception and stores ``str(exc)`` under
    ``findings``.
    """
    checked_words, _grid, size = _check_document(words, boxes, image_size)
    if isinstance(questions, str) or not isinstance(questions, Sequence) or not questions:
        raise TypeError("questions must be a non-empty sequence of str")
    checked = [_check_question(question) for question in questions]
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (answer takes one page)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {"id": names[0] if names else "page-0", "size": list(size), "n_words": len(checked_words)}
        ],
        "questions": checked,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    results: Sequence[Mapping[str, Any]],
    golds: Sequence[Sequence[str]] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``golds`` (one sequence of accepted answers per result, in order) the report carries the
    mean ``anls`` and the ``exact_match`` rate over the questions plus one per-question entry, verdict
    ``sample-sanity``; without golds it is ``not-measurable`` and says what labelled data would make
    the task measurable.
    """
    if not results:
        raise ValueError("results must contain at least one answer result")
    base = {
        "task": "question + OCR words/boxes -> extractive answer span (LayoutLM v1)",
        "score_semantics": (
            "score is the product of the start and end softmax probabilities of the chosen span within "
            "its window under the model's own head — a ranking signal over spans of this page, not a "
            "calibrated probability that the answer is right, and never a signal that the question is "
            "answerable; the model always returns its best span"
        ),
        "sample_kind": sample_kind,
        "n_questions": len(results),
        "scores": [float(result.get("score", 0.0)) for result in results],
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if golds is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no accepted answers were supplied for the evaluated questions",
            "needs": (
                "question/answer pairs with accepted answers on pages from the deployment domain "
                "(DocVQA-style annotations) scored with ANLS, plus the OCR the deployment will really use; "
                "no such labelled set ships with this repository"
            ),
        }
    if len(golds) != len(results):
        raise ValueError(f"golds has {len(golds)} entries for {len(results)} results")
    per_question = []
    for result, accepted in zip(results, golds, strict=True):
        if isinstance(accepted, str) or not accepted:
            raise ValueError("each golds entry must be a non-empty sequence of accepted answers")
        prediction = str(result["answer"])
        per_question.append(
            {
                "question": result.get("question"),
                "prediction": prediction,
                "score": float(result.get("score", 0.0)),
                "golds": list(accepted),
                "anls": anls(prediction, accepted),
                "exact_match": exact_match(prediction, accepted),
            }
        )
    metrics = [
        {
            "id": "anls",
            "value": sum(entry["anls"] for entry in per_question) / len(per_question),
            "threshold": ANLS_THRESHOLD,
            "normalisation": "lower-cased, punctuation removed, whitespace collapsed; max over golds",
            "estimation": f"{len(per_question)} question(s) on one page, no dispersion estimate",
        },
        {
            "id": "exact_match",
            "value": sum(entry["exact_match"] for entry in per_question) / len(per_question),
            "normalisation": "lower-cased, punctuation removed, whitespace collapsed",
            "estimation": f"{len(per_question)} question(s) on one page, no dispersion estimate",
        },
    ]
    return {
        **base,
        "metrics": metrics,
        "per_question": per_question,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(per_question)} authored question(s) on one tutorial page whose words and boxes you "
            "rendered yourself; plumbing evidence, not a DocVQA benchmark"
        ),
        "needs": (
            "a labelled question/answer set on pages from the deployment domain with the deployment's own "
            "OCR for any accuracy claim; the DocVQA benchmark itself is registration-gated and not bundled"
        ),
    }


def _window_bboxes(
    encoding: Any, window: int, grid_boxes: Sequence[Sequence[int]], sep_id: int
) -> list[list[int]]:
    """One 0..1000 box per token of a window: word tokens take their word's box, separators `[1000]*4`,
    everything else (the question and padding) `[0]*4` — the transformers document-question-answering
    pipeline's convention."""
    bbox = []
    for input_id, sequence_id, word_id in zip(
        encoding["input_ids"][window].tolist(),
        encoding.sequence_ids(window),
        encoding.word_ids(window),
        strict=True,
    ):
        if sequence_id == 1:
            bbox.append(list(grid_boxes[word_id]))
        elif input_id == sep_id:
            bbox.append([BOX_GRID] * 4)
        else:
            bbox.append([0] * 4)
    return bbox


@dataclass
class LayoutLMDocumentQAPipeline:
    """``_runner(question, words, grid_boxes)`` returns ``{"start": int, "end": int, "score": float}``
    (word indices, inclusive) or ``{"start": None, ...}`` when no span could be selected.
    ``_count_windows(question, words)`` returns how many 512-token windows the pair needs (1 = fits); both
    are injectable so the offline tests run without the model."""

    _runner: Callable[..., dict[str, Any]]
    device: str = "cpu"
    dtype: str = "float32"
    source: str = "injected"
    _count_windows: Callable[[str, list[str]], int] | None = field(default=None, repr=False)
    adapter: dict[str, Any] | None = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> LayoutLMDocumentQAPipeline:
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        common: dict[str, Any] = {"trust_remote_code": False}
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, common["local_files_only"], source = str(root), True, "local-snapshot"
        elif allow_download:
            location, common["revision"], source = MODEL_ID, MODEL_REVISION, "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import AutoTokenizer, LayoutLMForQuestionAnswering

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(location, **common)
        if not tokenizer.is_fast:
            raise RuntimeError("a fast tokenizer is required for word_ids/sequence_ids; snapshot has none")
        model = LayoutLMForQuestionAnswering.from_pretrained(location, dtype=torch.float32, **common)
        model = model.eval().to(resolved_device)
        for param in model.parameters():
            param.requires_grad_(False)
        sep_id = tokenizer.sep_token_id

        def encode(question: str, words: list[str]) -> Any:
            return tokenizer(
                text=question.split(),
                text_pair=words,
                is_split_into_words=True,
                max_length=MAX_SEQ_LEN,
                stride=DOC_STRIDE,
                truncation="only_second",
                return_overflowing_tokens=True,
                return_token_type_ids=True,
                padding="max_length",
                return_tensors="pt",
            )

        def count_windows(question: str, words: list[str]) -> int:
            return int(encode(question, words)["input_ids"].shape[0])

        def runner(question: str, words: list[str], grid_boxes: list[list[int]]) -> dict[str, Any]:
            encoding = encode(question, words)
            n_windows = int(encoding["input_ids"].shape[0])
            best: dict[str, Any] = {"start": None, "end": None, "score": 0.0, "n_windows": n_windows}
            model_device = next(model.parameters()).device
            for window in range(n_windows):
                sequence_ids = encoding.sequence_ids(window)
                word_ids = encoding.word_ids(window)
                inputs = {
                    "input_ids": encoding["input_ids"][window].unsqueeze(0).to(model_device),
                    "attention_mask": encoding["attention_mask"][window].unsqueeze(0).to(model_device),
                    "token_type_ids": encoding["token_type_ids"][window].unsqueeze(0).to(model_device),
                    "bbox": torch.tensor(_window_bboxes(encoding, window, grid_boxes, sep_id))
                    .unsqueeze(0)
                    .to(model_device),
                }
                with torch.inference_mode():
                    outputs = model(**inputs)
                # Only document tokens may start or end an answer (the pipeline's p_mask).
                allowed = torch.tensor([sid == 1 for sid in sequence_ids], device=model_device)
                start = outputs.start_logits[0].float().masked_fill(~allowed, float("-inf")).softmax(-1)
                end = outputs.end_logits[0].float().masked_fill(~allowed, float("-inf")).softmax(-1)
                candidates = start[:, None] * end[None, :]
                candidates = torch.triu(candidates) - torch.triu(candidates, diagonal=MAX_ANSWER_TOKENS)
                flat = int(candidates.argmax())
                s_index, e_index = divmod(flat, candidates.shape[1])
                score = float(candidates[s_index, e_index])
                if score > best["score"] and word_ids[s_index] is not None and word_ids[e_index] is not None:
                    best = {
                        "start": word_ids[s_index],
                        "end": word_ids[e_index],
                        "score": score,
                        "n_windows": n_windows,
                    }
            return best

        return cls(
            runner, resolved_device, "float32", source, count_windows, _model=model, _tokenizer=tokenizer
        )

    def answer(
        self,
        question: str,
        *,
        words: Sequence[str],
        boxes: Sequence[Sequence[float]],
        image_size: Sequence[int],
    ) -> dict[str, Any]:
        """Answer one question from the page's words and boxes; ``answer`` is the selected word span."""
        checked_words, grid, size = _check_document(words, boxes, image_size)
        checked_question = _check_question(question)
        raw = self._runner(checked_question, checked_words, grid)
        if not isinstance(raw, dict) or "start" not in raw or "end" not in raw or "score" not in raw:
            raise RuntimeError("runner must return a dict with 'start', 'end' and 'score'")
        start, end = raw["start"], raw["end"]
        if start is not None and not (0 <= int(start) <= int(end) < len(checked_words)):
            raise RuntimeError(f"runner returned an invalid word span {start}..{end}")
        span_words = checked_words[start : end + 1] if start is not None else []
        return {
            "answer": " ".join(span_words),
            "score": float(raw["score"]),
            "start": None if start is None else int(start),
            "end": None if end is None else int(end),
            "question": checked_question,
            "n_words": len(checked_words),
            "n_windows": int(raw.get("n_windows", 1)),
            "image_size": list(size),
            "device": self.device,
            "dtype": self.dtype,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    # ---- adaptation contract ---------------------------------------------------------------------------

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        return self._model, self._tokenizer

    def check_fit(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Split validated records into those whose question + page fit one 512-token window and those that
        would be windowed at inference (a windowed page is answerable but is not trained on; nothing is
        truncated)."""
        from .samples import validate_dataset

        if self._count_windows is None:
            raise ValueError("check_fit needs a pipeline built with from_pretrained() or from_artifact()")
        checked = validate_dataset(records, min_records=1, max_records=MAX_RECORDS_FIT)["records"]
        fitting, dropped = [], []
        for record in checked:
            if self._count_windows(record["question"], list(record["words"])) == 1:
                fitting.append(record)
            else:
                dropped.append(record["id"])
        return {"fitting": fitting, "dropped": dropped, "n_fitting": len(fitting), "n_dropped": len(dropped)}

    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Answer every record and score the predictions against its accepted answers (mean ANLS, exact-match
        rate)."""
        from .metrics import docqa_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        predictions = [
            self.answer(r["question"], words=r["words"], boxes=r["boxes"], image_size=r["image_size"])[
                "answer"
            ]
            for r in checked
        ]
        metrics = docqa_metrics(predictions, [[str(a) for a in r["answers"]] for r in checked])
        metrics.update(
            {
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def _trainable_names(self, trainable_encoder_layers: int) -> list[str]:
        if (
            isinstance(trainable_encoder_layers, bool)
            or not isinstance(trainable_encoder_layers, int)
            or not 1 <= trainable_encoder_layers <= ENCODER_LAYERS
        ):
            raise ValueError(f"trainable_encoder_layers must be an int in 1..{ENCODER_LAYERS}")
        model, _ = self._require_model()
        first = ENCODER_LAYERS - trainable_encoder_layers
        prefixes = tuple(f"layoutlm.encoder.layer.{k}." for k in range(first, ENCODER_LAYERS)) + (
            "qa_outputs.",
        )
        return [name for name, _p in model.named_parameters() if name.startswith(prefixes)]

    @staticmethod
    def _span_positions(encoded: Any, index: int, record: Mapping[str, Any]) -> tuple[int, int]:
        """Token start/end of the gold word span inside the page segment of one encoded example."""
        start = end = None
        for pos, (sid, word_id) in enumerate(
            zip(encoded.sequence_ids(index), encoded.word_ids(index), strict=True)
        ):
            if sid != 1:
                continue
            if start is None and word_id == record["answer_start"]:
                start = pos
            if word_id == record["answer_end"]:
                end = pos
        if start is None or end is None or end < start:
            raise ValueError(f"record {record['id']}: gold span does not map onto the encoded page tokens")
        return start, end

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 6,
        lr: float = 3e-5,
        batch_size: int = 16,
        trainable_encoder_layers: int = DEFAULT_TRAINABLE_ENCODER_LAYERS,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded supervised fine-tuning on a validated document-QA dataset.

        Only the last `trainable_encoder_layers` encoder blocks and the span head train (4 blocks by default:
        28,353,026 of 127,792,898 parameters; the word, position and 2-D box embeddings and the earlier blocks
        stay frozen). Start/end cross-entropy on the gold word span's first and last tokens, AdamW at a fixed
        learning rate with gradient clipping at 1.0, dynamic padding, no scheduler; every training record must
        fit one window (`check_fit`). Epoch 0 records the frozen model's validation ANLS; the epoch with the
        highest validation ANLS is kept."""
        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not (0.0 < lr <= 1e-3):
            raise ValueError("lr must be in (0, 1e-3]")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be an int in 1..64")
        names = self._trainable_names(trainable_encoder_layers)
        train_checked = validate_dataset(train)["records"]
        val_checked = (
            validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"] if val else []
        )
        if self._count_windows is not None:
            over = [
                r["id"] for r in train_checked if self._count_windows(r["question"], list(r["words"])) != 1
            ]
            if over:
                raise ValueError(
                    f"{len(over)} training record(s) need more than one window (use check_fit): {over[:5]}"
                )
        import torch

        torch.manual_seed(seed)
        model, tokenizer = self._require_model()
        sep_id = tokenizer.sep_token_id
        started = time.perf_counter()
        wanted = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in wanted)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        device = next(model.parameters()).device

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            return {k: v for k, v in self.evaluate(val_checked).items() if k in ("anls", "exact_match", "n")}

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}
        history.append(entry)
        if progress:
            progress(entry)
        best_anls = entry["val"]["anls"] if entry["val"] else -math.inf
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
        initial_state = {k: v.clone() for k, v in best_state.items()}
        best_epoch = 0
        generator = torch.Generator().manual_seed(seed)
        try:
            for epoch in range(1, epochs + 1):
                model.train()
                order = torch.randperm(len(train_checked), generator=generator).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    batch = [train_checked[i] for i in order[start : start + batch_size]]
                    encoded = tokenizer(
                        [r["question"].split() for r in batch],
                        [list(r["words"]) for r in batch],
                        is_split_into_words=True,
                        padding=True,
                        truncation="only_second",
                        max_length=MAX_SEQ_LEN,
                        return_token_type_ids=True,
                        return_tensors="pt",
                    )
                    grids = [
                        [normalize_box(box, r["image_size"][0], r["image_size"][1]) for box in r["boxes"]]
                        for r in batch
                    ]
                    bbox = torch.tensor(
                        [_window_bboxes(encoded, i, grids[i], sep_id) for i in range(len(batch))]
                    )
                    positions = [self._span_positions(encoded, i, r) for i, r in enumerate(batch)]
                    out = model(
                        input_ids=encoded["input_ids"].to(device),
                        attention_mask=encoded["attention_mask"].to(device),
                        token_type_ids=encoded["token_type_ids"].to(device),
                        bbox=bbox.to(device),
                        start_positions=torch.tensor([s for s, _e in positions], device=device),
                        end_positions=torch.tensor([e for _s, e in positions], device=device),
                    )
                    optimiser.zero_grad(set_to_none=True)
                    out.loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimiser.step()
                    losses.append(float(out.loss.detach()))
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": score_val()}
                history.append(entry)
                if progress:
                    progress(entry)
                current = entry["val"]["anls"] if entry["val"] else math.inf
                if current > best_anls or not entry["val"]:
                    best_anls = current
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
                    best_epoch = epoch
        except BaseException:
            # Transactional: a failure in training, validation or the progress callback leaves the base
            # exactly as it was, with every parameter frozen again.
            restore = dict(model.state_dict())
            restore.update(initial_state)
            model.load_state_dict(restore, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable_encoder_layers": trainable_encoder_layers,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "highest validation ANLS" if val_checked else "final epoch (no validation split)",
            "lr": lr,
            "batch_size": batch_size,
            "n_train": len(train_checked),
            "n_val": len(val_checked),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts ------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted encoder-block and span-head tensors as safetensors plus a base manifest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        model, _ = self._require_model()
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "weight_file": WEIGHT_FILE,
                "weight_sha256": WEIGHT_SHA256,
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def _check_artifact_manifest(self, root: Path, manifest: Mapping[str, Any]) -> Path:
        """Refuse an artifact whose manifest is not exactly the one this pipeline writes: the supported format
        and version, the pinned base (id, revision, weight file, digest), exactly one file entry named
        `adapter.safetensors` that resolves inside the artifact directory, and a recorded
        `trainable_encoder_layers` in range. Nothing is deserialised here. The digest check that follows
        detects corruption or drift of the weights relative to the adjacent manifest; it is not authenticity
        against an actor who can replace both files."""
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {manifest.get('format_version')!r} is not the supported "
                f"{ARTIFACT_FORMAT_VERSION!r}"
            )
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
            MODEL_ID,
            MODEL_REVISION,
            WEIGHT_SHA256,
        ):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        if base.get("weight_file", WEIGHT_FILE) != WEIGHT_FILE:
            raise ValueError("artifact was adapted from a different base weight file")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must list exactly one file")
        entry = files[0]
        if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact manifest must name exactly {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / entry["path"]).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weight path must resolve inside the artifact directory")
        adapter = manifest.get("adapter")
        layers = adapter.get("trainable_encoder_layers") if isinstance(adapter, Mapping) else None
        if isinstance(layers, bool) or not isinstance(layers, int) or not 1 <= layers <= ENCODER_LAYERS:
            raise ValueError("artifact manifest does not record an in-range integer trainable_encoder_layers")
        if not isinstance(manifest.get("tensors"), list):
            raise ValueError("artifact manifest must list its tensors")
        return weights_path

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, digest and exact tensor set **before** deserialising, then overwrite
        exactly the tensors it carries."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path = self._check_artifact_manifest(root, manifest)
        entry = manifest["files"][0]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        # The exact tensor set the recorded configuration implies — no subset, no extra, no other layer.
        expected = sorted(self._trainable_names(manifest["adapter"]["trainable_encoder_layers"]))
        if sorted(manifest["tensors"]) != expected:
            raise ValueError("artifact tensor list does not match its recorded configuration")
        model, _ = self._require_model()
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from its manifest")
        state = model.state_dict()
        for key, value in tensors.items():
            if key not in state or not (
                key.startswith("layoutlm.encoder.layer.") or key.startswith("qa_outputs.")
            ):
                raise ValueError(
                    f"artifact tensor {key} is not an adaptable encoder or span-head tensor of the base"
                )
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"artifact tensor {key} has shape {tuple(value.shape)}, "
                    f"base has {tuple(state[key].shape)}"
                )
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in tensors.items()})
        model.load_state_dict(merged, strict=True)
        model.eval()
        self.adapter = {
            **manifest["adapter"],
            "trainable_names": manifest["tensors"],
            "history": manifest.get("history", []),
        }
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> LayoutLMDocumentQAPipeline:
        pipeline = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipeline.load_artifact(artifact_dir)
        return pipeline
