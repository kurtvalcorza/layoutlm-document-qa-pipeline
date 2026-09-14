"""Extractive document question answering with the pinned ``impira/layoutlm-document-qa`` checkpoint.

The class loads the tokenizer and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the LayoutLM architecture comes from the pinned ``transformers`` release,
the weights are SafeTensors, and no model-repository code is executed.

LayoutLM (v1) reads **words and their boxes**, not pixels: OCR is an input provider outside the model.
The pipeline therefore takes ``words``/``boxes`` from the caller (bring-your-own OCR) and offers
``ocr_words_with_tesseract`` as an optional adapter that is imported only when called.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "impira/layoutlm-document-qa"
MODEL_REVISION = "beed3c4d02d86017ebca5bd0fdf210046b907aa6"
MODEL_LICENSE = "mit"
MODEL_KEY = "layoutlm-document-qa"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

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


@dataclass
class LayoutLMDocumentQAPipeline:
    """``_runner(question, words, grid_boxes)`` returns ``{"start": int, "end": int, "score": float}``
    (word indices, inclusive) or ``{"start": None, ...}`` when no span could be selected."""

    _runner: Callable[..., dict[str, Any]]
    device: str = "cpu"
    dtype: str = "float32"
    source: str = "injected"

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
        sep_id = tokenizer.sep_token_id

        def runner(question: str, words: list[str], grid_boxes: list[list[int]]) -> dict[str, Any]:
            encoding = tokenizer(
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
            n_windows = int(encoding["input_ids"].shape[0])
            best: dict[str, Any] = {"start": None, "end": None, "score": 0.0, "n_windows": n_windows}
            for window in range(n_windows):
                sequence_ids = encoding.sequence_ids(window)
                word_ids = encoding.word_ids(window)
                input_ids = encoding["input_ids"][window]
                bbox = []
                for input_id, sequence_id, word_id in zip(
                    input_ids.tolist(), sequence_ids, word_ids, strict=True
                ):
                    if sequence_id == 1:
                        bbox.append(grid_boxes[word_id])
                    elif input_id == sep_id:
                        bbox.append([BOX_GRID] * 4)
                    else:
                        bbox.append([0] * 4)
                inputs = {
                    "input_ids": input_ids.unsqueeze(0).to(resolved_device),
                    "attention_mask": encoding["attention_mask"][window].unsqueeze(0).to(resolved_device),
                    "token_type_ids": encoding["token_type_ids"][window].unsqueeze(0).to(resolved_device),
                    "bbox": torch.tensor(bbox).unsqueeze(0).to(resolved_device),
                }
                with torch.inference_mode():
                    outputs = model(**inputs)
                # Only document tokens may start or end an answer (the pipeline's p_mask).
                allowed = torch.tensor([sid == 1 for sid in sequence_ids], device=resolved_device)
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

        return cls(runner, resolved_device, "float32", source)

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
