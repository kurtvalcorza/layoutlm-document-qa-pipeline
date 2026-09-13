import hashlib
import json
import re
from pathlib import Path

import pytest

from layoutlm_document_qa_pipeline import (
    BOX_GRID,
    DEFAULT_WEIGHTS_DIR,
    DOC_STRIDE,
    MAX_ANSWER_TOKENS,
    MAX_IMAGE_SIDE,
    MAX_QUESTION_CHARS,
    MAX_SEQ_LEN,
    MAX_WORDS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    LayoutLMDocumentQAPipeline,
    normalize_box,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "impira/layoutlm-document-qa"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert MAX_SEQ_LEN == 512 and 0 < DOC_STRIDE < MAX_SEQ_LEN and MAX_ANSWER_TOKENS == 15
    assert MAX_WORDS == 2000 and MAX_QUESTION_CHARS == 256 and BOX_GRID == 1000
    manifest = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["modelId"] == MODEL_ID
        assert data["revision"] == MODEL_REVISION


def _write_snapshot(root: Path, content: bytes, sha: str | None = None, size: int | None = None) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_manifest(tmp_path):
    _write_snapshot(tmp_path, b'{"model_type": "layoutlm"}')
    info = verify_snapshot(tmp_path)
    assert info["revision"] == MODEL_REVISION and info["files"] == 1


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"model_type": "layoutlm"}'
    good = hashlib.sha256(content).hexdigest()
    flipped = ("0" if good[0] != "0" else "1") + good[1:]
    _write_snapshot(tmp_path, content, sha=flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_size_missing_file_and_revision(tmp_path):
    _write_snapshot(tmp_path, b"abc", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["revision"] = "0" * 40
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


DOCTAGS = (
    "<doctag><section_header_level_1><loc_32><loc_25><loc_231><loc_39>Report</section_header_level_1>\n"
    "<text><loc_32><loc_40><loc_319><loc_93>quarterly revenue by region</text>\n"
    "<otsl><loc_32><loc_151><loc_459><loc_303><ched>Region<ched>Q1<nl><fcel>North 1<fcel>21,132<nl></otsl>\n"
    "</doctag>"
)


WORDS = ["Invoice", "number:", "NW-2026-0417", "Customer:", "Blue", "Yonder", "Airlines"]
BOXES = [
    [70, 200, 140, 218],
    [146, 200, 220, 218],
    [300, 200, 420, 218],
    [70, 296, 160, 314],
    [300, 296, 340, 314],
    [346, 296, 410, 314],
    [416, 296, 490, 314],
]
SIZE = (850, 1100)


def _fake_pipeline(calls: list | None = None, span=(2, 2), score=0.9) -> LayoutLMDocumentQAPipeline:
    def runner(question, words, grid_boxes):
        if calls is not None:
            calls.append((question, list(words), [list(b) for b in grid_boxes]))
        return {"start": span[0], "end": span[1], "score": score, "n_windows": 1}

    return LayoutLMDocumentQAPipeline(runner, "cpu", "float32", "injected")


def test_normalize_box_maps_pixels_to_grid():
    assert normalize_box([85, 110, 170, 121], 850, 1100) == [100, 100, 200, 110]
    assert normalize_box([0, 0, 850, 1100], 850, 1100) == [0, 0, BOX_GRID, BOX_GRID]


def test_answer_output_fields_and_grid_boxes():
    calls: list = []
    pipe = _fake_pipeline(calls)
    result = pipe.answer("  What is the   invoice number? ", words=WORDS, boxes=BOXES, image_size=SIZE)
    assert result["answer"] == "NW-2026-0417" and (result["start"], result["end"]) == (2, 2)
    assert result["score"] == 0.9 and result["n_words"] == 7 and result["n_windows"] == 1
    assert result["question"] == "What is the invoice number?"  # whitespace collapsed
    assert result["image_size"] == [850, 1100]
    assert (result["model_id"], result["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert (result["device"], result["dtype"], result["source"]) == ("cpu", "float32", "injected")
    question, words, grid = calls[0]
    assert question == "What is the invoice number?" and words == WORDS
    assert grid[2] == normalize_box(BOXES[2], *SIZE) and all(0 <= v <= BOX_GRID for box in grid for v in box)


def test_answer_multi_word_span_and_no_span():
    pipe = _fake_pipeline(span=(4, 6), score=0.5)
    assert (
        pipe.answer("Who is the customer?", words=WORDS, boxes=BOXES, image_size=SIZE)["answer"]
        == "Blue Yonder Airlines"
    )
    none = LayoutLMDocumentQAPipeline(lambda *a: {"start": None, "end": None, "score": 0.0}, "cpu")
    result = none.answer("Who?", words=WORDS, boxes=BOXES, image_size=SIZE)
    assert result["answer"] == "" and result["start"] is None and result["score"] == 0.0


def test_answer_rejects_bad_inputs():
    pipe = _fake_pipeline()
    with pytest.raises(TypeError, match="question must be a str"):
        pipe.answer(None, words=WORDS, boxes=BOXES, image_size=SIZE)
    with pytest.raises(ValueError, match="non-whitespace"):
        pipe.answer("   ", words=WORDS, boxes=BOXES, image_size=SIZE)
    with pytest.raises(ValueError, match="MAX_QUESTION_CHARS"):
        pipe.answer("x" * (MAX_QUESTION_CHARS + 1), words=WORDS, boxes=BOXES, image_size=SIZE)
    with pytest.raises(TypeError, match="words must be a sequence"):
        pipe.answer("Who?", words="Invoice number", boxes=BOXES, image_size=SIZE)
    with pytest.raises(ValueError, match="MAX_WORDS"):
        pipe.answer(
            "Who?", words=["w"] * (MAX_WORDS + 1), boxes=[[0, 0, 1, 1]] * (MAX_WORDS + 1), image_size=SIZE
        )
    with pytest.raises(ValueError, match="non-empty str"):
        pipe.answer("Who?", words=["Invoice", ""], boxes=BOXES[:2], image_size=SIZE)
    with pytest.raises(ValueError, match="one pixel xyxy box per word"):
        pipe.answer("Who?", words=WORDS, boxes=BOXES[:-1], image_size=SIZE)
    with pytest.raises(ValueError, match=r"boxes\[1\] must be"):
        pipe.answer("Who?", words=WORDS[:2], boxes=[BOXES[0], [1, 2, 3]], image_size=SIZE)
    with pytest.raises(ValueError, match="not inside"):
        pipe.answer("Who?", words=WORDS[:1], boxes=[[0, 0, 900, 10]], image_size=SIZE)
    with pytest.raises(ValueError, match="not inside"):
        pipe.answer("Who?", words=WORDS[:1], boxes=[[50, 0, 10, 10]], image_size=SIZE)
    with pytest.raises(TypeError, match="image_size"):
        pipe.answer("Who?", words=WORDS, boxes=BOXES, image_size=(850.0, 1100))
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.answer("Who?", words=WORDS, boxes=BOXES, image_size=(MAX_IMAGE_SIDE + 1, 10))


def test_answer_rejects_malformed_runner_output():
    with pytest.raises(RuntimeError, match="start"):
        LayoutLMDocumentQAPipeline(lambda *a: {"answer": "x"}, "cpu").answer(
            "Who?", words=WORDS, boxes=BOXES, image_size=SIZE
        )
    with pytest.raises(RuntimeError, match="invalid word span"):
        LayoutLMDocumentQAPipeline(lambda *a: {"start": 5, "end": 99, "score": 1.0}, "cpu").answer(
            "Who?", words=WORDS, boxes=BOXES, image_size=SIZE
        )
