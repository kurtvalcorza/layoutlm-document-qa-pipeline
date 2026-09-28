"""OCR contract: TSV parsing, pinned closure, cache identity and (optionally) a real Tesseract run."""

import json
import os
from pathlib import Path

import pytest
import receipt_ocr as ocr
from receipt_common import ContractError, IntegrityError

from _receipt_fixtures import ROOT, render

LOCK = json.loads(
    (ROOT / "tutorials" / "receipt_intelligence" / "ocr_manifest.json").read_text(encoding="utf-8")
)
HEADER = "\t".join(ocr.TSV_COLUMNS)


def tsv(*rows):
    return "\n".join([HEADER, *["\t".join(str(v) for v in r) for r in rows]]) + "\n"


def test_parse_tsv_keeps_low_confidence_words_and_drops_invalid_geometry():
    text = tsv(
        (1, 1, 0, 0, 0, 0, 0, 0, 100, 100, -1, ""),
        (5, 1, 1, 1, 1, 1, 10, 10, 30, 12, 12.5, "TOTAL"),
        (5, 1, 1, 1, 1, 2, 60, 10, 25, 12, 96.2, "75.000"),
        (5, 1, 1, 1, 2, 1, 10, 30, 0, 12, 90, "bad"),
        (5, 1, 1, 1, 2, 2, 10, 30, 5, 5, 90, " "),
        (5, 1, 1, 1, 2, 3, 95, 30, 20, 12, -1, "edge"),
    )
    parsed = ocr.parse_tsv(text, (100, 100))
    assert [t["text"] for t in parsed["tokens"]] == ["TOTAL", "75.000", "edge"]
    assert parsed["tokens"][0]["confidence"] == 12.5 and parsed["tokens"][2]["confidence"] is None
    assert parsed["tokens"][2]["box"] == [95, 30, 100, 42]  # clipped to the page
    assert [t["id"] for t in parsed["tokens"]] == [0, 1, 2]
    assert parsed["structural_rows"] == 1 and {i["reason"] for i in parsed["invalid_words"]} == {
        "invalid_geometry",
        "empty_text",
    }


def test_parse_tsv_rejects_unexpected_header():
    with pytest.raises(ContractError):
        ocr.parse_tsv("level\ttext\n", (10, 10))


def test_frozen_ocr_closure_is_fully_pinned():
    assert LOCK["engine_version"] == "5.5.0" and LOCK["platform"] == "linux-64"
    assert len(LOCK["packages"]) == 28
    assert all(
        p["url"].startswith("https://conda.anaconda.org/conda-forge/")
        and len(p["sha256"]) == 64
        and p["bytes"] > 0
        and p["md5"]
        for p in LOCK["packages"]
    )
    assert {p["name"] for p in LOCK["packages"]} >= {"tesseract", "leptonica"}
    assert set(LOCK["languages"]) == {"eng", "ind"}
    assert all(len(v["sha256"]) == 64 and v["revision"] in v["url"] for v in LOCK["languages"].values())
    assert ocr.settings_from(LOCK) == {
        "oem": 1,
        "psm": 4,
        "language_order": "eng+ind",
        "timeout_seconds": 60,
        "omp_thread_limit": 1,
    }


def test_changed_ocr_settings_are_refused():
    changed = {**LOCK, "settings": {**LOCK["settings"], "psm": 6}}
    with pytest.raises(IntegrityError):
        ocr.settings_from(changed)


def test_cache_key_covers_pixels_orientation_engine_languages_and_settings():
    identity = {
        "executable_sha256": "e",
        "language_sha256": {"eng": "a", "ind": "b"},
        "settings": ocr.settings_from(LOCK),
    }
    base = ocr.cache_key("p", {"exif": 1}, identity)
    assert base != ocr.cache_key("q", {"exif": 1}, identity)
    assert base != ocr.cache_key("p", {"exif": 6}, identity)
    assert base != ocr.cache_key("p", {"exif": 1}, {**identity, "executable_sha256": "f"})
    assert base != ocr.cache_key("p", {"exif": 1}, {**identity, "language_sha256": {"eng": "a"}})
    assert base != ocr.cache_key(
        "p", {"exif": 1}, {**identity, "settings": {**identity["settings"], "psm": 6}}
    )


def test_missing_engine_stops_the_run(tmp_path: Path):
    with pytest.raises(IntegrityError):
        ocr.verify_engine(tmp_path / "tesseract", tmp_path, LOCK)


def test_test_double_engine_is_labelled_and_never_qualifying():
    image, tokens = render([["TOTAL", "10.000"]])
    from receipt_data import pixel_sha256

    engine, identity = ocr.test_double_engine({pixel_sha256(image): tokens}, LOCK)
    assert identity["test_double"] and not identity["qualifying_runtime"]
    assert engine(image)["tokens"] == tokens


@pytest.mark.skipif(
    not os.environ.get("RECEIPT_TESSERACT"), reason="set RECEIPT_TESSERACT/RECEIPT_TESSDATA for a real run"
)
def test_real_tesseract_reads_a_rendered_receipt(tmp_path: Path):
    """Development check with a local Tesseract 5.5.0 build; not hosted qualification."""
    from PIL import Image

    executable, tessdata = Path(os.environ["RECEIPT_TESSERACT"]), Path(os.environ["RECEIPT_TESSDATA"])
    identity = ocr.verify_engine(executable, tessdata, LOCK, development=True)
    image, _ = render([["SUBTOTAL", "70.000"], ["TOTAL", "77.000"]], width=520)
    image = image.resize((image.width * 3, image.height * 3), Image.Resampling.LANCZOS)
    result = ocr.ocr_image(image, executable, tessdata, identity["settings"])
    assert result["ocr_status"] == "ok"
    assert "77.000" in [t["text"] for t in result["tokens"]]
