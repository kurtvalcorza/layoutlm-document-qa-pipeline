"""Pinned Tesseract runtime and the OCR token contract (specification section 7).

The hosted install never uses a floating ``apt install``: micromamba and every conda-forge package of
the frozen closure are downloaded by exact URL and verified by SHA-256 (and byte count) against the
committed ``ocr_manifest.json``, then installed offline from those verified files. Language data are
the pinned ``tessdata_fast`` revision. Missing engines or languages stop the run (IntegrityError);
per-image problems become structured failure records.
"""

from __future__ import annotations

import csv
import io
import os
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from receipt_common import ContractError, IntegrityError, digest, download_pinned, file_hash

OCR_SCHEMA = "org.dimer.receipt-ocr-tokens.v1"
PREPROCESSING = {
    "id": "exif-transpose-rgb-png-v1",
    "steps": [
        "apply EXIF orientation",
        "convert to RGB",
        "write lossless PNG",
        "no crop, upscale, deskew, binarisation or trimming",
    ],
}
TSV_COLUMNS = [
    "level",
    "page_num",
    "block_num",
    "par_num",
    "line_num",
    "word_num",
    "left",
    "top",
    "width",
    "height",
    "conf",
    "text",
]
PACKAGE_HOSTS = ("conda.anaconda.org", "api.anaconda.org")
LANGUAGE_HOSTS = ("raw.githubusercontent.com",)


def settings_from(manifest: Mapping[str, Any]) -> dict[str, Any]:
    s = manifest["settings"]
    if (s["oem"], s["psm"], s["language_order"]) != (1, 6, "eng+ind"):
        raise IntegrityError("OCR manifest settings differ from the frozen design (OEM 1, PSM 6, eng+ind)")
    return dict(s)


def install_tesseract(root: Path, manifest: Mapping[str, Any], cache: Path) -> dict[str, Any]:
    """Linux x86-64 hosted install from the frozen closure. Returns the runtime identity."""
    import platform

    if platform.system() != "Linux" or platform.machine() != "x86_64" or manifest["platform"] != "linux-64":
        raise IntegrityError("The pinned OCR closure is linux-64 only")
    boot = manifest["bootstrap"]
    archive = download_pinned(
        boot["url"],
        cache / "micromamba.tar.bz2",
        {"sha256": boot["sha256"], "bytes": boot["bytes"]},
        PACKAGE_HOSTS,
        64 * 1024**2,
    )
    binary = root / "micromamba"
    with tarfile.open(archive) as bundle:
        member = bundle.getmember(boot["member"])
        if not member.isfile() or member.size > 64 * 1024**2:
            raise IntegrityError("Unexpected micromamba archive member")
        binary.write_bytes(bundle.extractfile(member).read())
    binary.chmod(0o700)
    if file_hash(binary) != boot["executable_sha256"]:
        raise IntegrityError("micromamba executable digest mismatch")
    packages = cache / "ocr-packages"
    lines = ["@EXPLICIT"]
    for pkg in manifest["packages"]:
        if not pkg["url"].startswith("https://conda.anaconda.org/conda-forge/linux-64/") and not pkg[
            "url"
        ].startswith("https://conda.anaconda.org/conda-forge/noarch/"):
            raise IntegrityError("Unexpected OCR package origin")
        local = download_pinned(
            pkg["url"],
            packages / pkg["filename"],
            {"sha256": pkg["sha256"], "bytes": pkg["bytes"]},
            PACKAGE_HOSTS,
            256 * 1024**2,
        )
        lines.append(local.resolve().as_uri() + "#" + pkg["md5"])
    explicit = root / "ocr-explicit.txt"
    explicit.write_text("\n".join(lines) + "\n", encoding="utf-8")
    prefix = root / "ocr-env"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CONDA", "MAMBA"))}
    subprocess.run(
        [
            str(binary),
            "--no-rc",
            "create",
            "-y",
            "--offline",
            "-r",
            str(root / "mamba-root"),
            "-p",
            str(prefix),
            "--file",
            str(explicit),
        ],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=900,
    )
    executable = prefix / manifest["executable"]
    tessdata = root / "tessdata"
    languages = {}
    for lang, pin in manifest["languages"].items():
        path = download_pinned(
            pin["url"],
            tessdata / pin["filename"],
            {"sha256": pin["sha256"], "bytes": pin["bytes"]},
            LANGUAGE_HOSTS,
            64 * 1024**2,
        )
        languages[lang] = {"sha256": file_hash(path), "revision": pin["revision"]}
    identity = verify_engine(executable, tessdata, manifest)
    identity.update(
        install_method="micromamba explicit offline create from SHA-256-verified package files",
        packages=len(manifest["packages"]),
        bootstrap_version=boot["version"],
    )
    return identity


def verify_engine(
    executable: Path, tessdata: Path, manifest: Mapping[str, Any], *, development: bool = False
) -> dict[str, Any]:
    """Assert executable version, language files and language availability; return the identity."""
    if not Path(executable).is_file():
        raise IntegrityError("Pinned Tesseract executable is missing")
    version = subprocess.run([str(executable), "--version"], capture_output=True, text=True, timeout=60)
    text = version.stdout + version.stderr
    first = text.strip().splitlines()[0] if text.strip() else ""
    if version.returncode or first != f"tesseract {manifest['engine_version']}":
        raise IntegrityError(f"Unexpected Tesseract version: {first!r}")
    languages = {}
    for lang, pin in manifest["languages"].items():
        path = Path(tessdata) / pin["filename"]
        if not path.is_file() or file_hash(path) != pin["sha256"]:
            raise IntegrityError(f"Language data {lang} missing or altered")
        languages[lang] = pin["sha256"]
    listed = subprocess.run(
        [str(executable), "--list-langs", "--tessdata-dir", str(tessdata)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    available = set((listed.stdout + listed.stderr).split())
    if not {"eng", "ind"} <= available:
        raise IntegrityError("Tesseract cannot load both eng and ind language data")
    leptonica = next((ln.strip() for ln in text.splitlines() if "leptonica" in ln), None)
    return {
        "engine_version": first,
        "leptonica": leptonica,
        "executable_sha256": file_hash(executable),
        "language_sha256": languages,
        "platform": manifest["platform"] if not development else "development",
        "settings": settings_from(manifest),
        "preprocessing": PREPROCESSING,
        "qualifying_runtime": not development,
    }


def cache_key(pixel_sha256: str, orientation: Mapping[str, Any], identity: Mapping[str, Any]) -> str:
    return digest(
        {
            "pixel_sha256": pixel_sha256,
            "orientation": orientation,
            "preprocessing": PREPROCESSING,
            "executable_sha256": identity["executable_sha256"],
            "language_sha256": identity["language_sha256"],
            "settings": identity["settings"],
        }
    )


def parse_tsv(text: str, image_size: tuple[int, int]) -> dict[str, Any]:
    """Word rows -> stable tokens. Low confidence never removes a word; invalid geometry does."""
    reader = csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE)
    header = next(reader, None)
    if header != TSV_COLUMNS:
        raise ContractError("Unexpected Tesseract TSV header")
    width, height = image_size
    tokens, structural, invalid = [], 0, []
    for row in reader:
        if not row:
            continue
        if len(row) < 12:
            invalid.append({"reason": "short_row"})
            continue
        if len(row) > 12:  # a literal tab inside text is not produced by Tesseract; keep it visible
            row = row[:11] + ["\t".join(row[11:])]
        level = int(row[0])
        if level != 5:
            structural += 1
            continue
        text_value = row[11]
        left, top, w, h = (int(float(v)) for v in row[6:10])
        if not text_value.strip():
            invalid.append({"reason": "empty_text", "line": row[2:5]})
            continue
        box = [max(0, left), max(0, top), min(width, left + w), min(height, top + h)]
        if w <= 0 or h <= 0 or box[0] >= box[2] or box[1] >= box[3]:
            invalid.append({"reason": "invalid_geometry", "text": text_value})
            continue
        conf = float(row[10])
        tokens.append(
            {
                "id": len(tokens),
                "text": text_value.strip(),
                "box": box,
                "line": [int(row[2]), int(row[3]), int(row[4])],
                "word": int(row[5]),
                "page": int(row[1]),
                "confidence": conf if conf >= 0 else None,
            }
        )
    return {"tokens": tokens, "structural_rows": structural, "invalid_words": invalid}


def ocr_image(
    image: Any, executable: Path, tessdata: Path, settings: Mapping[str, Any], workdir: Path | None = None
) -> dict[str, Any]:
    """Run the frozen configuration on one EXIF-oriented RGB image; recoverable failures are records."""
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        path = Path(tmp) / "page.png"
        image.save(path, format="PNG", compress_level=1)
        command = [
            str(executable),
            str(path),
            "stdout",
            "--oem",
            str(settings["oem"]),
            "--psm",
            str(settings["psm"]),
            "-l",
            settings["language_order"],
            "--tessdata-dir",
            str(tessdata),
            # Explicit variables: the named "tsv" config file lives in tessdata/configs, which the pinned
            # language-data directory does not ship, and a missing config silently falls back to text.
            "-c",
            "tessedit_create_tsv=1",
            "-c",
            "tessedit_create_txt=0",
        ]
        env = dict(os.environ, OMP_THREAD_LIMIT=str(settings.get("omp_thread_limit", 1)))
        try:
            done = subprocess.run(command, capture_output=True, timeout=settings["timeout_seconds"], env=env)
        except subprocess.TimeoutExpired:
            return {"ocr_status": "ocr_timeout", "tokens": [], "structural_rows": 0, "invalid_words": []}
    if done.returncode:
        return {
            "ocr_status": "ocr_error",
            "tokens": [],
            "structural_rows": 0,
            "invalid_words": [],
            "stderr_tail": done.stderr.decode("utf-8", "replace")[-500:],
        }
    parsed = parse_tsv(done.stdout.decode("utf-8"), image.size)
    return {"ocr_status": "ok" if parsed["tokens"] else "ocr_empty", **parsed}


def tesseract_engine(executable: Path, tessdata: Path, settings: Mapping[str, Any]):
    """The production OCR callable: EXIF-oriented RGB image -> OCR result record."""

    def engine(image: Any) -> dict[str, Any]:
        return ocr_image(image, executable, tessdata, settings)

    return engine


def test_double_engine(table: Mapping[str, list[dict[str, Any]]], manifest: Mapping[str, Any]):
    """Labelled test double for CPU fixtures: returns recorded tokens by pixel hash. Never evidence."""
    from receipt_data import pixel_sha256

    identity = {
        "engine_version": "test-double (not Tesseract)",
        "leptonica": None,
        "executable_sha256": digest(table),
        "language_sha256": {},
        "platform": "test-double",
        "settings": settings_from(manifest),
        "preprocessing": PREPROCESSING,
        "qualifying_runtime": False,
        "test_double": True,
    }

    def engine(image: Any) -> dict[str, Any]:
        tokens = table.get(pixel_sha256(image))
        if tokens is None:
            return {"ocr_status": "ocr_error", "tokens": [], "structural_rows": 0, "invalid_words": []}
        return {
            "ocr_status": "ok" if tokens else "ocr_empty",
            "tokens": [dict(t) for t in tokens],
            "structural_rows": 0,
            "invalid_words": [],
        }

    return engine, identity


def run_cohort(
    records: list[Mapping[str, Any]],
    load_image,
    identity: Mapping[str, Any],
    engine,
    cache_dir: Path,
    workers: int | None = None,
    progress=None,
) -> list[dict]:
    """OCR every record (in order), reusing only caches whose full key matches."""
    import json

    cache_dir.mkdir(parents=True, exist_ok=True)

    def one(record: Mapping[str, Any]) -> dict[str, Any]:
        key = cache_key(record["pixel_sha256"], record["orientation"], identity)
        cached = cache_dir / f"{key}.json"
        if cached.is_file():
            value = json.loads(cached.read_text(encoding="utf-8"))
            if value.get("cache_key") == key:
                # The key is pixel content, so another cohort (or the same image under a new id) may
                # have written this entry; the receipt id always comes from the current record.
                return {**value, "receipt_id": record["receipt_id"], "cache_hit": True}
        try:
            image, _ = load_image(record)
        except (ContractError, OSError, ValueError) as error:
            return {
                "receipt_id": record["receipt_id"],
                "image_size": record.get("image_size"),
                "ocr_status": "image_error",
                "error": type(error).__name__,
                "tokens": [],
                "cache_key": key,
                "cache_hit": False,
            }
        result = engine(image)
        value = {
            "schema": OCR_SCHEMA,
            "receipt_id": record["receipt_id"],
            "image_size": list(image.size),
            "pixel_sha256": record["pixel_sha256"],
            "cache_key": key,
            **result,
        }
        if result["ocr_status"] == "ok" or result["ocr_status"] == "ocr_empty":
            tmp = cached.with_suffix(".part")
            tmp.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
            tmp.replace(cached)
        return {**value, "cache_hit": False}

    workers = workers or max(1, os.cpu_count() or 1)
    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, value in enumerate(pool.map(one, records)):
            out.append(value)
            if progress and (index + 1) % 50 == 0:
                progress(index + 1, len(records))
    return out


def as_document(ocr: Mapping[str, Any], input_source: str = "actual_ocr") -> dict[str, Any]:
    """The only object predictors receive: tokens, page size and OCR status."""
    return {
        "receipt_id": ocr["receipt_id"],
        "image_size": list(ocr["image_size"] or [1, 1]),
        "tokens": [dict(t) for t in ocr.get("tokens", [])],
        "ocr_status": ocr["ocr_status"],
        "input_source": input_source,
    }
