"""Build-only asset resolver. Never called by the released notebook.

Downloads public source bytes, independently hashes them, freezes source-image
roles/duplicate exclusions, resolves an immutable OCR closure and Python lock.
No foundation-model inference, gradient training or release qualification.
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tutorials/receipt_intelligence"
CACHE = ROOT / ".cache/receipt-asset-build"
REV = "7f0115a4b758a71d6473b8d085751692da2fef98"
REPO = "naver-clova-ix/cord-v2"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def js(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "DIMER-receipt-asset-freezer/1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def get(url: str, path: Path, expected: str | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or (expected and file_sha(path) != expected):
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DIMER-receipt-asset-freezer/1"})
            with urllib.request.urlopen(req, timeout=180) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f, 1024 * 1024)
            if expected and file_sha(tmp) != expected:
                raise ValueError(f"Digest mismatch: {path.name}")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    return {"url": url, "bytes": path.stat().st_size, "sha256": file_sha(path)}


def write(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def freeze_data() -> None:
    import pyarrow.parquet as pq
    from PIL import Image, ImageOps

    tree = js(f"https://huggingface.co/api/datasets/{REPO}/tree/{REV}/data?recursive=true&expand=false")
    files, rows, counts, category_counts = [], [], collections.Counter(), collections.Counter()
    for item in sorted(tree, key=lambda x: x["path"]):
        path = item["path"]
        if not path.endswith(".parquet"):
            continue
        split = Path(path).name.split("-")[0]
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unexpected source partition {split}")
        local = CACHE / "data" / Path(path).name
        declared = item["lfs"]["oid"]
        fact = get(f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{path}", local, declared)
        if fact["bytes"] != item["size"]:
            raise ValueError("Source byte count changed")
        fact.update(path=path, official_split=split, publisher_sha256=declared,
                    independently_verified_full_file=True)
        files.append(fact)
        row_in_shard = 0
        for batch in pq.ParquetFile(local).iter_batches(batch_size=4, columns=["image", "ground_truth"]):
            for record in batch.to_pylist():
                rid = f"cord-v2:{split}:{counts[split]:04d}"
                counts[split] += 1
                item_row = {"receipt_id": rid, "official_split": split, "source_path": path,
                            "source_row": row_in_shard, "source_error": None}
                row_in_shard += 1
                try:
                    raw = record["image"]["bytes"]
                    with Image.open(io.BytesIO(raw)) as image:
                        orientation = int(image.getexif().get(274, 1))
                        original_size = list(image.size)
                        fmt = image.format
                        frames = getattr(image, "n_frames", 1)
                        corrected = ImageOps.exif_transpose(image).convert("RGB")
                        pixel_bytes = json.dumps(list(corrected.size), separators=(",", ":")).encode() + corrected.tobytes()
                        item_row.update(image_sha256=sha(raw), image_bytes=len(raw), image_format=fmt,
                                        original_size=original_size, oriented_size=list(corrected.size),
                                        exif_orientation=orientation, pixel_sha256=sha(pixel_bytes))
                        if fmt not in {"JPEG", "PNG"} or frames != 1:
                            item_row["source_error"] = "unsupported_image"
                        if len(raw) > 20 * 1024**2 or max(image.size) > 10000 or image.width * image.height > 20_000_000:
                            item_row["source_error"] = "image_limit"
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    item_row["source_error"] = f"image_decode:{type(exc).__name__}"
                gt = record["ground_truth"]
                item_row["annotation_sha256"] = sha(gt.encode("utf-8"))
                try:
                    annotation = json.loads(gt)
                    if not isinstance(annotation.get("valid_line"), list) or not isinstance(annotation.get("gt_parse"), dict):
                        raise ValueError("Missing v2 annotation fields")
                    item_row["source_image_id"] = str(annotation.get("meta", {}).get("image_id", ""))
                    item_row["annotation_canonical_sha256"] = sha(json.dumps(annotation, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
                    for line in annotation["valid_line"]:
                        category_counts[str(line.get("category", ""))] += 1
                except (ValueError, TypeError, KeyError) as exc:
                    item_row["source_error"] = f"annotation_decode:{type(exc).__name__}"
                rows.append(item_row)
    if dict(counts) != {"train": 800, "validation": 100, "test": 100}:
        raise ValueError(f"Unexpected complete population: {dict(counts)}")
    val = sorted((x["receipt_id"] for x in rows if x["official_split"] == "validation"),
                 key=lambda x: sha(("receipt-capstone-v1|42|" + x).encode()))
    model_val = set(val[:50])
    for row in rows:
        row["role"] = ("validation_model" if row["receipt_id"] in model_val else "validation_policy") if row["official_split"] == "validation" else row["official_split"]
        row["included"] = row["source_error"] is None
        row["exclusion_reason"] = row["source_error"]
    groups = collections.defaultdict(list)
    for row in rows:
        if row.get("pixel_sha256"):
            groups[row["pixel_sha256"]].append(row)
    duplicates = []
    for digest, family in sorted(groups.items()):
        if len(family) < 2:
            continue
        cross = len({r["role"] for r in family}) > 1
        ids = sorted(r["receipt_id"] for r in family)
        duplicates.append({"pixel_sha256": digest, "ids": ids, "cross_role": cross})
        for row in family:
            if cross or row["receipt_id"] != ids[0]:
                row["included"] = False
                row["exclusion_reason"] = "cross_role_exact_duplicate" if cross else "within_role_exact_duplicate"
    write("data_manifest.json", {"schema": "org.dimer.receipt-source.v1", "dataset_id": REPO,
          "revision": REV, "license": "CC-BY-4.0", "files": files, "rows": rows,
          "nominal_counts": dict(counts), "verification": "complete source bytes and decoded image hashes"})
    write("split_manifest.json", {"schema": "org.dimer.receipt-splits.v1", "salt": "receipt-capstone-v1|42|",
          "rows": [{k: r[k] for k in ("receipt_id", "official_split", "role", "included", "exclusion_reason")} for r in rows]})
    write("dataset_audit.json", {"schema": "org.dimer.receipt-source-audit.v1", "nominal_counts": dict(counts),
          "included_counts": dict(collections.Counter(r["role"] for r in rows if r["included"])),
          "exclusions": [{"receipt_id": r["receipt_id"], "reason": r["exclusion_reason"]} for r in rows if not r["included"]],
          "exact_duplicate_families": duplicates, "category_counts": dict(sorted(category_counts.items())),
          "near_duplicate_review": "not performed; exact-pixel audit is not template-disjointness",
          "human_annotation_review": "pending; no independent human review claimed",
          "numeric_reference_audit": "performed by the versioned parser during prepare; unresolved references remain unscoreable"})
    print("SOURCE_AUDIT", json.dumps({"counts": dict(counts), "bytes": sum(f["bytes"] for f in files),
          "included": sum(r["included"] for r in rows), "duplicate_families": len(duplicates)}, sort_keys=True), flush=True)


def freeze_ocr() -> None:
    metadata = js("https://api.anaconda.org/package/conda-forge/micromamba")
    choices = [x for x in metadata["files"] if x["version"] == "2.3.2" and x["basename"].startswith("linux-64/") and x["basename"].endswith(".tar.bz2")]
    chosen = sorted(choices, key=lambda x: x["basename"])[-1]
    url = "https:" + chosen["download_url"] if chosen["download_url"].startswith("//") else chosen["download_url"]
    archive = CACHE / "micromamba.tar.bz2"
    bootstrap = get(url, archive, chosen.get("attrs", {}).get("sha256") or chosen.get("sha256"))
    binary = CACHE / "micromamba"
    with tarfile.open(archive) as t:
        member = t.getmember("bin/micromamba")
        if not member.isfile():
            raise ValueError("Invalid micromamba binary member")
        binary.write_bytes(t.extractfile(member).read())
    binary.chmod(0o755)
    bootstrap.update(version="2.3.2", executable_sha256=file_sha(binary), member="bin/micromamba")
    env_path = CACHE / "ocr-env"
    subprocess.run([str(binary), "--no-rc", "create", "-y", "-r", str(CACHE / "mamba-root"), "-p", str(env_path),
                    "--override-channels", "-c", "conda-forge", "tesseract=5.5.0"], check=True)
    packages = []
    for meta_path in sorted((env_path / "conda-meta").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        package_url = meta.get("url") or (meta["channel"].rstrip("/") + "/" + meta["fn"])
        if not package_url.startswith("https://conda.anaconda.org/conda-forge/"):
            raise ValueError(f"Unexpected channel {package_url}")
        local = CACHE / "ocr-packages" / meta["fn"]
        fact = get(package_url, local, meta.get("sha256"))
        fact.update(name=meta["name"], version=meta["version"], build=meta["build"],
                    filename=meta["fn"], license=meta.get("license", "see package metadata"),
                    md5=hashlib.md5(local.read_bytes()).hexdigest())
        packages.append(fact)
    upstream = js("https://api.github.com/repos/tesseract-ocr/tessdata_fast")
    rev = js("https://api.github.com/repos/tesseract-ocr/tessdata_fast/commits/" + upstream["default_branch"])["sha"]
    languages = {}
    for lang in ("eng", "ind"):
        fact = get(f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/{rev}/{lang}.traineddata",
                   CACHE / "language" / f"{lang}.traineddata")
        languages[lang] = dict(fact, revision=rev, filename=f"{lang}.traineddata", license="Apache-2.0")
    version = subprocess.run([str(env_path / "bin/tesseract"), "--version"], capture_output=True, text=True, check=True)
    write("ocr_manifest.json", {"schema": "org.dimer.receipt-ocr-lock.v1", "platform": "linux-64",
          "bootstrap": bootstrap, "packages": packages, "languages": languages,
          "engine_version": "5.5.0", "observed_version": version.stdout + version.stderr,
          "executable": "bin/tesseract", "relocation_note": "Package bytes are pinned. Record relocated binary hash at installation and reverify for cache/replay.",
          "settings": {"oem": 1, "psm": 4, "language_order": "eng+ind", "timeout_seconds": 60, "omp_thread_limit": 1}})
    print("OCR_LOCK", len(packages), "packages", rev, flush=True)


def freeze_model() -> None:
    manifest = json.loads((ROOT / "weights/layoutlm-document-qa/dimer-base-manifest.json").read_text())
    for fact in manifest["files"]:
        local = CACHE / "model" / fact["path"]
        obs = get(f"https://huggingface.co/{manifest['modelId']}/resolve/{manifest['revision']}/{fact['path']}", local, fact["sha256"])
        if obs["bytes"] != fact["bytes"]:
            raise ValueError("Model byte count mismatch")
    manifest["verification"] = "all snapshot file bytes hashed; no inference performed"
    manifest["license"] = "MIT"
    write("model_manifest.json", manifest)
    print("MODEL_BYTES_VERIFIED", manifest["totalBytes"], flush=True)


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    freeze_data()
    freeze_ocr()
    freeze_model()
