"""Serving bundle export, pre-load verification and safe archive handling (spec section 15).

The bundle carries only the selected trained tensor subset (SafeTensors) plus serialized
configuration. It never contains receipt images, OCR transcripts, training examples or references.
Verification happens before any tensor is deserialised; executable code is never read from a bundle.
"""

from __future__ import annotations

import json
import shutil
import stat
import struct
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from receipt_common import ContractError, IntegrityError, canonical, digest, file_hash, read_json, write_json

ARTIFACT_SCHEMA = "org.dimer.receipt-intelligence.artifact.v1"
BUNDLE_FILES = (
    "manifest.json",
    "adapter.safetensors",
    "model_config.json",
    "field_schema.json",
    "questions.json",
    "preprocessing.json",
    "ocr_manifest.json",
    "number_format_policy.json",
    "keyword_rules.json",
    "review_policy.json",
    "runtime_manifest.json",
    "RECONSTRUCT.md",
    "LICENSES_AND_ATTRIBUTION.md",
)
MAX_BUNDLE_BYTES = 400 * 1024**2
SAFETENSOR_DTYPES = {"F32": "torch.float32"}

RECONSTRUCT = """# Reconstructing the receipt-intelligence workflow

1. Verify this directory with the capstone's trusted `receipt_artifact.verify_bundle` (file set, sizes,
   SHA-256, manifest schema, tensor names/shapes/dtypes, base identity) **before** loading anything.
2. Load a fresh `impira/layoutlm-document-qa` snapshot at the manifest revision and verify its digests.
3. Overwrite exactly the listed tensors (last four encoder blocks and `qa_outputs`) from
   `adapter.safetensors`; no other parameter changes.
4. Rebuild OCR (`ocr_manifest.json`, `preprocessing.json`), the questions, the number grammar, the
   keyword rules and the frozen total-review policy from their JSON files.
5. Run image -> OCR -> extraction -> normalisation -> routing. Routing states are `needs_review` and
   `total_unflagged`; neither is human verification or accounting approval.

This bundle is not an accepted DIMER production-worker artifact; that would need a separate contract
test. Hash checks detect changes relative to this manifest; they do not make an untrusted manifest
trustworthy. Executable reconstruction code is the notebook's embedded implementation, never code
from a bundle.
"""

LICENSES = """# Licences and attribution

- Base model: `impira/layoutlm-document-qa` (MIT). Only fine-tuned tensor values are included.
- Training data: CORD v2 (NAVER CLOVA, CC BY 4.0), Indonesian receipts. No images, OCR text, examples
  or references are included; model weights may still encode information learned from them.
- OCR: Tesseract 5.5.0 (Apache-2.0) with `tessdata_fast` eng/ind (Apache-2.0); not bundled, identified
  by digest in `ocr_manifest.json`.
- Capstone implementation: repository licence of `kurtvalcorza/layoutlm-document-qa-pipeline`.
"""


def safetensors_header(path: Path) -> dict[str, Any]:
    """Parse the SafeTensors JSON header without deserialising tensors."""
    with Path(path).open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise IntegrityError("Truncated SafeTensors file")
        (length,) = struct.unpack("<Q", raw)
        if length <= 0 or length > 16 * 1024**2:
            raise IntegrityError("Implausible SafeTensors header size")
        header = json.loads(stream.read(length).decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def export_bundle(
    destination: Path,
    tensors: Mapping[str, Any],
    parts: Mapping[str, Any],
    base_identity: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the bundle atomically-by-directory; ``parts`` holds the JSON configuration objects."""
    from safetensors.torch import save_file

    destination.mkdir(parents=True, exist_ok=False)
    ordered = {k: tensors[k].detach().to("cpu").contiguous() for k in sorted(tensors)}
    save_file(ordered, str(destination / "adapter.safetensors"), metadata={"format": "pt"})
    for name in (
        "model_config",
        "field_schema",
        "questions",
        "preprocessing",
        "ocr_manifest",
        "number_format_policy",
        "keyword_rules",
        "review_policy",
        "runtime_manifest",
    ):
        write_json(destination / f"{name}.json", parts[name])
    (destination / "RECONSTRUCT.md").write_text(RECONSTRUCT, encoding="utf-8", newline="\n")
    (destination / "LICENSES_AND_ATTRIBUTION.md").write_text(LICENSES, encoding="utf-8", newline="\n")
    files = {
        name: {"bytes": (destination / name).stat().st_size, "sha256": file_hash(destination / name)}
        for name in BUNDLE_FILES
        if name != "manifest.json"
    }
    manifest = {
        "schema": ARTIFACT_SCHEMA,
        "base_model": dict(base_identity),
        "tensors": [{"name": k, "shape": list(v.shape), "dtype": "F32"} for k, v in ordered.items()],
        "files": files,
        "training": dict(training),
        "adaptation": "bounded fine-tuning of the last four encoder blocks and the QA span head; "
        "not LoRA, not a full checkpoint, not OCR training",
        "contains": "trained tensor subset and configuration only; no images, OCR text or references",
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def verify_bundle(
    root: Path,
    *,
    expected_base: Mapping[str, Any],
    allowed_tensors: set[str] | None = None,
    expected: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Refuse before loading: exact file set, no links, sizes, digests, schema, tensors, base identity.

    ``expected`` optionally pins configuration digests (e.g. policy/grammar hashes from the selection
    record) so a changed threshold, grammar, language file or tokenizer fails verification.
    """
    root = Path(root)
    present = sorted(p.name for p in root.iterdir())
    if present != sorted(BUNDLE_FILES):
        raise IntegrityError(f"Unexpected bundle file set: {present}")
    for name in BUNDLE_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise IntegrityError(f"Bundle entry is not a regular file: {name}")
    if sum((root / n).stat().st_size for n in BUNDLE_FILES) > MAX_BUNDLE_BYTES:
        raise IntegrityError("Bundle exceeds the size ceiling")
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema") != ARTIFACT_SCHEMA:
        raise IntegrityError("Unsupported bundle schema")
    files = manifest.get("files", {})
    if sorted(files) != sorted(n for n in BUNDLE_FILES if n != "manifest.json"):
        raise IntegrityError("Manifest file list differs from the bundle")
    for name, pin in files.items():
        if (root / name).stat().st_size != pin["bytes"] or file_hash(root / name) != pin["sha256"]:
            raise IntegrityError(f"Bundle file altered: {name}")
    base = manifest.get("base_model", {})
    for key in ("model_id", "revision", "weight_sha256", "tokenizer_sha256"):
        if base.get(key) != expected_base.get(key):
            raise IntegrityError(f"Bundle base identity differs: {key}")
    header = safetensors_header(root / "adapter.safetensors")
    declared = {t["name"]: (t["shape"], t["dtype"]) for t in manifest["tensors"]}
    observed = {k: (v["shape"], v["dtype"]) for k, v in header.items()}
    if declared != observed:
        raise IntegrityError("SafeTensors header differs from the manifest tensor list")
    if any(dtype not in SAFETENSOR_DTYPES for _, dtype in observed.values()):
        raise IntegrityError("Unsupported tensor dtype in bundle")
    if allowed_tensors is not None and set(observed) != set(allowed_tensors):
        raise IntegrityError("Bundle tensors are not exactly the allowed trainable subset")
    for name, want in (expected or {}).items():
        if digest(read_json(root / name)) != want:
            raise IntegrityError(f"Bundle configuration {name} differs from the frozen selection record")
    return manifest


def load_tensors(root: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    return load_file(str(Path(root) / "adapter.safetensors"))


def safe_extract(
    archive: Path,
    destination: Path,
    *,
    max_members: int = 1100,
    max_compressed: int = 2 * 1024**3,
    max_expanded: int = 5 * 1024**3,
    max_ratio: float = 200.0,
    allowed_suffixes: tuple[str, ...] | None = None,
) -> list[str]:
    """Extract a ZIP refusing traversal, absolute paths, links, duplicates, bombs and oversize input."""
    archive = Path(archive)
    if archive.stat().st_size > max_compressed:
        raise ContractError("Archive exceeds the compressed-size ceiling")
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        if len(infos) > max_members:
            raise ContractError("Archive has too many members")
        names, total = set(), 0
        for info in infos:
            name = info.filename
            if name.endswith("/"):
                continue
            path = PurePosixPath(name)
            if (
                not name
                or "\\" in name
                or path.is_absolute()
                or ":" in name
                or any(part in ("", ".", "..") for part in name.split("/"))
            ):
                raise ContractError(f"Unsafe archive path: {name!r}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (mode and not stat.S_ISREG(mode) and stat.S_IFMT(mode)):
                raise ContractError(f"Links and special files are not allowed: {name!r}")
            if name.lower() in names:
                raise ContractError(f"Duplicate archive entry: {name!r}")
            names.add(name.lower())
            if allowed_suffixes is not None and not name.lower().endswith(allowed_suffixes):
                raise ContractError(f"Unexpected file type in archive: {name!r}")
            total += info.file_size
            if total > max_expanded:
                raise ContractError("Archive expands beyond the ceiling")
            if info.compress_size and info.file_size / info.compress_size > max_ratio:
                raise ContractError(f"Suspicious compression ratio for {name!r}")
        destination.mkdir(parents=True, exist_ok=False)
        written = []
        for info in infos:
            if info.filename.endswith("/"):
                continue
            target = destination.joinpath(*PurePosixPath(info.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as sink:
                copied = shutil.copyfileobj(source, sink, 1024 * 1024)
            del copied
            if target.stat().st_size != info.file_size:
                raise ContractError("Archive member size mismatch")
            written.append(info.filename)
    return sorted(written)


def config_digests(root: Path) -> dict[str, str]:
    return {
        name: digest(read_json(Path(root) / name))
        for name in BUNDLE_FILES
        if name.endswith(".json") and name != "manifest.json"
    }


def canonical_bytes(value: Any) -> bytes:
    return canonical(value)
