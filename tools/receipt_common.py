"""Shared, dependency-light integrity and stage-ownership primitives.

The functions here never infer trust from an uploaded manifest. Downloads use
caller-owned, immutable pins; source files carried in the notebook are trusted code.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

SCHEMA = "org.dimer.receipt-intelligence.v1"
FIELDS = ("total_amount", "subtotal_amount", "tax_amount", "service_charge")
QUESTIONS = {
    "total_amount": "What is the total amount?",
    "subtotal_amount": "What is the subtotal?",
    "tax_amount": "What is the tax amount?",
    "service_charge": "What is the service charge?",
}
ROLES = ("train", "validation_model", "validation_policy", "test", "inference")
SEED = 42


class ContractError(ValueError):
    """An incompatible input, rather than an expected per-receipt OCR failure."""


class IntegrityError(RuntimeError):
    """A global integrity failure; it must stop the experiment."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ContractError(f"Non-finite JSON number: {value}")
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ContractError(f"Duplicate JSON key: {key}")
            out[key] = value
        return out
    return json.loads(Path(path).read_text("utf-8"), object_pairs_hook=unique_pairs,
                      parse_constant=reject_constant)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value) + b"\n"
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text("utf-8").splitlines():
        if line.strip():
            value = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(
                ContractError("Non-finite JSON number")))
            if not isinstance(value, dict):
                raise ContractError("Each JSONL record must be an object")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b"".join(canonical(row) + b"\n" for row in rows))


def spreadsheet_safe(value: Any) -> str:
    text = "" if value is None else str(value)
    if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n")):
        return "'" + text
    return text


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: spreadsheet_safe(row.get(key)) for key in columns})


def safe_relative(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ContractError("Expected a nonempty POSIX relative path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("..", ".", "") for part in name.split("/")):
        raise ContractError(f"Unsafe relative path: {name!r}")
    if ":" in name or any(ord(c) < 32 for c in name):
        raise ContractError("Drive names and control characters are forbidden in paths")
    return path


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ContractError("IDs must be 1–80 nonidentifying ASCII letters/digits/._-")
    return value


def verify_file(path: str | Path, pin: Mapping[str, Any]) -> None:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise IntegrityError(f"Missing or symlinked asset: {path.name}")
    size = path.stat().st_size
    if pin.get("bytes") is not None and size != pin["bytes"]:
        raise IntegrityError(f"Wrong byte count: {path.name}")
    if pin.get("sha256") and file_hash(path) != pin["sha256"]:
        raise IntegrityError(f"SHA-256 mismatch: {path.name}")
    if pin.get("git_blob_sha1"):
        h = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != pin["git_blob_sha1"]:
            raise IntegrityError(f"Git-blob mismatch: {path.name}")
    if not pin.get("sha256") and not pin.get("git_blob_sha1"):
        raise IntegrityError("An immutable content digest is required")


def download_pinned(url: str, destination: str | Path, pin: Mapping[str, Any],
                    allowed_hosts: tuple[str, ...], max_bytes: int = 3 * 1024**3) -> Path:
    """Read only caller-owned URLs. Validate caches as rigorously as downloads."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts or parsed.username:
        raise IntegrityError("Unapproved asset origin")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_file(destination, pin)
        return destination
    for attempt in range(3):
        fd, temporary = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "DIMER-Receipt-Capstone/1"})
            with os.fdopen(fd, "wb") as output, urllib.request.urlopen(request, timeout=120) as response:
                transferred = 0
                while chunk := response.read(1024 * 1024):
                    transferred += len(chunk)
                    if transferred > max_bytes:
                        raise IntegrityError("Asset exceeds download ceiling")
                    output.write(chunk)
            verify_file(temporary, pin)
            os.replace(temporary, destination)
            return destination
        except IntegrityError:
            raise
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        finally:
            Path(temporary).unlink(missing_ok=True)
    raise AssertionError("Unreachable")


def inventory(root: Path, exclude: tuple[str, ...] = ("stage_receipt.json",)) -> dict[str, Any]:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise IntegrityError("Symlinks are not allowed in stage outputs")
        if path.is_file() and path.name not in exclude:
            out[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size,
                                                     "sha256": file_hash(path)}
    return out


class StageStore:
    """Atomic success receipts with recursive verification and retry invalidation."""

    def __init__(self, root: str | Path, code_digest: str, configuration: dict[str, Any],
                 graph: dict[str, list[str]]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.code_digest = code_digest
        self.configuration = configuration
        self.graph = graph

    def invalidate(self, stage: str) -> None:
        victims = {stage}
        while True:
            extended = victims | {name for name, deps in self.graph.items() if victims.intersection(deps)}
            if extended == victims:
                break
            victims = extended
        for name in victims:
            directory = self.root / safe_id(name)
            if directory.exists():
                shutil.rmtree(directory)

    def verify(self, stage: str) -> dict[str, Any]:
        directory = self.root / safe_id(stage)
        receipt = read_json(directory / "stage_receipt.json")
        if receipt["code_digest"] != self.code_digest or receipt["configuration_digest"] != digest(self.configuration):
            raise IntegrityError(f"Stage {stage} belongs to different source/configuration")
        if receipt["state"] != "success" or inventory(directory) != receipt["outputs"]:
            raise IntegrityError(f"Stage {stage} outputs are missing or altered")
        expected = {dep: digest(self.verify(dep)) for dep in self.graph[stage]}
        if receipt["dependencies"] != expected:
            raise IntegrityError(f"Stage {stage} has stale dependencies")
        return receipt

    @contextmanager
    def stage(self, name: str) -> Iterator[Path]:
        if name not in self.graph:
            raise ContractError(f"Unknown stage {name}")
        # Invalidate first: even a dependency failure cannot preserve old success.
        self.invalidate(name)
        dependency_hashes = {dep: digest(self.verify(dep)) for dep in self.graph[name]}
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=self.root))
        started = time.perf_counter()
        try:
            yield temporary
            receipt = {"schema": SCHEMA + ".stage", "stage": name, "state": "success",
                       "code_digest": self.code_digest, "configuration_digest": digest(self.configuration),
                       "dependencies": dependency_hashes, "outputs": inventory(temporary),
                       "elapsed_seconds": time.perf_counter() - started}
            write_json(temporary / "stage_receipt.json", receipt)
            os.replace(temporary, self.root / name)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def finite_score(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
