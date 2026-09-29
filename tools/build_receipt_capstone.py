"""Generate the standalone Small-Business Receipt Intelligence capstone notebook.

The notebook is never hand-edited: ``python tools/build_receipt_capstone.py`` writes it and
``--check`` enforces byte parity (CI). All DIMER-owned logic is carried inside the notebook with
SHA-256 checks; nothing is cloned, pip-installed from this repository or fetched as source at run time.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "DIMER_Small_Business_Receipt_Intelligence_Capstone.ipynb"
NOTEBOOK_REVISION = "0.1.0-candidate"
REPOSITORY = "kurtvalcorza/layoutlm-document-qa-pipeline"
# The badge must point at a branch where the notebook exists; switch to "main" in the merge commit.
BRANCH_FOR_BADGE = "feat/small-business-receipt-capstone"
MODULES = ("receipt_common.py", "receipt_fields.py", "receipt_data.py", "receipt_ocr.py", "receipt_models.py",
           "receipt_training.py", "receipt_metrics.py", "receipt_policy.py", "receipt_artifact.py", "receipt_byod.py")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def carried_files() -> dict[str, str]:
    files = {"capstone.py": (ROOT / "tools" / "receipt_capstone.py").read_text(encoding="utf-8")}
    for name in MODULES:
        files[name] = (ROOT / "tools" / name).read_text(encoding="utf-8")
    for name in ("data", "split", "model", "ocr"):
        text = (ROOT / "tutorials" / "receipt_intelligence" / f"{name}_manifest.json").read_text(encoding="utf-8")
        # Compact JSON keeps the carrier small; content (not whitespace) is what the runner verifies.
        files[f"{name}_manifest.json"] = json.dumps(json.loads(text), sort_keys=True, separators=(",", ":")) + "\n"
    files["requirements.txt"] = (ROOT / "tools" / "receipt-requirements.lock").read_text(encoding="utf-8")
    files["licenses/code.txt"] = (ROOT / "LICENSE").read_text(encoding="utf-8")
    files["licenses/DATA_LICENSE.md"] = (ROOT / "tutorials" / "receipt_intelligence" / "DATA_LICENSE.md").read_text(encoding="utf-8")
    return files


PREFLIGHT = r"""
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import zipfile

SESSION_START = time.perf_counter()
if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise RuntimeError('Use a fresh Google Colab Linux x86-64 T4 runtime for this notebook.')
try:
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
except FileNotFoundError as exc:
    raise RuntimeError('No GPU is attached. Select Runtime > Change runtime type > T4 GPU, then Run all.') from exc
if gpu.returncode or 'T4' not in gpu.stdout:
    raise RuntimeError('The canonical run requires a T4 GPU (found: %r). Change the runtime type, then Run all.' % gpu.stdout.strip())
print('Device:', gpu.stdout.strip(), '| CPUs:', os.cpu_count())
RUN_ID = uuid.uuid4().hex[:12]
ROOT = Path.cwd() / OUTPUT_DIR / RUN_ID
ROOT.mkdir(parents=True)
if shutil.disk_usage(ROOT).free < 15 * 1024**3:
    raise RuntimeError('At least 15 GiB of free disk is needed for the CORD shards, CUDA wheels, OCR and outputs.')
print('Run directory:', ROOT)
"""

CARRIER = r"""
for name, text in CARRIED_FILES.items():
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')
    if hashlib.sha256(path.read_bytes()).hexdigest() != CARRIED_HASHES[name]:
        raise RuntimeError('Carried file integrity failure: ' + name)
(ROOT / 'run_config.json').write_text(json.dumps({'run_id': RUN_ID, 'source': 'cord', 'mode': 'canonical',
                                                  'number_format_policy': 'cord_mixed_v1', 'currency': 'unspecified'}),
                                      encoding='utf-8')
print(f'Embedded implementation verified: {len(CARRIED_FILES)} files, notebook revision {NOTEBOOK_REVISION}.')
"""

BOOTSTRAP = r"""
UV_URL = 'https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
UV_SHA256 = 'aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60'
step = time.perf_counter()
for attempt in range(3):
    try:
        with urllib.request.urlopen(UV_URL, timeout=90) as response:
            wheel = response.read(20081405)
        break
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        if attempt == 2:
            raise
        time.sleep(2 ** attempt)
if len(wheel) != 20081404 or hashlib.sha256(wheel).hexdigest() != UV_SHA256:
    raise RuntimeError('uv wheel size/hash mismatch')
with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
    member = next(name for name in archive.namelist() if name.endswith('.data/scripts/uv'))
    UV = ROOT / 'uv'
    UV.write_bytes(archive.read(member))
UV.chmod(0o700)
ENV = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
for name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'):
    ENV.pop(name, None)
ENV['MPLBACKEND'] = 'Agg'
print('Creating the isolated Python 3.12.12 environment...', flush=True)
subprocess.run([str(UV), 'venv', '--managed-python', '--python', '3.12.12', str(ROOT / 'env')], env=ENV, check=True)
PYTHON = ROOT / 'env/bin/python'
print('Installing the hashed dependency lock (several minutes; CUDA wheels are large)...', flush=True)
subprocess.run([str(UV), 'pip', 'install', '--python', str(PYTHON), '--require-hashes', '--only-binary', ':all:',
                '--index-url', 'https://pypi.org/simple', '-r', str(ROOT / 'requirements.txt')], env=ENV, check=True)
check = subprocess.run([str(PYTHON), '-c', 'import sys,torch,transformers; print("Python", sys.version.split()[0]); print("torch", torch.__version__, "CUDA", torch.version.cuda, "| transformers", transformers.__version__); assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU", torch.cuda.get_device_name(0))'],
                       env=ENV, capture_output=True, text=True)
print(check.stdout.strip())
if check.returncode:
    print(check.stderr[-3000:])
    raise RuntimeError('The isolated environment cannot use the GPU. Preserve this output.')
print(f'Environment ready in {time.perf_counter() - step:.0f} s.')

def run_stage(stage, *options, root=None):
    root = root or ROOT
    label = stage or options[0].lstrip('-')
    log = root / (label + '.log')
    print('Running', label, 'in a separate process. Log:', log, flush=True)
    with log.open('w', encoding='utf-8') as output:
        process = subprocess.Popen([str(PYTHON), '-u', str(root / 'capstone.py'), '--root', str(root), *(['--stage', stage] if stage else []), *options],
                                   env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            output.write(line)
            output.flush()
            if line.strip() and len(line) < 2000:
                print(line.rstrip(), flush=True)
        process.wait()
    if process.returncode:
        print(log.read_text(encoding='utf-8')[-7000:])
        raise RuntimeError(f'{stage} failed with exit {process.returncode}. Preserve the diagnostics; do not weaken checks.')
"""

HELPERS = r"""
import csv
from IPython.display import HTML, Image, Markdown, display

def _fmt(value):
    if value is None or value == '':
        return 'undefined'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).replace('|', '/').replace('\n', ' ')
    if str(value).lstrip('-').isdigit():
        return str(value)
    return f'{number:.4f}'

def show_table(name, columns=None, limit=25, root=None):
    path = (root or ROOT) / 'outputs' / name
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        available = reader.fieldnames or []
    if not rows:
        raise RuntimeError('Expected a nonempty evidence table: ' + name)
    columns = columns or available
    missing = set(columns) - set(available)
    if missing:
        raise RuntimeError(f'Missing columns in {name}: {sorted(missing)}')
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
    lines += ['| ' + ' | '.join(_fmt(row[c]) for c in columns) + ' |' for row in rows[:limit]]
    display(Markdown('\n'.join(lines)))
    print('File:', path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path)
    if len(rows) > limit:
        print(f'Showing {limit} of {len(rows)} rows; open the CSV for all rows.')

def show_json(name, keys=None, root=None):
    path = (root or ROOT) / 'outputs' / name
    value = json.loads(path.read_text(encoding='utf-8'))
    items = [(k, v) for k, v in value.items() if (keys is None or k in keys) and not isinstance(v, (dict, list))]
    display(Markdown('\n'.join(['| Evidence | Value |', '| --- | --- |'] + [f'| {k} | {_fmt(v)} |' for k, v in items])))
    print('File:', path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path)
    return value

def show_figure(name, root=None):
    path = (root or ROOT) / 'outputs' / 'figures' / name
    if not path.exists():
        raise RuntimeError('Expected figure is missing: ' + name)
    display(Image(filename=str(path)))
"""


def build() -> dict:
    files = carried_files()
    target = ROOT / "tutorials" / NAME
    revision = (
        json.loads(target.read_text(encoding="utf-8"))["metadata"]["dimer"]["generated_from"]["base_revision"]
        if target.exists()
        else subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    )
    source = {"repository": REPOSITORY, "base_revision": revision, "generator": "build_receipt_capstone.py/1",
              "generator_sha256": sha((ROOT / "tools" / "build_receipt_capstone.py").read_text(encoding="utf-8")),
              "files": {name: sha(text) for name, text in files.items()}}
    files["source.json"] = json.dumps(source, indent=2, sort_keys=True)
    audit = json.loads((ROOT / "tutorials" / "receipt_intelligence" / "dataset_audit.json").read_text(encoding="utf-8"))
    counts = audit["included_counts"]
    cells: list[dict] = []

    def md(text: str) -> None:
        cells.append({"cell_type": "markdown", "id": f"md-{len(cells):02d}", "metadata": {},
                      "source": text.strip().splitlines(True)})

    def code(text: str, **metadata: object) -> None:
        ast.parse(text)
        cells.append({"cell_type": "code", "id": f"code-{len(cells):02d}", "metadata": metadata,
                      "execution_count": None, "outputs": [], "source": text.strip().splitlines(True)})

    md(f"""# From Receipts to Records: Small-Business Document Intelligence

[![DIMER](https://img.shields.io/badge/DIMER-Applied_AI-165b80)](https://training.dimer1.asti.dost.gov.ph/signin)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPOSITORY}/blob/{BRANCH_FOR_BADGE}/tutorials/{NAME})

**Profile:** E2E · **Mode:** GUIDED · **Notebook standard:** 2.2 · **Revision:** {NOTEBOOK_REVISION} · **Release status:** Candidate (built; hosted qualification pending).

**Central question.** Can an AI-assisted workflow extract receipt amounts reliably, and how does referring uncertain totals for human review change the error rate among the totals left unflagged?

## 0. Orientation

This self-paced notebook starts from receipt **photographs**, recognises their words with OCR, extracts four amounts (total, subtotal, tax, service charge) into auditable records, measures errors against published annotations and develops a validation-selected policy for referring uncertain **totals** for human review.

| Input | System | Output |
| --- | --- | --- |
| Receipt image | Image checks → Tesseract OCR (words, boxes, confidences) → three extractors: **A** fixed keyword rules, **B** frozen LayoutLM document QA, **C** receipt-adapted LayoutLM → conservative amount normalisation → total-review policy | CSV/JSON records, evaluation evidence, a trained adapter and a fresh-process reload check |

It answers three separate questions: (1) how accurately the complete image → OCR → extraction workflow recovers annotated amounts; (2) whether bounded fine-tuning improves on a transparent rule baseline and the unchanged pretrained model; and (3) what coverage–error trade-off a validation-selected review policy achieves on held-out receipts.

**What this notebook does not establish.** The data are CORD v2, a public sample of **Indonesian** receipts (CC BY 4.0). Nothing here supports a claim about Philippine businesses, unseen merchants or templates, production use or accounting-grade accuracy. It does not parse line items, merchants, dates or handwriting, convert currencies, check tax compliance, reconcile bank records or post to a ledger. A missing annotation never means a charge was absent or zero. A `total_unflagged` record is **not** human-verified. Readable OCR, a confident-looking answer and a numerically correct business record are three different achievements; keeping them apart is the main lesson.

**Learning outcomes.** By the end you should be able to:

1. explain the difference between OCR, field extraction, numeric normalisation and human verification;
2. trace an extracted amount back to its recognised words and boxes on the image;
3. compare a rule baseline, a frozen model and a genuinely fine-tuned model on the same receipts;
4. explain why an extractive model cannot restore a digit that OCR failed to recognise;
5. interpret total exact match, annotation-conditional field accuracy, review coverage and selective error;
6. change one review-policy setting on validation receipts without touching the held-out experiment;
7. export a trained adapter and reconstruct matching predictions from serialized files; and
8. name the additional evidence needed before using such a workflow on local business records.

**Prerequisites:** basic Python and Colab; no document-AI experience is assumed. Reflection prompts are optional personal aids, **not required submissions**; no certificate is implied.

**AI Assistance Disclosure:** Generative AI assisted this notebook's code and technical writing under maintainer direction. The maintainer reviews the implementation, validates results and decides release. AI assistance is not independent verification, endorsement or release approval.
""")
    md("""## 1. How to use this notebook

1. Select **Runtime → Change runtime type → T4 GPU**, start a fresh runtime, then **Runtime → Run all**. No login, token, upload, repository clone, DIMER service or restart is needed; the default run never opens a file dialog.
2. Read each **Predict** prompt before its results appear. Worked answers are folded under *Show a worked answer*.
3. Infrastructure cells are collapsed. Their carried source stays inspectable; every stage runs in a separate process and writes a success receipt that later stages verify.

**Roadmap:** setup → data card → OCR → rules → frozen model → alignment and training → review policy → freeze and test → diagnose → export → fresh reload → change one thing → optional BYOD → conclusion.

**Expected cost.** Everything is measured by the run itself (see the runtime table in §12): downloads (~2.3 GB of CORD shards, CUDA wheels, a 0.5 GB checkpoint), OCR of all 1,000 source receipts on Colab's CPUs (usually the longest step), four training epochs and evaluation. No elapsed time is promised before a qualified hosted run has measured it.

### Glossary

| Term | Meaning here |
| --- | --- |
| OCR | Optical character recognition: image → recognised words with boxes and confidences (Tesseract 5.5.0, LSTM engine, `eng+ind`). |
| Bounding box | A word's pixel rectangle `x0, y0, x1, y1` on the EXIF-oriented image. |
| Extractive span | A contiguous run of OCR words selected as the answer; the model cannot write text that OCR did not produce. |
| Annotation | CORD's published labels (categories, value words, parsed values), used only for training, selection and evaluation. |
| Fine-tuning | Gradient updates to the last four LayoutLM encoder blocks and the span head, from a fresh pinned checkpoint. |
| Exact match (EM) | The canonical decimal amount equals the reference exactly; no rounding tolerance. |
| Validation / test | Validation receipts develop choices (epoch, threshold); the official test receipts are scored once after everything is frozen. |
| Calibration | Whether a score behaves like a probability of being right. The scores here are **not** calibrated. |
| Coverage | The share of receipts whose total is left unflagged by the review policy. |
| Selective error | The error rate among unflagged totals that have usable references. |
| Adapter | The exported trained tensor subset (SafeTensors) plus the configuration needed to rebuild the workflow. |
| Provenance | Recorded identities (revisions, SHA-256 digests) of every data, model, OCR and code input. |

The controls below have safe defaults and never block **Run all**.
""")
    code("""# @title Run controls (safe defaults)
OUTPUT_DIR = 'receipt_capstone_outputs' # @param {type:'string'}
ACTIVITY_TARGET = 0.90 # @param {type:'number'}
USE_BYOD = False # @param {type:'boolean'}
BYOD_MODE = 'inference' # @param ['inference', 'adapt']
BYOD_PATH = '' # @param {type:'string'}
BYOD_AUTHORIZED = False # @param {type:'boolean'}
DOWNLOAD_RESULTS = False # @param {type:'boolean'}
print('Controls set. BYOD is', 'on' if USE_BYOD else 'off (default).')""", cellView="form")
    md("""## 2. Setup — infrastructure (collapsed)

These cells check the runtime, write the carried implementation (verified by SHA-256), create an isolated **Python 3.12** environment from a fully hashed dependency lock, install the pinned Tesseract OCR closure and verify the pinned base checkpoint. No DIMER package is installed and no source is downloaded.

Network hosts: PyPI (pinned `uv` wheel and hashed lock), the CPython build `uv` downloads, `api.anaconda.org` / `conda.anaconda.org` (micromamba and the 28 OCR packages, each SHA-256-pinned), `raw.githubusercontent.com` (pinned `tessdata_fast` eng/ind), and `huggingface.co` (the CORD v2 shards and `impira/layoutlm-document-qa` at fixed revisions). No receipt content is sent anywhere.

**Input:** a fresh T4 runtime. **System:** verified carrier, locked Python environment, pinned OCR and model. **Output:** a unique run directory and identity reports. Integrity failures stop execution; do not disable checks to continue.
""")
    code(PREFLIGHT, cellView="form")
    code("NOTEBOOK_REVISION = " + repr(NOTEBOOK_REVISION) + "\nCARRIED_FILES = " + repr(files) + "\nCARRIED_HASHES = "
         + repr({name: sha(text) for name, text in files.items()}) + "\n" + CARRIER,
         cellView="form", dimer={"embedded_sources": True})
    code(BOOTSTRAP + HELPERS, cellView="form")
    code("run_stage('ocr_runtime')\nshow_json('ocr_runtime.json', keys=['engine_version', 'leptonica', 'executable_sha256', 'platform', 'install_method', 'qualifying_runtime'])\nrun_stage('model')", cellView="form")
    md(f"""## 3. Data card and source audit

**What counts as a receipt and a reference?** The default population is the complete public CORD v2 sample: 800 official training, 100 validation and 100 test receipts. The notebook preserves those official partitions and splits the official validation set into two roles *before* any exclusion, by ascending SHA-256 of `receipt-capstone-v1|42|<receipt id>`:

| Role | Source | Permitted use |
| --- | --- | --- |
| `train` | official train | gradient updates, keyword development, supervised OCR alignment |
| `validation_model` | first 50 official validation IDs by hash | choosing the trained epoch |
| `validation_policy` | remaining 50 | choosing review thresholds; the change-one-thing activity |
| `test` | official test | one frozen evaluation of every predetermined system |

The frozen source audit found two exact-pixel duplicate pairs inside training; the lexically later copy of each is excluded, giving **{counts['train']} / {counts['validation_model']} / {counts['validation_policy']} / {counts['test']}** receipts. Near-duplicates (same merchant or template) were **not** audited, so the split is not template-disjoint. Every shard and every row is re-hashed against the committed manifest; a changed byte stops the run.

Each receipt–field reference has one state: `present_usable` (a unique amount agreed by the labelled value words and the parsed value), `not_annotated` (**not** the same as absent or zero), `ambiguous_reference` or `invalid_reference`. Only `present_usable` references enter correctness denominators. References are stored separately from model inputs, and each access records its purpose. Published annotations are references, not newly verified financial truth; no independent human review is claimed.

**Predict:** which field will most often be `not_annotated`, and why might that be different from "the receipt has no such charge"?
""")
    code("run_stage('prepare')\nrun_stage('references')\nshow_table('data_summary.csv')\nshow_table('reference_states.csv', limit=16)\nshow_table('reference_examples.csv', limit=20)")
    md("""**What to notice:** the counts include every frozen receipt; later OCR failures, long documents or wrong predictions will **not** remove validation or test receipts. The examples (training receipts only) show the raw reference text and the canonical amount produced by the same conservative grammar that will be applied to predictions.

<details><summary>Show a worked answer</summary>Service charge is usually the least annotated field: many receipts print none, but CORD labels only what annotators marked, so a missing label is "unknown", not "zero". Predictions on `not_annotated` fields are therefore reported as unscored and unverified, never as false positives or correct nulls.</details>
""")
    md("""## 4. OCR inspection: can the amount be recovered from the recognised words?

LayoutLM reads **words and boxes, not pixels**, so every learned system depends on OCR. One Tesseract configuration serves all three systems: LSTM engine (`--oem 1`), uniform-block segmentation (`--psm 6`), languages `eng+ind`, TSV word output, a 60-second limit per image, and the full EXIF-oriented image (no cropping, deskewing, binarisation or upscaling). Low-confidence words are kept. Empty OCR, a timeout or more than 2,000 words becomes a structured failure record, never a silent drop.

**Input:** oriented images. **System:** pinned Tesseract. **Output:** tokens with IDs, text, boxes, line identifiers and raw confidence, cached under a key that includes the pixel hash and every OCR identity.

**Predict:** on a thermal receipt photo, which characters do you expect OCR to confuse, and what would that do to an amount such as `75.000`?
""")
    code("run_stage('ocr')\nshow_table('ocr_summary.csv')\nshow_figure('ocr_inspection.png')\nshow_table('ocr_tokens_example.csv', columns=['token_id', 'text', 'line', 'confidence', 'parseable_amount'], limit=30)")
    md("""**What to notice:** the inspection uses a *training* receipt, so no evaluation receipt is examined early. Grey boxes are all recognised words; orange boxes are words the amount grammar can parse. A total that OCR split, merged or misread cannot be selected later by any extractor.

<details><summary>Show a worked answer</summary>Typical confusions are `0`/`O`, `1`/`l`, `5`/`S`, a dropped separator or a separator read as a comma. The normaliser never repairs these (no `O`→`0` substitution, no invented digits), so a misread total stays wrong or unparseable. That is deliberate: a silent repair could turn an unreadable amount into a confident wrong record.</details>

### How amounts are normalised

The frozen `cord_mixed_v1` grammar accepts non-negative integers, consistent three-digit grouping with `,` or `.`, and an optional two-digit decimal part after the *other* separator: `75,000`, `75.000`, `75,000.00` and `75.000,00` all become `75000`; `1,23,456`, percentages, signs, several amounts in one span and letters are `ambiguous`, `unsupported` or `parse_failed`, never guessed. A separately recognised `Rp` marker is removed and recorded as the currency; no currency is inferred from the country. Canonical amounts are decimal strings, never floating-point numbers.
""")
    md("""## 5. System A: fixed keyword rules

**What does a transparent extractor get right or miss?** The keyword dictionary is derived from **training annotations only**: a key phrase becomes a field keyword when it appears at least three times and at least 80% of its training occurrences label that field. Matching is case- and punctuation-insensitive over whole OCR words, longest phrase first, so `SUB TOTAL` is never read as `TOTAL`, and cash, change and discount phrases cannot become totals. For each keyword the rule takes the rightmost parseable amount on its OCR line (receipts right-align prices), or on the next line if none; distinct values at the best rank are reported as `ambiguous`, not picked arbitrarily. Its ranking score is the lowest OCR confidence among the supporting key and amount words.

**Predict:** will the rules do better on totals or on tax, and what receipt layout would defeat them?
""")
    code("run_stage('keywords')\nshow_table('keyword_rules.csv', limit=30)\nrun_stage('rules')\nshow_table('validation_comparison.csv')")
    md("""**Interpret:** these are `validation_model` numbers (development data), not independent evidence. Each field's denominator is its count of usable references. A simple baseline winning is a legitimate result.

<details><summary>Show a worked answer</summary>Totals usually carry a distinctive keyword and sit at the right edge, so rules often do well on them; tax and service lines vary more (`PB1`, `PPN`, `SVC`, percentages next to amounts). Layouts that put the amount on a different line far below the label, abbreviations absent from training, or two `TOTAL` lines with different values defeat the rules, and the last case is reported as `ambiguous`.</details>
""")
    md("""## 6. System B: frozen LayoutLM

**Does layout-aware extraction add value?** `impira/layoutlm-document-qa` (MIT) at a pinned revision answers each fixed question — *What is the total amount? What is the subtotal? What is the tax amount? What is the service charge?* — by choosing the best span of at most 15 tokens over overlapping 512-token windows (stride 128), exactly as the repository pipeline does. All four questions are asked for every receipt; references never decide which questions are asked. Its span score is a ranking signal, not a probability, and the model has no learned "no answer" option: it always returns its best span. These frozen predictions are saved **before** any training, so the baseline can never come from a model that was trained in memory.

**Predict:** will the frozen model's answers often include label words such as `Total`, and what would that do to exact match?
""")
    code("run_stage('frozen')\nshow_table('validation_comparison.csv')")
    md("""**Interpret:** a span such as `TOTAL 75.000` is readable but is not an amount, so the conservative normaliser returns `parse_failed` and the answer counts as wrong. That is the gap between *finding the right region* and *producing a usable record*.

<details><summary>Show a worked answer</summary>The Impira checkpoint was fine-tuned for general document questions, not for returning bare amounts on Indonesian receipts, so it can drift to label words, quantities or neighbouring prices. Because matching is exact, a near-miss earns no credit.</details>
""")
    md("""## 7. Alignment and training (System C)

**What can be learned from noisy OCR?** A span extractor can learn only from a gold amount that exists in the recognised words. For each usable training reference the notebook searches contiguous OCR spans (same line, at most four words) whose normalised amount equals the reference and whose box overlaps the labelled value region (intersection-over-area ≥ 0.5). Nested matches collapse to the shortest; two separate matches are skipped as ambiguous; a missing or misread amount is skipped as `target_not_recoverable_from_ocr`. Reference text never replaces OCR tokens, and skips never become fake "no answer" examples.

The frozen recipe starts from a **fresh** pinned checkpoint (no earlier CORD adapter) and trains only the last four encoder blocks and the span head: AdamW, learning rate 3e-5, weight decay 0.01, batch 8, **4 epochs, all executed**, float32, gradient clipping 1.0, seed 42, no scheduler. The run asserts positive optimiser steps, finite loss, the exact trainable tensor list, a change in permitted tensors and **no** change in frozen tensors. The adapted candidate is chosen from **epochs 1–4** by `validation_model` total exact match (earliest epoch on ties); epoch 0 is not eligible, so the notebook always exports a trained model — even one that loses to System B.

**Predict:** what fraction of usable training totals will align to OCR, and will validation total EM rise every epoch?
""")
    code("run_stage('train')\nshow_table('alignment_summary.csv')\nhistory = show_json('training_history.json', keys=['selected_epoch', 'n_trainable', 'n_total', 'optimizer_steps', 'changed_tensors', 'adapter_sha256', 'seconds'])\nshow_figure('training_curve.png')\nshow_table('validation_comparison.csv')")
    md("""**Interpret:** read the alignment coverage beside everything that follows: fields with few aligned examples cannot be learned well. The selected epoch is the one validation favoured, not necessarily the last. Validation numbers are still development evidence.

<details><summary>Show a worked answer</summary>Alignment is limited by OCR: when Tesseract drops or garbles a digit the target is not recoverable, and the model never sees that receipt's total. Validation EM often rises then plateaus or dips; the selection rule keeps the best of epochs 1–4 without looking at test receipts.</details>
""")
    md("""## 8. Review-policy development

**What changes when uncertain totals are referred?** A total always needs review when processing failed, no valid span exists, or the amount is unparsable or ambiguous. Otherwise it is left unflagged when its score reaches a system-specific threshold. Thresholds are chosen **only on `validation_policy` receipts**: maximise coverage subject to at least **95%** empirical accuracy, at least **50%** coverage and at least **25** unflagged receipts (illustrative teaching settings, not business service levels), searching every observed score plus accept-all and refer-all. If no threshold qualifies, the policy is **refer all** with `policy_feasible = false`; the target is never loosened automatically. States are `needs_review` and `total_unflagged` only — never "verified" or "approved".

**Predict:** which system will reach the 95% target with the most coverage, and could a system with lower accuracy still achieve a usable policy?
""")
    code("run_stage('select_policy')\nshow_table('policy_selection.csv')\nshow_figure('policy_validation.png')")
    md("""**Interpret:** each curve comes from about fifty receipts, so one decision moves accuracy by roughly two percentage points; the Wilson interval shows how uncertain the selected accuracy is. A feasible policy on validation is not a guarantee on new receipts, and an infeasible (refer-all) policy is a valid, reportable result.

<details><summary>Show a worked answer</summary>Coverage at a fixed accuracy target depends on how well a score *ranks* right answers above wrong ones, not on raw accuracy alone. A system with modest accuracy can still achieve useful coverage if its errors have low scores; a more accurate system can fail if its confident answers include errors.</details>
""")
    md("""## 9. Freeze, then test once

Before any test scoring, `selection_record.json` fixes the cohort, role and data identities, model/OCR identities, preprocessing, grammar, keyword, question and policy hashes, the training recipe, the chosen epoch, the adapter digest, every threshold and the metric definitions, sealed by its own SHA-256. Evaluation refuses to run if any of them changed.

**Primary endpoint:** total EM on test receipts with a usable total reference; OCR, model, parse and ambiguity failures count as incorrect. **Predetermined contrast:** adapted minus frozen LayoutLM, with both learned systems also compared with the rules. Intervals are paired receipt-level bootstrap (2,000 seeded resamples) for differences and Wilson intervals for single rates; they ignore merchant/template dependence and annotation bias. Secondary measures are per-field EM (with denominators), field-macro EM, the all-annotated-fields match (which says nothing about unannotated fields), text EM/ANLS and coverage.

**Predict:** will fine-tuning beat the frozen model on test totals, and will the difference's interval exclude zero?
""")
    code("run_stage('freeze')\nshow_json('selection_record.json', keys=['record_digest', 'selected_epoch', 'adapter_sha256', 'frozen_at', 'test_scored_before_freeze'])\nrun_stage('evaluate')\nshow_table('system_comparison.csv', columns=['system', 'frozen_receipts', 'total_usable', 'total_correct', 'total_em', 'field_macro_em', 'all_annotated_fields_match'])\nshow_table('paired_bootstrap.csv')\nshow_figure('test_comparison.png')\nshow_table('field_metrics.csv', columns=['system', 'field', 'usable_references', 'correct', 'em', 'parse_coverage', 'unscored_on_not_annotated'], limit=12)\nshow_table('review_metrics.csv', columns=['system', 'policy_feasible', 'threshold', 'usable_total_references', 'scored_coverage', 'selective_error', 'unflagged_scored', 'unflagged_wrong', 'operational_routing_coverage', 'review_count', 'hard_failures'])")
    md("""**Interpret:** keep the unfiltered primary EM beside the selective metrics: a low selective error on a small unflagged set does not make the whole workflow accurate. *Scored coverage* uses receipts with usable truth; *operational routing coverage* counts every frozen receipt, including those whose correctness is unknown. Do not revise any setting after reading this section — doing so starts a new experiment on an already-inspected test set.

<details><summary>Show a worked answer</summary>Fine-tuning on OCR-aligned examples usually teaches the model to return bare amounts, which the frozen model often does not, so an improvement is plausible but not guaranteed. With about one hundred test receipts, an interval that includes zero means the sample cannot distinguish the systems; that is a finding, not a failure.</details>
""")
    md("""## 10. Diagnose: OCR, extraction, parsing or the reference?

**Reference-text diagnostic.** The same frozen weights and questions are re-run on CORD's *annotated* words and boxes (categories, key flags and parsed values stripped). This changes recognition, geometry and reading order together, so it is an input-source sensitivity check, not a causal decomposition or an upper bound; an OCR-trained model may do worse on reference text. It is labelled `reference_text_diagnostic` and never replaces the image-based result.

**OCR recoverability (evaluator-only).** For usable references, is the correct amount present in a spatially compatible OCR span? The primary score is never filtered by it and no predictor sees it.

**Failure panel.** After scores are fixed, one test receipt per category is chosen deterministically (success, OCR loss, extraction error, numeric ambiguity, failure/no candidate); an empty category is reported as empty rather than manufactured.
""")
    code("run_stage('diagnose')\nshow_table('reference_text_diagnostic.csv', limit=12)\nshow_table('ocr_recoverability.csv', limit=12)\nshow_table('failure_panel.csv')\nfor category in ('success', 'ocr_loss', 'extraction_error', 'numeric_ambiguity', 'failure_or_no_candidate'):\n    if (ROOT / 'outputs' / 'figures' / f'panel_{category}.png').exists():\n        show_figure(f'panel_{category}.png')")
    md("""**Interpret:** if a large share of test totals is *not recoverable* from OCR, no extractor could have got them right; improving OCR (or image capture) would matter more than a better extractor. Conditional EM on recoverable totals is secondary evidence only.

<details><summary>Show a worked answer</summary>An extractive model selects words; it cannot restore a digit Tesseract never produced. When the reference-text diagnostic scores far higher than actual OCR, recognition is the bottleneck; when both are low, extraction or the amount grammar is the bottleneck.</details>
""")
    md("""## 11. Infer and export

The export stage writes machine-readable records for the evaluation roles — `predictions.jsonl` (exact raw strings, token IDs, canonical amounts, parse states, review decision) and spreadsheet-safe `records.csv` (formula-like strings are neutralised; JSON keeps the exact text). It also writes the serving bundle: the selected trained tensors in SafeTensors, the base identity, questions, OCR lock, preprocessing, number grammar, keyword rules, review policy, reconstruction notes and licences. The bundle contains no receipt image, OCR transcript, training example or reference, but weights may still encode what they learned. It is not a DIMER production-worker artifact.
""")
    code("run_stage('export')\nshow_table('records.csv', columns=['receipt_id', 'role', 'system_id', 'total_amount_raw_text', 'total_amount_normalized', 'total_amount_parse_status', 'total_review_state'], limit=9)\nshow_table('artifact_inventory.csv')")
    md("""## 12. Fresh reload: can the serialized workflow reproduce its output?

A **new process** verifies the bundle's file set, sizes, digests, schema, tensor names/shapes and base identity *before* loading, loads a fresh verified checkpoint, applies only the allowed tensors, rebuilds OCR, questions, grammar, rules and policy from JSON, then re-runs three deterministically chosen held-out receipts through the whole image → OCR → record path with **no labels supplied**. OCR tokens, spans, amounts, parse states and review decisions must match exactly; scores within `atol=1e-6, rtol=1e-5`. These are demonstration replays of held-out images, not a second test set.
""")
    code("run_stage('replay')\nshow_json('replay_report.json', keys=['pid', 'receipts', 'all_parity', 'labels_supplied', 'note'])\nshow_table('replay_records.csv')\nrun_stage('report')\nshow_table('runtime_summary.csv', limit=20)\nshow_json('archive_verification.json')\ndisplay(Markdown((ROOT / 'outputs' / 'summary.md').read_text(encoding='utf-8')))\nprint('Results ZIP (derived evidence + small bundle; no receipt images):', ROOT / 'outputs' / 'results.zip')\nif DOWNLOAD_RESULTS:\n    from google.colab import files as colab_files\n    colab_files.download(str(ROOT / 'outputs' / 'results.zip'))")
    md("""## 13. Change one thing: the review accuracy target

**Predict → change one thing → run → observe → explain.** Only the minimum accuracy target changes (the `ACTIVITY_TARGET` control, default 0.90 instead of 0.95), on the cached `validation_policy` predictions. No training, no test data, and the canonical threshold, model, selection record and test results are untouched; results go to `outputs/activity/`.

**Predict:** lowering the target from 95% to 90% — how much coverage do you gain, and how many more errors are left unflagged?
""")
    code("run_stage(None, '--activity-target', repr(float(ACTIVITY_TARGET)))\nfolder = ROOT / 'outputs' / 'activity' / f'target_{float(ACTIVITY_TARGET):.2f}'\nshow_table(str(folder.relative_to(ROOT / 'outputs') / 'activity_comparison.csv'))\ndisplay(Image(filename=str(folder / 'activity.png')))")
    md("""**Explain:** a lower target admits lower-scoring totals, raising coverage and usually the number of wrong totals left unflagged. On about fifty validation receipts the change may be a handful of decisions; this is a teaching comparison, not an estimate of a real service level.

<details><summary>Show a worked answer</summary>Whether coverage grows depends on how many validation totals have scores between the two thresholds. If the canonical policy was already refer-all because no threshold reached 95% with 25 receipts, a 90% target may make a policy feasible — at the cost of accepting more errors.</details>
""")
    md("""## 14. Optional: bring your own receipts (BYOD)

Off by default. **Privacy and authority:** process only receipts you are authorised to process in this hosted runtime, remove unnecessary personal or payment details first, and review every export before sharing. Nothing is sent to an OCR or LLM API; "local" here means this Colab runtime, not your own computer.

Upload a ZIP in the Files panel, set `USE_BYOD`, `BYOD_MODE`, `BYOD_PATH` and `BYOD_AUTHORIZED` in §1, then run this section. The ZIP holds `manifest.json` (schema `org.dimer.receipt-byod.v1`, `authorized: true`, `number_format_policy` = `dot_decimal_comma_grouping` or `comma_decimal_dot_grouping`, a three-letter `currency` such as `PHP`, and one entry per image with a nonidentifying `receipt_id` and relative `image` path) and the JPEG/PNG images. Limits: 1,000 images, 2 GiB compressed, 5 GiB expanded, 20 MiB / 20 MP per image; traversal, links, duplicates and undeclared files are refused.

* **`inference`**: the canonical bundle extracts four candidates; the CORD-selected threshold is **not validated for your receipts**, so every total defaults to `needs_review` and the transferred decision is only shown for comparison. No accuracy is reported without checked references.
* **`adapt`**: adds `annotations.jsonl` (per receipt and field: `status` `present`/`not_annotated`, `raw_value`, `value_boxes` in original-image pixels, `provenance`) and explicit disjoint roles `train`, `validation_model`, `validation_policy`, `test` (optional `inference`) with a `group_id` per receipt; groups may not cross roles. The full workflow then runs in a separate directory from a fresh base: at least eight aligned training examples and nonempty validation and test roles are required, and missing prerequisites are refused. Small samples are small-sample evidence; a Philippine BYOD run is a new evaluation, not an extension of the CORD score.
""")
    code(r"""# @title Optional BYOD run (off unless enabled in §1)
if not USE_BYOD:
    print('Optional BYOD is off. The canonical capstone is complete.')
else:
    if not BYOD_AUTHORIZED:
        raise ValueError('Confirm in §1 that you are authorised to process these receipts (BYOD_AUTHORIZED).')
    archive = Path(BYOD_PATH).expanduser().resolve()
    if not archive.is_file():
        raise ValueError('Upload a ZIP in the Colab Files panel and set BYOD_PATH to its path.')
    inspect = subprocess.run([str(PYTHON), str(ROOT / 'capstone.py'), '--byod-inspect', str(archive), '--byod-mode', BYOD_MODE],
                             env=ENV, capture_output=True, text=True)
    if inspect.returncode:
        raise ValueError('BYOD archive refused: ' + inspect.stderr.strip().splitlines()[-1])
    contract = json.loads(inspect.stdout)
    byod_root = ROOT.parent / (ROOT.name + '-byod-' + uuid.uuid4().hex[:6])
    byod_root.mkdir()
    for name, text in CARRIED_FILES.items():
        path = byod_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8', newline='\n')
    config = {'run_id': RUN_ID + '-byod', 'source': 'byod', 'byod_archive': str(archive), 'byod_mode': BYOD_MODE,
              'number_format_policy': contract['policy'], 'currency': contract['currency'],
              'cache_dir': str(ROOT / 'cache'), 'ocr_runtime_from': str(ROOT / 'outputs' / 'ocr_runtime.json'),
              'keyword_rules_from': str(ROOT / 'stages' / 'keywords' / 'keyword_rules.json')}
    if BYOD_MODE == 'inference':
        config.update(mode='byod_inference', bundle_from=str(ROOT / 'stages' / 'export' / 'receipt_intelligence_artifact'))
        stages = ('prepare', 'ocr_runtime', 'model', 'ocr', 'byod_infer')
    else:
        config.update(mode='canonical')
        stages = ('prepare', 'references', 'ocr_runtime', 'model', 'ocr', 'keywords', 'rules', 'frozen', 'train',
                  'select_policy', 'freeze', 'evaluate', 'diagnose', 'export', 'replay', 'report')
    (byod_root / 'run_config.json').write_text(json.dumps(config), encoding='utf-8')
    for stage in stages:
        run_stage(stage, root=byod_root)
    if BYOD_MODE == 'inference':
        show_table('byod_records.csv', root=byod_root, limit=20)
    else:
        show_table('system_comparison.csv', root=byod_root)
        show_table('review_metrics.csv', root=byod_root, columns=['system', 'policy_feasible', 'scored_coverage', 'selective_error', 'unflagged_scored'])
    print('Separate BYOD outputs (never merged into the canonical results):', byod_root / 'outputs')""", cellView="form")
    md("""## 15. Conclude with evidence

Complete this optional scaffold in your own words:

> **Task and population:** On [n] held-out CORD v2 Indonesian receipts, extracting [fields] from photographs through Tesseract OCR…
> **Principal result and baseline:** [system] reached total EM [value, interval] versus [rules] and [frozen model]; the adapted-minus-frozen difference was [value, interval].
> **One failure mode:** [OCR loss / label words in spans / ambiguous separators / …], seen in [evidence].
> **Review trade-off:** the validation-selected policy left [scored coverage] unflagged with selective error [value] (policy feasible: [yes/no]); lowering the target to [x] changed [coverage, errors] on validation.
> **Uncertainty:** [small validation sets, template dependence, annotation bias, unknown pretraining overlap].
> **Evidence needed for transfer:** [authorised local receipts with checked references and boxes, merchant-disjoint splits, a locally validated review policy, human audit of references…].

Limits that bound every conclusion: published CORD annotations are references, not verified accounting truth; missing annotations do not establish absence; scores are uncalibrated; the review policy routes totals only and never certifies a record; the arithmetic `subtotal + tax + service = total` is not assumed (discounts, inclusive tax and rounding are outside this schema); and a losing fine-tuned model, poor OCR or an infeasible 95% policy are valid, informative outcomes.

### References

- Park, S., Shin, S., Lee, B., Lee, J., Surh, J., Seo, M., & Lee, H. (2019). *CORD: A consolidated receipt dataset for post-OCR parsing*. Document Intelligence Workshop at NeurIPS. [Publisher repository](https://github.com/clovaai/cord) (no DOI verified). Dataset: [naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2), CC BY 4.0.
- Xu, Y., Li, M., Cui, L., Huang, S., Wei, F., & Zhou, M. (2020). LayoutLM: Pre-training of text and layout for document image understanding. *Proceedings of KDD 2020*, 1192–1200. https://doi.org/10.1145/3394486.3403172
- Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory, 16*(1), 41–46. https://doi.org/10.1109/TIT.1970.1054406 (the error–rejection idea; not a claim that this uncalibrated threshold is optimal).
- Engineering sources: [impira/layoutlm-document-qa model card](https://huggingface.co/impira/layoutlm-document-qa); [Tesseract command-line usage](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html) and [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast); [DIMER notebook standard](https://github.com/kurtvalcorza/ml-worker/blob/main/integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md); [this repository](https://github.com/kurtvalcorza/layoutlm-document-qa-pipeline).

### Troubleshooting

| Observation | Response |
| --- | --- |
| No T4 or too little disk | Start a fresh T4 runtime; do not substitute a CPU or another accelerator silently. |
| Hash, count or schema refusal | Stop and keep the log; never switch to `main`, a mirror, another dataset version or reference OCR. |
| OCR install or language check fails | Keep the log; the pinned closure must install as declared (this is an open qualification item). |
| Training refuses (< 8 aligned examples) | The data contract failed; do not synthesise supervision. |
| Policy is refer-all | A valid result: no threshold met the declared targets on validation. |
| Replay parity fails | Keep outputs; resolve the mismatch before any release claim. |
| An undefined metric | Read its denominator; do not replace it with a favourable number. |
""")
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "accelerator": "GPU",
            "colab": {"name": NAME, "gpuType": "T4", "provenance": []},
            "dimer": {"notebook_spec": "2.2", "profile": "E2E", "notebook_profile": "E2E", "pedagogical_mode": "GUIDED",
                      "notebook_mode": "GUIDED", "standalone": True, "requires_dimer_worker": False,
                      "release_status": "Candidate", "notebook_revision": NOTEBOOK_REVISION, "generated_from": source},
        },
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "tutorials" / NAME
    content = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != content:
            raise SystemExit("Receipt capstone differs from generated sources; run tools/build_receipt_capstone.py")
        print("Receipt capstone parity: PASS")
    else:
        target.write_text(content, encoding="utf-8", newline="\n")
        print(target)


if __name__ == "__main__":
    main()
