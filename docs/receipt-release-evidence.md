# Receipt capstone — release evidence ladder

Each state is reported separately. A green CI run or a local CPU chain is not hosted qualification.

| State | Status | Evidence |
| --- | --- | --- |
| specified | done | [`receipt-capstone-spec.md`](receipt-capstone-spec.md) v1.0, preserved byte-for-byte |
| built | done | Generated notebook `tutorials/DIMER_Small_Business_Receipt_Intelligence_Capstone.ipynb`, revision `0.1.0-candidate`, from `tools/build_receipt_capstone.py` |
| source-checked | done (local) | `ruff check src tests tools`; `python tools/build_receipt_capstone.py --check`; `python tools/validate_release_assets.py`; see the build record below |
| CPU-tested | done (local, test doubles) | Full `pytest`, including the stage chain on synthetic receipts. The chain uses recorded OCR tokens and a tiny random LayoutLM, then is repeated once with a local Tesseract 5.5.0 build. These are mechanics checks only |
| hosted-executed | **pending** | No Colab T4 run of any revision exists yet |
| annotation-reviewed | **not started** | [`receipt-annotation-audit.md`](receipt-annotation-audit.md): no human audit performed |
| release-qualified | **no** | Needs the hosted run, a BYOD positive run and a BYOD negative case, measured resources, and the annotation audit |
| published / merged | **no** | Draft PR only; no merge authority implied |

## Build record — 2026-09-28 (local Windows workstation, CPU only)

Environment: Python 3.12.14, torch 2.14.0+cpu, transformers 4.57.6, pillow 11.3.0, pyarrow 25.0.1, matplotlib 3.10.6, pytest 8.4.2, ruff 0.16.6. This is not the notebook's locked runtime (torch 2.9.1 CUDA 12.8 on Linux).

- `ruff check src tests tools`: all checks passed.
- `python tools/build_receipt_capstone.py --check`: parity PASS. `python tools/build_notebook.py --check`: the existing tutorial is unchanged.
- `python tools/validate_release_assets.py`: PASS. Static source validation only.
- `pytest`: 169 passed and 2 skipped, both skips expected. `test_model_backed` needs the git-ignored base snapshot. The real-Tesseract test runs only when `RECEIPT_TESSERACT`/`RECEIPT_TESSDATA` are set; it was run separately and passed with a local conda-forge win-64 Tesseract 5.5.0 and the pinned `tessdata_fast` eng/ind files.
- Stage chain with test doubles: all 16 stages succeeded on 28 synthetic receipts, and fresh-process replay parity passed. Gold non-interference was also checked. With every test total changed, the adapter bytes, keyword rules, review policy, all three systems' predictions and the selection record (excluding its timestamp and digest) stayed byte-identical.
- Stage chain with local Tesseract (development OCR, `qualifying_runtime=false`): all 16 stages succeeded. OCR misread some small-font synthetic amounts, and 12 training targets were correctly skipped as `target_not_recoverable_from_ocr`. Fresh-process replay reproduced the OCR tokens, spans and decisions exactly.

**Not run locally:** the pinned CORD v2 shards and the pretrained `impira/layoutlm-document-qa` snapshot could not be downloaded in the build sandbox (network offline). No real-CORD reference audit, OCR coverage, pretrained inference, T4 training, timing or memory figure exists. The Linux micromamba offline install of the OCR closure has never been executed.

## First hosted run — what to check

1. Fresh Colab T4, **Runtime → Run all** with defaults, on the exact PR head. Keep the executed `.ipynb`.
2. §2: `ocr_runtime.json` shows `tesseract 5.5.0` and `qualifying_runtime: true`. The OCR install is the most likely first failure; keep the log.
3. §3: role counts 798 / 50 / 50 / 100 and the reference-state table (`reference_audit.json` reason counts show which CORD notations the grammar rejects).
4. §4: OCR summary (timeouts, empty pages, >2,000 words) and the measured OCR time on Colab's CPUs.
5. §7: alignment coverage per field, four epochs, the selected epoch, GPU peak memory.
6. §9–§12: the test comparison, policy feasibility, replay `all_parity: true`, `runtime_summary.csv`.
7. Separately: a BYOD `inference` run, a BYOD `adapt` run and one deliberately invalid ZIP (refusal).

Record each executed notebook, pass or fail, under `docs/execution-evidence/<date>/` with a section below.

## Recorded executions

None yet.
