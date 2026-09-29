# Receipt capstone — release evidence ladder

Each state is reported separately. A green CI run or a local CPU chain is not hosted qualification.

| State | Status | Evidence |
| --- | --- | --- |
| specified | done | [`receipt-capstone-spec.md`](receipt-capstone-spec.md) v1.0, preserved byte-for-byte |
| built | done | Generated notebook `tutorials/DIMER_Small_Business_Receipt_Intelligence_Capstone.ipynb`, revision `0.1.0-candidate`, from `tools/build_receipt_capstone.py` |
| source-checked | done (local) | `ruff check src tests tools`; `python tools/build_receipt_capstone.py --check`; `python tools/validate_release_assets.py`; see the build record below |
| CPU-tested | done (local, test doubles) | Full `pytest`, including the stage chain on synthetic receipts. The chain uses recorded OCR tokens and a tiny random LayoutLM, then is repeated once with a local Tesseract 5.5.0 build. These are mechanics checks only |
| hosted-executed | **default journey done** (`fc553d4`) | Kaggle Tesla T4 clean-runtime Run all of the exact PR head, 2026-09-29: 17/17 code cells, 941.8 s. See *Recorded executions*. BYOD inference, BYOD adapt and the refused-ZIP case are **not yet run** |
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

| Date | Revision / notebook blob | Runtime | Journeys | Wall | Result |
| --- | --- | --- | --- | --- | --- |
| 2026-09-29 | `fc553d4` / `e54dd1e7` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-small-business-receipt-intelligence-caps` v1) | default + change-one-thing activity (0.90) | 941.8 s | **PASSED** (execution): 17/17 code cells ok; findings R1–R5 below |

### Kaggle T4 clean-runtime execution of revision `fc553d4` — 2026-09-29

- **Files.** [`execution-evidence/2026-09-29/receipt_capstone_fc553d4_kaggle-t4.ipynb`](execution-evidence/2026-09-29/receipt_capstone_fc553d4_kaggle-t4.ipynb), SHA-256 `1f2c65b1dfc99b060f183098bdaf05db31a8397a4b72065a9046220cdd7ea236`, and the executor's `receipt_capstone_fc553d4_run_summary.json`, SHA-256 `1e061cca0e5f337bbeb9a4281066b3fcc6e2bcdc5072f855f209a7e857462d6e`. Both are byte-for-byte copies of the Kaggle output.
- **Source match.** The executor downloaded the notebook from GitHub at `fc553d4d68aa220c3cc84e2aa66c20333f6334ea` and verified its git blob `e54dd1e77c9c0cb35a262eddb6e18fd0b4d7ea7b` before execution. The executed notebook matches the PR-head notebook exactly: 42/42 cells, same ids and order, and no source diff (no `# @param` toggle was changed).
- **Runtime.** Kaggle Batch, image `gcr.io/kaggle-gpu-images/python@sha256:37c64f7d…`, Python 3.12.13, 4 CPUs, Tesla T4 (15,360 MiB, driver 580.159.04). Run all happened in a fresh `nbclient` interpreter with an empty Hugging Face cache and no repository checkout; no restart was needed. The notebook built its isolated CPython 3.12.12 environment from the hashed lock (52 packages, about 70 s), so the image's own `torch 2.10.0+cu128` and `transformers 5.0.0` were not used.
- **§2 OCR closure.** `tesseract 5.5.0` (leptonica 1.83.1), executable SHA-256 `b323272f…`, micromamba explicit offline create from SHA-256-verified packages, `qualifying_runtime: true`. This is the first execution of the Linux OCR install. Base checkpoint `impira/layoutlm-document-qa@beed3c4d02d8` was verified.
- **§3 data.** Frozen cohort train 798 / validation_model 50 / validation_policy 50 / test 100, with 2 train exclusions. Usable total references: 767 / 49 / 49 / 94. Reference audit: 24 `unparseable_source_value`, 11 `invalid_or_missing_value_words`, 4 `multiple_incompatible_values`.
- **§4 OCR.** 998 receipts in 390 s. `ocr_empty` for 162 / 9 / 12 / 22 receipts (train / validation_model / validation_policy / test), with 0 timeouts, errors or image errors. Median 13–15 words per receipt.
- **§7 training.** 432 aligned OCR examples from 257 receipts; 922 targets were skipped as `target_not_recoverable_from_ocr` and 374 as `ocr_empty`. Validation_model total EM was 0.265 / 0.245 / 0.265 / 0.265 over epochs 1–4, so **epoch 1** was selected. 216 optimizer steps in 52 s; 28,353,026 of 127,792,898 parameters trainable; adapter SHA-256 `e1f83717…`.
- **§8 policy.** At the 0.95 target no system has a feasible threshold on validation_policy, so all three are `refer_all` (coverage 0).
- **§9 test (evaluated once, selection record `e510b603b85b2f5b`).**

  | System | Total EM | Correct / usable | Field-macro EM |
  | --- | --- | --- | --- |
  | rules_baseline | 0.1809 | 17 / 94 | 0.1651 |
  | layoutlm_frozen | 0.2128 | 20 / 94 | 0.1952 |
  | layoutlm_adapted | 0.2340 | 22 / 94 | 0.2069 |

  Paired receipt bootstrap (2,000 resamples): adapted − frozen 0.0213 [−0.0322, 0.0851]; frozen − rules 0.0319 [−0.0319, 0.0957]; adapted − rules 0.0532 [0.0106, 0.1064].
- **§10 diagnostics.** With the reference text in place of OCR, total EM is 0.82 / 0.88 / 0.87 (rules / frozen / adapted). The test total appears in the OCR for only 26 of 94 receipts (27.7%); subtotal 18.5%, tax 17.9%, service 25.0%.
- **§11–§12 export and replay.** 66 trained tensors (113,419,976 bytes). A fresh-process replay (pid 1304) of 3 held-out receipts reached `all_parity: 1.0`. Results ZIP: 32 members, 105,374,681 bytes, CRC and member digests pass.
- **Resources.** The largest stage was OCR, at 389.7 s. Peak RSS was 3.84 GiB (prepare) and peak GPU allocation 1.67 GiB (train).
- **Evidence boundary.** Each cell's output was inspected in the executed notebook. This is one clean-runtime execution. It was not independently repeated, and it is not a Colab run.

| Journey | Verdict |
| --- | --- |
| Default Run all (canonical path, §1–§12) | **PASS**: executes end to end; metrics above |
| Change-one-thing activity (`ACTIVITY_TARGET = 0.90`, default) | **PASS (mechanics)**: ran and wrote `activity/target_0.90/`. At both 0.95 and 0.90 every system stays infeasible, so the activity shows no contrast (R1) |
| BYOD inference / BYOD adapt / refused ZIP | not assessed in this run |
| `DOWNLOAD_RESULTS` (Colab `files.download`) | not assessed (Kaggle, default off) |
| Repeated Run all in a warm runtime | not assessed |

**Findings from this run and their dispositions.** The OCR page segmentation changed after this run (PSM 4 → 6, deviation 15 in [`receipt-capstone.md`](receipt-capstone.md)), so **this run no longer matches the PR head** and a re-run on the new head is required.

- **R1 (major, teaching contract).** The review policy is infeasible for every system at both 0.95 and 0.90, so §8, the §9 review table and the change-one-thing activity all read "refer all, coverage 0" on the default path, and the activity demonstrates nothing. The cause is the low OCR recoverability (R2), not the policy code. *Disposition:* expected to change with the R2 fix; to be re-assessed on the re-run.
- **R2 (major, OCR yield).** 20.5% of receipts come back `ocr_empty`, and the test total is recoverable from OCR on only 27.7% of receipts. The §4 figure (train:0296) shows CORD's privacy blurring of most receipt text, which is a plausible main cause, but the blurring has not been shown to be the only cause. OCR is the dominant error source: reference-text EM is about 0.85, against about 0.2 on OCR. *Disposition:* diagnosed as Tesseract PSM 4 layout analysis dropping text blocks; blurring was ruled out as the main cause. Fixed by PSM 6, selected on the validation receipts only (aligned totals 19 → 40 of 98, empty OCR 21 → 1).
- **R3 (minor).** Fine-tuning on 432 aligned examples is not separable from frozen LayoutLM on test (the adapted − frozen interval includes 0), and validation EM is flat across epochs, so epoch 1 is selected. *Disposition:* to be re-assessed on the re-run with more aligned examples.
- **R4 (cosmetic).** Missing values render as `undefined` in the notebook's Markdown tables (threshold, accuracy interval, empty cells). *Disposition:* no change. `undefined` is the notebook's own term for an undefined metric (see the glossary: read its denominator, never substitute a favourable number).
- **R5 (cosmetic).** The activity cell prints `Running None in a separate process` because the stage label is missing. *Disposition:* fixed. `run_stage` now prints the option-derived label it already used for the log name.

**Still open before promotion.** BYOD inference, BYOD adapt and one refused ZIP; dispositions of R1–R5; the annotation audit; and a human decision on promotion. Status stays **Candidate**.
