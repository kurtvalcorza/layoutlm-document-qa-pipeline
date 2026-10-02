# Receipt capstone — release evidence ladder

Each state is reported separately. A green CI run or a local CPU chain is not hosted qualification.

| State | Status | Evidence |
| --- | --- | --- |
| specified | done | [`receipt-capstone-spec.md`](receipt-capstone-spec.md) v1.0, preserved byte-for-byte |
| built | done | Generated notebook `tutorials/DIMER_Small_Business_Receipt_Intelligence_Capstone.ipynb`, revision `0.2.0-candidate` (review fixes, 2026-10-02; `0.1.0-candidate` before), from `tools/build_receipt_capstone.py` |
| notebook-reviewed | done; **Needs revision → fixes built, not yet hosted** | Notebook Review Framework v1 review of `ad47ef8` (blob `329fbfe8`), 2026-10-02: one blocker (RC-B1), one major (RC-M1, the earlier R1), five minor. Report and probes in [`reviews/2026-10-02-notebook-review/`](reviews/2026-10-02-notebook-review/); fixes in revision `0.2.0-candidate` (see *Review fixes* below) |
| source-checked | done (local) | `ruff check src tests tools`; `python tools/build_receipt_capstone.py --check`; `python tools/validate_release_assets.py`; see the build record below |
| CPU-tested | done (local, test doubles) | Full `pytest`, including the stage chain on synthetic receipts. The chain uses recorded OCR tokens and a tiny random LayoutLM, then is repeated once with a local Tesseract 5.5.0 build. These are mechanics checks only |
| hosted-executed | **`0.1.0-candidate`: all journeys done at `7bd7f87`** (Kaggle T4, CORD stand-ins for BYOD); **`0.2.0-candidate`: not yet executed** | 2026-09-29 clean-runtime Run all at `7bd7f87`: BYOD inference + refused ZIP (18/18 code cells) and BYOD adapt (17/17), each also repeating the canonical path with results identical to the `78ef7c3` default run. BYOD used CORD stand-ins whose OCR came from the cache, so fresh OCR of new BYOD images is not evidenced. A maintainer-supplied Colab T4 default run of `7bd7f87` reproduces the Kaggle results exactly. No BYOD run on real learner receipts yet. See *Recorded executions* |
| annotation-reviewed | **not started** | [`receipt-annotation-audit.md`](receipt-annotation-audit.md): no human audit performed |
| release-qualified | **no** | Needs a hosted run of `0.2.0-candidate` covering every journey, a BYOD run on images OCR has not seen, and the annotation audit |
| published / merged | `0.1.0-candidate` merged | PR #9 merged to `main` as `ad47ef8` on 2026-09-29 (notebook blob `329fbfe8`). The `0.2.0-candidate` review fixes are on `fix/receipt-capstone-review`, not merged. A merge is not release qualification |

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
| 2026-09-29 | `fc553d4` / `e54dd1e7` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-small-business-receipt-intelligence-caps` v1) | default + change-one-thing activity (0.90) | 941.8 s | **PASSED** (execution): 17/17 code cells ok; findings R1–R5 below; superseded by `78ef7c3` |
| 2026-09-29 | `78ef7c3` / `18cfc43b` | Kaggle Tesla T4 (same kernel, v2) | default + change-one-thing activity (0.90) | 1077.4 s | **PASSED** (execution): 17/17 code cells ok; R2 and R5 fixed; R1 persists (structural) |
| 2026-09-29 | `78ef7c3` / `18cfc43b` | Kaggle Tesla T4 (same kernel, v3) | BYOD inference (12 CORD stand-ins) + refused ZIP | — | **FAILED** at `code-40`, the BYOD `ocr` stage: `KeyError: 'cord-v2:test:0000'` (R6). The refused-ZIP harness cell did not run |
| 2026-09-29 | `78ef7c3` / `18cfc43b` | Kaggle Tesla T4 (same kernel, v4) | BYOD adapt (133 CORD stand-ins) | — | **FAILED** at `code-40`, the BYOD `ocr` stage: `KeyError: 'cord-v2:test:0001'` (R6) |
| 2026-09-29 | `7bd7f87` / `329fbfe8` | Kaggle Tesla T4 (same kernel, v5) | canonical + BYOD inference (12 stand-ins) + refused ZIP | 1111.3 s | **PASSED**: 18/18 code cells ok, including 1 appended harness cell |
| 2026-09-29 | `7bd7f87` / `329fbfe8` | Kaggle Tesla T4 (same kernel, v6) | canonical + BYOD adapt (133 stand-ins) | 1125.8 s | **PASSED**: 17/17 code cells ok |
| 2026-09-29 | `7bd7f87` / `329fbfe8` | Maintainer-supplied Google Colab Tesla T4 (2 CPUs) | default + change-one-thing activity (0.90) | stages ≈1540 s + 97 s install | **PASSED**: 17/17 code cells ok; adapter bytes identical to Kaggle |

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

### Kaggle T4 clean-runtime executions of revision `78ef7c3` — 2026-09-29

**Default journey (kernel v2): PASSED.** [`execution-evidence/2026-09-29/receipt_capstone_78ef7c3_kaggle-t4.ipynb`](execution-evidence/2026-09-29/receipt_capstone_78ef7c3_kaggle-t4.ipynb), SHA-256 `8bccc5c7f9ea03906175a9756021a9916f123bd5a0194880759d4d7f537d40d2`; run summary SHA-256 `dbac31b1f18cd9ee132e6a35cab6fe6b7dfd97ee2ca73f47a7290c5e6493ad5e`.
- **Source match.** Blob `18cfc43bb17cb62b6dd7bb436a9f0206cce8e662` was verified before execution. The executed source is identical to the PR head (42/42 cells, no toggle diffs), and execution counts run 1..17. The runtime is the same Kaggle image and T4 as the `fc553d4` run.
- **Compared with `fc553d4` (PSM 4 → 6):**

  | Measure | `fc553d4` (PSM 4) | `78ef7c3` (PSM 6) |
  | --- | --- | --- |
  | `ocr_empty` (all roles) | 205 / 998 | 22 / 998 |
  | median words per receipt | 13–15 | 27–29 |
  | OCR time | 390 s | 536 s |
  | aligned training examples (receipts) | 432 (257) | 636 (378) |
  | selected epoch; validation_model total EM by epoch | 1; 0.265 / 0.245 / 0.265 / 0.265 | 3; 0.347 / 0.367 / 0.408 / 0.408 |
  | test total EM rules / frozen / adapted | 0.181 / 0.213 / 0.234 | 0.298 / 0.372 / 0.372 |
  | adapted − frozen, 95% paired interval | 0.021 [−0.032, 0.085] | 0.000 [−0.053, 0.053] |
  | test totals recoverable from OCR | 27.7% | 38.3% |
  | review policy (0.95 and activity 0.90) | refer-all for every system | refer-all for every system |

- Selection record `c4ca8bbc632df673`, adapter SHA-256 `da69be11…`, replay `all_parity: 1.0`. Peak GPU allocation 2.15 GiB (train); peak RSS 3.29 GiB (prepare).
- **R5 confirmed fixed.** The activity cell prints `Running activity-target in a separate process`.
- **R1 is structural.** `receipt_policy.DEFAULT_TARGETS` requires coverage ≥ 0.50, accuracy ≥ the target, and at least 25 accepted. With validation total EM of about 0.41, accuracy at 50% coverage cannot exceed about 0.82, even with perfect ranking. Refer-all therefore follows at any accuracy target ≥ 0.82, and an activity that changes only the accuracy target cannot show a contrast on CORD. This is a design decision for the maintainer.

**BYOD journeys (kernels v3 and v4): FAILED, recorded as evidence.**
- **Harness.** [`patch_byod.py`](execution-evidence/2026-09-29/patch_byod.py) patches the executor preamble. It still downloads the committed bytes and asserts the blob. It then stages the archives from the private Kaggle dataset `kurtvalcorza/dimer-receipt-byod-standins`; they are stored as `.bin` because Kaggle unpacks `.zip`. Finally it replaces only the `code-02` form toggles (`USE_BYOD`, `BYOD_MODE`, `BYOD_PATH`, `BYOD_AUTHORIZED`), each guarded by the SHA-256 of the committed cell. The applied diffs are in `…_harness-diff.txt`. For B1 it also appends one harness cell that re-executes `code-40`'s committed source against a path-traversal ZIP. The executor also saved the committed bytes, which are identical to the PR-head notebook.
- **Data.** CORD v2 receipts (CC BY 4.0) repackaged as `org.dimer.receipt-byod.v1` stand-ins by [`build_byod_standins.py`](execution-evidence/2026-09-29/build_byod_standins.py). They are **not independent data**. Inference uses 12 test-shard receipts. Adapt uses 133 validation and test receipts whose total is expressible under `dot_decimal_comma_grouping`, split 68 / 20 / 20 / 25 with one group per receipt. Fields not expressible under that policy are left `not_annotated`. Before upload, the notebook's own `prepare_byod` accepted both archives and refused the third with `Unsafe archive path: '../escape.png'`.
- **Files.** `receipt_capstone_78ef7c3_byod-inference_FAILED.ipynb` (SHA-256 `268e8a17…`) and `receipt_capstone_78ef7c3_byod-adapt_FAILED.ipynb` (SHA-256 `6e653ce2…`), each with its run summary and harness diff.
- **Result.** Cells 1–37 passed in both runs, reproducing the canonical path. `code-40` failed in the BYOD `ocr` stage (`stage_ocr`, `roles[r["receipt_id"]]`): `KeyError: 'cord-v2:test:0000'` for inference and `'cord-v2:test:0001'` for adapt.
- **R6 (major, BYOD).** `receipt_ocr.run_cohort` returns a cached OCR row as stored, including the `receipt_id` of the receipt that first wrote it. The cache key is pixel content plus OCR identity, and the BYOD run shares the canonical cache (`cache_dir`). A BYOD image whose pixels match an earlier receipt therefore comes back under the wrong id. That covers a re-upload under a new id, or these stand-ins, which are pixel-identical to CORD receipts. The stage then fails. *Fixed in the next commit:* a cache hit now takes `receipt_id` from the current record.

| Journey at `78ef7c3` | Verdict |
| --- | --- |
| Default Run all | **PASS** |
| Change-one-thing activity | **PASS (mechanics)**; no contrast (R1, structural) |
| BYOD inference | **FAIL** (R6) |
| BYOD adapt | **FAIL** (R6) |
| Refused ZIP | not assessed; the preceding BYOD cell failed first |

### Kaggle T4 clean-runtime BYOD executions of revision `7bd7f87` — 2026-09-29

The R6 fix (`7bd7f87`: an OCR cache hit keeps the current record's `receipt_id`) was followed by a re-run of both BYOD journeys. The harness, stand-in archives and private dataset are the same as for `78ef7c3` (see above). Before execution, the executor verified blob `329fbfe87258d5dd6bd076c1c202f0a4f82865fe` and saved the committed bytes, which are identical to the PR-head notebook. Runtime: the same Kaggle image and Tesla T4 as before.

| Run | File (SHA-256) | Source vs PR head | Cells | Wall |
| --- | --- | --- | --- | --- |
| v5: BYOD inference + refused ZIP | [`receipt_capstone_7bd7f87_byod-inference.ipynb`](execution-evidence/2026-09-29/receipt_capstone_7bd7f87_byod-inference.ipynb) (`9f522c76868ced999935d053ab06108b7ef821033cca54c927f36b1d093eb89e`) | `code-02` toggles only, plus the declared harness cell `kaggleharness-refused-zip` | 18/18 ok, counts 1..18 | 1111.3 s |
| v6: BYOD adapt | [`receipt_capstone_7bd7f87_byod-adapt.ipynb`](execution-evidence/2026-09-29/receipt_capstone_7bd7f87_byod-adapt.ipynb) (`9a07c16e3edf62813abfdacd858f391321ec84d070316017202d652456097c64`) | `code-02` toggles only | 17/17 ok, counts 1..17 | 1125.8 s |

- **Canonical path (both runs).** Results are identical to the `78ef7c3` default run: adapter SHA-256 `da69be11…`; test total correct 35 / 35 / 28 of 94 (adapted / frozen / rules); 998 receipts OCR'd; 636 aligned training examples; epoch 3 selected. Only the selection-record digest differs, because it includes the freeze timestamp. This is therefore a default-journey execution of `7bd7f87` as well. The source diff from `78ef7c3` is limited to the OCR cache change.
- **BYOD inference (v5): PASS.** Cohort `{'inference': 12}`. The stages were prepare, ocr_runtime, model, ocr and byod_infer, and 12 records were written to a separate `…-byod-…/outputs` directory. Every `total_review_state` is `needs_review` (the inference-mode default), and the notebook states that no accuracy is reported without checked references. Five totals normalised; the rest are `undefined` because the conservative grammar refused OCR text such as `28, 006`.
- **Refused ZIP (v5 harness cell): PASS.** `code-40`'s committed source (SHA-256-guarded) raised `BYOD archive refused: receipt_common.ContractError: Unsafe archive path: '../escape.png'`, and no BYOD run directory was created.
- **BYOD adapt (v6): PASS.** Cohort train 68 / validation_model 20 / validation_policy 20 / test 25, and all 16 stages ran. 55 aligned examples came from 31 train receipts. Validation_model total EM by epoch was 0.30 / 0.30 / 0.40 / 0.35. Selection record `5b94226e6f0505e3`; the held-out test was evaluated once. Test total correct: rules 5, frozen 8, adapted 7 of 25; the intervals are wide. The policy was infeasible for every system (R1). The replay of 3 receipts reached parity PASS.
- **Evidence boundary.** The stand-ins are pixel-identical to CORD receipts, so both BYOD OCR stages were served from the canonical OCR cache (12 receipts in 0 s; 133 in 1 s). That exercises the R6 path, but **fresh Tesseract OCR of previously unseen BYOD images is not evidenced**, and no real learner receipts were used. Saved outputs were inspected; execution was not independently repeated.

| Journey at `7bd7f87` | Verdict |
| --- | --- |
| Default Run all | **PASS** (canonical cells of v5 and v6, identical to `78ef7c3` v2) |
| Change-one-thing activity | **PASS (mechanics)**; no contrast (R1, structural, open) |
| BYOD inference | **PASS** (stand-ins; OCR from cache) |
| BYOD adapt | **PASS** (stand-ins; OCR from cache) |
| Refused ZIP | **PASS** |
| Colab runtime (default journey) | **PASS**: maintainer-supplied run, see below |
| Colab `DOWNLOAD_RESULTS` | not assessed (off in the Colab run) |

**Still open before promotion.** R1 disposition (a review-policy design decision); a BYOD run with images not seen before; the annotation audit; and a human promotion decision. Status stays **Candidate**.

### Maintainer-supplied Colab execution of revision `7bd7f87` — 2026-09-29

- **File.** [`execution-evidence/2026-09-29/receipt_capstone_7bd7f87_colab-t4.ipynb`](execution-evidence/2026-09-29/receipt_capstone_7bd7f87_colab-t4.ipynb). SHA-256 `1dc9cf1bd400cfd93797d49e78203008c0c5a57b7f4bf710dd9a8836e5932fef`, a byte-for-byte copy of the downloaded notebook.
- **Source match.** The notebook is identical to the PR-head notebook (blob `329fbfe8…`): 42/42 cells, same ids and order, and no source diff. Every control is at its default (BYOD off). Execution counts run 1..17 with no errors, so this was one Run all.
- **Runtime.** Google Colab, Tesla T4, **2 CPUs**. The notebook's isolated environment ran CPython 3.12.12, torch 2.9.1+cu128 (CUDA 12.8) and transformers 4.57.6; the environment was ready in 97 s. Tesseract 5.5.0 (leptonica 1.83.1) with `eng+ind` was verified.
- **Results equal the Kaggle runs of `7bd7f87` and `78ef7c3`.** Adapter SHA-256 `da69be11e69fc023bf956e6ffdc24b5ae1ab17810487b9194da0c40c99495549` (byte-identical across Colab and Kaggle); test total correct 35 / 35 / 28 of 94 (adapted / frozen / rules); 636 aligned examples from 378 receipts; validation_model EM 0.347 / 0.367 / 0.408 / 0.408, with epoch 3 selected. OCR returned `ocr_empty` for 2 / 19 / 0 / 1 receipts (test / train / validation_model / validation_policy); replay of 3 receipts passed parity. Results ZIP: 32 members, CRC and member digests pass. Selection record `59a789600cf117ef`; only its freeze timestamp differs from the Kaggle runs.
- **Resources.** OCR 1,096 s on 2 CPUs (536 s on Kaggle's 4); stage total ≈1,540 s plus the 97 s install. Peak GPU allocation 2.15 GiB (train); peak RSS 3.28 GiB (prepare).
- **Journeys.** Default Run all: **PASS**. Change-one-thing activity: PASS (mechanics), with no contrast (R1). BYOD and `DOWNLOAD_RESULTS`: not assessed in this run.
- **Evidence boundary.** Saved outputs were inspected; execution was not independently repeated. This run is a second runtime family for the default journey, not new BYOD evidence.

## Review fixes — revision `0.2.0-candidate` (2026-10-02)

A Notebook Review Framework v1 review of `ad47ef8` (notebook blob `329fbfe8`) concluded **Needs revision**. The report, its probes and the fix report are archived in [`reviews/2026-10-02-notebook-review/`](reviews/2026-10-02-notebook-review/). Each fix has a regression test in `tests/test_receipt_review_fixes.py`.

| Finding | Severity | Fix |
| --- | --- | --- |
| RC-B1 the Colab badge pointed at the deleted feature branch and opened an error | Blocker | `BRANCH_FOR_BADGE = "main"` |
| RC-M1 the review-policy activity could never differ from the refer-all canonical policy (the earlier **R1**) | Major | Maintainer decision (2026-10-02): keep the canonical 95% / 50% / 25 targets; the §13 activity now varies the share of `validation_policy` totals left unflagged (`ACTIVITY_COVERAGE`, default 0.30) and reports accuracy and wrong unflagged totals per system. §8, §13 and §15 text rewritten; the infeasible "90% may make a policy feasible" answer removed. Deviation 17 in [`receipt-capstone.md`](receipt-capstone.md) |
| RC-m1 the failure panel filed OCR misreads (`BOO`) as numeric ambiguity | Minor | `panel_category`: `parse_failed` is an OCR loss when not recoverable and an extraction error when recoverable; §10 defines every category |
| RC-m2 no rerun instructions for the §1 controls | Minor | §1, §13 and §14 tell the learner to run the controls cell again; the "BYOD is off" message says how |
| RC-m3 stale learner text after the hosted runs | Minor | Measured 2026-09-29 runtimes (labelled, with environment) in §1; status line and troubleshooting updated |
| RC-m4 booleans displayed as `1.0000` / `0.0000` | Minor | `_fmt` renders `true` / `false` |
| RC-m5 this ladder and note 16 described the pre-merge state | Minor | Ladder rows and note 16 updated |

Suggestions RC-S1–S3 are not addressed. R1 is resolved by RC-M1 as a design change. The other earlier dispositions are unchanged.

**User-visible changes.** The control `ACTIVITY_TARGET` is replaced by `ACTIVITY_COVERAGE`; the runner option `--activity-target` by `--activity-coverage`; activity outputs move from `outputs/activity/target_<x>/` to `outputs/activity/coverage_<x>/`, with new columns (`unflagged_wrong`, `accuracy_interval`, `reached_requested`). The failure panel may assign a different receipt to `ocr_loss`, `extraction_error` and `numeric_ambiguity`. Training, the review targets, policy selection and the test evaluation are unchanged in logic, but the revision string is part of the selection record and bundle manifest, so their digests change.

**Evidence boundary.** These fixes are source-checked and CPU-tested (test doubles) only. No hosted run of `0.2.0-candidate` exists; the hosted evidence above is for `0.1.0-candidate`. Status stays **Candidate**.
