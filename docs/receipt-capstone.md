# Receipt Intelligence Capstone — implementation notes

**Notebook:** `tutorials/DIMER_Small_Business_Receipt_Intelligence_Capstone.ipynb` (revision `0.1.0-candidate`)
**Specification:** [`receipt-capstone-spec.md`](receipt-capstone-spec.md) (v1.0, unchanged) · **Profile / mode:** `E2E` / `GUIDED`, Notebook Specification 2.2
**Status:** built, source-checked and CPU-tested with labelled test doubles. The default journey was hosted-executed on a clean-runtime Kaggle T4 at `fc553d4` on 2026-09-29. BYOD journeys are not yet run, and the notebook is **not** annotation-reviewed or release-qualified. See [`receipt-release-evidence.md`](receipt-release-evidence.md).

## Source layout

| File | Role |
| --- | --- |
| `tools/build_receipt_capstone.py` | Deterministic generator; `--check` enforces byte parity (CI and `validate_release_assets.py`). Never hand-edit the notebook. |
| `tools/receipt_capstone.py` | Stage runner, carried as `capstone.py`. Each stage runs in its own process and commits atomically through `StageStore`. |
| `tools/receipt_common.py` | Canonical JSON/digests, pinned downloads, spreadsheet-safe CSV, stage receipts with downstream invalidation. |
| `tools/receipt_data.py` | Frozen-manifest CORD loading, role/exclusion re-check, EXIF transforms, evaluator-owned references, purpose-logged reference reader. |
| `tools/receipt_fields.py` | Amount grammars, geometry helpers, training-derived keyword dictionary and System A. **New in this build** (see below). |
| `tools/receipt_ocr.py` | Pinned Tesseract install (micromamba + 28 SHA-256-pinned conda-forge packages, offline explicit create), engine verification, TSV contract, cache keys, labelled test double. |
| `tools/receipt_models.py` | Capstone-owned LayoutLM extractor (Systems B and C) reproducing the host's windowing/p-mask/score semantics without importing the package. |
| `tools/receipt_training.py` | Train-only OCR↔reference alignment and the bounded recipe with epoch 1–4 selection. |
| `tools/receipt_metrics.py`, `tools/receipt_policy.py` | Evaluator-only metrics and bootstrap; review routing, validation-only threshold selection, the activity. |
| `tools/receipt_artifact.py`, `tools/receipt_byod.py` | Serving bundle export/verification, safe ZIP extraction, BYOD intake. |
| `tutorials/receipt_intelligence/*.json` | Frozen data, split, audit, model and OCR manifests from `freeze_receipt_assets.py` (unchanged). |
| `tests/test_receipt_*.py` | CPU contracts, negative controls, full stage chain, gold non-interference, notebook structure. |

Regenerate after any change to `tools/receipt_*.py`, the manifests, the lock or the generator:

```powershell
python tools/build_receipt_capstone.py
python tools/build_receipt_capstone.py --check
```

## Stage graph

`ocr_runtime`, `model`, `prepare` → `references` → `ocr` → `keywords` → `rules`, `frozen` → `train` → `select_policy` → `freeze` → `evaluate` → `diagnose`, `export` → `replay` → `report`. A stage refuses to start unless every upstream receipt, output digest, carried-source digest and run configuration verifies; starting a stage deletes its own and all downstream outputs first, so a failed retry cannot leave an old success receipt behind. The activity (`--activity-target`) and BYOD run outside the canonical graph and never write into it.

## Resolved specification details and deviations

These are build-time resolutions of points the specification left open, or places where the implementation differs from a literal reading. Check this list before flagging drift.

1. **`receipt_fields.py` is a new module.** The draft named in the handoff (13,038 bytes, SHA-256 `bcf62b4f…`) was never published and was not available to this build. The module here was written from specification §§8–9. It is not a reconstruction of that draft; if the draft is recovered, compare it before merging.
2. **Single source of truth for data.** The committed frozen manifests govern shard pins, receipt IDs (`cord-v2:<split>:NNNN`), per-row image/pixel/annotation hashes, roles and exclusions. The draft loader's own pins and `cord-<split>-NNNN` IDs were removed. Runtime recomputes roles (`sha256("receipt-capstone-v1|42|" + id)`) and duplicate exclusions and refuses any disagreement with `split_manifest.json`. The pixel hash is the freezer's definition: SHA-256 of the compact JSON size followed by the RGB bytes.
3. **Canonical amounts.** Decimal strings only. An all-zero fraction is dropped (`75,000.00` → `75000`), and a nonzero fraction keeps two digits (`12,50` → `12.50`). Only a leading `Rp`, `Rp.`, `IDR`, `PHP` or `₱` marker is recognised; it is recorded as the currency and never inferred from the country.
4. **Reference boxes.** Annotation quadrilaterals become axis-aligned boxes clipped to the page. When several lines of one category all normalise to the same amount, the reference is `present_usable` and every value box is kept. Differing values make it `ambiguous_reference`. Values are never summed.
5. **Keyword dictionary (System A).** Derived programmatically from training `is_key` phrases. A phrase is kept when it has support ≥ 3 and purity ≥ 0.8 for one target category. Matching is NFKC-casefolded and punctuation-stripped, on whole OCR words, longest phrase first.
6. **Rule ranking.** Candidates are ranked by key-phrase priority first. Within that, the same line comes before the next line, and the next line is used only if its vertical gap is ≤ 1.5 × the key height. On a line, the rightmost parseable amount wins, because receipts right-align prices. The specification's "distance" criterion is therefore read as right alignment. Distinct values at the best rank are `ambiguous`.
7. **Training alignment.** Candidates are contiguous spans on one OCR line, at most four words long. Nested matches collapse to the shortest, so `Rp 75.000` becomes `75.000`. Two or more non-nested survivors are skipped as `ambiguous_alignment`.
8. **OCR TSV.** Output is requested with `-c tessedit_create_tsv=1 -c tessedit_create_txt=0`, not with the `tsv` config name. With a `--tessdata-dir` holding only the pinned language files, the config file is absent, and Tesseract silently falls back to plain text. This was observed on a local Tesseract 5.5.0 build and is guarded by `tests/test_receipt_ocr.py` when a local engine is configured.
9. **OCR install.** The frozen closure is installed with `micromamba create --offline` from `file://` explicit-spec lines carrying each package's md5. Each package is SHA-256-verified before that step. It ran successfully on a clean-runtime Kaggle T4 on 2026-09-29 (`fc553d4`; `qualifying_runtime: true`).
10. **Predictions before freeze.** The train stage also writes the adapted model's predictions for `validation_policy` and `test`. Test references are read only after `freeze`, which the access-log test enforces.
11. **§14.3 demonstration.** The three-receipt held-out input-only demonstration is the fresh-process replay itself (§12 of the notebook). The export stage only writes the bundle and the expectations.
12. **Arithmetic.** No reconciliation contract exists in v1, so every record carries `arithmetic_check.state = not_evaluable`.
13. **BYOD.** In inference mode the canonical bundle is used and every total defaults to `needs_review`; the transferred CORD decision is shown only for comparison. Adaptation mode reuses the canonical training-derived keyword dictionary, because BYOD annotations carry no key phrases. The reference-text diagnostic is unavailable for BYOD (no reference words are supplied).
14. **Images on disk.** Verified source images are written once to a content-addressed store (about 2.3 GB) and re-hashed on every read. The notebook requires 15 GiB free disk.
15. **Page segmentation is PSM 6, not the specification's initial PSM 4.** The first hosted run (`fc553d4`, PSM 4) returned no words for 205 of 998 receipts, and every system became refer-all. A diagnosis with a local Tesseract 5.5.0 and the pinned `tessdata_fast` files traced this to layout analysis, not recognition. A tight crop of a crisp `46,000` returns nothing at PSM 6/7 and reads correctly at PSM 8/13. EXIF (all orientation 1), privacy blurring (recall of the unblurred annotated words is only about 0.40) and the spatial alignment check were ruled out as the main cause. Upscaling, grayscale/autocontrast and Sauvola thresholding did not help. The change was **selected on the official validation receipts only**: on the 98 validation receipts with a usable total, PSM 4 → 6 moves empty OCR from 21 to 1 and aligned total references from 19 to 40. The test receipts had been inspected during diagnosis; that observation is evaluator-side and was not used to choose. `tessdata_best` was not adopted. Its gain was small and would have needed new language pins. One configuration still serves all three systems.
16. **Colab badge.** The badge points at `feat/small-business-receipt-capstone`, because the notebook does not exist on `main` yet. Switch `BRANCH_FOR_BADGE` to `main` and regenerate in the merge commit.

## Known scientific limits

CORD v2 is an Indonesian receipt sample, so no Philippine, unseen-merchant, production or accounting-grade claim follows. The split preserves the official partitions, but it is not merchant- or template-disjoint, and near-duplicates were not audited. Overlap with the checkpoint's pretraining is unknown. The published annotations are references, not verified financial truth, and no independent human review has been performed (see [`receipt-annotation-audit.md`](receipt-annotation-audit.md)). Scores are uncalibrated. The validation roles hold about fifty receipts each, so a single decision moves a percentage by roughly two points.
