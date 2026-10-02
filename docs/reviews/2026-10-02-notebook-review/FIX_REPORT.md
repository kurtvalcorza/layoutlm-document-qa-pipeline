# Fix report: receipt capstone review, 2026-10-02

Reviewed revision: `ad47ef8` (notebook blob `329fbfe8`, `0.1.0-candidate`). Fixed revision: `0.2.0-candidate` on branch `fix/receipt-capstone-review`. All changes are made in the generator (`tools/build_receipt_capstone.py`) and the carried modules; the notebook is regenerated and `--check` parity passes. The review report and its probe ZIP are in this folder.

**Readiness: Verification pending.** The fixes are source-checked and CPU-tested. A maintainer-supplied Colab T4 run of `4b950a1` (2026-10-02) passed the default journey and the §13 activity, with default results identical to `0.1.0-candidate`; see `docs/receipt-release-evidence.md`. A second commit (`4b950a1`) split the 1.18 MB carrier line that froze Colab while opening the notebook. BYOD has not yet been run at this revision.

## Fixes per finding

| Finding | Change | Cells / files touched | Acceptance check (test) |
| --- | --- | --- | --- |
| **RC-B1** (Blocker): Colab badge → deleted branch | `BRANCH_FOR_BADGE = "main"` | `md-00`; generator | `test_rc_b1_colab_badge_points_at_the_default_branch` |
| **RC-M1** (Major): the review-policy activity could never show a change | Maintainer decision 2026-10-02, option (a). The canonical 95% / 50% / 25 targets stay. The §13 activity now leaves each system's top-scored share of `validation_policy` totals unflagged (`ACTIVITY_COVERAGE`, default 0.30; `receipt_policy.coverage_operating_point`; `--activity-coverage`) and reports accepted totals, accuracy with a Wilson interval, wrong totals left unflagged and `reached_requested`, beside each system's canonical row. Question (3), outcome 6, the §8 Predict and worked answer, §13 and the §15 scaffold were rewritten. The "90% may make a policy feasible" answer was removed; §8's worked answer now states the bound (≈48% of totals must be correct). | `md-00`, `code-02`, `md-08`, `md-23`, `md-25`, `md-36`, `code-37`, `md-38`, `md-41`; `receipt_policy.py`, `receipt_capstone.py` (`run_activity`, `_policy_figure`, CLI) | `test_rc_m1_*`; `test_coverage_activity_*` (4) in `test_receipt_policy_metrics.py`; the stage-chain activity test |
| **RC-m1**: the panel filed OCR misreads as numeric ambiguity | `panel_category()`: `parse_failed` is `ocr_loss` when not recoverable and `extraction_error` when recoverable; `numeric_ambiguity` only for `ambiguous`/`unsupported`. §10 defines each category and how panels trace words to boxes | `md-29`; `receipt_capstone.py` (`stage_diagnose`) | `test_rc_m1_failure_panel_categories` (9 cases), `test_rc_m1_panel_categories_are_defined_for_the_learner` |
| **RC-m2**: no rerun instructions for controls | §1 item 4, §13 and §14 say to run the controls cell again; the "BYOD is off" message says how; the controls cell prints the activity coverage and BYOD mode | `md-01`, `code-02`, `md-36`, `md-39`, `code-40` | `test_rc_m2_controls_have_rerun_instructions` |
| **RC-m3**: stale learner text | Measured 2026-09-29 runtimes with their environments (UX12); the status line now reads "hosted runs recorded; not release-qualified"; the OCR-install troubleshooting row was updated | `md-00`, `md-01`, `md-41` | `test_rc_m3_no_stale_learner_text` |
| **RC-m4**: booleans shown as `1.0000` | `_fmt` renders `true`/`false` | `code-06` | `test_rc_m4_booleans_render_as_words` |
| **RC-m5**: evidence records were pre-merge | Ladder rows (merged as `ad47ef8`, a new reviewed state), note 16, a new notes 17–18, a *Review fixes* section, and the README status cell | `docs/receipt-release-evidence.md`, `docs/receipt-capstone.md`, `tutorials/README.md` | `test_rc_m5_evidence_records_are_current` |

Not addressed (suggestions): RC-S1 (a §10 tracing prompt; partly covered by the new panel note), RC-S2 (empty-stderr guard in `code-40`), RC-S3 (disclose the PSM selection history in §4).

## User-visible changes

- `ACTIVITY_TARGET` → `ACTIVITY_COVERAGE`. The runner option `--activity-target` → `--activity-coverage`. Activity outputs move from `outputs/activity/target_<x>/` to `outputs/activity/coverage_<x>/`.
- `activity_comparison.csv` columns are now `system, setting, threshold, accepted, coverage, empirical_accuracy, unflagged_wrong, accuracy_interval, reached_requested, selection_support`; settings are `canonical_95pct_policy` and `activity_coverage_<x>`.
- The failure panel may pick different receipts for `ocr_loss`, `extraction_error` and `numeric_ambiguity`.
- `NOTEBOOK_REVISION` is `0.2.0-candidate`. It is recorded in the selection record, bundle manifest and summary, so their digests differ from `0.1.0-candidate` runs. The adapter, predictions and test metrics should not change: training, selection and evaluation logic are untouched. This is inferred, and the next hosted run checks it.

## Verification (labelled by evidence basis)

Environment: Windows, CPU only, Python 3.12 venv with CI's pinned packages (torch 2.14.0+cpu, transformers 4.57.6, pyarrow 25.0.1, matplotlib 3.10.6, pytest 8.4.2, ruff 0.16.6).

| Check | Baseline `ad47ef8` | After fixes |
| --- | --- | --- |
| `ruff check src tests tools` | pass | pass |
| `pytest` | 171 passed, 2 skipped | 191 passed, 2 skipped (+4 policy tests replacing 1, +17 review-fix tests, the chain activity test rewritten) |
| `tools/build_receipt_capstone.py --check` | PASS | PASS |
| `tools/build_notebook.py --check` | PASS | PASS (unchanged tutorial) |
| `tools/validate_release_assets.py` | PASS | PASS |

- **Stand-in models (direct execution, not performance evidence):** the stage chain on 28 synthetic receipts with a tiny random LayoutLM ran `--activity-coverage 0.3`. It wrote six rows to `activity/coverage_0.30/` and `activity.png`, and left `review_policy.json` byte-identical. The rules baseline accepted 4 of 4 at 30%. The random LayoutLMs had no eligible totals (`refer_all`, `reached_requested` false). The panel labelled a recoverable `parse_failed` span as `extraction_error` and an `ambiguous` span as `numeric_ambiguity`.
- **Real CORD data:** not executed. The 0.30 default was read off the `validation_policy` sweep figures saved in the 2026-09-29 Colab and Kaggle runs. At that coverage each system had eligible totals: rules ≈0.93 accuracy, frozen ≈0.80, adapted ≈0.86, read by eye.
- **First-time-learner re-read** (source inspection) of every touched cell: the new control is explained in §1 and §13 before it is used; prose matches the new columns; the §8 worked answer was corrected during the re-read to state the real bound; the essential path is unchanged and §13/§14 remain optional.

## Remaining gates

1. A hosted Run all of `0.2.0-candidate` on a fresh Colab T4 covering every journey: the default path, the §13 activity at the default and one other coverage (following the rerun instruction), BYOD inference and adapt, and a refused ZIP. Check that the adapter digest and test metrics equal the `0.1.0-candidate` runs.
2. A BYOD run on images OCR has not seen (not served from the cache).
3. The annotation audit.
4. A human promotion decision. Status stays **Candidate**.
