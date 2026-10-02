# Notebook review — From Receipts to Records (receipt intelligence capstone)

Review date: 2026-10-02 · Framework: Notebook Review Framework v1 (sha256 `a7cc6611…c9b375`) · Requirements baseline: DIMER Notebook Specification 2.2 (ml-worker `main` `b1cfe13`) and `docs/receipt-capstone-spec.md` v1.0 · Finding prefix: `RC`

**Readiness decision: Needs revision.** One Blocker (the only Colab link opens an error dialog) and one Major (the review-policy half of the central question is never demonstrated on the default path) are open. A `MUST` is unmet (SRC3, stale learner text), and three verification gates remain.

## 1. Scope and evidence

| Contract item | Value |
| --- | --- |
| Repository / revision | `kurtvalcorza/layoutlm-document-qa-pipeline` @ `ad47ef84d417cbb070c98b95bd6eb82bc4d7b7fd` (`main`, confirmed via GitHub API `commits/main`) |
| Notebook | `tutorials/DIMER_Small_Business_Receipt_Intelligence_Capstone.ipynb`, git blob `329fbfe87258…` (last changed at `7bd7f87`), file SHA-256 `dd7c8f57…8b42`, 42 cells (17 code) |
| Specification / profile / mode | Notebook Spec 2.2 · `E2E` · `GUIDED` · revision `0.1.0-candidate` · release status Candidate |
| Audience and prerequisites | Self-paced learners with basic Python and Colab; no document-AI experience assumed |
| Supported runtime | Fresh Google Colab Linux x86-64 with a T4 GPU and 15 GiB free disk (enforced in `code-04`) |
| Promised outcomes | The two-part central question (extraction reliability; effect of review referral on unflagged error), three sub-questions, eight learning outcomes, export and fresh reload, and optional BYOD in `inference` and `adapt` modes |

**Execution evidence covering this exact blob** (`329fbfe8`, recorded in `docs/receipt-release-evidence.md`):

- Kaggle T4 v5: canonical path + BYOD inference + refused ZIP.
- Kaggle T4 v6: canonical path + BYOD adapt.
- Maintainer-supplied Colab T4: default Run all, 17/17 cells.

All three passed. BYOD used CORD stand-ins whose OCR came from the cache. Older runs (`fc553d4`, `78ef7c3`) do not cover this blob.

**What I did myself:**

- **Source inspection** of every learner-facing cell and the carried implementation (`capstone.py`, `receipt_policy.py`, `receipt_fields.py`, `receipt_byod.py`, `receipt_artifact.py`, `receipt_ocr.py`).
- **Direct CPU execution** on Windows, Python 3.12.10, stdlib only:
  - `tools/build_receipt_capstone.py --check` printed `Receipt capstone parity: PASS` (exit 0).
  - Probes P1–P7 (`run_probes.py`): all ran, and each reproduces its finding.
- **Direct browser observation** in signed-out Colab of the badge URL, the same notebook on `main`, and a control notebook.
- No GPU execution (workspace rule) and no learner observation.

| Journey | Evidence basis | Result |
| --- | --- | --- |
| First-time learner | Source inspection | Strong orientation, glossary, Predict prompts and worked answers. Entry fails at the badge (RC-B1). The review-policy sections promise a contrast the run cannot show (RC-M1). The diagnosis panel teaches one wrong category (RC-m1). |
| Clean default | Documented execution (Colab T4 + 2 Kaggle T4 runs, same blob); direct observation of the entry link | Run all **passes** from a fresh runtime, and results are byte-identical across runtimes (adapter `da69be11…`). Opening via the notebook's own badge **fails** (RC-B1). |
| Active learning | Documented execution | Mechanics pass. The output shows **no contrast**: every row is refer-all at 0.95 and 0.90 (RC-M1). The learner-facing rerun procedure was never exercised, because the harness edited cell source (RC-m2). |
| Reuse and recovery | Documented execution (Kaggle stand-ins); direct CPU probe of the refusal path | BYOD inference and adapt passed, and the traversal ZIP is refused with an actionable message. **Not verified:** fresh OCR of unseen BYOD images, real learner receipts, BYOD through the Colab form path, `DOWNLOAD_RESULTS`, and a repeated Run all in a warm runtime. |

**Limitations:**

- The Colab observations were made signed out, so cells could not be executed there.
- The validation_policy correct-total counts are not printed by the notebook. The policy bound in RC-M1 therefore combines the source logic (P4) with documented validation_model counts.

## 2. Separate judgments

**Technical correctness: strong.**

- **Carrier integrity.** All 19 carried files match their embedded hashes and metadata, and the 11 Python files equal `tools/` byte for byte after CRLF normalisation (P1).
- **Pipeline discipline.** Stages verify upstream receipts and invalidate downstream outputs. The selection record is sealed before test scoring.
- **Fresh reload.** The replay really runs fresh OCR: `stage_replay` calls `engine(image)` directly, not the cache. It applies only allowed tensors to a re-verified base, and checks parity on tokens, spans, parse states and routing.
- **BYOD intake.** Unsafe archives are refused before any run directory is created (P7, Kaggle v5).
- **Defects found:** the panel categorisation (RC-m1) and boolean display (RC-m4).

**Scientific and experimental validity: sound and candid.**

- Official partitions are preserved, with a hash-split validation into model and policy roles. Duplicates are excluded before evaluation, and only usable references enter the denominators. Test is scored once after freezing.
- Intervals are paired bootstrap and Wilson. The primary EM is never filtered.
- The headline result is reported honestly: adapted − frozen = 0.000 [−0.053, 0.053] on 94 test totals. OCR is identified as the bottleneck: 58/94 test totals are not recoverable from OCR, while the reference-text diagnostic reaches 0.88–0.95.
- The one validity caveat is disclosed in the docs but not in the notebook. The PSM 4 → 6 change was selected on validation receipts, after test receipts had been seen during diagnosis (RC-S3).

**Promise fulfilment: partial.**

- Delivered as stated:
  - Question 1 (workflow accuracy) and question 2 (fine-tuning against the baselines).
  - Outcomes 1–4, 7 and 8.
- Not delivered on the default path:
  - Question 3 (the coverage–error trade-off).
  - The second half of the central question.
  - Outcome 5 for the coverage and selective-error metrics.
  - Outcome 6.

**Learner experience: good structure, two real barriers.**

- Every GDL item is present:
  - roadmap, glossary and the Input → System → Output contract;
  - infrastructure cells labelled and collapsed;
  - Predict prompts with folded answers;
  - a troubleshooting table and a conclusion scaffold.
- The barriers are the broken entry link (RC-B1) and a "change one thing" activity whose only possible output on CORD is "nothing changed", accompanied by a worked answer suggesting otherwise (RC-M1).

**Specification conformance:**

- **SRC3 (MUST) unmet.** Knowingly stale instructions survive (RC-m3).
- **SRC10 (SHOULD) unmet.** The badge points at a deleted branch (RC-B1).
- **§1 open-and-Run-all contract.** Unmet through the notebook's own link; met when the notebook is opened from `main`.
- **REL12 partly evidenced.** Representative input was accepted and an incompatible input refused. Fresh OCR of user images was not exercised.
- **Met:**
  - DAT10–DAT19 by source inspection;
  - REL1, REL2, REL5–REL7 and REL10 through the recorded runs;
  - UNC1/UNC2: scores are labelled as uncalibrated ranking signals;
  - GDL1–GDL15: GDL10 is present but its observable result is degenerate (RC-M1).

## 3. Findings

### Blocker

**RC-B1 — The notebook's only Colab link opens an error.**

- **Cell/section:** `md-00`, the Open in Colab badge (generator: `tools/build_receipt_capstone.py:23`, `BRANCH_FOR_BADGE = "feat/small-business-receipt-capstone"`).
- **Observed issue:** the badge targets `feat/small-business-receipt-capstone`, which was deleted when PR #9 merged. Implementation note 16 in `docs/receipt-capstone.md` ("Switch `BRANCH_FOR_BADGE` to `main` … in the merge commit") was not carried out. The repository has no other Colab link to this notebook: `tutorials/README.md` lists it without one.
- **Consequence:** a learner who follows the documented entry gets "There was an error loading this notebook" and cannot start. The spec §1 requires that a user can open the notebook in a supported fresh runtime.
- **Evidence:**
  - *Direct observation, 2026-10-02:* signed-out Colab at the badge URL showed that dialog with 0 cells. The same path on `/blob/main/` loaded all 42 cells. Control: the repo's workshop notebook loaded 68 cells.
  - *Direct check (P2):* GitHub API `branches/feat%2Fsmall-business-receipt-capstone` returns 404.
- **Recommended correction:** set `BRANCH_FOR_BADGE = "main"`, regenerate, and add a test that the badge branch is the default branch.
- **Acceptance check:**
  1. The badge URL in `md-00` contains `/blob/main/`.
  2. Opening it in a signed-out Colab shows 42 cells with no error dialog.
  3. A test fails if `BRANCH_FOR_BADGE` is not `main`.
- **Spec:** §1 (MUST), SRC10.

### Major

**RC-M1 — The review-policy half of the central question cannot be demonstrated on CORD, and the activity shows no contrast while its worked answer implies one.**

- **Cell/section:** `md-00` (central question, question 3, outcomes 5–6), §8 `md-23`/`code-24`, §9 review table `code-27`, §13 `md-36`/`code-37`/`md-38`.
- **Observed issue:**
  - `DEFAULT_TARGETS` (`receipt_policy.py:16`) requires accuracy ≥ target **and** coverage ≥ 0.50 **and** at least 25 accepted.
  - With 49 usable validation totals, feasibility at target *t* needs at least ⌈25·t⌉ correct totals even with perfect ranking. That means 24 at 0.95 and 23 at 0.90, an EM of ≥ 0.47.
  - The systems' validation_model EMs are 15, 19 and 20 of 49 (0.31–0.41).
  - On every recorded run, all three systems are `refer_all` at 0.95 (`policy_selection.csv`) and again at the activity's 0.90 (`activity_comparison.csv`: six identical rows).
- **Consequence:**
  - §8's Predict ("which system will reach the 95% target with the most coverage") has no answer.
  - The §9 review table is all zeros and `undefined`.
  - The Predict → Change → Run → Observe → Explain activity always observes "nothing changed".
  - The §13 worked answer says "a 90% target may make a policy feasible", which the bound rules out for these systems.
  - Outcome 6 and the coverage/selective-error half of outcome 5 are not exercised. The notebook does say refer-all is a valid result, so learners are not told something false about the data. The demonstrated trade-off itself is missing.
- **Evidence:**
  - *Documented execution:* Colab T4 and Kaggle v5/v6 at blob `329fbfe8`.
  - *Direct CPU execution (P4):* the real `receipt_policy.select` on 49 synthetic totals with perfect ranking is infeasible at 0.95 and 0.90 with 20 correct, and feasible at 0.80.
  - *Prior record:* this was disposition R1 in `receipt-release-evidence.md`, left open as "a design decision for the maintainer".
- **Recommended correction (maintainer decision; pick one):**
  - (a) Make the activity vary `min_coverage` (or both targets), with a default chosen on validation_policy that produces a visible change.
  - (b) Lower the canonical targets to values that remain teaching settings but are attainable on CORD, and document the choice.
  - (c) Keep the targets but rewrite the central question, question 3, outcomes 5–6, the §8 Predict and the §13 worked answer, so that the lesson is "no threshold meets these targets; here is the sweep and why". Then show the sweep table, not only a figure.

  In every option, delete or correct the "may make a policy feasible" answer.
- **Acceptance check:**
  - (1) On a hosted default run at the fixed revision, `activity_comparison.csv` has at least one system whose `accepted` or `coverage` differs between the canonical and activity rows; **or** the learner-facing text no longer promises a coverage change or a feasible policy (verified by grep for "may make a policy feasible" and by reading `md-23`, `md-36`, `md-38`).
  - (2) A unit test pins the chosen targets against the documented validation counts, so a future regeneration cannot silently return to an infeasible-by-construction activity.
- **Spec:** UX1 (objectives correspond to executed code: they execute, but produce no observable outcome), GDL10, UX5.

### Minor

**RC-m1 — The failure panel files OCR misreads as "numeric ambiguity".**

- **Cell/section:** §10 `code-30` / `md-29` (`capstone.py`, the `stage_diagnose` panel labels).
- **Observed issue:** `parse_failed` is routed to `numeric_ambiguity` *before* recoverability is checked. A total that OCR turned into letters is therefore never labelled `ocr_loss`. On the Colab run, the `numeric_ambiguity` panel is `cord-v2:test:0059`: prediction `BOO` (probably "800"), `parse_failed`, `not_recoverable`. That is an OCR loss.
- **Consequence:** the section whose purpose is to separate OCR, extraction and parsing (the notebook's stated "main lesson") shows a mislabelled example in its table and its figure title.
- **Evidence:** source inspection plus documented execution (P3 reproduces both).
- **Recommended correction:** keep `numeric_ambiguity` for `ambiguous`/`unsupported`. Send `parse_failed` to `ocr_loss` when not recoverable and to `extraction_error` when recoverable. Alternatively, add a `parse_failure` category and define each category in `md-29`.
- **Acceptance check:** a unit test where a `parse_failed`, `not_recoverable` total is labelled `ocr_loss`, and `md-29` defines every panel category.

**RC-m2 — No rerun instructions for the §1 controls.**

- **Cell/section:** §13 `md-36`, §14 `md-39`; controls in `code-02`.
- **Observed issue:** §14 says "set `USE_BYOD`, … in §1, then run this section", and §13 never says how to change `ACTIVITY_TARGET`. The `# @param` fields have no `run: "auto"`. In Colab, editing a form field rewrites the code but does not re-execute the cell, so the variables keep their old values.
- **Consequence (inferred):**
  - A learner who follows §14 literally sees "Optional BYOD is off. The canonical capstone is complete."
  - A learner who follows §13 reruns the activity at the old target.
- **Evidence:** source inspection. Not verified in Colab: the Kaggle BYOD runs patched the `code-02` source directly, so this path was never exercised.
- **Recommended correction:** state "change the value, run the §1 cell, then run this cell", or give §13 and §14 their own form fields.
- **Acceptance check:** the instruction is present in `md-36` and `md-39`, and a recorded Colab check changes `ACTIVITY_TARGET` through the form and gets a new `activity/target_<x>` folder.

**RC-m3 — Stale learner text after the hosted runs.**

- **Cell/section:** `md-00`, `md-01`, `md-41`.
- **Observed issue:**
  - `md-01` says "No elapsed time is promised before a qualified hosted run has measured it". The Colab T4 run measured about 1,540 s of stages plus a 97 s install, with OCR taking 1,096 s on 2 CPUs.
  - The troubleshooting table calls the OCR install "an open qualification item", but it has now run on Kaggle and Colab.
  - The header says "hosted qualification pending".
- **Consequence:** learners start a run of roughly 30 minutes with no time expectation.
- **Evidence:** source inspection (P6) plus documented execution.
- **Recommended correction:** state the measured time with its environment, labelled as a measurement (UX12). Drop the open-item note and update the status line.
- **Acceptance check:** those phrases are absent, and `md-01` gives a measured duration naming Colab T4 with 2 CPUs.
- **Spec:** SRC3 (MUST), UX12.

**RC-m4 — Booleans are displayed as numbers.**

- **Cell/section:** `code-06` `_fmt`, used by `show_json`.
- **Observed issue:** `float(True)` is tried before any boolean check, so `all_parity`, `labels_supplied` and `test_scored_before_freeze` show as `1.0000` / `0.0000`.
- **Consequence:** small, but it sits in exactly the cells that carry integrity claims.
- **Evidence:** direct execution of the notebook's own `_fmt` (P5).
- **Recommended correction:** render `bool` as `true`/`false` before trying `float`.
- **Acceptance check:** `_fmt(True) == 'true'` and `_fmt(False) == 'false'`.

**RC-m5 — The evidence records still describe the pre-merge state.**

- **Cell/section:** `docs/receipt-release-evidence.md` (ladder row "published / merged: **no** — Draft PR only") and `docs/receipt-capstone.md` note 16.
- **Observed issue:** PR #9 merged as `ad47ef8`.
- **Consequence:** REL10 records are less trustworthy as the promotion record.
- **Evidence:** source inspection.
- **Recommended correction:** update both alongside RC-B1.
- **Acceptance check:** the ladder names `ad47ef8`, and note 16 records the badge switch.

### Suggestions

- **RC-S1.** §10 gives no learner task for outcome 2 ("trace an extracted amount back to its recognised words and boxes"). Add a prompt that asks the learner to read one panel overlay and say where the pipeline failed.
- **RC-S2.** `code-40` reads `inspect.stderr.strip().splitlines()[-1]`. If the inspect process dies without stderr (for example, killed), this raises `IndexError` instead of the actionable refusal. Guard the empty case.
- **RC-S3.** Disclose in §4 what `docs/receipt-capstone.md` note 15 already records: PSM 6 was chosen on validation receipts after a first run, and test receipts were viewed during that diagnosis.

## 4. Readiness

**Needs revision.**

- Required to reach Ready: RC-B1 and RC-M1 resolved; RC-m3 resolved (SRC3 MUST).
- Still required before promotion (unchanged from the evidence ladder):
  - a hosted run at the fixed revision;
  - a BYOD run on images not seen before, so OCR is not served from the cache;
  - the annotation audit (`docs/receipt-annotation-audit.md`: not started);
  - a human promotion decision.
- Minor findings: fix when practical.

## 5. Verified versus inferred

- **Verified directly:**
  - the badge failure and the main-branch success in Colab (signed out);
  - branch absence via the GitHub API;
  - carrier and generator parity;
  - the policy-feasibility bound (on the real `select`);
  - the boolean rendering;
  - the panel-categorisation rule;
  - the BYOD traversal refusal.
- **From documented execution:** every hosted result quoted above (same blob `329fbfe8`; saved outputs read, not re-run).
- **Inferred:**
  - Colab form fields do not re-execute their cell (RC-m2).
  - The validation_policy correct counts sit in the same 15–20 range as validation_model's (RC-M1). The notebook does not print them; the refer-all outputs are consistent with this.
- **Finding most likely to be wrong:** the **severity** of RC-B1. It is a one-line generator fix, and a learner who reaches the notebook through Colab's GitHub search or the DIMER platform never touches the badge. If DIMER distributes its own link, Major is defensible. The defect itself is not in doubt.

## Probe package

`DIMER_Small_Business_Receipt_Intelligence_Capstone_Review_Probes.zip` holds three files:

- `run_probes.py`: stdlib only. P2 needs network access; everything else is offline.
- `results.json`.
- `source_manifest.json`: the SHA-256 of every inspected file at `ad47ef8`.

To reproduce from a `git archive ad47ef8` export:

```
python run_probes.py --src <export> --evidence <export>/docs/execution-evidence/2026-09-29
```
