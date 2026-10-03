# DIMER Notebook: Document Question Answering (LayoutLM vs Pix2Struct) — Review

Review of `tutorials/DIMER_Document_QA_LayoutLM_vs_Pix2Struct_Workshop.ipynb` against the Notebook Review
Framework v1 and DIMER Notebook Specification 2.2.

**Reviewed revision:** `kurtvalcorza/layoutlm-document-qa-pipeline@07e4c08dd5ba0a6983236b0e67ea019487728b6e`
(`main` on 2026-10-02, confirmed through the GitHub API). Notebook blob `f698fb73a132eeec8fdd7a2646d19391a78c48bf`.
Its code cells are identical to blob `256c9d9a` (commit `fb8415f`), which the 2026-09-26 hosted run executed; the
only later notebook commit, `49d9757`, changed markdown.
**Probe bundle:** `DIMER_Document_QA_LayoutLM_vs_Pix2Struct_Workshop_Review_Probes.zip` (`run_probes.py`,
`results.json`, `source_manifest.json`).

## Executive assessment

The default comparison is sound and reproducible. The page split, the 229 held-out questions, both baselines and
the LayoutLM results reproduced exactly on CPU in this review (test-set digest `b2c58a97…`, LayoutLM ANLS 0.843 /
exact match 0.782, the same values as the hosted T4 run). The notebook is honest about the evaluation's bias toward
extractive answers and about uncalibrated scores. **Readiness: Needs revision.** Four Major and five Minor
findings remain open.

- **DQA-M1 (Major):** the optional BYOD branch states none of its limits. It does not accept the ZIP format that
  the repository's tutorial registry advertises. Most invalid inputs are either accepted or refused with a raw
  Python error that does not name the record. Some are refused only after LayoutLM has answered every record.
  DAT12, DAT17 and DAT19 (`MUST`) fail.
- **DQA-M2 (Major):** a BYOD run neither shows nor saves the answers. The learner sees a count, a verdict and two
  mean ANLS values (DAT15).
- **DQA-M3 (Major):** the notebook never shows a page. The qualitative panels, the main teaching view for
  "inspect answer localization and answers not found in OCR", are only written to disk. No example of a
  Pix2Struct answer that is absent from the OCR words is selected.
- **DQA-M4 (Major):** the only "Try it yourself" activity asks the learner to rerun an experiment that has
  already run, with nothing to change. It asks which question types are most affected, but the output never
  breaks results down by field. Rerunning the cell as instructed raises `NameError`, because LayoutLM was released
  in §15.
- The Minor findings concern control validation, rejection probes that are recorded but never executed, runtime
  orientation, under-reported experiment subsets, and terminology and metric explanation.

## 1. Review contract and evidence

| Field | Value |
|---|---|
| Repository / notebook | `layoutlm-document-qa-pipeline` / `tutorials/DIMER_Document_QA_LayoutLM_vs_Pix2Struct_Workshop.ipynb` (68 cells, 29 code) |
| Revision | `07e4c08` (blob `f698fb73`) |
| Specification | NOTEBOOK_SPEC 2.2 (`ml-worker` `origin/main`, last changed in `50b7eb4`). The notebook declares 2.1 (see S1) |
| Repository workshop spec | `docs/document-qa-workshop-spec.md` |
| Profile / mode | `TASK-INFERENCE` / `WORKSHOP`, `MULTI-MODEL`, standalone (no repository import, digest-pinned downloads) |
| Audience | "Learners who can run Python cells in Colab/Jupyter and are new to document QA"; OCR familiarity helpful, not required |
| Runtime | GPU recommended (Tesla T4); the code silently falls back to CPU |
| Promised outcomes | 8 learning objectives (cell 0): distinguish extractive from generative QA; ANLS / EM / empty rate / per-field on one test set; inspect localization and answers not in OCR; paraphrase sensitivity; unanswerable behaviour; LayoutLM with and without layout; Pix2Struct on clean and degraded pages; quality vs OCR dependency, traceability and cost. Optional BYOD (§30) |

**Execution evidence that covers this revision.**
- `docs/release-verification.md` records a maintainer-supplied **Colab T4** Run all, dated 2026-09-26, archived at
  `docs/execution-evidence/2026-09-26/…Workshop.ipynb`. All 29 code cells ran with no errors, and its code-cell
  sources match `fb8415f`. The code cells are unchanged at `07e4c08`. Results: last-number 0.406, keyword 0.635,
  LayoutLM 0.843, Pix2Struct 0.801 ANLS. BYOD was off. The exported files were not supplied.
- Nothing hosted covers BYOD (REL12 open). Neither the "Try it yourself" activity nor any non-default control
  setting has been run.

**Direct execution in this review** (CPU only, CUDA hidden; Windows; anaconda Python 3.13, torch 2.9.1,
transformers 4.57.6; `run_probes.py`):
- The notebook's own cells, top to bottom, with the **real pinned LayoutLM** checkpoint, verified by the notebook's
  own staging code from a local snapshot.
- The **real CORD-v2 `ground_truth` column** of the two pinned shards. Page pixels were not available, so each page
  image is a blank stand-in of the annotated size. LayoutLM does not read pixels.
- A **deterministic Pix2Struct stand-in**. Nothing here is evidence about Pix2Struct's answers.
- The install cell's pip step and the parquet download were bypassed, so dependency resolution was not exercised.
- The BYOD cell on 11 synthetic datasets: 1 valid labelled, 1 unlabelled, 9 invalid, plus 2 ZIP cases.

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Good framing of the two information boundaries, the bias caveat, and score semantics. Gaps: M3, m3, m4, m5 |
| Clean default | Documented execution (Colab T4, 2026-09-26, same code) + direct CPU execution (LayoutLM real, Pix2Struct stand-in) | **Passed.** On CPU, split, digest, baselines and LayoutLM metrics equal the hosted values exactly (P01) |
| Active learning | Direct execution | **Fails:** the Try-it rerun raises `NameError`, there is no learner-changeable variable, and there is no field-level view (M4). Control defects: m1 |
| Reuse and recovery | Direct execution (CPU, synthetic BYOD, stand-in Pix2Struct) | A valid BYOD folder runs, but its results are not usable (M2). Invalid inputs are mostly not refused actionably (M1). The documented ZIP is refused. The model stages on GPU are **not verified** |

## 2. Separate judgments

- **Technical correctness:** sound on the default path.
  - Model files and CORD shards are digest-checked.
  - Image/annotation size agreement is enforced.
  - Pages are deduplicated before the seeded split, and page leakage is checked.
  - Models are released between stages.
  - One metric implementation is shared by all four systems, and it matches the repository's `metrics.anls`
    (`anls("1250.00", ["1,250.00"]) = 0.875` in both).

  The defects are in recovery:
  - BYOD validation (M1) and output (M2);
  - unvalidated or stale controls (m1);
  - `input_manifest.json` claims refusal checks that never ran (m2).
- **Promise fulfilment:** the comparison, metrics, per-field table, agreement counts, paraphrase, unanswerable,
  layout ablation, degradation and resource objectives are delivered. "Inspect answer localization and answers not
  found in OCR" is delivered only as files on disk and an aggregate rate (M3). The BYOD promise "the two models
  still run" holds, but the learner gets no answers back (M2).
- **Learner experience:**
  - The orientation, information-boundary explanation and interpretation sections are clear and appropriately
    cautious.
  - The learner never sees a receipt or an individual answer (M3).
  - The single activity cannot be completed as written (M4).
  - Experiment means come without their sample size (m4).
  - ANLS is named but never explained (m5).
- **Spec conformance (2.2):**
  - Fails `MUST`: DAT12, DAT17 and DAT19 (M1); DAT15 (M2); REL12 (BYOD never verified, so it is open regardless).
  - Unmet `SHOULD`: GDL6, GDL7, GDL10, GDL15 (m5, M3, M4); UX4, UX6 and UX10 (M3, M4, M1); VAL3 and VAL6 `MUST`
    within BYOD (M1).
  - REL1 is met for the default path by the 2026-09-26 run.
  - Release status stays **Candidate**.

## 3. Findings

### DQA-M1 (Major): BYOD contract incomplete; refusals late, generic or missing

- **Cell/section:** md `291a71b6` (§30), code `77f434e8`, `tutorials/README.md` workshop row.
- **Observed issue:**
  1. §30 states the folder layout and required keys, but none of the limits the code enforces or that the
     repository workshop spec §76 sets: 1..500 records, 1..100 pages, 1..2000 words, 256-character questions,
     image sides 16..4096.
  2. It never says that the data stays in the runtime (DAT17).
  3. `tutorials/README.md` and workshop spec §75 describe a **ZIP** of `pages/` plus `records.jsonl`. `BYOD_PATH`
     accepts only an existing folder, and the notebook does not say how to put a folder into Colab.
  4. Results of the probe matrix (P04):

     | Input | What happened |
     |---|---|
     | Missing page file | Raw `FileNotFoundError` with the absolute path, no record id |
     | 3-number box | `not enough values to unpack (expected 4, got 3)` |
     | Blank question | Refused only after LayoutLM was loaded, `Question must contain 1..256 characters`, no record id |
     | 5000-px-wide page | Passes validation. LayoutLM loads and answers, then Pix2Struct refuses with `Pix2Struct image side outside 16..4096`, no record id |
     | Duplicate `id`; 101 pages; `answers` given as a string; mixed labelled and unlabelled records; an empty OCR word | All **accepted**. The two `answers` cases silently turn the whole run into `not-measurable` |
     | Records that fail validation | Messages say `BYOD record {i}` (the 0-based line index) rather than the record's `id` |
- **Consequence:** a learner bringing receipts cannot tell what the notebook accepts, or which record to fix. One
  oversized page costs a full LayoutLM pass before it is refused. A malformed `answers` field silently removes the
  evaluation.
- **Evidence:** direct execution (CPU, synthetic data, stand-in Pix2Struct, real LayoutLM), P04; source inspection.
- **Spec:** DAT12, DAT17, DAT19 (`MUST`); VAL3, VAL6 (`MUST`); UX10.
- **Recommended correction:**
  - State every limit, the folder **and** ZIP layouts, how to upload to Colab, and data locality in §30.
  - Validate the whole dataset before any model loads: unique ids; 1..100 pages; files present; page size
    16..4096; non-empty question ≤256 characters; 1..2000 non-empty words; four finite numbers per box inside the
    page; `answers` either a list of non-empty strings on every record or absent from every record. Each message
    names the record `id` and the failed rule.
  - Accept a `.zip`, extracted safely with member paths confined to the extraction folder.
- **Acceptance check:** each P04 invalid case is refused before any model load, with a message containing the
  record id (or file name) and the rule. The ZIP form of the valid dataset is accepted, and a ZIP member path
  outside the root is refused. §30 contains the five limits and a data-locality sentence.

### DQA-M2 (Major): BYOD results are neither shown nor exported

- **Cell/section:** code `77f434e8`.
- **Observed issue:** after both models answer the user's questions, the cell prints only
  `{'n', 'evaluation_verdict', 'layoutlm_anls', 'pix2struct_anls'}`. Individual answers, exact match, empty
  answers and answer-in-OCR are never shown. Nothing is written to disk, because the export cell (§29) runs
  earlier and the BYOD branch has no export of its own (P05: `byod_files_written: []`). An unlabelled run prints
  only `{'n': 1, 'evaluation_verdict': 'not-measurable'}`. Workshop spec §77 says "answers are exported". Pix2Struct
  also stays resident after the branch.
- **Consequence:** the promised "evaluate compatible user-supplied document QA data" yields no usable result. An
  unlabelled run, the common case, produces nothing at all.
- **Evidence:** direct execution, P05; source inspection.
- **Spec:** DAT15 (`MUST`: the same output contract as the sample path); OUT1, OUT4.
- **Recommended correction:**
  - Build BYOD rows with the same columns as the sample `predictions.csv` (record id, question, both answers, and,
    when labelled, ANLS and EM).
  - Print a per-record table and the per-model metrics.
  - Write `byod/predictions.csv` and `byod/metrics.json` under the output folder, separate from the canonical
    exports.
  - Release Pix2Struct at the end of the branch.
- **Acceptance check:** a labelled synthetic BYOD run prints every record id with both answers and writes
  `byod/predictions.csv` with ANLS and EM. An unlabelled run prints `not-measurable`, still prints and writes both
  answers, and leaves the canonical export files byte-identical.

### DQA-M3 (Major): the learner never sees a page, an answer location or an answer that left the OCR

- **Cell/section:** code `52ae9160` (§27), md `8e20a437`, objectives in cell 0.
- **Observed issue:**
  - No cell displays an image. The §27 panels, the main teaching view in workshop spec §46, are saved as PNGs and
    only their paths are printed. The hosted run's output is just four paths.
  - The panels omit the LayoutLM span score and the Pix2Struct truncation flag that spec §46 lists.
  - Selection covers only the four agreement categories. Spec §45's "Pix2Struct answer not present in OCR" and
    "LayoutLM high span score but incorrect" are not selected, and no individual answer appears anywhere in the
    notebook's output.
  - The pre-results exercise in spec §84 (show a receipt, ask which evidence each model receives) is absent; no
    receipt is shown before the metrics.
- **Consequence:** the objective "inspect answer localization and answers not found in OCR" is reduced to an
  aggregate rate (Pix2Struct 0.812 in OCR on T4) and files the learner must find in Colab's file browser. Answer
  traceability, the notebook's central contrast, is asserted rather than seen.
- **Evidence:**
  - Documented execution: the 2026-09-26 output of cell `52ae9160` is four file paths.
  - Direct execution, P06: 0 images displayed; categories `layoutlm_only`, `neither_exact`.
  - Source inspection.
- **Spec:** UX1 (objective tied to executed code), UX3, UX11; GDL7.
- **Recommended correction:**
  - Display the panels inline, with span score and truncation on them.
  - Add the two spec §45 selections (Pix2Struct answer not in OCR; LayoutLM's highest-scoring wrong answer).
  - Print a short table of Pix2Struct answers absent from the OCR words.
  - Before the first model result, show one test receipt with its OCR boxes and the spec §84 prediction questions.
- **Acceptance check:** a default run displays at least one page image before the first LayoutLM result and at
  least four panels in §27. `selected_examples` includes a `pix2struct_not_in_ocr` category whenever such answers
  exist. The not-in-OCR table lists record id, question, gold answer and Pix2Struct answer.

### DQA-M4 (Major): "Try it yourself" cannot be carried out as written

- **Cell/section:** md `guided-01`; code `aada1835` (§13), `c01e6b81` (§15).
- **Observed issue:**
  - The activity says: remove spatial layout, "Predict which question types should be most affected, rerun that
    comparison". The default path already runs exactly that ablation, so there is no variable for the learner to
    change.
  - The §13 output prints only two means. Its rows carry no `field`, so "which question types" cannot be answered.
  - The activity sits after the terminal summary. By then §15 has deleted the LayoutLM model and tokenizer, so
    rerunning §13 raises `NameError: name 'layout_tokenizer' is not defined` (P02).
- **Consequence:** the notebook's only Predict → Change → Run → Observe → Explain activity fails at the "Run"
  step. Even when it runs, it cannot answer its own question.
- **Evidence:** direct execution, P02; source inspection.
- **Spec:** GDL10, UX6 (`WORKSHOP`), UX7; framework dimension 7.
- **Recommended correction:**
  - Add `field` to the §13 rows and print a per-field breakdown.
  - Give the activity its own optional cell, off by default, so it does not change the default run. The cell
    should have a learner-chosen layout change (e.g. zero / shuffled / coarse boxes) and a record cap. It loads
    LayoutLM itself, writes only under `activity/`, prints a per-field comparison, and releases the model.
  - Make a call to the released model fail with an actionable message instead of `NameError`.
- **Acceptance check:** after a default Run all, running the activity cell with the activity enabled completes,
  names at least one question field in its output, and leaves the canonical export files byte-identical. The §13
  output names fields. Calling `layout_answer` after §15 raises an error that says how to reload.

### DQA-m1 (Minor): controls unvalidated or stale on partial rerun

- **Cell/section:** code `5f810ac8` (§3), `c598d7a7` (§16).
- **Observed issue:**
  - `ROBUSTNESS_MAX_RECORDS` is not validated. 0 is accepted and gives empty experiments with `NaN` means; −3 is
    accepted and silently selects 226 records (P03).
  - `pix_answer(…, max_new_tokens=MAX_NEW_TOKENS)` freezes the budget when §16 runs. If a learner changes
    `MAX_NEW_TOKENS` and reruns §17, the old budget is used while `provenance.json` records the new one.
- **Consequence:** wrong-sized experiments, or provenance that does not match what ran.
- **Evidence:** direct execution, P03.
- **Spec:** UX10; OUT7.
- **Recommended correction:** bound `ROBUSTNESS_MAX_RECORDS` to 1..50 with a message, and read `MAX_NEW_TOKENS` at
  call time.
- **Acceptance check:** the config cell refuses 0 and −3. `pix_answer`'s default is `None` and resolves to the
  current `MAX_NEW_TOKENS`.

### DQA-m2 (Minor): the rejection probes in `input_manifest.json` were never run

- **Cell/section:** code `b1722b90` (§29).
- **Observed issue:** `input_manifest.json` lists four refusals as literal strings next to `"verdict": "accepted"`
  ("LayoutLM: empty question is rejected", …). No cell ever executes them. Workshop spec §74 asks for six
  non-blocking checks, including box outside page and image outside the size ceiling.
- **Consequence:** a machine-readable export claims checks that were never performed.
- **Evidence:** direct execution, P07; source inspection.
- **Spec:** REL8 (static claims are not execution evidence); workshop spec §74.
- **Recommended correction:** run the six refusals in the definition cells without blocking the run, and export
  each one's observed message.
- **Acceptance check:** `input_manifest.json.rejection_probes` holds six objects, each with `refused: true` and the
  observed message.

### DQA-m3 (Minor): runtime expectations unclear; silent CPU fallback

- **Cell/section:** md `guided-00`, md `e02df353` (§4), code `7984f79d`.
- **Observed issue:**
  - "Use the documented GPU runtime" names no runtime, and no total download size is given: 0.51 GB + 1.13 GB of
    weights plus 0.48 GB of CORD shards.
  - No measured time is given, although the hosted run measured LayoutLM at 10 s and Pix2Struct at 152 s of
    evaluation on a T4.
  - On a CPU runtime the notebook proceeds without any warning.
- **Consequence:** a learner who forgets to select a GPU waits through 229 Pix2Struct generations on CPU with no
  explanation.
- **Evidence:** documented execution (resource row of the 2026-09-26 run); direct execution, P08.
- **Spec:** RUN11, ENV4, UX12.
- **Recommended correction:** name the Colab T4, give download sizes and the measured evaluation times with their
  environment, and print a clear warning when no GPU is detected.
- **Acceptance check:** the opening states the GB to download and a measured time with its environment, and a CPU
  run prints a "No GPU detected" warning.

### DQA-m4 (Minor): experiment subsets reported without size or coverage

- **Cell/section:** code `aada1835` (§13), `b1a10dd3` (§18), `6206457c` (§23).
- **Observed issue:** the ablation, degradation and paraphrase experiments use the first 20 records by id. These
  come from **5 pages** and cover 7 of 10 fields (P01). The cells print bare means, with no record or page count
  and no "What to notice" note. The subset is easier than the full set (LayoutLM 0.913 on the subset against 0.843
  overall in the T4 run), so a learner comparing "no layout 0.676" with the headline 0.843 draws the wrong
  difference.
- **Consequence:** over-reading small, unrepresentative subsets.
- **Evidence:** direct execution, P01 and P09; documented execution.
- **Spec:** GDL8, UX4; EVAL6.
- **Recommended correction:** print n records, pages and fields with each experiment, and add a "What to notice"
  note after each one that names the subset and says to compare against the subset's own canonical mean.
- **Acceptance check:** each of the three outputs includes "pages", and a "What to notice" markdown cell follows
  each one.

### DQA-m5 (Minor): metric explanation, terms and terminology

- **Cell/section:** md `f3f4db23` (§9), md `guided-05` (Glossary), code `8ad583a7`.
- **Observed issue:**
  - §9 names ANLS "with the standard 0.5 threshold" but never explains how it is computed or how answers are
    normalised. The normalisation removes punctuation, so `1250.00` against gold `1,250.00` scores ANLS 0.875 but
    fails exact match.
  - The glossary, placed at the end, lacks OCR, word box, span, Exact Match, token window, patch and uncalibrated
    score.
  - The terminal summary prints "DIMER Document Question Answering Workshop".
- **Consequence:** learners compare ANLS against EM without knowing why they differ.
- **Evidence:** source inspection; direct execution, P10.
- **Spec:** EVAL3 (`MUST`: principal metrics explained); GDL6, GDL15.
- **Recommended correction:** add a worked ANLS/EM explanation in §9, extend the glossary and point to it from the
  opening, and rename the summary heading.
- **Acceptance check:** §9 explains edit distance, normalisation and the 0.5 cut-off with an example. The glossary
  contains the seven terms. The summary heading does not say "Workshop".

### Suggestions (not release requirements)

- **DQA-S1:** declare NOTEBOOK_SPEC 2.2 in the metadata and opening. The notebook already carries the 2.2 guided
  layer.
- **DQA-S2:** add a near-miss agreement view (workshop spec §44, ANLS ≥ 0.5 but not exact).
- **DQA-S3:** add Files-pane or ZIP download guidance for `outputs/document_qa_comparison/`.

## 4. Readiness

**Needs revision.** Four Major findings (M1–M4) are open, and the `MUST`s DAT12, DAT15, DAT17, DAT19 and EVAL3 are
unmet. The default path has hosted evidence for the reviewed code cells. After fixes the notebook will be a new blob
and will need:
- a fresh Colab T4 Run all of the fix head, covering the default path and the activity;
- a labelled BYOD run with one rejected input (REL12).

The status stays **Candidate** until a human promotes it.

## 5. What was verified versus inferred

- **Verified by direct execution (CPU):**
  - the default cell chain with real LayoutLM and real CORD annotations, with results equal to the hosted run;
  - BYOD acceptance and refusal behaviour;
  - the Try-it rerun failure;
  - the control defects.
- **Verified from documented execution:** the hosted default run's outputs, including the panel cell printing only
  paths.
- **Inferred:**
  - Pix2Struct's behaviour on BYOD pages (stand-in only);
  - the impact of the CPU fallback on wall time (not measured on CPU for Pix2Struct);
  - how learners would actually respond (no learner observation).
- **The finding I'd most expect to be wrong:** DQA-M3's severity. A facilitator in a live workshop could open the
  PNGs from the Files pane, which would make it Minor for an instructor-led session. For self-paced use, which
  the opening invites, it remains Major.
