# LayoutLM Document QA (E2E) Notebook — Review

**Verdict: Needs revision**
**Review date:** 3 October 2026 (relay batch 2026-10-02, row 39)
**Repository:** `kurtvalcorza/layoutlm-document-qa-pipeline`
**Notebook:** `tutorials/layoutlm_document_qa_colab.ipynb`
**Reviewed commit:** `9c5ccfd891a8cabea28abb1f361ed6e5d6f2c21e` (origin/main, confirmed via GitHub API)
**Notebook Git blob:** `ad2dea7fda3e6f4595f2a505ed4c6a626b927740`
**Finding prefix:** `LDQ` (the workshop and receipt-capstone reviews in this folder use `DQA` and `RC`)

## Executive assessment

The notebook is a careful, standalone E2E workflow. It carries the three package modules byte-for-byte (the generator `--check` passes). It digest-verifies an 8-file safetensors snapshot and reads one digest-pinned column of CORD-v2 without downloading images. It splits 199 receipts by page into 595 / 152 / 229 questions, scores two non-neural baselines and the frozen model, fine-tunes the last four encoder blocks with validation-ANLS epoch selection, and exports a manifested safetensors adapter that it reloads from files. The prose about score semantics, OCR exclusion and what one seeded split cannot show is unusually honest. This review reproduced the data path and the frozen inference contract on CPU exactly.

Three Major problems remain:

1. **Run all needs a manual restart.** On the only hosted record of this exact blob, Run all stopped at the install cell with a restart instruction. The run passed only after the executor restarted the kernel. `docs/release-verification.md` describes that restart as "expected", and the README and STATUS call the result Release-grade.
2. **The documented experiments break the notebook.** "Set `TRAINABLE_ENCODER_LAYERS = 2`" and "raise `EPOCHS`" re-run `pipe.adapt` on the already-adapted in-memory model. With the 2-layer setting, the exported artifact omits blocks 8–9, which still hold the first run's changes. The reload-parity assertion then fails: 7/8 identical answers in a CPU stand-in.
3. **A BYOD re-run produces mislabelled numbers.** After the tutorial, BYOD as instructed re-runs Section 6 on that same adapted model. The learner's "frozen model" baseline is then the CORD-adapted model, and the BYOD adapter is trained on top of the CORD fine-tune.

This review does **not** establish whether Colab, rather than the recorded Kaggle T4, needs the restart, or how the full 6-epoch CPU default behaves. Neither was run.

## 1. Review contract and evidence

| Item | Scope |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED`; declares NOTEBOOK_SPEC **2.0** (`metadata.dimer.notebook_spec`). Reviewed against fleet spec **2.2 (2026-09-26)**, ml-worker origin/main `b1cfe13` |
| Intended learner | Basic Python and PIL; knows what extractive span QA over OCR tokens is, why a softmax product is not calibrated, and what ANLS measures (Prerequisites cell) |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU float32 default, CUDA used automatically |
| Default task | Frozen and fine-tuned extractive DocQA with `impira/layoutlm-document-qa@beed3c4d…` on templated CORD-v2 questions (seed 42; `EPOCHS=6`, `LEARNING_RATE=3e-5`, `BATCH_SIZE=16`, `TRAINABLE_ENCODER_LAYERS=4`) |
| Promised outcomes | Pinned install; digest-verified snapshot; one-column corpus fetch; validated, page-disjoint split; fit check; inference contract on a rendered invoice with input manifest and rejection probe; baselines and frozen ANLS/EM; bounded fine-tuning; held-out four-way comparison and per-field breakdown; invoice re-read; safetensors adapter with verified reload parity |
| Optional paths | `USE_BYOD` upload (JSON/JSONL); "Optional experiments" (2 trainable blocks, more epochs, unsupported question, BYOD) |
| Generator | `tools/build_notebook.py` (build_notebook.py/2) and `tools/notebook_template.py`; generated from revision `7a9a150`; carried module sha256 `fbcc4758…`; `--check` OK at `9c5ccfd` |
| Out of scope | `DIMER_Document_QA_LayoutLM_vs_Pix2Struct_Workshop.ipynb` and `DIMER_Small_Business_Receipt_Intelligence_Capstone.ipynb` (reviewed separately) |

### Evidence actually obtained

**Source inspection:** all 25 cells, the three carried modules (`pipeline.py` adapt/save/load paths, `samples.py` validation/split/BYOD loader), the generator template, `tutorials/README.md`, `docs/release-verification.md`, `README.md` and `STATUS.md`.

**Documented execution evidence:** Kaggle Tesla T4, commit `8541181`, notebook blob `ad2dea7f`, which is **identical to the reviewed blob**. Kernel `dimer-nb2-layoutlm-document-qa` v2, 19 Sep 2026, archived at `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-layoutlm-document-qa/v2/evidence/` (sha256s are in `source_manifest.json`).

- **Pass 1** stopped at cell 3 with `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.` (`run_summary.json`: `restarted_after_install_cell: true`).
- **Pass 2** ran 11/11 code cells on `cuda:0`. It used a `google.colab` shim.
- Recorded outputs: frozen test ANLS 0.843 / EM 0.782; adapted 0.940 / 0.921; validation ANLS 0.819 → 0.952, with best epoch 5; adaptation took 81.9 s; reload parity 8/8.

The CPU default path has only the local pre-flight harness record. Colab has **no** record.

**Direct execution (CPU, this review):** Windows, CPython 3.12 conda env `eo-notebook-test` with torch **2.13.0+cpu** (not the pinned 2.14.0), transformers 4.57.6 and Pillow 12.3.0 (not the pinned 11.3.0). Nothing was installed. The notebook's own module cells 5/7/9 were exec'd from the committed `.ipynb`. The snapshot and the CORD-v2 column cache were read from a pre-staged local copy; `verify_snapshot` and `fetch_corpus` re-checked the digests. Six probes ran in about 85 s in the foreground:

| Probe | What it ran | Result |
|---|---|---|
| P01 | Static checks | See LDQ-m2 and LDQ-m5 |
| P02 | Cell 13 sample branch | 100 + 100 rows; 595 / 152 / 229 over 119 / 30 / 50 pages; digests `d063f100…` / `07e37d51…` / `496e39d8…`, exactly as recorded |
| P03 | Cell 15 verbatim | Fit check dropped 0; invoice answered 5/5 exactly; all sanity checks `True`; frozen ANLS 0.830 on the first 60 test questions (a subset, not the 229) |
| P04 | Active-learning stand-in | See LDQ-M2 |
| P05 | BYOD split minimum | See LDQ-m1 |
| P06 | BYOD invalid inputs | See LDQ-m1 |

**Not verified:**
- a Colab Run all;
- the full 6-epoch CPU default (about 15 minutes; out of the probe time cap);
- the BYOD upload widget;
- any hosted BYOD run.

## 2. Separate judgments

- **Technical correctness:** **Strong on the default path; broken on reruns.** The data, validation, split, fit-check and inference paths reproduced exactly. Adaptation is transactional on failure. Artifact loading verifies the manifest before deserialising. However, `pipe.adapt` always mutates the live model and nothing reloads the base, so every documented rerun inherits a previous adaptation (LDQ-M2, LDQ-M3).
- **Promise fulfilment:** **Default promises are met on the recorded T4 run.** Two promises fail: "Run all … in a fresh supported runtime" without a restart (LDQ-M1), and the optional experiments and the BYOD re-run as instructed (LDQ-M2, LDQ-M3).
- **Learner experience:** **Clear but passive.** The explanations and Look-for notes are good. However, there are no prediction prompts, no checkpoints, no troubleshooting section and no Infrastructure labelling, and the experiments are prose-only (LDQ-m5). Some literal `{{…}}` schema strings render wrongly (LDQ-m2).
- **Spec conformance (2.2):** These `MUST`s are unresolved:
  - RUN1, RUN10 and ENV6: the restart (LDQ-M1);
  - VER4/VER5 under the documented experiment, and UX7 (LDQ-M2);
  - EVAL8/OUT8 semantics under the BYOD rerun (LDQ-M3);
  - REL12: no BYOD verification (LDQ-m1);
  - REL11/REL10 currency: no Colab record and a spec 2.0 declaration (LDQ-m3).

## 3. Prioritized findings

### LDQ-M1 — Major: Run all stops at the install cell and needs a manual restart on the recorded hosted run

- **Cell/section:** Cell 3 (§1 Install the pinned runtime); `docs/release-verification.md` procedure step 4; README/STATUS "Release-grade … (1 restart after install cell)".
- **Observed issue:** The cell pip-installs `torch==2.14.0`, `numpy==2.5.3` and others into the running kernel. When a preloaded distribution changes, it raises `RuntimeError … Restart the runtime, then rerun from the top`. On the recorded Kaggle T4 run of this exact blob it raised for `cuda-bindings` and `numpy`. The run passed only after the executor restarted the kernel.
  - Release-verification step 4 calls this restart "expected".
  - README and STATUS still grade the notebook Release-grade with "(1 restart after install cell)".
- **Consequence:** A learner who presses Run all gets an error at the first code cell and must restart and run again. Spec §1 lists "manually restarting the runtime" as something Run all must not need. A restart is therefore not an acceptable release condition.
- **Evidence:** Documented execution evidence: `run_summary.json` `runner.passes[0].ok=false`, error text above, `restarted_after_install_cell: true`. Source inspection: cell 3 lines 80–95; `docs/release-verification.md` step 4. Not verified on Colab, whose preinstalled torch/numpy differ; the restart there is inferred to be likely but is unrecorded.
- **Recommended correction:** Replace the in-kernel install with the uv isolated-environment pattern this repository already uses for its own workshop at `cf98b6e`: `tools/build_document_qa_workshop_carrier.py` and `tools/document-qa-workshop-requirements.lock`. The fleet reference is `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb`. The pattern:
  - a carrier cell bootstraps uv;
  - it creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`;
  - it installs a hash-locked requirements file with `uv pip install --require-hashes --only-binary :all:`;
  - it runs the workload in that environment, so the kernel's preloaded torch/numpy are never replaced.

  Make the change in `tools/notebook_template.py` (install cell and carrier) and `tools/build_notebook.py`. Then remove "restart … is expected" from `docs/release-verification.md` step 4, and return README/STATUS/tutorials README to Candidate until a no-restart run is recorded.
- **Acceptance check:** A fresh hosted run of the regenerated blob (Colab CLI or Kaggle) completes every code cell in a single pass, with no restart and no `RuntimeError` in any cell. Its run summary shows `restarted_after_install_cell: false`, and no release document describes a restart as expected.
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11.

### LDQ-M2 — Major: The documented experiments re-adapt the already-adapted model, and the 2-layer experiment breaks reload parity

- **Cell/section:** Interpretation, "Optional experiments" ("set `TRAINABLE_ENCODER_LAYERS = 2` and compare the artifact size and the test scores; raise `EPOCHS` …"). Cells 19, 21 and 23. `pipe.adapt` in the carried `pipeline.py`.
- **Observed issue:** `pipe.adapt` trains the live `pipe._model` in place. No cell reloads the base before it, so a learner who changes a knob and re-runs Section 7 onward does the following:
  1. Fine-tunes the **already fine-tuned** model. Epoch 0, labelled `'note': 'frozen model'`, is in fact the adapted model.
  2. With `TRAINABLE_ENCODER_LAYERS = 2`, trains only blocks 10–11 and the head. `save_artifact` writes those 34 tensors, but blocks 8–9 in memory still carry the first run's changes. `from_artifact` overlays the 34 tensors on a fresh base, so the reloaded model differs from the in-memory one, and the cell 23 assertion `identical_answers == of` can fail.
  3. Compares an "adapted" test score that reflects stacked 4-layer + 2-layer training against the original frozen score. The "artifact size and test scores" comparison therefore does not measure the 2-layer configuration the learner was asked about.
- **Consequence:** The one GUIDED experiment teaches a wrong conclusion about trainable depth, and it can end with an `AssertionError` that the notebook does not explain. Raising `EPOCHS` silently continues training from the adapted state.
- **Evidence:** Direct execution (CPU stand-in, P04). Real pinned weights, 48 training records, 1 epoch per run, on torch 2.13 CPU:
  - first `adapt` with 4 layers and no validation split, so the trained weights are kept;
  - second `adapt` with 2 layers as the experiment says: epoch 0 is labelled `frozen model`; `n_trainable` 14,177,282;
  - artifact has 34 tensors;
  - blocks 8–9 differ from base by up to 9.1e-5 in memory and by 0.0 in the reloaded pipeline;
  - reload parity on `test_records[:8]` is **7/8**, which fails the cell 23 assertion; 39/40 on 40 records; maximum score delta 0.115.

  With the default 6 epochs, the first-run drift is larger, so a failure is more likely. That is inferred: the full schedule was not run. Source inspection: `adapt` (`pipeline.py` lines 667–821) has no base reload; cells 19, 21 and 23 call it on the same `pipe`.
- **Recommended correction:** Make each experiment start from the verified base. Either:
  - add `pipe = LayoutLMDocumentQAPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)` at the top of cell 19 (cheap: the snapshot is local and verified); or
  - have `adapt` refuse to run when `self.adapter is not None` unless `from_base=True`, with a message naming the reload.

  Rewrite the experiment text as an explicit Predict → Change one thing → Run → Observe → Explain activity that names the cells to re-run. Put the change in `tools/notebook_template.py` (cell 19 and Interpretation prose) and, if the guard is chosen, in `src/layoutlm_document_qa_pipeline/pipeline.py`, then regenerate.
- **Acceptance check:**
  1. After a full Run all, set `TRAINABLE_ENCODER_LAYERS = 2` and re-run cell 19 onward as the notebook instructs.
  2. Epoch 0's validation ANLS must equal the original frozen value.
  3. The adapter must have 34 tensors, and every tensor outside them must equal the base.
  4. The cell 23 reload-parity assertion must pass.
  5. Repeat with `EPOCHS = 8` and check that the reported history starts from the frozen base.
- **Spec:** GDL10, UX7, SRC2, VER4, VER5, OUT8, ART8.

### LDQ-M3 — Major: BYOD "after the tutorial completes … re-run from that cell" scores the CORD-adapted model as "frozen" and stacks the BYOD adaptation on the CORD fine-tune

- **Cell/section:** Cell 0 BYOD paragraph and Interpretation "Baselines first". Cells 13 → 17 → 19 → 23.
- **Observed issue:** The notebook tells the learner to run BYOD after the default path, by setting `USE_BYOD = True` in cell 13 and re-running from there. At that point `pipe` holds the CORD-adapted weights, so:
  - Cell 17's `frozen_test`, `frozen_fields` and the "frozen" row of the comparison are computed with the **CORD-adapted** model, not the pinned base.
  - Cell 19 continues training from the CORD fine-tune, and epoch 0 is again labelled "frozen model".
  - The exported adapter's manifest names the pinned base and records BYOD training only. Its four blocks actually encode CORD + BYOD training, while the export's `data_source` says BYOD.
- **Consequence:** The notebook's central transfer advice ("the frozen model's score on *your* gold spans … are the numbers to read before any adapted one") is exactly what the BYOD path gets wrong. A learner reads an inflated "frozen" baseline and an understated adaptation gain, and ships an adapter whose provenance does not describe its training.
- **Evidence:** Source inspection: cells 13, 17, 19 and 23; `adapt` and `save_artifact` in `pipeline.py` (no base reload; the manifest records `self.adapter` only). The mechanism is the same one P04 shows by direct execution. BYOD itself was not executed end to end (not verified).
- **Recommended correction:** Reload the verified base whenever the dataset changes. Either add a `pipe = LayoutLMDocumentQAPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)` line in cell 13's BYOD branch, or apply the LDQ-M2 guard. Alternatively, document BYOD as "set `USE_BYOD = True`, then Runtime → Run all" and drop the "re-run from that cell" instruction. Put the change in `tools/notebook_template.py` (cells 0, 13 and Interpretation).
- **Acceptance check:**
  1. After a full default Run all, enable BYOD with a valid 60-record file and follow the notebook's instruction.
  2. Cell 17's frozen ANLS must equal a fresh base pipeline's ANLS on the same BYOD test split.
  3. Cell 19's epoch 0 must equal that value.
  4. The exported adapter must be reproducible from the base plus BYOD training alone: a base-only reload of the artifact gives the same answers as the in-memory model.
- **Spec:** DAT13, DAT14, EVAL8, EVAL10, OUT8, ART8, SRC2.

### LDQ-m1 — Minor: BYOD friction and no BYOD evidence

- **Cell/section:** Prerequisites "Data contract" and cell 13 BYOD branch; `split_dataset` / `load_byod_dataset` in `samples.py`.
- **Observed issue:**
  - **Undocumented minimum.** The contract says "a dataset needs 8..20,000 records", but cell 13 validates every split with the 8-record minimum. A one-question-per-page BYOD file needs **50** records: 8, 20, 40 and 49 records all fail. 8 records fail at the split ("split leaves 5 training records"). 20 / 40 / 49 records fail with `ValueError: 4 records; 8..20000 are required` (6 / 7 for the larger files). That message names neither the split nor the remedy.
  - **Parse errors lack a file line.** A malformed JSONL line raises a bare `JSONDecodeError … line 1 column 2`, with no file line number.
  - **Colab-only upload.** The branch imports `google.colab` unconditionally, so it fails with `ModuleNotFoundError` on the "Jupyter" runtime the Prerequisites name, and there is no location field.
  - **No BYOD run on record.** No BYOD run is recorded at any revision.
- **Consequence:** A learner whose small but valid dataset meets the stated contract is rejected with a message that seems to contradict it. Local-Jupyter users cannot reach BYOD at all.
- **Evidence:** Direct execution (P05, P06; synthetic records). Missing key and wrong extension get actionable messages (`records[0] is missing 'answer_end'`; `BYOD datasets must be .json or .jsonl`). Source inspection: cell 13. Documented evidence: `docs/release-verification.md` has no BYOD row.
- **Recommended correction:**
  1. State the effective minimum (about 50 records, or "≥ 8 per split") in the Prerequisites, or validate the evaluation splits with `min_records=1`.
  2. Prefix split-level errors with the split name.
  3. Have `load_byod_dataset` report the JSONL line number.
  4. Add a `BYOD_PATH` form field that is read before falling back to `google.colab.files.upload()` when available.
  5. Record one hosted BYOD run with one rejected input.

  Make the changes in `tools/notebook_template.py` and `src/layoutlm_document_qa_pipeline/samples.py`.
- **Acceptance check:**
  - A 20-record one-question-per-page file either passes or fails with a message naming the split and the documented minimum.
  - A malformed JSONL line error names the line number.
  - BYOD works with `BYOD_PATH` set outside Colab.
  - `docs/release-verification.md` lists a BYOD run with one rejection.
- **Spec:** DAT12, DAT19, EXE2, UX10, REL12.

### LDQ-m2 — Minor: Literal double braces in the rendered schema

- **Cell/section:** Cell 0 (BYOD paragraph) and cell 1 (Data contract, twice).
- **Observed issue:** The record schema renders as `{{id, page_id, question, words, boxes, image_size, answer_start, answer_end}}`, and the id pattern as `[A-Za-z0-9_.:-]{{1,64}}`. The template escapes braces as if for `str.format`, but these strings are not formatted. The doubled braces leak into the notebook, and the id pattern as printed (`{{1,64}}`) is not the regex the code uses (`{1,64}`).
- **Consequence:** A learner who copies the schema or the id pattern copies the wrong text.
- **Evidence:** Direct execution (P01: 1 occurrence in cell 0, 2 in cell 1). Source: `tools/notebook_template.py` lines 41 and 128.
- **Recommended correction:** Use single braces in those template strings, or route them through the same formatting as the other strings.
- **Acceptance check:** `"{{" not in` any markdown cell of the regenerated notebook.
- **Spec:** SRC3.

### LDQ-m3 — Minor: Release status and spec version are stale relative to the evidence

- **Cell/section:** `metadata.dimer.notebook_spec` / cell 0 ("Specification 2.0"); `tutorials/README.md` row; README; STATUS; `docs/release-verification.md`.
- **Observed issue:**
  - The notebook declares spec 2.0; the fleet spec is 2.2.
  - Release-grade rests on one Kaggle T4 CUDA run that needed a restart (LDQ-M1).
  - The advertised default runtime is a Colab **CPU** runtime, and the release table lists Colab as "the runtime the tutorial is written for". Neither a Colab run nor a hosted CPU run is recorded; the only CPU record is the local pre-flight, which the document itself marks as not promotion evidence.
  - Cell 3 still embeds `repository_revision` `7a9a150`. That is consistent with the generator, but the reviewer must resolve it by hand.
- **Consequence:** The release token overstates what was verified for the supported runtime.
- **Evidence:** Source inspection of the files named; `run_summary.json`.
- **Recommended correction:** Return the status to Candidate until a no-restart Colab (or Colab CLI) run of the current blob is recorded. Regenerate against spec 2.2 and update the tutorials README row.
- **Acceptance check:** The status token matches a recorded single-pass hosted run of the current blob on the stated default runtime, and the spec version equals the fleet spec.
- **Spec:** REL1, REL10, REL11, SRC3.

### LDQ-m4 — Minor: Expected-result notes drift from the recorded run, and per-field losses go unmentioned

- **Cell/section:** Cell 18 ("validation ANLS from about 0.82 to about 0.96"); `docs/release-verification.md` ("≈ 0.82 → ≈ 0.94"); cell 20 and Interpretation ("gain … concentrated in the fields the frozen model missed"); cells 17 and 21 example printouts.
- **Observed issue:**
  - The recorded run of this blob gave 0.819 → 0.952, and the two documents give different targets.
  - In the same run, `sub_total.discount_price` (n = 4) **dropped** from 1.00 to 0.75 after adaptation. Neither the notes nor the Interpretation tell the learner that some fields can regress, or that n = 2 and n = 4 fields cannot be read.
  - The three "frozen vs adapted vs gold" examples are `test_records[:3]`, which both models answer correctly, so they illustrate nothing.
- **Consequence:** A learner is primed to see only gains and may over-read tiny-n fields.
- **Evidence:** Documented execution evidence (`executed.ipynb` cells 17, 19, 21); source inspection.
- **Recommended correction:**
  - Quote one consistent observed range in both documents.
  - In cell 20, add a Look-for note about fields that lose score and about tiny n.
  - Print examples where the frozen and adapted answers differ.

  Make the changes in `tools/notebook_template.py`.
- **Acceptance check:** Cell 18 and `docs/release-verification.md` state the same range, and that range contains the recorded value. The cell 21 printout includes at least one disagreement when one exists. The prose names regression and small-n as things to look for.
- **Spec:** GDL8, UX4, EVAL15.

### LDQ-m5 — Minor: The GUIDED layer is thin for spec 2.2

- **Cell/section:** Whole notebook.
- **Observed issue:**
  - No cell asks the learner to predict anything (P01: 0 occurrences of "predict").
  - There are no interpretation checkpoints with sample answers and no troubleshooting section.
  - The 46 KB carried-module cell is not labelled Infrastructure "you may run without reading".
  - There is no Input → Model → Output contract box and no conclusion template.
  - The optional experiments are one prose sentence, with no cell. "Ask the adapted model *What is the delivery address?*" needs code the learner must write.
- **Consequence:** The notebook teaches by exposition. The learner runs cells and reads numbers, but is never asked to apply the score semantics or the baseline-first rule it explains.
- **Evidence:** Source inspection; P01.
- **Recommended correction:** Add the following in `tools/notebook_template.py`:
  - a prediction prompt before Sections 6 and 8;
  - one runnable experiment cell for the unsupported-question probe, which prints the span and its score;
  - an Infrastructure callout on Section 2;
  - a short troubleshooting section (restart, download or digest failure, memory, BYOD refusals);
  - a conclusion template.
- **Acceptance check:**
  - At least one Predict → Change → Run → Observe → Explain activity has a runnable cell.
  - Section 2 is labelled Infrastructure.
  - A troubleshooting heading exists.
  - The Interpretation section ends with a fill-in conclusion prompt.
- **Spec:** GDL4, GDL7, GDL9, GDL10, GDL11, GDL13, GDL14, UX5.

### Suggestions

- **LDQ-S1:** Compare reload parity on scores with an explicit tolerance as well as answer strings, over more than 8 records. P04 showed 39/40 identical strings while the scores differed by up to 0.115. (VER4)
- **LDQ-S2:** Print `n` beside each per-field ANLS in the comparison, and flag fields with n < 10 as unreadable.
- **LDQ-S3:** In Section 6, show one keyword-lookup failure next to the frozen model's answer, so the baseline is concrete rather than a number.

## 4. Promise and objective trace

| Claim / objective | Cell | Observable result | Status |
|---|---|---|---|
| Run all in a fresh runtime, no restart | 3 | Kaggle T4: RuntimeError, then restart | **Not met** (LDQ-M1) |
| Digest-verified pinned snapshot, no pickle | 11 | 8 files verified, `local-snapshot` (Kaggle; P03 local) | Met |
| One-column CORD-v2 fetch, digest-pinned | 13 | 100 + 100 rows, column digests match (Kaggle; P02 cache path) | Met |
| Page-disjoint 595 / 152 / 229 split, refusal probes | 13 | Exact counts and digests; 4 probes rejected (Kaggle, P02) | Met |
| Fit check drops, never truncates | 15 | 0 dropped (Kaggle, P03) | Met |
| Inference contract, score semantics | 15 | 5/5 invoice answers, sanity checks True, `sample-sanity` (Kaggle, P03) | Met |
| Baselines and frozen score | 17 | 0.406 / 0.635 / 0.843 ANLS (Kaggle) | Met |
| Bounded fine-tuning with validation epoch selection | 19 | Best epoch 5 of 6 (Kaggle) | Met (default only) |
| Held-out four-way comparison and per-field breakdown | 21 | 0.94 adapted; per-field table (Kaggle) | Met; regression unmentioned (LDQ-m4) |
| Adapter export and verified reload | 23 | 66 tensors, 8/8 parity (Kaggle) | Met on default; **fails under the documented experiment** (LDQ-M2) |
| Objective: run a bounded fine-tuning with explicit hyperparameters | 19 | Form fields reach `adapt` | Met for the first run; reruns are invalid (LDQ-M2) |
| Objective: read `score` correctly | 14/15 | Prose explains it; no learner activity applies it | Partially (LDQ-m5) |
| BYOD through the same stages | 13→23 | Not executed; rerun semantics invalid | **Not met as instructed** (LDQ-M3, LDQ-m1) |

## 5. Journeys

| Journey | Evidence basis | Outcome |
|---|---|---|
| First-time learner | Source inspection | Strong prose and Look-for notes; first cell errors with a restart (LDQ-M1); schema strings misrender (LDQ-m2); passive (LDQ-m5) |
| Clean default | Documented execution (Kaggle T4, exact blob) + direct CPU execution of the data and frozen-inference paths | PASS after one restart. Data path and invoice reproduced exactly on CPU. The full CPU 6-epoch path and Colab were **not verified** |
| Active learning | Direct execution, CPU stand-in (P04) | The 2-layer experiment produces an artifact that does not reproduce the in-memory model (7/8 parity → assertion failure); epoch 0 is mislabelled "frozen model" |
| Reuse and recovery | Documented reload (Kaggle 8/8); direct execution of BYOD validators on synthetic files (P05, P06) | Artifact reload works on the default path. BYOD rerun semantics are invalid (LDQ-M3). Small datasets are rejected with an unlabelled message. The upload widget and hosted BYOD were **not verified** |

## 6. Readiness

**Needs revision.** Three Major findings are open. Each is a gate:

1. **LDQ-M1:** the notebook needs a single-pass, no-restart hosted run of the regenerated blob.
2. **LDQ-M2:** the experiments need a base reload or a guard, plus the parity acceptance check.
3. **LDQ-M3:** BYOD needs the frozen baseline and artifact provenance corrected.

These applicable `MUST`s are unmet: RUN1/RUN10/ENV6, VER4/VER5 (under the experiment), UX7, REL12 and REL11 currency. The current "Release-grade" token should return to Candidate.

## 7. Verified vs inferred

- **Verified by direct execution (CPU, torch 2.13, not pinned):**
  - data path counts and digests;
  - the 5/5 invoice answers and sanity checks;
  - rerun stacking and the 7/8 parity failure in a 1-epoch stand-in;
  - BYOD minimum and error messages;
  - literal braces;
  - generator `--check`.
- **Verified from documented evidence:** the Kaggle T4 restart and all recorded metrics of this exact blob.
- **Inferred:**
  - that Colab also needs the restart;
  - that the full 6-epoch default makes the LDQ-M2 parity failure more likely than the stand-in;
  - that the BYOD rerun (LDQ-M3) behaves as the code reads; BYOD was not executed end to end.
- **Only Kurt can confirm:** a Colab run, and whether BYOD is intended to be run after the default path or instead of it.
- **Most likely to be wrong:** the *severity* of LDQ-M2's parity failure under the default schedule. The stand-in proves the mechanism (blocks 8–9 differ between memory and the reload). Whether the 8 checked answers diverge at full schedule is inferred, and the cell could pass by chance while the artifact still misrepresents the trained model.

## Probes

`layoutlm_document_qa_colab_Review_Probes.zip` contains:
- `run_probes.py`: the CPU probes P01–P06; usage is in its docstring;
- `results.json`: the outputs, environment and timings;
- `source_manifest.json`: the sha256 of every inspected file, the Kaggle evidence and the spec.
