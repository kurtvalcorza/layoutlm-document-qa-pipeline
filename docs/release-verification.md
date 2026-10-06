# Release verification

`tutorials/layoutlm_document_qa_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` (/2.1) output for its recorded revision; exactly two kernel cells — the isolated install
  (uv wheel size/SHA-256, managed CPython 3.12.12, `--require-hashes --only-binary :all:` from the carried lock) and
  the router to the isolated worker (whose `google.colab` stubs carry a module spec); Sections 1–3 labelled
  Infrastructure and collapsed; the guided-layer markers; no bare `assert` in learner cells; no doubled braces in
  markdown; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the pinned
  CORD-v2 dataset revision is the one other 40-hex string allowed);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `LayoutLMDocumentQAPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path,
  `read_corpus` + `build_sample_dataset(seed=SPLIT_SEED)` / `load_byod_dataset`, `validate_dataset` per split,
  `check_split_disjoint`, `write_dataset_jsonl`, `pipe.check_fit` per split, the ceiling print, `validate_inputs`
  with the box-outside-page refusal probe, `pipe.answer` with the sanity checks and the per-page
  `evaluation_report` on the rendered invoice, `last_number_baseline`, `keyword_lookup_baseline`, `pipe.evaluate`
  on the frozen model (after `reset_to_pretrained()`, refusing an adapted model) and on the validation and test splits after adaptation with the recorded outcomes, `pipe.adapt`
  with its explicit hyperparameters, `evaluation_report` on the invoice after adaptation, `pipe.save_artifact`,
  `LayoutLMDocumentQAPipeline.from_artifact` and the reload-parity check that raises an explained `RuntimeError`, and the provenance fields
  `weight_format`, `weight_sha256` and the `corpus` block), the seven expected `outputs/` paths, the learner-facing
  statements (MIT weights, the model never sees pixels, adaptation with gold spans, the CC BY 4.0 corpus, the span
  score as a product of two softmax probabilities that is not a calibrated probability, the two non-neural
  baselines, no dispersion estimate, the OCR exclusion, the snapshot note, the windowed-not-trained rule) and the
  gated-off BYOD default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import
  on the primary path, a mutable `revision='main'`, direct `from transformers import` /
  `LayoutLMForQuestionAnswering` / `AutoTokenizer` / `start_logits` / `from huggingface_hub import` /
  `get_hf_file_metadata` / `urllib.request` / `pyarrow` / `pytesseract` / `safetensors` / `torch.optim` /
  `.backward(` / `pipe._model` use **outside the carried module cells**, `trust_remote_code=True`, `pickle.load`,
  `torch.load(` without `weights_only=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `numpy`, `pillow`,
`huggingface-hub` and `pyarrow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the
offline unit suite (`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; injected runner, window counter and corpus fetcher,
temporary manifests, no weights — `tests/test_model_backed.py` is skipped without the snapshot). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU or GPU runtime, Linux x86_64 (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/layoutlm-document-qa/` or the corpus cache `weights/cord-v2/` (the standalone path writes the
   manifest itself, stages the missing files from the Hub, and reads the pinned CORD-v2 columns from the Hub, so
   neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `EPOCHS = 6`, `LEARNING_RATE = 3e-5`, `BATCH_SIZE = 16`,
   `TRAINABLE_ENCODER_LAYERS = 4`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`,
   `pillow==11.3.0`, `huggingface-hub==0.36.2`, `pyarrow==25.0.1`, imported from the isolated CPython 3.12.12
   environment Section 1 builds from `tutorials/requirements-colab.lock.txt`; every code cell must complete in a
   **single pass with no restart** of the runtime and no `RuntimeError` (a run that needs a restart is not promotion
   evidence);
5. verify every default-path stage completes:
   - the isolated environment built from the carried hash lock with no GitHub access, and every later cell routed to it;
   - the three carried module cells execute (defining `LayoutLMDocumentQAPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `anls`, `exact_match`, `normalize_box`,
     `docqa_metrics`, `last_number_baseline`, `keyword_lookup_baseline`, `fetch_corpus`, `read_corpus`,
     `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`, `split_dataset`, `load_byod_dataset`,
     `write_dataset_jsonl`, `gold_texts` and the ceilings) with no import of the repository package and no import of
     `pytesseract`;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 8 manifest entries fetched from `impira/layoutlm-document-qa` at the
     immutable revision on a clean runtime, `verify_snapshot` returning its dict (8 files), and
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory with `source` `local-snapshot`;
   - Section 4: `fetch_corpus` checking both shards' declared size and SHA-256 against the pins, reading only the
     `ground_truth` column of each (100 + 100 rows) and matching the pinned column digests `b499e58a…` /
     `adf8303e…`; the seeded split into 595 / 152 / 229 questions over 119 / 30 / 50 receipts with
     `check_split_disjoint` reporting no shared page and the three dataset digests `d063f100…` / `07e37d51…` /
     `496e39d8…`; `outputs/…_train.jsonl` written; the four dataset refusal probes each raising `ValueError`;
   - Section 5: the fit check dropping nothing; the ceilings (`MIN_IMAGE_SIDE` 1, `MAX_IMAGE_SIDE` 10000,
     `MAX_WORDS` 2000, `MAX_QUESTION_CHARS` 256, `MAX_SEQ_LEN` 512, `DOC_STRIDE` 128, `MAX_ANSWER_TOKENS` 15,
     `BOX_GRID` 1000, `MIN_RECORDS` 8, `MAX_RECORDS` 20000) surfaced; the 850×1100 invoice rendered with 73 words;
     `validate_inputs` writing `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding
     from the box-outside-page probe); `pipe.answer` on the five authored questions with every sanity check `True`
     and the per-page `evaluation_report` verdict `sample-sanity` (the card-pass smoke answered all five exactly —
     `NW-2026-0417`, `Blue Yonder Airlines`, `$1,099.20`, `11 April 2026`, `40`; a different span on another
     runtime is a finding to record, not a failure);
   - Section 6: the last-number baseline (ANLS ≈ 0.41, exact match ≈ 0.25), the keyword-lookup baseline
     (≈ 0.64 / ≈ 0.59) and the frozen model's test score (ANLS ≈ 0.84, exact match ≈ 0.78) with the per-field
     breakdown (`n` and the `too_few_to_read` flag) and `adapted: False`; the comparisons are recorded, not asserted;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 28,353,026 trainable of 127,792,898 parameters,
     595 training questions, and a six-epoch history with validation ANLS rising from about 0.82 to about 0.95 (the
     recorded Kaggle Tesla T4 run of `ad2dea7f`: 0.819 → 0.952, epoch 5 kept; a CPU pre-flight reached 0.960 at
     epoch 5);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison, the per-field
     breakdown, the fields that lost score (the T4 run: `sub_total.discount_price` 1.00 → 0.75 on n = 4), the questions
     where the two models disagree, the reading line and the run history, and `outputs/…_evaluation_report.json`
     written (on the sample ≈ 0.94 versus ≈ 0.84);
   - Section 9: the five invoice questions answered by the adapted model with the `sample-sanity` report,
     `outputs/…_answers.csv` and `outputs/…_annotated.png` written; `pipe.save_artifact` writing
     `outputs/…_adapter/{adapter.safetensors,manifest.json}` (66 tensors, about 113 MB) and
     `LayoutLMDocumentQAPipeline.from_artifact` reloading it with 8/8 identical answers (the cell raises an explained
     error otherwise); Section 10 printing a span and score for each of the three unsupported questions;
     `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity and licence, the snapshot block
     (`weight_format`, `weight_sha256`), the `corpus` block, the inference-contract items, the comparison, the
     artifact digest, the reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable
   `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `layoutlm_document_qa_colab.ipynb` (`E2E`) | `8541181` / `ad2dea7f` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-layoutlm-document-qa` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | **Completed in two passes, not promotion evidence** — pass 1 stopped at the in-kernel install cell (`RuntimeError`: cuda-bindings 12.9.4 → 13.4.2 and numpy 2.0.2 → 2.5.3 changed under loaded modules) and pass 2 ran 11/11 code cells after the executor restarted the kernel; NOTEBOOK_SPEC 2.2 §5 does not accept a manual restart; 20 files, 515 MB staged from the Hub into a clean cache; comparison {anls: {last_number: 0.406, keyword_lookup: 0.635, frozen: 0.843, adapted: 0.94}, exact_match: {last_number: 0.253, keyword_lookup: 0.594, frozen: 0.782, adapted: 0.921}, delta_vs_frozen: {anls: 0.097, exact_match: 0.14}, by_field: {menu.nm: {n: 50, frozen: 0.79, adapted: 0.92}, sub_total.discount_price: {n: 4, frozen: 1, adapted: 0.75}, sub_total.service_price: {n: 2, frozen: 1, adapted: 1}, sub_total.subtotal_price: {n: 27, frozen: 0.98, adapted: 1}, sub_total.tax_price: {n: 17, frozen: 0.92, adapted: 0.94}, total.cashprice: {n: 35, frozen: 0.85, adapted: 0.9}, total.changeprice: {n: 29, frozen: 0.87, adapted: 1}, total.creditcardprice: {n: 7, frozen: 0.96, adapted: 1}, total.menuqty_cnt: {n: 12, frozen: 0.17, adapted: 0.83}, total.total_price: {n: 46, frozen: 0.91, adapted: 0.95}}}; reload parity {identical_answers: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-layoutlm-document-qa/v2/evidence/` in the workspace |
| `layoutlm_document_qa_colab.ipynb` (`E2E`) | generated at `7a9a150` / blob `c068cc6f1b8e` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed, snapshot and corpus cache pre-staged) | PASS — pre-flight only, **not** promotion evidence |
| `layoutlm_document_qa_colab.ipynb` (`TASK-INFERENCE`, superseded) | `232fc8d` / `935148fc5c95` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-layoutlm-document-qa` v1) | PASSED — 8/8 code cells, 239.0 s; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/layoutlm_document_qa_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/layoutlm_document_qa_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-06 | `4c211df` / `c64d5241` | Google Colab CLI 0.7.4, fresh Colab Tesla T4 session (`colab new --gpu T4`, `colab exec -f`, `colab stop`); isolated CPython 3.12.12 environment built from the 48-package hash lock (kernel Python 3.13.15), `torch 2.14.0+cu130`, `transformers 4.57.6`, `cuda:0` | Default sample path (form defaults), sequential execution of every code cell in one kernel with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before the session); not a browser Run all, no execution counts (order from `exec.log`, 1/14 … 14/14) | 235.7 s | **One pass, no restart, 0 errors** — 14/14 code cells; 8 snapshot files fetched and verified, split 595 / 152 / 229 over 119 / 30 / 50 receipts with the pinned digests, baselines 0.406 / 0.635, frozen test ANLS 0.843 (exact match 0.782), validation ANLS 0.8191 → 0.9523 (epoch 5 kept), adapted test ANLS 0.940 (exact match 0.921), `sub_total.discount_price` 1.00 → 0.75 (n = 4), reload parity 8/8; see [Colab CLI execution of revision 4c211df](#colab-cli-execution-of-revision-4c211df--2026-10-06) |
| 2026-09-19 | `8541181` / `ad2dea7f` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-layoutlm-document-qa` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 339.2 s | **Completed in two passes, not promotion evidence** — pass 1 stopped at the in-kernel install cell (`RuntimeError`: cuda-bindings 12.9.4 → 13.4.2 and numpy 2.0.2 → 2.5.3 changed under loaded modules) and pass 2 ran 11/11 code cells after the executor restarted the kernel; NOTEBOOK_SPEC 2.2 §5 does not accept a manual restart; 20 files, 515 MB staged from the Hub into a clean cache; comparison {anls: {last_number: 0.406, keyword_lookup: 0.635, frozen: 0.843, adapted: 0.94}, exact_match: {last_number: 0.253, keyword_lookup: 0.594, frozen: 0.782, adapted: 0.921}, delta_vs_frozen: {anls: 0.097, exact_match: 0.14}, by_field: {menu.nm: {n: 50, frozen: 0.79, adapted: 0.92}, sub_total.discount_price: {n: 4, frozen: 1, adapted: 0.75}, sub_total.service_price: {n: 2, frozen: 1, adapted: 1}, sub_total.subtotal_price: {n: 27, frozen: 0.98, adapted: 1}, sub_total.tax_price: {n: 17, frozen: 0.92, adapted: 0.94}, total.cashprice: {n: 35, frozen: 0.85, adapted: 0.9}, total.changeprice: {n: 29, frozen: 0.87, adapted: 1}, total.creditcardprice: {n: 7, frozen: 0.96, adapted: 1}, total.menuqty_cnt: {n: 12, frozen: 0.17, adapted: 0.83}, total.total_price: {n: 46, frozen: 0.91, adapted: 0.95}}}; reload parity {identical_answers: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-layoutlm-document-qa/v2/evidence/` in the workspace |
| 2026-09-19 | generated at `7a9a150` / blob `c068cc6f1b8e` | Local Windows-venv harness (`run_nb_local.py`: nbclient, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6, snapshot and CORD-v2 column cache pre-staged | Default sample path, all 11 code cells: pinned install skipped (pre-installed), `stage_missing_files` reported nothing to fetch, `verify_snapshot` PASS (8 files), corpus columns read from the pre-staged cache and split 595 / 152 / 229 over 119 / 30 / 50 receipts, fit check dropped nothing, five invoice answers exact (`sample-sanity`), baselines 0.406 / 0.635, frozen test ANLS 0.843 (41.0 s), six epochs 934.3 s (validation ANLS 0.819 → 0.870 → 0.908 → 0.894 → 0.925 → 0.960 → 0.956, epoch 5 kept), adapted test ANLS 0.942 / exact match 0.930, invoice 5/5 after adaptation, adapter 113,419,976 B / 66 tensors, reload parity 8/8, 7 outputs written; the committed blob differs from the executed one in markdown prose only (CPU timing estimates corrected after this run) | 1218.7 s | PASS — pre-flight only; not promotion evidence |

## Current status

**Candidate** — the `E2E` notebook was regenerated on 2026-10-04 to fix the 2026-10-02 Notebook Review Framework v1 findings (LDQ-M1..M3, LDQ-m1..m5; review PR #13): it now builds a uv isolated environment from a hash lock (no in-kernel install, no restart; Linux x86_64 only), starts every adaptation from the pretrained model, checks a BYOD dataset's real minimum and adds the guided layer. The new blob `c64d5241` (`4c211df`) then ran once on a fresh Colab Tesla T4 on 2026-10-06 via the Colab CLI (sequential execution, not a browser Run all): 14/14 code cells in one pass, no restart, 0 errors, 235.7 s, with every printed metric equal to the 2026-09-19 Kaggle T4 run (test ANLS 0.843 → 0.940); the BYOD run with one rejected input (REL12) and the Section 11 activity are still open. The previous blob `ad2dea7f` (`8541181`) ran on a Kaggle Tesla T4 on 2026-09-19 only after a manual restart of the kernel (two passes), which is not promotion evidence; see `docs/release-verification.md`. The local pre-flight rows above remain history.

Facts a reviewer should still weigh: the frozen model is already strong on receipt totals (it was fine-tuned on DocVQA), so the gain is measured per field (item count 0.17 → 0.83, item name 0.79 → 0.92, change 0.87 → 1.00 on the T4 run; overall ANLS 0.843 → 0.940) and is several points, not a rescue; the questions are templated from CORD's field categories, not written by people; CORD's words and boxes are annotations, cleaner than any OCR engine's output on a photographed receipt; and the adapted model's answers on the rendered invoice (5/5 on the T4 run) are one page of evidence about behaviour outside the corpus, not a measurement.

### Review fixes of 2026-10-02 (LDQ-M1..M3, LDQ-m1..m5) — default path executed on Colab T4 2026-10-06 (see below)

Applies to the primary `E2E` notebook. The [review](https://github.com/kurtvalcorza/layoutlm-document-qa-pipeline/pull/13) of `9c5ccfd` (blob `ad2dea7f`) found 3 Major and 5 Minor issues; the fixes are on the review branch of PR #13.

- **Runtime (LDQ-M1).** Generator /2.1 (the fleet's, as in florence2-vision-language-pipeline `9c4e95a`): Section 1 verifies a pinned uv 0.12.15 wheel by size and SHA-256, builds a managed CPython 3.12.12 environment, installs `tutorials/requirements-colab.lock.txt` with `--require-hashes --only-binary :all:` and routes every later cell to one worker there. The lock is florence2-vision-language-pipeline's byte for byte (identical pins; Colab T4 PASS at `9c4e95a`), and its 48 versions and hashes equal this repository's workshop lock (Colab CLI T4 PASS at `cf98b6e`). No kernel install, no restart; **Linux x86_64 only**.
- **Re-runs (LDQ-M2, LDQ-M3).** `reset_to_pretrained()` at the top of Sections 5, 6 and 7 reloads the verified snapshot when the model in memory is adapted; Section 6 refuses an adapted model; `adapt()` refuses an already adapted pipeline. The documented experiments and the BYOD re-run therefore always start from the base, epoch 0 is the frozen model, and the exported adapter reproduces the evaluated model.
- **BYOD (LDQ-m1).** `split_minimums()` (8 / 2 / 2) and `byod_record_limits()` (12 records with one question per page); refusals name the split, the counts and the minimum; a malformed JSONL line is refused with its line number; `BYOD_PATH` reads a file on any runtime; a cancelled upload or a runtime without the Colab dialog gets an actionable message.
- **Prose (LDQ-m2, LDQ-m4, LDQ-m5).** Single braces in the schema and id pattern; one expected validation ANLS quoting the recorded run (0.819 → 0.952, Kaggle Tesla T4) with the run-to-run spread; the discount regression (1.00 → 0.75, n = 4) stated; fields that lose score and model disagreements printed; outcomes recorded instead of asserted; the guided layer added (see `tutorials/README.md`).
- **Local verification (not clean-runtime evidence).** A CPU harness executed the regenerated notebook's own code cells in order in one namespace (torch 2.14.0+cpu, transformers 4.57.6; the real pinned snapshot and the CORD-v2 column cache pre-staged; `DIMER_NOTEBOOK_CI_PREINSTALLED=1` in place of the Linux-only isolated environment), at reduced scale (160 / 40 / 60 records, two epochs at `LEARNING_RATE = 1e-4`): the default pass adapted the model (validation ANLS 0.750 → 0.882, test 0.830 → 0.882); the 2-block experiment re-run from Section 7 reloaded the base, started at epoch 0 = 0.750, exported 34 tensors with every other tensor equal to the base, and reloaded with 8/8 (and 40/40) identical answers; a 60-record BYOD re-run from Section 4 through `BYOD_PATH` (no `google.colab`) scored the frozen model exactly as a fresh base pipeline (test ANLS 0.717, validation 0.900) and reloaded 8/8. Split counts and digests equal the recorded run. Not run here: the uv bootstrap and lock install, the full six-epoch schedule, the upload dialog.
- **Hosted default-path run: passed; optional paths pending.** The new blob ran in one pass with no restart on a fresh Colab T4 (see [Colab CLI execution of revision 4c211df — 2026-10-06](#colab-cli-execution-of-revision-4c211df--2026-10-06)). Still required: the Section 11 activity and one BYOD run with one rejected input (REL12). Status is **Candidate**.

### Colab CLI execution of revision 4c211df — 2026-10-06

Applies to the primary `E2E` notebook only.

- **Evidence.** [`execution-evidence/2026-10-06/layoutlm_document_qa_colab_4c211df_colab-cli-t4.ipynb`](execution-evidence/2026-10-06/layoutlm_document_qa_colab_4c211df_colab-cli-t4.ipynb) (SHA-256 `252d3d49fd4c8a57763d4b21d4ab92fe5d99c955d401dc54617eaa7dacfd3700`), [`run_summary.json`](execution-evidence/2026-10-06/layoutlm_document_qa_colab_4c211df_run_summary.json) (`cbcd3c429e43a69af1501f037e6bcd1c5f937d34e211212d067277ce38b7407b`) and [`exec.log`](execution-evidence/2026-10-06/layoutlm_document_qa_colab_4c211df_exec.log) (`c48a9e58c9401f6879168c7737d0beff0a3d9bd948bef59e3756823e9420d509`), archived byte-for-byte.
- **Revision.** PR #13 head `4c211dfe826a50de14315bd46000842e53975248`, notebook blob `c64d524183cd08fe57d5df04914b08714b7030a8`, downloaded from GitHub at that commit and blob-verified before the session. The executed notebook's 14 code-cell sources equal the committed ones; Section 1 reports `repository_revision` `d14e372…`, equal to `metadata.dimer.generated_from`.
- **Executor.** Google Colab CLI 0.7.4 on a fresh Colab Tesla T4 session via the workspace `colab-cli-serial-test-suite`. Code cells ran in order in one kernel; this is not a browser Run all, and the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log (1/14 … 14/14, sequential and complete). The session was stopped after the run.
- **Path exercised.** Default sample path with the form defaults (`USE_BYOD = False`, `SPLIT_SEED = 42`, `EPOCHS = 6`, `LEARNING_RATE = 3e-5`, `BATCH_SIZE = 16`, `TRAINABLE_ENCODER_LAYERS = 4`). Not exercised: BYOD (upload or `BYOD_PATH`) with one rejected input (REL12), the Section 11 activity and the optional experiments.
- **Environment.** Section 1 built the isolated CPython 3.12.12 environment from the 48-package hash lock in 58 s (kernel Python 3.13.15) and routed every later cell to it; `torch 2.14.0+cu130`, `transformers 4.57.6`, CUDA available, model on `cuda:0` from `local-snapshot`. No restart.
- **Outcome.** **One pass, no restart, 0 errors** — 14/14 code cells; 235.7 s wall for the execution, including the environment build and the downloads. Cells 4–6 (the carried `pipeline.py`, `metrics.py` and `samples.py` modules) print nothing by design; every other cell printed output.
- **Results.** All 8 manifest files fetched from `impira/layoutlm-document-qa` at `beed3c4d…` (513,814,827 B) and verified. Corpus: 100 + 100 rows, column digests `b499e58a…` / `adf8303e…`; split 595 / 152 / 229 questions over 119 / 30 / 50 receipts, digests `d063f100…` / `07e37d51…` / `496e39d8…`; the four refusal probes refused. Fit check dropped nothing; the invoice (850×1100, 73 words) answered `NW-2026-0417`, `Blue Yonder Airlines`, `$1,099.20`, `11 April 2026`, `40` (scores 1.0000 / 1.0000 / 0.9998 / 1.0000 / 0.9994), every check `True`, verdict `sample-sanity`. Test ANLS / exact match: last-number 0.406 / 0.253, keyword lookup 0.635 / 0.594, frozen 0.843 / 0.782 (7.4 s). Adaptation: 28,353,026 of 127,792,898 parameters trainable, validation ANLS 0.8191 → 0.8594 → 0.9023 → 0.9237 → 0.9337 → 0.9523 → 0.9518 (training loss 1.3726 → 0.3775), epoch 5 kept, 78.9 s. Adapted test ANLS 0.940, exact match 0.921 (delta +0.097 / +0.14); per field item count 0.17 → 0.83 (n = 12), first item 0.79 → 0.92 (n = 50), change 0.87 → 1.00 (n = 29); `fields_that_lost_score`: `sub_total.discount_price` 1.00 → 0.75 (n = 4); the two models answered 44 of 229 test questions differently. Invoice after adaptation 5/5 (ANLS 1.0); adapter 66 tensors, 113,419,976 B; reload parity 8/8; 7 outputs written. Section 10 (frozen model): `14 Harbour Road,` (0.2774), `$183.20` (0.9997), `Northwind Traders Ltd.` (0.9892); no empty answers.
- **Comparison with the worked answers and the 2026-09-19 Kaggle T4 run.** Equal at printed precision: baselines, frozen and adapted test scores, the full per-field table, the validation curve quoted in Section 7 (0.819 → 0.859 → 0.902 → 0.924 → 0.934 → 0.952 → 0.952, epoch 5 kept), the discount regression, the invoice answers and scores, and the Section 10 frozen spans and scores. Only timings differ (six epochs 78.9 s against 82 s). No stated direction of change is contradicted.
- **Boundary.** Saved outputs were inspected; BYOD with one rejected input (REL12) and the Section 11 activity remain open. Status remains **Candidate**; `metadata.dimer.clean_runtime_evidence` is unchanged (editing it would change the verified blob).

## Supplemental document question answering workshop — `tutorials/DIMER_Document_QA_LayoutLM_vs_Pix2Struct_Workshop.ipynb`

This entry applies only to the supplemental workshop notebook, not the primary tutorial executions above.

### Maintainer-supplied successful Colab run — 2026-09-26

The maintainer supplied the [executed notebook](execution-evidence/2026-09-26/DIMER_Document_QA_LayoutLM_vs_Pix2Struct_Workshop.ipynb) and authorized merging PR #7 (merge commit `9697355`). The file is archived byte-for-byte, SHA-256 `37b941bf2d09446b953b2d934ff3ec256b434dce42fce7d3940c2128102d6e61`. All 29 code cells have execution counts, 47 saved outputs and zero saved errors. Code-cell sources match commit `fb8415f59f5b0f926d54431c6c16d6d263c4cec2`, tutorial blob `256c9d9af0e68fc7b63803591eacddc2f5c017af`, apart from Colab-inserted `# @title` lines. Later commits on `main` that touch the notebook (`49d9757` (AI Use Disclosure)) change only markdown cells; its code cells are identical to the executed revision. This evidence commit does not change tutorial code.

Scope: Default path: 50 test pages, 229 question-answer records over 10 fields; LayoutLM (OCR-based) against Pix2Struct (OCR-free) with last-number and keyword-lookup baselines. BYOD was not exercised.

Saved runtime: Python 3.13.15, torch 2.14.0+cu130, Transformers 4.57.6, huggingface_hub 0.36.2, pyarrow 25.0.1, NumPy 2.1.3 (preloaded by the host kernel, retained), CUDA Tesla T4. Execution reaches the final completion summary. The separate exported files were not supplied, so their bytes/digests were not independently inspected. Saved counts run sequentially from 1 to 29; runtime freshness and absence of manual restarts/reruns are not independently established by the artifact.

Results (sample-sanity measures on the built-in data, not general model rankings): ANLS / exact match: last-number baseline 0.406 / 0.253, keyword lookup 0.635 / 0.594, LayoutLM 0.843 / 0.782, Pix2Struct 0.801 / 0.659; no empty answers. Mean latency LayoutLM 0.045 s, Pix2Struct 0.666 s; Pix2Struct truncation rate 0.

Status remains **Candidate**. Merge approval and this successful default-path run do not close the optional-path (FULL/BYOD) or REL12 qualification gates, and `metadata.dimer.clean_runtime_evidence` in the notebook stays `pending` as authored (editing it would change the verified blob).

### Review fixes of 2026-10-02 — default path executed on Colab T4 2026-10-03 (see below)

The [2026-10-02 notebook review](reviews/2026-10-02-notebook-review/DIMER_Document_QA_LayoutLM_vs_Pix2Struct_Workshop_Review.md) of `07e4c08` (4 Major, 5 Minor) is addressed on branch `fix/dqa-workshop-review`. The fixes change code cells (BYOD validation and outputs, inline panels, the layout activity, refusal checks, control validation), so the 2026-09-26 Colab run above **no longer covers the notebook once they merge**. Offline checks on CPU only, which are not clean-runtime evidence: the notebook's own cells ran top to bottom with the real pinned LayoutLM checkpoint, the real CORD-v2 annotations (blank stand-in page images), and a stand-in for Pix2Struct; split, test digest, baselines and LayoutLM metrics were identical to the hosted run; BYOD and the activity were exercised with synthetic data. Still required: a fresh Colab T4 Run all of the fix head, plus the layout activity and a labelled BYOD dataset with one rejected input (REL12). Status remains **Candidate**.

### 2026-10-03 uv isolated environment — default path executed on Colab T4 2026-10-03 (see below)

Applies only to the supplemental workshop notebook: blob `67390f7f426fac5e9f5f0d11124bf89337877ff4` (review-fix head `059b250`) → `5dbd6ff874363a02eb2fb41790eb4b4909a01460`.

- **Change.** The in-kernel `pip install` and its "Restart session, then Run all" guard are removed. The notebook carries `tools/document_qa_workshop.py` (every model and evaluation step; the former cell functions moved unchanged except that each model loads inside the step that uses it and results are written under `work/document_qa_workshop/state/`) and `tools/document-qa-workshop-requirements.lock` (`uv pip compile --generate-hashes` for CPython 3.12 on `x86_64-manylinux_2_28`) in carrier cell `uvcarrier`, written by `tools/build_document_qa_workshop_carrier.py` in pieces of at most 1,000 characters. Cell `7984f79d` verifies a pinned uv 0.12.15 wheel by size and SHA-256, builds a CPython 3.12.12 venv (`--managed-python`) and installs with `--require-hashes --only-binary :all:`; every step runs as a separate process with that venv's interpreter (`MPLBACKEND=Agg`; `PYTHONPATH`, `PYTHONHOME`, `PYTHONSTARTUP` and HF tokens dropped). Display cells read the step JSON/CSV/PNG outputs. A warm runtime reuses the venv when its recorded lock hash matches.
- **Pins.** Same direct versions as the former install cell (torch 2.14.0, torchvision 0.29.0, torchaudio 2.11.0, transformers 4.57.6, safetensors 0.8.0, numpy 2.5.3, pillow 11.3.0, huggingface-hub 0.36.2, pyarrow 25.0.1); nothing added. The former exception that kept a NumPy 2.x already loaded by the hosted kernel (2.1.3 in the 2026-09-26 run) no longer applies: the environment always uses numpy 2.5.3.
- **User-visible.** Linux x86_64 only (Google Colab, Kaggle, Linux Jupyter); Windows and macOS kernels are refused with a message. The first run builds the environment (several minutes). No split, seed, model, metric or output file changes.
- **Local verification (not clean-runtime evidence).** A CPU harness on Windows (CUDA hidden) executed every code cell in order, each step as a real subprocess. Stand-ins: the platform check was patched; this host's interpreter (torch 2.9.1, not the lock) replaced the uv venv; the real pinned CORD-v2 annotations with blank page images replaced the shard download; Pix2Struct used its real pinned weights but answered blank pages, so its scores are not meaningful. Split 119 / 30 / 50 pages and 595 / 152 / 229 questions, test digest `b2c58a97…`, last-number baseline 0.406 / 0.253, keyword lookup 0.635 / 0.594 and LayoutLM 0.843 / 0.782 (ANLS / exact match) equal the 2026-09-26 hosted run exactly; all 11 exports were written. The layout activity and a labelled synthetic BYOD archive ran after the default path, and a rejected archive stopped with a message naming the record and the rule. `pytest` 229 passed / 2 skipped before, 255 passed / 2 skipped after (same two skips). The hash-locked install was dry-run for the Linux target (`uv pip install --dry-run --require-hashes --only-binary :all:`).
- **Hosted default-path run: passed; optional paths pending.** The hosted Colab T4 run of this blob passed on the default path, including the uv bootstrap and the hash-locked install (see [Colab CLI execution of revision cf98b6e — 2026-10-03](#colab-cli-execution-of-revision-cf98b6e--2026-10-03)). Still pending: the layout activity, a labelled BYOD dataset and one rejected input (REL12). Status remains **Candidate**.

### Colab CLI execution of revision cf98b6e — 2026-10-03

- **Evidence.** [`execution-evidence/2026-10-03/DIMER_Document_QA_LayoutLM_vs_Pix2Struct_Workshop_cf98b6e_colab-cli-t4.ipynb`](execution-evidence/2026-10-03/DIMER_Document_QA_LayoutLM_vs_Pix2Struct_Workshop_cf98b6e_colab-cli-t4.ipynb), archived byte-for-byte, SHA-256 `2f94d5d26231ed529abaa9b6d14462f2197bbefdb1db3d362724aece9eb89f24`.
- **Revision.** Branch head `cf98b6e48fd6f68e110e6407cfca84439f68679c`, notebook blob `5dbd6ff874363a02eb2fb41790eb4b4909a01460`, downloaded from GitHub at the PR head and blob-verified before the session. It includes the 2026-10-02 review fixes above.
- **Executor.** Google Colab CLI 0.7.4 on a fresh Colab Tesla T4 session via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`). Code cells ran in order in one kernel; this is not a browser Run all, and the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log (1/33 … 33/33, sequential and complete). Session stopped after the run.
- **Path exercised.** Default path with the paraphrase experiment, unanswerable probe and modality-robustness checks enabled (all on by default), plus the LayoutLM and Pix2Struct refusal checks (6 refused as expected). The layout activity was not run (`Activity not run.`); BYOD was not run (`BYOD disabled on the canonical Run all path.`).
- **Environment.** The uv bootstrap succeeded on the hosted runtime: carried stage file and hash lock verified; an isolated CPython 3.12.12 venv was built and the hashed lock installed, reporting `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `transformers 4.57.6`, `huggingface_hub 0.36.2`, `pyarrow 25.0.1`, `numpy 2.5.3` (hash-locked; the 2026-09-26 run used the host kernel's 2.1.3 on Python 3.13.15), `pillow 11.3.0`, `cuda:0` Tesla T4. No restart. Every model step ran in its own venv process. The saved outputs carry no per-cell timings; the venv build and install fall inside the 418.4 s wall time of the whole run.
- **Outcome.** **PASSED** — 33/33 code cells, no errors; 418.4 s wall.
- **Results.** Split 119 / 30 / 50 pages, 595 / 152 / 229 questions, test digest `b2c58a97…`. ANLS / exact match: last-number baseline 0.406 / 0.253, keyword lookup 0.635 / 0.594, LayoutLM 0.843 / 0.782, Pix2Struct 0.801 / 0.659 (0.8012455 / 0.6593886); no empty answers; answer in OCR 0.996 (LayoutLM) and 0.812 (Pix2Struct, 43 of 229 answers not a contiguous OCR span). Agreement: both exact 129, LayoutLM only 50, Pix2Struct only 22, neither 28. Paraphrase subset (20 records): LayoutLM 0.913 → 0.863, stability 0.95; Pix2Struct 0.871 → 0.871, stability 1.0. Modality robustness: LayoutLM 0.913 canonical → 0.676 without layout; Pix2Struct 0.871 → 0.764 degraded. Pix2Struct mean new tokens 7.52, max 16, truncation rate 0; LayoutLM 229/229 one window. Mean latency LayoutLM 0.033 s, Pix2Struct 0.624 s.
- **Comparison with the 2026-09-26 run.** Every metric above is equal to the 2026-09-26 hosted run at full printed precision, including the Pix2Struct figures, the per-field table, agreement counts, paraphrase and robustness means and the unanswerable-probe answers (10 of 10 identical). Only resource figures differ: mean latency 0.033 s against 0.045 s (LayoutLM) and 0.624 s against 0.666 s (Pix2Struct), and load seconds, as expected between sessions. This run is the first hosted run of the Pix2Struct step in its own uv-environment process; the earlier local check could not score it (blank stand-in pages). The 2026-09-26 run printed an `HF_TOKEN` secret-timeout warning; this run printed none. The inline panels and the not-in-OCR table added by the review fixes have no 2026-09-26 counterpart.
- **Boundary.** Saved outputs were inspected; the journeys not exercised (layout activity, BYOD with one rejected input/REL12) remain open. Status remains **Candidate**; `metadata.dimer.clean_runtime_evidence` is unchanged.
