# Release verification

`tutorials/layoutlm_document_qa_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
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
  on the frozen model and on the validation and test splits after adaptation with the ANLS assertions, `pipe.adapt`
  with its explicit hyperparameters, `evaluation_report` on the invoice after adaptation, `pipe.save_artifact`,
  `LayoutLMDocumentQAPipeline.from_artifact` and the reload-parity assertion, and the provenance fields
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
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
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
   `pillow==11.3.0`, `huggingface-hub==0.36.2`, `pyarrow==25.0.1` (an interpreter restart after the install is
   expected where the runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
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
     (≈ 0.64 / ≈ 0.59) and the frozen model's test score (ANLS ≈ 0.84, exact match ≈ 0.78 on CPU float32) with the
     per-field breakdown, and the cell's assertion that the frozen ANLS beats the last-number baseline;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 28,353,026 trainable of 127,792,898 parameters,
     595 training questions, and a six-epoch history with validation ANLS rising (≈ 0.82 → ≈ 0.94 in the build
     record's runs; `best_epoch` in the last epochs);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison, the per-field
     breakdown and `outputs/…_evaluation_report.json` written (the cell asserts the adapted test ANLS exceeds the
     frozen one — on the sample ≈ 0.94 versus ≈ 0.84);
   - Section 9: the five invoice questions answered by the adapted model with the `sample-sanity` report,
     `outputs/…_answers.csv` and `outputs/…_annotated.png` written; `pipe.save_artifact` writing
     `outputs/…_adapter/{adapter.safetensors,manifest.json}` (66 tensors, about 113 MB) and
     `LayoutLMDocumentQAPipeline.from_artifact` reloading it with 8/8 identical answers (the cell asserts it);
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
| `layoutlm_document_qa_colab.ipynb` (`E2E`) | `8541181` / `ad2dea7f` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-layoutlm-document-qa` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 20 files, 515 MB staged from the Hub into a clean cache; comparison {anls: {last_number: 0.406, keyword_lookup: 0.635, frozen: 0.843, adapted: 0.94}, exact_match: {last_number: 0.253, keyword_lookup: 0.594, frozen: 0.782, adapted: 0.921}, delta_vs_frozen: {anls: 0.097, exact_match: 0.14}, by_field: {menu.nm: {n: 50, frozen: 0.79, adapted: 0.92}, sub_total.discount_price: {n: 4, frozen: 1, adapted: 0.75}, sub_total.service_price: {n: 2, frozen: 1, adapted: 1}, sub_total.subtotal_price: {n: 27, frozen: 0.98, adapted: 1}, sub_total.tax_price: {n: 17, frozen: 0.92, adapted: 0.94}, total.cashprice: {n: 35, frozen: 0.85, adapted: 0.9}, total.changeprice: {n: 29, frozen: 0.87, adapted: 1}, total.creditcardprice: {n: 7, frozen: 0.96, adapted: 1}, total.menuqty_cnt: {n: 12, frozen: 0.17, adapted: 0.83}, total.total_price: {n: 46, frozen: 0.91, adapted: 0.95}}}; reload parity {identical_answers: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-layoutlm-document-qa/v2/evidence/` in the workspace |
| `layoutlm_document_qa_colab.ipynb` (`E2E`) | generated at `7a9a150` / blob `c068cc6f1b8e` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed, snapshot and corpus cache pre-staged) | PASS — pre-flight only, **not** promotion evidence |
| `layoutlm_document_qa_colab.ipynb` (`TASK-INFERENCE`, superseded) | `232fc8d` / `935148fc5c95` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-layoutlm-document-qa` v1) | PASSED — 8/8 code cells, 239.0 s; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/layoutlm_document_qa_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/layoutlm_document_qa_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-19 | `8541181` / `ad2dea7f` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-layoutlm-document-qa` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 339.2 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 20 files, 515 MB staged from the Hub into a clean cache; comparison {anls: {last_number: 0.406, keyword_lookup: 0.635, frozen: 0.843, adapted: 0.94}, exact_match: {last_number: 0.253, keyword_lookup: 0.594, frozen: 0.782, adapted: 0.921}, delta_vs_frozen: {anls: 0.097, exact_match: 0.14}, by_field: {menu.nm: {n: 50, frozen: 0.79, adapted: 0.92}, sub_total.discount_price: {n: 4, frozen: 1, adapted: 0.75}, sub_total.service_price: {n: 2, frozen: 1, adapted: 1}, sub_total.subtotal_price: {n: 27, frozen: 0.98, adapted: 1}, sub_total.tax_price: {n: 17, frozen: 0.92, adapted: 0.94}, total.cashprice: {n: 35, frozen: 0.85, adapted: 0.9}, total.changeprice: {n: 29, frozen: 0.87, adapted: 1}, total.creditcardprice: {n: 7, frozen: 0.96, adapted: 1}, total.menuqty_cnt: {n: 12, frozen: 0.17, adapted: 0.83}, total.total_price: {n: 46, frozen: 0.91, adapted: 0.95}}}; reload parity {identical_answers: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-layoutlm-document-qa/v2/evidence/` in the workspace |
| 2026-09-19 | generated at `7a9a150` / blob `c068cc6f1b8e` | Local Windows-venv harness (`run_nb_local.py`: nbclient, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6, snapshot and CORD-v2 column cache pre-staged | Default sample path, all 11 code cells: pinned install skipped (pre-installed), `stage_missing_files` reported nothing to fetch, `verify_snapshot` PASS (8 files), corpus columns read from the pre-staged cache and split 595 / 152 / 229 over 119 / 30 / 50 receipts, fit check dropped nothing, five invoice answers exact (`sample-sanity`), baselines 0.406 / 0.635, frozen test ANLS 0.843 (41.0 s), six epochs 934.3 s (validation ANLS 0.819 → 0.870 → 0.908 → 0.894 → 0.925 → 0.960 → 0.956, epoch 5 kept), adapted test ANLS 0.942 / exact match 0.930, invoice 5/5 after adaptation, adapter 113,419,976 B / 66 tensors, reload parity 8/8, 7 outputs written; the committed blob differs from the executed one in markdown prose only (CPU timing estimates corrected after this run) | 1218.7 s | PASS — pre-flight only; not promotion evidence |

## Current status

**Release-grade.** The `E2E` notebook blob `ad2dea7f` (committed at `8541181`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok (1 restart after install cell), 339.2 s, 20 files, 515 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.

Facts a reviewer should still weigh: the frozen model is already strong on receipt totals (it was fine-tuned on DocVQA), so the gain is measured per field (item count 0.17 → 0.83, item name 0.79 → 0.92, change 0.87 → 1.00 on the T4 run; overall ANLS 0.843 → 0.940) and is several points, not a rescue; the questions are templated from CORD's field categories, not written by people; CORD's words and boxes are annotations, cleaner than any OCR engine's output on a photographed receipt; and the adapted model's answers on the rendered invoice (5/5 on the T4 run) are one page of evidence about behaviour outside the corpus, not a measurement.
