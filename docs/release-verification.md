# Release verification

`tutorials/layoutlm_document_qa_colab.ipynb` (`TASK-INFERENCE`, **standalone** carrier) is a
**release candidate** until the exact notebook revision has executed top-to-bottom in a clean
supported runtime. Unit tests, JSON validation, code-cell compilation, the generator parity checks
and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under
DIMER Notebook Specification 2.0. This file is the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that
  profile, spec `2.0`, a pedagogical mode, `standalone: true` and `generated_from` (repository, revision, module
  SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on
  the primary path; exactly one cell tagged `embedded_module` equal to
  `src/layoutlm_document_qa_pipeline/pipeline.py` after the generator's documented rewrites; the
  inline `MANIFEST` equal to the committed snapshot manifest and the inline `PINS` equal to the
  `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to `tools/build_notebook.py`
  output for its recorded revision; the pinned-install cell with its restart-on-stale-import guard;
  `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline
  manifest, which the notebook asserts against the module before fetching), the revision is a 40-hex
  immutable commit, and the same identity string appears in `README.md`, `MODEL_CARD.md`, and
  `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `LayoutLMDocumentQAPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `answer`,
  `evaluation_report`, the `ocr_words_with_tesseract` adapter on the BYOD branch), the ceiling print
  (`MIN_IMAGE_SIDE`, `MAX_IMAGE_SIDE`, `MAX_WORDS`, `MAX_QUESTION_CHARS`, `MAX_SEQ_LEN`, `DOC_STRIDE`,
  `MAX_ANSWER_TOKENS`, `BOX_GRID`), the exports, the learner-facing statements (the model never sees
  pixels, the span score is an uncalibrated within-page product of softmaxes, the model always returns
  a span, no DocVQA benchmark, ANLS as sanity check, no OCR installed, capability exclusions) and the
  gated-off BYOD default listed in the validator; forbidden patterns (credential-in-URL, any `git clone`
  / `github.com` / repository import on the primary path, a mutable `revision='main'`, direct
  `from transformers import` / `LayoutLMForQuestionAnswering` / `AutoTokenizer` / `import pytesseract`
  / `start_logits` / `from huggingface_hub import` use **outside the carried module cell**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `numpy` and
`pillow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit
suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`;
injected runner, no weights). These are source/provenance and unit checks. They are **not** execution
evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (**no repository checkout is needed — the notebook is standalone**) |
| Local Windows-venv harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded
   in `metadata.dimer.generated_from` and that the installed core package versions equal the inline
   `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines `LayoutLMDocumentQAPipeline`, `validate_inputs`,
     `evaluation_report`, `anls`, `exact_match`, `normalize_box`, `ocr_words_with_tesseract`,
     `verify_snapshot`, `stage_missing_files`) with no import of the repository package and no import of
     `pytesseract` on the default path;
   - synthetic 850×1100 invoice-style form rendered in code with 73 words and their renderer-recorded
     boxes, its RGB SHA-256 printed and the ceilings (`MIN_IMAGE_SIDE` 1, `MAX_IMAGE_SIDE` 10000,
     `MAX_WORDS` 2000, `MAX_QUESTION_CHARS` 256, `MAX_SEQ_LEN` 512, `DOC_STRIDE` 128,
     `MAX_ANSWER_TOKENS` 15, `BOX_GRID` 1000) surfaced;
   - pinned `impira/layoutlm-document-qa` acquisition at the immutable revision through the carried
     module: the inline `MANIFEST` is asserted against the module identity and written to
     `weights/layoutlm-document-qa/`, `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports
     all 8 manifest entries on a clean runtime, `verify_snapshot` returns its summary dict, and
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loads from the verified directory;
   - `validate_inputs` writes `outputs/layoutlm_document_qa_input_manifest.json` (verdict `accepted`,
     73 words, five checked questions, one recorded rejection finding from the box-outside-page probe);
   - `answer` returning one span per question with `n_windows` 1; record the answers and scores (the
     card-pass CPU smoke answered all five authored questions exactly — `NW-2026-0417`,
     `Blue Yonder Airlines`, `$1,099.20`, `11 April 2026`, `40` — at scores 0.999–1.000; a materially
     different result is a finding to record, not a failure by itself, because no metric is asserted);
   - `evaluation_report` writes `outputs/layoutlm_document_qa_evaluation_report.json` with verdict
     `sample-sanity`, an `anls` entry, an `exact_match` entry and five per-question entries on the
     synthetic sample (`not-measurable` on BYOD), stated as such;
   - `outputs/layoutlm_document_qa_result.json`, `outputs/layoutlm_document_qa_answers.csv` and
     `outputs/layoutlm_document_qa_annotated.png` written with `NOTEBOOK_SOURCE`, model revision, model
     licence, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/layoutlm_document_qa_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/layoutlm_document_qa_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Local pre-flight evidence (not a supported runtime)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | notebook blob `7f7c851fdc14` (commit `a757ef7`, generated at `da06cfb`; `NOTEBOOK_SOURCE.repository_revision` = `da06cfb…`) | Local Windows-venv harness (`run_nb_local.py`: nbclient 0.11.0, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6; no `pytesseract` in the venv | Default synthetic path, all 8 code cells: pinned install skipped (pre-installed), `stage_missing_files` fetched all 8 manifest entries (514 MB) from the Hub cache at the pinned revision into the scratch `weights/`, `verify_snapshot` PASS (8 files), renderer supplied 73 words + boxes (no OCR call), five `answer` calls → `NW-2026-0417`, `Blue Yonder Airlines`, `$1,099.20`, `11 April 2026`, `40` at scores 0.999–1.000, `n_windows` 1, `evaluation_report` `sample-sanity` (`anls` 1.0, `exact_match` 1.0), 5 outputs written | 57.0 s | PASS — pre-flight only; not promotion evidence |

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `232fc8d` / `935148fc5c95` | Kaggle CPU (`kurtvalcorza/dimer-nb2-layoutlm-document-qa` v1) | Default sample path | 239.0 s | **PASSED** — 8/8 ok code cells executed cleanly, 18 files, 514 MB staged |

## Current status

No clean-runtime execution in a **supported** runtime (Colab or Kaggle) has been recorded yet; clean execution evidence is now recorded below. What exists: static validation (`tools/validate_release_assets.py`), the generator parity
checks (`--check` OK), the offline unit suite, and one **local fresh-kernel execution** of the generated
notebook (table above) that exercised the standalone carrier, the real `hf_hub_download` staging path
into an empty `weights/` directory, verification, answering, the evaluation report and every export —
which is necessary but not promotion evidence because the workstation is not a supported runtime. The
registry status remains **Candidate** until a reviewer confirms a recorded supported-runtime run against
the notebook blob under review and an integrator promotes it. Facts a reviewer should weigh: the CUDA
path has not been executed; the default path's OCR is the renderer's own word boxes, so the 5/5 result
measures the pipeline plumbing on perfect OCR and says nothing about Tesseract or any other OCR on real
scans — no OCR is installed or exercised anywhere in this repository; the span score is uncalibrated and
an unanswerable question ("What is the delivery address?") still received `14 Harbour Road,` at 0.28 in
the smoke run, so any rejection threshold is the deployment's to validate; and a 657-word page was
answered across 4 overlapping windows in 0.47 s with the same span, which is the only long-page evidence.