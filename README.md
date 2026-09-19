# LayoutLM document question answering pipeline (impira)

DIMER inference and fine-tuning wrapper for **`impira/layoutlm-document-qa`**, Impira's LayoutLM (v1) encoder fine-tuned on SQuAD 2.0 and DocVQA for extractive question answering over documents, pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The model reads a question plus the page's **OCR words and boxes** — never the pixels — and returns the best answer span with its word indices and an (uncalibrated) span score. OCR is therefore an input the caller provides: bring your own words and boxes, or use the optional `ocr_words_with_tesseract` adapter when `pytesseract` and a Tesseract binary exist on the host. Nothing here installs Tesseract. The repository adds a bounded adaptation contract on top: a digest-pinned real receipt corpus (CORD-v2, read one column at a time), mean ANLS and exact match with two non-neural baselines, supervised fine-tuning of the last encoder blocks and the span head with validation-ANLS epoch selection, and a safetensors adapter that reloads onto the digest-verified base.

## Upstream alignment

- Model: `impira/layoutlm-document-qa`
- Revision: `beed3c4d02d86017ebca5bd0fdf210046b907aa6`
- Upstream weight license: MIT
- Upstream task: document question answering (extractive, over OCR tokens with 2-D positions)
- Repository adaptation: bounded supervised fine-tuning of the last *k* encoder blocks plus the span head (`adapt`, default 4 of 12 = 28,353,026 of 127,792,898 parameters) on caller-supplied or pinned CORD-v2 records; the word, position and 2-D box embeddings and the vocabulary are never modified; the adapter carries only the trained tensors and is bound to the base `model.safetensors` SHA-256

## Quick start

```python
from layoutlm_document_qa_pipeline import LayoutLMDocumentQAPipeline, anls

pipe = LayoutLMDocumentQAPipeline.from_pretrained()      # stages + verifies weights/layoutlm-document-qa first
words = ["Invoice", "number:", "NW-2026-0417", "Customer:", "Blue", "Yonder", "Airlines"]   # from your OCR
boxes = [[70, 200, 140, 218], [146, 200, 220, 218], [300, 200, 420, 218], [70, 296, 160, 314], [300, 296, 340, 314], [346, 296, 410, 314], [416, 296, 490, 314]]
result = pipe.answer("Who is the customer?", words=words, boxes=boxes, image_size=(850, 1100))
print(result["answer"], result["start"], result["end"], round(result["score"], 3))   # 'Blue Yonder Airlines' 4 6 …

# optional OCR adapter (needs pytesseract + a Tesseract binary; not part of the pinned runtime)
# from layoutlm_document_qa_pipeline import ocr_words_with_tesseract
# words, boxes = ocr_words_with_tesseract(Image.open("page.png"))

print(anls(result["answer"], ["Blue Yonder Airlines"]))   # DocVQA's ANLS against an accepted answer
```

Adaptation on the pinned CORD-v2 sample (CPU, about sixteen minutes of training after the snapshot is staged):

```python
from layoutlm_document_qa_pipeline import (
    LayoutLMDocumentQAPipeline, fetch_sample_dataset, check_split_disjoint, last_number_baseline, keyword_lookup_baseline,
)

splits = fetch_sample_dataset()            # the ground_truth column of two pinned Hub shards (~0.4 MB), digest-verified, cached under weights/cord-v2/
check_split_disjoint(splits)               # 595 / 152 / 229 questions over 119 / 30 / 50 receipts, no page shared between splits
pipe = LayoutLMDocumentQAPipeline.from_pretrained()
fit = {name: pipe.check_fit(part)['fitting'] for name, part in splits.items()}   # drops records that need more than one window (none here)
print(keyword_lookup_baseline(fit['test'])['anls'], pipe.evaluate(fit['test'])['anls'])   # 0.635, 0.843 in the recorded run
pipe.adapt(fit['train'], fit['validation'])                                               # last 4 encoder blocks + span head, 6 epochs, best validation ANLS kept
print(pipe.evaluate(fit['test'])['anls'])                                                 # 0.942 in the recorded run
artifact = pipe.save_artifact('outputs/adapter')                                          # adapter.safetensors (~113 MB) + manifest.json
again = LayoutLMDocumentQAPipeline.from_artifact(artifact)                                # verifies base digest + artifact digest before applying
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the offline test suite (no weights needed). On a fresh clone the manifest is committed but the weights are not: `LayoutLMDocumentQAPipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

`answer(question, *, words, boxes, image_size)` takes a non-empty question of at most 256 characters, 1..2,000 non-empty words with one pixel `[x0, y0, x1, y1]` box each inside the page, and the page size; pages over 512 tokens are answered window by window (stride 128) and every result carries `answer`, `start`/`end` (inclusive word indices), `score`, `n_words`, `n_windows`, `device`, `source`, `model_id` and `model_revision`. `evaluate(records)` answers a validated dataset and reports mean ANLS, the exact-match rate and the empty rate (`measured` / `measured-small-sample`); `check_fit(records)` drops — never truncates — the records that would need more than one window; `adapt(train, val, *, epochs=6, lr=3e-5, batch_size=16, trainable_encoder_layers=4, seed=0)` fine-tunes the last encoder blocks and the span head and keeps the best-validation-ANLS epoch; `save_artifact` / `from_artifact` export and reload the trained tensors as safetensors with a manifest bound to the base weight digest. Dataset helpers (`fetch_corpus`, `read_corpus`, `page_from_ground_truth`, `questions_for_page`, `build_sample_dataset`, `validate_dataset`, `split_dataset`, `check_split_disjoint`, `load_byod_dataset`, `write_dataset_jsonl`) live in `samples.py`; `docqa_metrics` and the baselines (`last_number_baseline`, `keyword_lookup_baseline`) in `metrics.py`; records are 8–20,000 mappings `{id, page_id, question, words, boxes, image_size, answer_start, answer_end}` whose gold span lies inside the words, and every ceiling is a refusal, never a silent cut.

## Weights layout

```
weights/layoutlm-document-qa/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256 (8 files)
  config.json                # LayoutLMForQuestionAnswering, 12 layers, max_position_embeddings 514
  merges.txt  vocab.json  tokenizer.json  tokenizer_config.json  special_tokens_map.json   # RoBERTa BPE
  model.safetensors          # git-ignored, 511,200,628 bytes
  README.md
weights/cord-v2/             # git-ignored: the decoded ground_truth columns of the two pinned shards (~1.2 MB), digest-checked on read
```

`pytorch_model.bin` and `tf_model.h5` exist upstream and are deliberately not listed.

## Input ceilings and request parameters

`MAX_WORDS = 2000` words with one pixel `[x0, y0, x1, y1]` box each inside the page (`MIN_IMAGE_SIDE = 1`, `MAX_IMAGE_SIDE = 10000`), normalised to the `BOX_GRID = 1000` grid; `MAX_QUESTION_CHARS = 256`; encoding in `MAX_SEQ_LEN = 512`-token windows with `DOC_STRIDE = 128` overlap; answer spans of at most `MAX_ANSWER_TOKENS = 15` tokens (the transformers document-question-answering pipeline's conventions). No score threshold is applied; the caller owns one. Training records must fit one window (`check_fit`). See `MODEL_CARD.md` for the score semantics, the OCR boundary and the measured CPU timings.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: they use an injected fake runner, window counter and corpus fetcher plus temporary manifests, never the weights (36 tests plus 5 notebook-parity tests). `tests/test_model_backed.py` (6 tests: `evaluate` against gold spans, a one-epoch adaptation of the last encoder block with an artifact round trip, the final-epoch policy, the loader's scope check, the transactional guarantee, and — where CUDA is visible — answer, adaptation and reload on the accelerator) runs only when `weights/layoutlm-document-qa/` is staged.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/layoutlm-document-qa-pipeline/blob/main/tutorials/layoutlm_document_qa_colab.ipynb)

`tutorials/layoutlm_document_qa_colab.ipynb` is declared `E2E` (mode `GUIDED`) under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the three pipeline modules, model identity, manifest digests and runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`; see `tutorials/README.md`). Its default path reads the `ground_truth` column of two digest-pinned CORD-v2 shards from the Hub (about 0.4 MB, CC BY 4.0, refused on any size, file-digest or column-digest mismatch), turns the 199 unique receipts into 595 / 152 / 229 templated gold-span questions over 119 / 30 / 50 page-disjoint receipts with four refusal probes, stages the git-ignored `model.safetensors` with `stage_missing_files(..., allow_download=True)` and digest-verifies the snapshot, runs the fit check (nothing dropped), exercises the inference contract with its input manifest and sanity checks on a rendered invoice, scores the last-number and keyword-lookup baselines and the frozen model on the test receipts (ANLS 0.406 / 0.635 / 0.843 in the recorded run), fine-tunes the last four encoder blocks and the span head for six epochs with validation-ANLS epoch selection (934.3 s on CPU), re-scores the test receipts (ANLS 0.942, exact match 0.782 → 0.930) with a per-field breakdown, re-reads the invoice with the adapted model, exports a ~113 MB safetensors adapter and reloads it with 8/8 identical answers. Every number is one seeded split with no dispersion estimate. BYOD (`{id, page_id, question, words, boxes, image_size, answer_start, answer_end}` as a JSON array or JSONL) is optional and gated off by default. See `docs/release-verification.md` for the release gate.

## Release status

**Release-grade** — the `E2E` notebook blob `ad2dea7f` (committed at `8541181`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok (1 restart after install cell), 339.2 s); the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted run is. A later change to the carried modules or the notebook returns the status to Candidate until re-verified.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests, input/output and adaptation contract, measured runtime.
- `docs/WEIGHTS.md` — weight provenance, OCR boundary, adapter and corpus notes.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are MIT; the tutorial corpus is CC BY 4.0 and is not redistributed; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
