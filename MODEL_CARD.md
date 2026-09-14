---
license: mit
model_card_spec: "1.1"
pipeline_tag: document-question-answering
base_model: impira/layoutlm-document-qa
date_published: "2022-08-07"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt` 2022-08-07T21:07:19Z, https://huggingface.co/api/models/impira/layoutlm-document-qa); the LayoutLM architecture paper is arXiv:1912.13318 (2019-12) and the pinned revision is the Hub's `main` as of 2026-09-14"
---

# LayoutLM Document QA, impira — Extractive Document Question Answering (Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-impira%2Flayoutlm--document--qa-ffcc4d?style=flat)](https://huggingface.co/impira/layoutlm-document-qa)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-microsoft%2Funilm%20(layoutlm)-181717?style=flat&logo=github&logoColor=white)](https://github.com/microsoft/unilm/tree/master/layoutlm)
[![arXiv Paper](https://img.shields.io/badge/arXiv-1912.13318-b31b1b.svg)](https://arxiv.org/abs/1912.13318)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — stage and verify the pinned upstream revision in a fresh runtime, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/layoutlm-document-qa-pipeline/blob/main/tutorials/layoutlm_document_qa_colab.ipynb) [`layoutlm_document_qa_colab.ipynb`](https://github.com/kurtvalcorza/layoutlm-document-qa-pipeline/blob/main/tutorials/layoutlm_document_qa_colab.ipynb)  
  *Five authored questions over an invoice-style form rendered in code — whose renderer supplies the words and boxes, so no OCR runs — with the pinned `impira/layoutlm-document-qa` weights: one answer span per question with its word indices and span score, and `anls` / `exact_match` against the authored answers as sanity evidence only — no DocVQA benchmark.*

---

#### Description

`impira/layoutlm-document-qa` is Impira's fine-tune of Microsoft's LayoutLM (v1) for extractive question answering over documents — "LayoutLM: Pre-training of Text and Layout for Document Image Understanding" (Xu et al., arXiv:1912.13318) fine-tuned, per the snapshot README, on SQuAD 2.0 and DocVQA — pinned here to revision `beed3c4d02d86017ebca5bd0fdf210046b907aa6` (the Hub's `main` on 2026-09-14). The snapshot `config.json` declares `LayoutLMForQuestionAnswering`: a 12-layer BERT-style encoder (hidden size 768, 12 heads, intermediate width 3072) whose input embeddings add 2-D position embeddings for each token's box on a 0–1000 grid (`max_2d_position_embeddings` 1024) to a RoBERTa-style token embedding (vocabulary 50,265, `max_position_embeddings` 514, RoBERTa byte-level BPE tokenizer with `<s>`/`</s>` specials) and a span head predicting start and end logits — about 128M parameters in the 511 MB float32 `model.safetensors`. **The model reads words and boxes, not pixels**: the caller supplies the page's OCR tokens with their pixel boxes, the pipeline normalises the boxes to the grid, encodes `<s> question </s></s> words </s>` (question tokens and specials at box `[0,0,0,0]`, separators at `[1000]*4`, word tokens at their boxes — the transformers document-question-answering pipeline's convention), splits documents longer than 512 tokens into overlapping windows (stride 128), and selects the span of at most 15 tokens whose start×end softmax product is highest across windows, returning it as a word span with indices. Nothing is trained or adapted here. What this repository adds is packaging: `verify_snapshot` and `stage_missing_files` (manifest digest checking and fresh-clone staging), `LayoutLMDocumentQAPipeline.from_pretrained` (verified local loading with `trust_remote_code=False`), `answer` (input validation, box normalisation, windowing, span decoding), `normalize_box`, the optional `ocr_words_with_tesseract` adapter (imports `pytesseract` only when called; not part of the pinned runtime), `normalize_answer`, `anls`, `exact_match`, and the `validate_inputs` and `evaluation_report` stage helpers.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is extractive document question answering over OCR output: input one question (up to 256 characters), the page's words (1–2000) with one pixel `[x0, y0, x1, y1]` box per word and the page size; output the best answer span as text, its inclusive start/end word indices (so the answer's boxes are known), the span score and the number of windows used. Envisioned applications are field extraction from forms, invoices, receipts, letters and reports that already pass through an OCR engine — the DocQuery use case Impira built the model for — where the fields can be phrased as questions and the returned indices localise the answer on the page for review; triage of scanned business documents; and interactive lookup over an OCR'd page. Within DIMER the pipeline is an inference component and a zero-configuration baseline for OCR-based document QA, not a certified extractor for any specific document family, and it ships no OCR.

###### Primary Intended Users

Intended users are machine-learning engineers, document-processing developers, and data analysts integrating document question answering into research prototypes, internal enterprise document tooling, or the DIMER workbench, who already have an OCR stage (Tesseract, a cloud OCR, a PDF text layer with positions). A user is expected to understand that the answer can only ever be a contiguous span of the words they supplied — a mis-read, split or mis-ordered OCR word propagates unchanged — that the span score is the product of two within-window softmaxes under the model's own head, a ranking signal that is not calibrated and says nothing about whether the page can answer the question (the model always returns its best span), that boxes must be in the page's pixel frame and are coarsened to a 1000-step grid, that pages longer than 512 tokens are answered window by window, that the fine-tuning data is English business documents so other languages, handwriting and unusual layouts are distribution shifts, and that accuracy can only be measured on labelled question/answer pairs with the deployment's own OCR. Users who need OCR, abstention, multi-page reasoning, computed answers or batch throughput are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** no OCR (words and boxes are inputs; the Tesseract adapter is optional and unpinned), no abstention or "no answer" output (the model returns its best span for every question), no answers that are not a contiguous span of the supplied words, no arithmetic, comparison or multi-hop reasoning, no multi-page or PDF handling (one page per call), no batching, and no confidence beyond the uncalibrated span score.
2. **Input boundary:** `answer` rejects non-string or empty questions and questions longer than `MAX_QUESTION_CHARS = 256`, more than `MAX_WORDS = 2000` words or empty words, box lists that do not match the words, boxes that are not four numbers, boxes outside the page or inverted, and page sizes outside `MIN_IMAGE_SIDE = 1`..`MAX_IMAGE_SIDE = 10000` px or not integer pairs (`TypeError`/`ValueError`). Documents beyond 512 tokens are windowed with a 128-token stride, so an answer that straddles a window boundary without falling wholly inside either window is missed; spans are capped at `MAX_ANSWER_TOKENS = 15` tokens, so a long free-text answer is cut.
3. **Input boundary:** the fine-tuning data is SQuAD 2.0 (English Wikipedia text) and DocVQA (scanned English business documents from the UCSF Industry Documents Library with short-span answers) — and the OCR conventions of DocVQA's provided tokens. Pages in other languages and scripts, handwriting, OCR that tokenises differently (sub-word fragments, merged cells), reading orders that differ from the OCR's, and questions whose answer is not a span on the page fall outside what the upstream authors report and what this repository measured; results on them are undefined, not merely degraded. The tutorial's perfect renderer-supplied OCR is not evidence about any real OCR engine.
4. **Decision boundary:** not for autonomous decisions that act on extracted values — automated invoice payment, claims adjudication, KYC or identity checks, clinical or legal record extraction — without a human comparing the answer span (its boxes are returned for exactly this purpose) with the page, and a locally measured ANLS on the deployment's own labelled pages with the deployment's own OCR.

#### Factors

###### Groups

This pipeline is not human-centric by design: it selects a span of OCR words that answers a question and never classifies, identifies or scores people. The fine-tuning data (SQuAD 2.0's Wikipedia paragraphs; DocVQA's 12,767 scanned industry documents with 50,000 questions, per the dataset paper) contains no evaluation groups in the demographic sense, and neither Impira nor this repository audited it for anything of the kind. What does vary is the document population: DocVQA is English-language, typed or printed, mid-twentieth-century-onward industry correspondence and forms, so pages in other languages and scripts, handwritten or historical typesetting, modern designer layouts and non-Western document conventions are the groups whose answer accuracy is unknown, not known to be equal — compounded by whatever OCR the operator uses, whose own error rates vary by script and print quality. Where pages carry personal data — medical records, HR files, identity documents, correspondence naming individuals — a question turns the pipeline into a targeted extractor; the operator who processes such documents is responsible for a fairness and privacy audit on their own page set, stratified by document family, before relying on the output.

###### Instrumentation

The upstream fine-tuning "instrument" is the OCR that produced DocVQA's word tokens and boxes over archival scans, plus SQuAD's clean text with no layout at all; the model learned the token boundaries, reading order and box conventions of that OCR. Inference inputs arrive from whatever OCR the operator runs — Tesseract at some page-segmentation mode, a cloud OCR, a PDF text layer — and each tokenises, orders and boxes words differently; the pipeline coarsens boxes to a 1000-step grid and validates only that they are numbers inside the page. It cannot detect a mis-read word, a wrong reading order, a merged or split token, or a box on the wrong line, and every such error passes through to the answer. The synthetic tutorial form is the opposite extreme: its words and boxes come from the renderer (`ImageDraw.textbbox`), an instrument that is exact by construction and that no OCR reproduces, so the 5/5 result bounds nothing about real OCR.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, float32 on CPU; CUDA is used automatically when visible (float32) but was not exercised for this card. No OCR engine is installed: `pytesseract` is not a dependency and the smoke run never called the adapter. Measured on the reference machine with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`) and the Hub offline (`HF_HUB_OFFLINE=1`): `verify_snapshot` on the 8-file, 514 MB snapshot 0.27 s; load 5.10 s; five questions over a 73-word page 0.29 s (first call) then 0.11–0.12 s each; an unanswerable question 0.12 s; a 657-word page (the form's words repeated nine times) answered across 4 windows in 0.47 s — cost is one encoder pass per window per question and is small. Data environment: the model assumes English, machine-readable OCR of a business document in reading order with one box per word, and a question answerable by a short span; the synthetic form satisfies that assumption exactly and is where the measured behaviour holds. Real OCR noise, other languages, unusual layouts and unanswerable questions violate it to degrees this repository did not measure, and the pipeline reports no signal when they do.

#### Metrics

###### Performance Measures

The pipeline reports no accuracy measure. Each answer carries `score`, the product of the start-position and end-position softmax probabilities of the chosen span within its window under the model's own span head — a ranking signal over spans of one page, not a probability that the answer is right and not an answerability signal. The repository ships DocVQA's own metric because it is what a caller would use to evaluate: `anls(prediction, golds)` — Average Normalised Levenshtein Similarity (Biten et al., 2019), `1 − lev / max(len)` over lower-cased, punctuation-stripped, whitespace-collapsed strings, maximised over the accepted answers, scored 0 below `ANLS_THRESHOLD = 0.5` — and `exact_match` after the same normalisation. Both need labelled question/answer pairs with matching conventions that the caller must supply, together with the OCR the deployment will really use; the DocVQA benchmark itself is registration-gated and not bundled. The public `evaluation_report(results, golds=None)` stage returns that report in machine-readable form: the mean `anls` and the `exact_match` rate over the questions plus one per-question entry (prediction, span score, accepted answers, both metrics) with the verdict `sample-sanity`, or the verdict `not-measurable` naming the labelled set that would be required when no accepted answers are supplied. Impira publishes no benchmark number for this checkpoint and none is claimed here.

###### Decision thresholds

No threshold is applied inside the pipeline: `answer` always returns the highest-scoring span (the transformers document-question-answering pipeline's default behaviour, without its `handle_impossible_answer` option) and reports its `score`. The only fixed decision parameters are encoding conventions taken from that pipeline — `MAX_SEQ_LEN = 512`, `DOC_STRIDE = 128`, `MAX_ANSWER_TOKENS = 15` — none of them tuned here. The smoke run shows why a caller might want a threshold and why this repository does not supply one: the five answerable questions scored 0.999–1.000 while the unanswerable "What is the delivery address?" scored 0.28 for `14 Harbour Road,` — suggestive on one clean page, but the score is uncalibrated and the gap will not hold on noisy OCR. A deployment owns choosing and validating a rejection threshold (and the OCR) on its own labelled pages, and re-validating whenever the document source or the OCR engine changes. The evaluation helper's `ANLS_THRESHOLD = 0.5` is DocVQA's published convention, not a value tuned here.

###### Approaches to uncertainty and variability

This repository reports no central metric value and therefore no dispersion: the smoke run records timings, spans and scores for five authored questions, one unanswerable question and one long page, on one synthetic form, not accuracy. Run-to-run variability comes only from floating-point kernel selection across CPU builds and accelerators; there is no sampling and no seed to set, so a fixed input on fixed hardware is repeatable but not guaranteed bitwise-identical across machines, and the synthetic form's own bytes and word boxes depend on the Pillow build's bundled font. The `score` is a product of softmaxes, not a calibrated confidence: 0.999–1.000 on five right answers and 0.28 on a fabricated one is one observation, not a calibration curve. A caller who needs calibrated confidences must fit a calibration map on their own labelled pages; a caller who needs an accuracy estimate must supply labelled question/answer pairs with their own OCR and compute ANLS over many pages or bootstrap resamples themselves.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The snapshot README states fine-tuning on SQuAD 2.0 and DocVQA; SQuAD's paragraphs come from English Wikipedia, and DocVQA's pages from the UCSF Industry Documents Library — public archives of tobacco-industry correspondence, forms and reports that name real people and organisations — so personal data in the fine-tuning corpus is present by construction. LayoutLM's own pre-training used the IIT-CDIP scanned-document collection from the same archive. Neither was audited here. This repository distributes code, tests, and documentation; it does not distribute the 511,200,628-byte `model.safetensors`, which is staged locally under `weights/layoutlm-document-qa/` and git-ignored, and it ships no sample documents — the tutorial form is rendered in code with fictitious names. The operator must audit the pages they submit for personal, proprietary, or otherwise restricted content; the pipeline performs no such check and will answer "What is the patient's name?" as readily as "What is the invoice number?".

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, Impira, Microsoft, or any regulator. Foreseeable but unintended sensitive uses — extracting amounts and payees for automated payment, reading identity documents for verification, pulling diagnoses or dosages from clinical records, screening applications by fields read from scans — would be admissible only with human comparison of every answer span against the page (the indices and boxes exist for that), a locally measured ANLS on the deployment's own labelled pages with its own OCR, a validated rejection threshold, and whatever regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks all 8 listed files' byte sizes and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True`, always passes `trust_remote_code=False`, requires the fast tokenizer (needed for word/sequence ids), and the smoke run loaded and answered with `HF_HUB_OFFLINE=1`. The upstream `pytorch_model.bin` (pickle) and `tf_model.h5` are neither listed nor loaded. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused; the import-boundary tests assert that a missing or tampered snapshot is refused before `torch` or `transformers` is imported.
- **Input integrity:** the public `validate_inputs(words, boxes, questions, *, image_size)` stage applies exactly the checks `answer` applies (both route through the same private checkers) and returns an input manifest recording the schema, the ceilings, the page size and word count, the checked questions and the verdict; words, boxes (count, shape, numeric, inside the page, non-inverted), page size and questions are validated before any encoding; `answer` raises on a malformed runner result or an out-of-range span; `evaluation_report` rejects mismatched or empty accepted-answer lists.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; no sampling; the encoding conventions are module constants; every result carries `model_id`, `model_revision`, the checked question, the span indices, the score and the window count.
- **Refusals:** no OCR inside the pinned runtime (the adapter is opt-in and imports lazily), no batching, no download without the explicit flag, no pickle deserialisation, no hidden threshold, no attempt to guess whether the OCR is right or the question answerable.
- No statistical mitigation (class balancing, subsampling) applies: no training happens in this repository.

###### Risks and harms

- **Fabricated spans for unanswerable questions:** the model has no abstention — the smoke run returned `14 Harbour Road,` for a question the page cannot answer — so a wrong page, a mis-phrased field or an absent value yields a plausible span; downstream consumers that trust the string inherit the error silently unless they validate a threshold themselves.
- **OCR errors pass through:** a transposed digit, a merged token or a wrong reading order in the OCR becomes the answer; the tutorial's perfect OCR hides this entirely, and ANLS on labelled pages with the real OCR is the only way to see the rate.
- **Automation bias:** exact answers at 0.999+ on a clean form invite trust that an uncalibrated within-page score has not earned.
- **Windowing and span-length misses:** an answer straddling a 512-token window boundary or longer than 15 tokens is missed or cut, silently.
- **Targeted extraction and privacy exposure:** a question turns any OCR'd page into a lookup for a named field — including personal data — with no content check; returned boxes make the value easy to crop and reuse.
- **Bias amplification:** any document family DocVQA and the operator's OCR under-represent (non-English, handwritten, modern designer layouts, non-Western conventions) is reproduced as uneven accuracy, undetected because no per-family evaluation exists.
- **Resource use:** a 511 MB model and ~0.1–0.3 s per question per window on the reference CPU; cheap, but a question-heavy workload over long pages scales with windows × questions.

###### Use cases

Prohibited even where the model would work: querying documents in order to extract personal data for surveillance, profiling, social scoring, or unlawful discrimination in employment, housing, credit, insurance, education, or healthcare access; processing documents the operator has no right to process, or paywalled and licence-restricted material in breach of its terms; deceptive uses that present extracted spans as verified document facts or as evidence; and any use that violates the upstream MIT licence terms, the DIMER deployment terms, or the consent and data-protection obligations attached to the documents processed. Autonomous high-consequence actions triggered by unreviewed answers are prohibited by the intended-use contract above.

## Immutable provenance

- Model: `impira/layoutlm-document-qa`
- Revision: `beed3c4d02d86017ebca5bd0fdf210046b907aa6`
- Snapshot manifest: `weights/layoutlm-document-qa/dimer-base-manifest.json`, 8 files, `totalBytes` 513814827
- `model.safetensors` SHA-256: `e4bbad3e4a1b5ae50c787b7afd6049a0bfa99fd823b50436e444e092ae2347b9` (511,200,628 bytes, float32)
- `config.json` SHA-256: `6d0fc068193109d0d053fa4de00963778beffbde05067c3e9f3454235044380f` (789 bytes; `LayoutLMForQuestionAnswering`, `max_2d_position_embeddings` 1024, `tokenizer_class` RobertaTokenizer)
- `tokenizer.json` SHA-256: `33465117406b9007673e8ba283f7f1383d9b5094df947481af60eec94ed7d7bd` (1,355,881 bytes; the fast RoBERTa BPE tokenizer the pipeline requires)
- Weight format: SafeTensors; loader `LayoutLMForQuestionAnswering.from_pretrained(<dir>, local_files_only=True, trust_remote_code=False, dtype=float32)` with `AutoTokenizer` from the same directory. The upstream `pytorch_model.bin` and `tf_model.h5` are not part of the manifest and are never loaded.

## Input/output contract

- `LayoutLMDocumentQAPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — stages missing manifest files (only with `allow_download=True`), verifies digests, loads; `device` defaults to `cuda:0` when visible, else `cpu`; float32 on both.
- `answer(question, *, words, boxes, image_size) -> dict` with keys `answer` (span words joined by spaces; empty when no span), `score`, `start`, `end` (inclusive word indices or `None`), `question` (whitespace-collapsed), `n_words`, `n_windows`, `image_size`, `device`, `dtype`, `source`, `model_id`, `model_revision`.
- `normalize_box(box, width, height) -> [int]*4` (pixel xyxy → 0..1000 grid); `ocr_words_with_tesseract(image, *, lang="eng") -> (words, boxes)` (optional; needs `pytesseract` + Tesseract on the host); `normalize_answer(text) -> str`; `anls(prediction, golds, *, threshold=0.5) -> float`; `exact_match(prediction, golds) -> bool`.
- Ceilings and constants: `MIN_IMAGE_SIDE = 1`, `MAX_IMAGE_SIDE = 10000`, `MAX_WORDS = 2000`, `MAX_QUESTION_CHARS = 256`, `MAX_SEQ_LEN = 512`, `DOC_STRIDE = 128`, `MAX_ANSWER_TOKENS = 15`, `BOX_GRID = 1000`, `ANLS_THRESHOLD = 0.5`, `INPUT_SCHEMA`.
- `validate_inputs(words, boxes, questions, *, image_size, names) -> dict`; `evaluation_report(results, golds=None, *, sample_kind) -> dict` where `golds` holds one sequence of accepted answers per result; `verify_snapshot(path=None) -> dict`; `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`; Python 3.12. `pytesseract` and Tesseract are **not** pinned or installed.
- Precision: float32; encoding `<s> question </s></s> words </s>` in 512-token windows with 128-token stride, boxes on the 0..1000 grid; span of ≤ 15 tokens with the highest start×end softmax product.
- Measured 2026-09-14 in the Windows venv (`torch 2.14.0+cu130`) with `CUDA_VISIBLE_DEVICES=-1` and `HF_HUB_OFFLINE=1`, device `cpu`: `verify_snapshot` 0.27 s (8 files, 514 MB); load 5.10 s; `answer` over a synthetic 850×1100 invoice-style form whose renderer recorded 73 words with their boxes (header, supplier, five labelled fields, a four-row line-item table, totals and a payment-terms line, rendered with Pillow's bundled font) → "What is the invoice number?" `NW-2026-0417` (words 12..12, score 1.000, 0.29 s); "Who is the customer?" `Blue Yonder Airlines` (24..26, 1.000, 0.11 s); "What is the total due?" `$1,099.20` (59..59, 0.9998, 0.11 s); "What is the due date?" `11 April 2026` (20..22, 1.000, 0.12 s); "How many cargo pallets were invoiced?" `40` (38..38, 0.9994, 0.11 s); `evaluation_report` against the authored answers: `anls` 1.0, `exact_match` 1.0, verdict `sample-sanity`; unanswerable "What is the delivery address?" → `14 Harbour Road,` (score 0.2774, 0.12 s); the same words repeated nine times (657 words) → `NW-2026-0417` across 4 windows (score 0.9999, 0.47 s).
- Tutorial execution: `tutorials/layoutlm_document_qa_colab.ipynb` ran top-to-bottom in a fresh local kernel (all 8 code cells, 57.0 s including the 514 MB staging, same five answers as the smoke run); recorded in `docs/release-verification.md` as pre-flight, not supported-runtime evidence.
- Tests: `pytest -q -o addopts= tests` — offline, no weights required; `ruff check src tests tools` clean.
- Not executed: CUDA path, the `ocr_words_with_tesseract` adapter (no Tesseract on the reference machine), any real OCR output, any ANLS measurement against labelled pages, scans or photographs, non-Latin scripts, answers straddling a window boundary.

## References

- Xu et al. LayoutLM: Pre-training of Text and Layout for Document Image Understanding. KDD 2020. https://arxiv.org/abs/1912.13318
- Mathew, Karatzas, Jawahar. DocVQA: A Dataset for VQA on Document Images. WACV 2021. https://arxiv.org/abs/2007.00398
- Rajpurkar, Jia, Liang. Know What You Don't Know: Unanswerable Questions for SQuAD (SQuAD 2.0). ACL 2018. https://arxiv.org/abs/1806.03822
- Biten et al. Scene Text Visual Question Answering (the ANLS metric). ICCV 2019. https://arxiv.org/abs/1905.13648
- Upstream architecture code: https://github.com/microsoft/unilm/tree/master/layoutlm
- Upstream card and DocQuery: https://huggingface.co/impira/layoutlm-document-qa · https://github.com/impira/docquery
- Transformers `LayoutLM` and document-question-answering pipeline documentation: https://huggingface.co/docs/transformers/model_doc/layoutlm · https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.DocumentQuestionAnsweringPipeline
