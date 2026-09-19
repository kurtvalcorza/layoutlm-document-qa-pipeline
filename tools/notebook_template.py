"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, metrics.py, samples.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E document-QA workflow: the pinned impira/layoutlm-document-qa snapshot
is digest-verified and loaded, the `ground_truth` column of two digest-pinned CORD-v2 receipt shards is
fetched, turned into templated question/answer records over real OCR words and boxes, validated and
split by page, five authored questions over a rendered invoice are answered through the inference
contract, the frozen model is scored on the held-out receipts beside two non-neural baselines, a
bounded fine-tuning of the last encoder blocks and the span head runs in the kernel, the held-out
split is scored again, the adapted model re-reads the invoice, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "layoutlm_document_qa_pipeline",
    "repo_name": "layoutlm-document-qa-pipeline",
    "stem": "layoutlm_document_qa",
    "notebook_name": "layoutlm_document_qa_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `impira/layoutlm-document-qa` snapshot (safetensors, 511 MB), reads the `ground_truth` column of two "
        "digest-pinned CORD-v2 receipt shards from the Hugging Face Hub (about 0.4 MB over HTTP range requests, no credential), "
        "turns the 199 unique receipts into templated question/answer records over their real OCR words and boxes and cuts "
        "them by receipt into 119 / 30 / 50 training, validation and test pages, drops any record that would not fit one "
        "512-token window, answers five authored questions over a rendered invoice through the inference contract with an "
        "input manifest and a rejection probe, scores the frozen model on the test receipts with ANLS and exact match beside "
        "the last-number and keyword-lookup baselines, runs a bounded fine-tuning of the last four encoder blocks and the span "
        "head on the training receipts with validation-ANLS epoch selection, scores the held-out receipts again, re-reads the "
        "invoice with the adapted model, exports the adapter as safetensors with a manifest, and reloads that artifact into a "
        "fresh pipeline to verify answer parity. The default path needs no repository clone, no DIMER worker or service, no "
        "credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes about "
        "twenty minutes of model time after the downloads; a CUDA runtime is used automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "labelled pages as a JSON array or a JSONL file of `{{id, page_id, question, words, boxes, image_size, answer_start, "
        "answer_end}}` records — the page's OCR words with one pixel `[x0, y0, x1, y1]` box each, the page size, and the "
        "inclusive word indices of the gold span. They pass through the same validation, seeded page-disjoint split, fit "
        "check, baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells as the CORD-v2 sample. "
        "The expected schema and the ceilings are stated in the Prerequisites and in Section 4, and uploaded files stay inside "
        "this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "LayoutLMDocumentQAPipeline",
    "weights_key": "layoutlm-document-qa",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers"],
    "title": "LayoutLM Document QA (impira) — DIMER E2E document question-answering fine-tuning tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/layoutlm-document-qa-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/layoutlm-document-qa-pipeline/blob/main/tutorials/layoutlm_document_qa_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-impira%2Flayoutlm--document--qa-ffcc4d?style=flat",
            "https://huggingface.co/impira/layoutlm-document-qa",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-microsoft%2Funilm%20(layoutlm)-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/microsoft/unilm/tree/master/layoutlm",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-1912.13318-b31b1b.svg", "https://arxiv.org/abs/1912.13318"),
    ],
    "capability": "extractive document question answering over OCR words and boxes and bounded supervised fine-tuning of the last encoder blocks and the span head on a gold-span receipt dataset, using the pinned `impira/layoutlm-document-qa` weights",
    "intro": (
        "`impira/layoutlm-document-qa` is Impira's LayoutLM (v1) — a 12-layer BERT-style encoder with 2-D position "
        "embeddings, 128 M parameters, RoBERTa vocabulary — fine-tuned on SQuAD 2.0 and DocVQA for extractive document "
        "question answering and published under the **MIT** licence. At inference the encoder reads the question and the "
        "page's **words with their boxes** (normalised to a 0–1000 grid) as one token sequence and emits a start and an end "
        "logit per token; the carried module scores every span of at most 15 tokens by the product of the start and end "
        "softmax probabilities and returns the best one as a word span with its indices. **The model never sees pixels:** "
        "OCR is an input provider outside the model, so words and boxes come from the caller (bring-your-own OCR), and the "
        "model has no abstention — it returns its best span for every question.\n\n"
        "What this notebook adds to inference is **adaptation with gold spans**. The dataset is real: CORD-v2 (Park et al., "
        "2019; NAVER Clova; **CC BY 4.0**) — photographed Indonesian receipts whose every printed word carries a pixel box and "
        "a field category. Because the model needs words and boxes only, the notebook reads just the `ground_truth` column of "
        "two pinned Hub parquet shards (about 0.4 MB of the 476 MB the shards hold with their images) and turns each receipt "
        "into templated questions — *What is the total amount?*, *How much change was given?*, *What is the name of the first "
        "item?* — whose gold answer is the value words of the matching field, a contiguous span of the page's OCR words by "
        "construction. The frozen model already reads most receipt totals (the build record measured ANLS 0.84 on the test "
        "receipts), so the fine-tuning question is narrower and more honest than a rescue: does a bounded adaptation of the "
        "last four encoder blocks and the span head on 119 receipts lift the fields it gets wrong — item names, quantities, "
        "cash and change lines — on held-out receipts? Two metrics are implemented in the carried modules (mean **ANLS**, "
        "DocVQA's official normalised-Levenshtein score, and the **exact-match** rate after the same normalisation), and two "
        "**non-neural baselines** — last number on the page and keyword lookup — show where a reader with no model sits. "
        "Nothing here is a quality claim about your documents: it is one seeded split of one corpus with one OCR convention.\n\n"
        "**Snapshot note:** the pinned revision ships a fast `tokenizer.json` (an 8-file manifest) and a float32 "
        "`model.safetensors` — no pickle is opened anywhere in this notebook; the upstream `pytorch_model.bin` and "
        "`tf_model.h5` are neither listed nor fetched. Section 3 stages and digest-verifies those eight files before the "
        "tokenizer or the model is constructed."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, metrics and dataset modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; fetch one column of a digest-pinned receipt corpus without downloading "
        "its images, turn it into gold-span records and validate and split it by page without leakage; drop the records the "
        "window ceiling would refuse rather than truncating them; answer through the public API over a rendered page and read "
        "`answer`, `start`, `end` and `score` correctly (a product of softmax masses, not a calibrated probability); score the "
        "frozen model against gold spans beside two non-neural baselines; run a bounded fine-tuning with explicit "
        "hyperparameters and validation-based epoch selection; evaluate on a page-disjoint test split; re-read a page from a "
        "different document family with the adapted model; and export a safetensors adapter that reloads against the pinned "
        "base with verified parity."
    ),
    "exclusions": (
        "OCR itself (the notebook installs no Tesseract and ships no OCR model; the default path uses the corpus's annotated "
        "words and boxes and the invoice renderer's own boxes), PDF or multi-page documents (one page per call), abstention "
        "(the model always returns its best span, even for an unanswerable question), answers that are not a contiguous span "
        "of the OCR words, arithmetic or reasoning, evaluation on the DocVQA benchmark (registration-gated and not bundled), "
        "full-model or embedding fine-tuning, training on pages that need more than one window, languages and scripts other "
        "than the Latin-script receipts and English questions used here, and any claim that a CORD-v2 split stands in for your "
        "documents. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is adequate but not fast: the build record measured about 7 s to load and digest-verify the 514 MB snapshot, about 41 s to score the 229 test questions, and about two and a half minutes per epoch of fine-tuning the last four encoder blocks and the span head on 595 training questions (validation scoring included; 934 s for six epochs in the build record). The pinned `torch==2.14.0` install and the 511 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what extractive (span) question answering over OCR tokens is; why a start/end softmax product is not a calibrated confidence; what normalised Levenshtein similarity (ANLS) measures and why neither it nor exact match is a human judgement.",
        "- **Data contract:** records are `{{id, page_id, question, words, boxes, image_size, answer_start, answer_end}}` — the page's OCR words (1..`MAX_WORDS` = 2,000, non-empty) with one pixel `[x0, y0, x1, y1]` box per word inside the page, the page size in pixels (`MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` = 1..10,000), a question of at most `MAX_QUESTION_CHARS` (256) characters, and the inclusive word indices of the gold span; an optional `answers` list must start with the span text. Ids match `[A-Za-z0-9_.:-]{{1,64}}` and are unique; a dataset needs 8..20,000 records; every question on the same page lands in the same split so a test page is never trained on; a question + page that needs more than one 512-token window is dropped from training by the fit check, never truncated. BYOD accepts a JSON array or JSONL in that shape.",
        "- **Validation is structural, not semantic:** nothing checks that a question is answerable from its page beyond the gold span lying inside the words, or that a gold span is the *best* answer — a mislabelled corpus is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — an internal document set with its field annotations is exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path reads the `ground_truth` column of two objects in the Hub dataset repository `naver-clova-ix/cord-v2` at the immutable revision `7f0115a4…` (`data/test-…parquet`, 234,202,795 bytes, SHA-256 `51c65f17…`, and `data/validation-…parquet`, 242,080,800 bytes, SHA-256 `0d0f6dac…`): the declared size and SHA-256 of each file are checked against the pins before any byte is read, only the parquet footer and that one column are fetched over HTTPS range requests, and the decoded column is refused unless its own SHA-256 matches; the corpus is CC BY 4.0 (Park et al., 2019). No image is downloaded.",
    ],
    "cells": [
        {
            "md": (
                "## 4. CORD-v2 receipt corpus, validation and split\n\n"
                "`fetch_corpus` reads the pinned shards' `ground_truth` column (or the cache under `weights/cord-v2/`): for "
                "each shard it first checks the byte size and SHA-256 the Hub declares for the file against the pins, then "
                "reads only the parquet footer and the one column chunk through `pyarrow` over HTTPS range requests, and "
                "refuses the decoded column unless its SHA-256 matches. `read_corpus` turns each row into a page — the words "
                "of every annotated line in order, each box the clipped axis-aligned hull of the annotated quadrilateral, and "
                "the lines' field categories with their value-word spans (`is_key` 0; a printed key such as `TOTAL` is on the "
                "line but not in the answer). `build_sample_dataset` keeps every receipt that supports at least one templated "
                "question, drops the one receipt whose words repeat another shard's, shuffles the 199 unique receipts with "
                "`SPLIT_SEED` and cuts them **by receipt** into 119 / 30 / 50 training, validation and test pages, then expands "
                "each page into its questions (a category that occurs on exactly one line becomes one question; *first item* "
                "is the topmost single-line `menu.nm`). `validate_dataset` checks every record against the contract, "
                "`check_split_disjoint` asserts no page id and no page content is shared, and the training split is written "
                "to `outputs/{stem}_train.jsonl` in the shape BYOD expects.\n\n"
                "Look for: 100 + 100 raw rows, three digests, splits 595 / 152 / 229 questions over 119 / 30 / 50 receipts, "
                "the field mix, and four refusal probes — a duplicate id, a gold span outside the words, a box outside the "
                "page and a dataset too small to split — each rejected before `torch` does anything."
            ),
            "code": (
                "import collections\n"
                "import hashlib\n"
                "import io\n"
                "import json\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    corpus_rows = fetch_corpus(cache_dir='weights/cord-v2')\n"
                "    raw_rows = {{name: len(rows) for name, rows in corpus_rows.items()}}\n"
                "    splits = build_sample_dataset(read_corpus(corpus_rows), seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} {{CORPUS_RELEASE}} ({{CORPUS_LICENSE}})'\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "pages = {{name: len({{r['page_id'] for r in part}}) for name, part in splits.items()}}\n"
                "fields = {{name: dict(collections.Counter(r.get('field', 'byod') for r in part)) for name, part in splits.items()}}\n"
                "write_dataset_jsonl(splits['train'], 'outputs/{stem}_train.jsonl')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'pages': pages, 'column_sha256': {{name: spec['column_sha256'][:16] + '...' for name, spec in CORPUS_FILES.items()}}}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'unique_pages': manifest['unique_pages'], 'words_per_page': manifest['words_per_page'], 'answer_words': manifest['answer_words'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "print({{'fields_test': fields['test']}})\n"
                "example = splits['train'][0]\n"
                "print({{'example': {{'id': example['id'], 'page_id': example['page_id'], 'question': example['question'], 'answers': example['answers'], 'span': [example['answer_start'], example['answer_end']], 'words': ' '.join(example['words'][:18]) + ' ...'}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in splits['train'][:8]],\n"
                "    'gold span outside the words': [{{**splits['train'][0], 'answer_end': len(splits['train'][0]['words'])}}, *splits['train'][1:8]],\n"
                "    'box outside the page': [{{**splits['train'][0], 'boxes': [[0, 0, splits['train'][0]['image_size'][0] + 1, 5], *splits['train'][0]['boxes'][1:]]}}, *splits['train'][1:8]],\n"
                "    'too small': splits['train'][:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Fit check, then answer through the inference contract\n\n"
                "The window ceiling needs the real tokenizer, so it is applied now that the model is loaded: `pipe.check_fit` "
                "partitions each split into the records whose question and page fit one 512-token window and the ones that "
                "would be **windowed at inference and are therefore not trained on** — nothing is truncated. The dropped ids "
                "are printed and the fitting records are what every later cell uses (the build record dropped none of the 976 "
                "sample records; receipts are short).\n\n"
                "Then the inference contract is exercised as the inference-only tutorial exercised it: an invoice-style form is "
                "rendered in code with Pillow's bundled font and the renderer records the pixel box of **every word it draws** "
                "— perfect OCR by construction, a different document family from the receipts, and a page the model will be "
                "asked to read again after adaptation. `validate_inputs` applies exactly the checks `answer` applies (page "
                "size, 1..`MAX_WORDS` non-empty words with one box each inside the page, questions up to `MAX_QUESTION_CHARS`) "
                "and returns an input manifest; a box outside the page is validated too and its rejection recorded as a "
                "finding. `answer` returns the word span, its inclusive `start`/`end` indices (so the answer's boxes are "
                "known), `score`, `n_words`, `n_windows` and the model identity. **Score semantics:** the `score` is the "
                "**product of the start and end softmax probabilities of the chosen span within its window** — a ranking "
                "signal over spans of this page, **not a calibrated probability** that the answer is right, and never a "
                "signal that the question is answerable. Whether the answers are *right* is what Section 6 measures on 229 "
                "gold spans, not what five authored pairs can tell you."
            ),
            "code": (
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "fit = {{name: pipe.check_fit(part) for name, part in splits.items()}}\n"
                "train_records, val_records, test_records = fit['train']['fitting'], fit['validation']['fitting'], fit['test']['fitting']\n"
                "print({{'fit_check': {{name: {{'fitting': f['n_fitting'], 'dropped': f['dropped']}} for name, f in fit.items()}}}})\n"
                "ceilings = {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_WORDS': MAX_WORDS, 'MAX_QUESTION_CHARS': MAX_QUESTION_CHARS, 'MAX_SEQ_LEN': MAX_SEQ_LEN, 'DOC_STRIDE': DOC_STRIDE, 'MAX_ANSWER_TOKENS': MAX_ANSWER_TOKENS, 'BOX_GRID': BOX_GRID, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS}}\n"
                "print(ceilings)\n\n\n"
                "def synthetic_form(width=850, height=1100):\n"
                "    \"\"\"Invoice-style form rendered with Pillow's bundled font; every drawn word is recorded with its pixel box.\"\"\"\n"
                "    page = Image.new('RGB', (width, height), 'white')\n"
                "    d = ImageDraw.Draw(page)\n"
                "    body, bold, head = ImageFont.load_default(size=18), ImageFont.load_default(size=20), ImageFont.load_default(size=30)\n"
                "    words, boxes = [], []\n\n"
                "    def put(x, y, text, font, fill='black'):\n"
                "        for token in text.split():\n"
                "            x0, y0, x1, y1 = d.textbbox((x, y), token, font=font)\n"
                "            d.text((x, y), token, fill=fill, font=font)\n"
                "            words.append(token)\n"
                "            boxes.append([float(x0), float(y0), float(x1), float(y1)])\n"
                "            x = x1 + d.textlength(' ', font=font)\n\n"
                "    put(70, 60, 'INVOICE', head)\n"
                "    put(70, 110, 'Northwind Traders Ltd.', bold)\n"
                "    put(70, 136, '14 Harbour Road, Portsmouth PO1 3AX', body, (40, 40, 40))\n"
                "    y = 200\n"
                "    for label, value in [('Invoice number:', 'NW-2026-0417'), ('Invoice date:', '12 March 2026'), ('Due date:', '11 April 2026'), ('Customer:', 'Blue Yonder Airlines'), ('Purchase order:', 'PO-88213')]:\n"
                "        put(70, y, label, bold)\n"
                "        put(300, y, value, body)\n"
                "        y += 32\n"
                "    d.rectangle([70, 400, 780, 640], outline='black', width=2)\n"
                "    cols = [70, 420, 540, 660, 780]\n"
                "    rows = [('Description', 'Qty', 'Unit price', 'Amount'), ('Cargo pallets (standard)', '40', '$18.50', '$740.00'), ('Shrink wrap rolls', '12', '$9.25', '$111.00'), ('Handling fee', '1', '$65.00', '$65.00')]\n"
                "    for r, row in enumerate(rows):\n"
                "        yy = 400 + r * 48\n"
                "        if r:\n"
                "            d.line([(70, yy), (780, yy)], fill=(120, 120, 120), width=1)\n"
                "        for c, cell in enumerate(row):\n"
                "            put(cols[c] + 10, yy + 14, cell, bold if r == 0 else body)\n"
                "    for c in cols[1:-1]:\n"
                "        d.line([(c, 400), (c, 640)], fill=(120, 120, 120), width=1)\n"
                "    put(540, 670, 'Subtotal:', bold)\n"
                "    put(680, 670, '$916.00', body)\n"
                "    put(540, 700, 'VAT (20%):', bold)\n"
                "    put(680, 700, '$183.20', body)\n"
                "    put(540, 736, 'Total due:', head)\n"
                "    put(680, 736, '$1,099.20', head)\n"
                "    put(70, 900, 'Payment terms: 30 days from invoice date. Bank: Solent Mutual, sort code 40-11-22.', body, (40, 40, 40))\n"
                "    qa = [\n"
                "        ('What is the invoice number?', ['NW-2026-0417']),\n"
                "        ('Who is the customer?', ['Blue Yonder Airlines']),\n"
                "        ('What is the total due?', ['$1,099.20', '1,099.20']),\n"
                "        ('What is the due date?', ['11 April 2026']),\n"
                "        ('How many cargo pallets were invoiced?', ['40']),\n"
                "    ]\n"
                "    return page, words, boxes, qa\n\n\n"
                "image, words, boxes, qa = synthetic_form()\n"
                "questions, golds = [q for q, _ in qa], [g for _, g in qa]\n"
                "image_name = 'synthetic_invoice_850x1100.png'\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "input_manifest = validate_inputs(words, boxes, questions, image_size=image.size, names=[image_name])\n"
                "try:\n"
                "    validate_inputs(words[:1], [[0, 0, image.width + 1, 10]], questions, image_size=image.size)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'box-outside-page-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'page': image_name, 'size': image.size, 'rgb_sha256': image_sha256[:16] + '...', 'n_words': len(words), 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n"
                "results = []\n"
                "for question in questions:\n"
                "    started = time.perf_counter()\n"
                "    result = pipe.answer(question, words=words, boxes=boxes, image_size=image.size)\n"
                "    results.append({{'seconds': round(time.perf_counter() - started, 3), **result}})\n"
                "    print(f\"Q: {{result['question']}}\\n   A: {{result['answer']!r}}  score {{result['score']:.4f}}  words {{result['start']}}..{{result['end']}}  windows {{result['n_windows']}}\")\n"
                "checks = {{\n"
                "    'one_result_per_question': len(results) == len(questions),\n"
                "    'span_is_the_indexed_words': all(r['answer'] == ' '.join(words[r['start'] : r['end'] + 1]) for r in results if r['start'] is not None),\n"
                "    'indices_inside_the_page': all(0 <= r['start'] <= r['end'] < r['n_words'] for r in results if r['start'] is not None),\n"
                "    'scores_in_unit_interval': all(0.0 <= r['score'] <= 1.0 for r in results),\n"
                "    'one_window': all(r['n_windows'] == 1 for r in results),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'answer output failed a sanity check: {{checks}}')\n"
                "frozen_form = evaluation_report(results, golds, sample_kind='synthetic')\n"
                "print({{'checks': checks, 'frozen_invoice': {{m['id']: round(m['value'], 3) for m in frozen_form['metrics']}}, 'verdict': frozen_form['verdict']}})"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model's score on the test receipts\n\n"
                "Three numbers frame the adaptation. The **last-number baseline** answers every question with the last word "
                "on the page that contains a digit — receipts end with their totals, so it is right more often than chance "
                "and wrong for everything else. The **keyword-lookup baseline** finds the page word sharing the longest prefix "
                "with a content word of the question (`TOTAL` for *What is the total amount?*) and answers with the next "
                "numeric word after it — the find-the-label-then-read-the-value heuristic a person applies, with no model. "
                "The **frozen model** answers the 229 test questions and is scored with the same two metrics: mean **ANLS** "
                "(DocVQA's normalised Levenshtein similarity, 0 below 0.5) and the **exact-match** rate, both after "
                "lower-casing, punctuation removal and whitespace collapsing. Expect the frozen model to be strong already — "
                "it was fine-tuned on DocVQA, and receipt totals are what it reads best — and read the per-field breakdown: "
                "the build record measured ANLS 0.98 on subtotals but 0.17 on *How many items were bought?* and 0.79 on item "
                "names, which is where Section 7 has room to move."
            ),
            "code": (
                "baseline_last = last_number_baseline(test_records)\n"
                "baseline_lookup = keyword_lookup_baseline(test_records)\n"
                "print({{'last_number_baseline': {{'anls': round(baseline_last['anls'], 3), 'exact_match': round(baseline_last['exact_match'], 3), 'n': baseline_last['n']}}}})\n"
                "print({{'keyword_lookup_baseline': {{'anls': round(baseline_lookup['anls'], 3), 'exact_match': round(baseline_lookup['exact_match'], 3), 'n': baseline_lookup['n']}}}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "print({{'frozen_model_test': {{'anls': round(frozen_test['anls'], 3), 'exact_match': round(frozen_test['exact_match'], 3), 'n': frozen_test['n'], 'verdict': frozen_test['verdict']}}, 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n\n\n"
                "def by_field(pipeline, records):\n"
                "    scores = collections.defaultdict(list)\n"
                "    for record in records:\n"
                "        item = pipeline.answer(record['question'], words=record['words'], boxes=record['boxes'], image_size=record['image_size'])\n"
                "        scores[record.get('field', 'byod')].append(anls(item['answer'], gold_texts(record)))\n"
                "    return {{field: {{'n': len(values), 'anls': round(sum(values) / len(values), 2)}} for field, values in sorted(scores.items())}}\n\n\n"
                "frozen_fields = by_field(pipe, test_records)\n"
                "print({{'frozen_by_field': frozen_fields}})\n"
                "for record in test_records[:3]:\n"
                "    item = pipe.answer(record['question'], words=record['words'], boxes=record['boxes'], image_size=record['image_size'])\n"
                "    print({{'question': record['question'], 'frozen': item['answer'], 'gold': gold_texts(record)}})\n"
                "assert frozen_test['anls'] > baseline_last['anls']"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the last encoder blocks and the span head\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_ENCODER_LAYERS` encoder blocks plus the span head "
                "`qa_outputs` — four blocks by default, 28,353,026 of 127,792,898 parameters; the word, position and 2-D box "
                "embeddings and the earlier blocks stay frozen — with start/end cross-entropy on the first and last token of "
                "the gold word span, AdamW at a fixed learning rate, gradient clipping at 1.0, dynamic padding, seeded "
                "shuffling and no scheduler. Nothing is truncated: every training record passed the fit check. Epoch 0 records "
                "the frozen model's validation ANLS; every epoch is scored on the validation receipts, and the epoch with the "
                "highest validation ANLS is kept.\n\n"
                "Watch validation ANLS move from about 0.82 to about 0.96 over six epochs (about two and a half minutes per epoch on "
                "CPU, validation scoring included; the build record kept epoch 5). The build record's counter-examples are in the model card — two trainable "
                "blocks reach about 0.91 test ANLS in the same six epochs; the default is the configuration that captured most "
                "of the gain at the same CPU cost."
            ),
            "code": (
                "EPOCHS = 6  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 3e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 16  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_ENCODER_LAYERS = 4  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row['val_anls'] = round(entry['val']['anls'], 4)\n"
                "        row['val_exact_match'] = round(entry['val']['exact_match'], 4)\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_encoder_layers=TRAINABLE_ENCODER_LAYERS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test receipts were never used for training or epoch selection, and no test page — by id or by word "
                "content — appears in the training or validation splits. The adapted model is scored exactly as the frozen "
                "model was in Section 6, the four numbers are put side by side and the per-field breakdown is repeated. Look "
                "for an ANLS gain of several points concentrated in the fields the frozen model missed — the cell asserts the "
                "adapted test ANLS is above the frozen one — and for the same three questions answered by the adapted model. "
                "Two hundred and twenty-nine questions over 50 receipts from one seeded split of one corpus give **no "
                "dispersion estimate**; the deltas are sample-sanity evidence that the adaptation contract works, not a "
                "benchmark, and a gain on CORD receipts with their annotated OCR says nothing about your documents or your OCR "
                "until you measure it there."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "adapted_fields = by_field(pipe, test_records)\n"
                "comparison = {{\n"
                "    'anls': {{'last_number': round(baseline_last['anls'], 3), 'keyword_lookup': round(baseline_lookup['anls'], 3), 'frozen': round(frozen_test['anls'], 3), 'adapted': round(adapted_test['anls'], 3)}},\n"
                "    'exact_match': {{'last_number': round(baseline_last['exact_match'], 3), 'keyword_lookup': round(baseline_lookup['exact_match'], 3), 'frozen': round(frozen_test['exact_match'], 3), 'adapted': round(adapted_test['exact_match'], 3)}},\n"
                "    'delta_vs_frozen': {{'anls': round(adapted_test['anls'] - frozen_test['anls'], 3), 'exact_match': round(adapted_test['exact_match'] - frozen_test['exact_match'], 3)}},\n"
                "    'by_field': {{field: {{'n': frozen_fields[field]['n'], 'frozen': frozen_fields[field]['anls'], 'adapted': adapted_fields[field]['anls']}} for field in frozen_fields}},\n"
                "}}\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "for record in test_records[:3]:\n"
                "    item = pipe.answer(record['question'], words=record['words'], boxes=record['boxes'], image_size=record['image_size'])\n"
                "    print({{'question': record['question'], 'adapted': item['answer'], 'gold': gold_texts(record)}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'pages': pages,\n"
                "    'fit_check': {{name: {{'fitting': f['n_fitting'], 'dropped': f['dropped']}} for name, f in fit.items()}},\n"
                "    'baselines': {{'last_number': baseline_last, 'keyword_lookup': baseline_lookup}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['anls'] > frozen_test['anls']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Re-read the invoice, export the adapter and reload it\n\n"
                "The five authored invoice questions from Section 5 are answered again by the adapted model — a page from a "
                "different document family than the receipts it was tuned on, so this is a small look at what the adaptation "
                "did *outside* its corpus (the build record kept 5/5; a lost answer here is a finding to record, not a "
                "failure) — and scored with the per-page `evaluation_report`, whose verdict is `sample-sanity` because five "
                "authored pairs on one perfectly-OCR'd page carry no dispersion estimate. The answers are written as CSV and "
                "an annotated PNG draws the adapted answer spans' boxes on the page.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the last four encoder blocks and the span head, about "
                "113 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id "
                "and revision, the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the "
                "training configuration and the epoch history (OUT8). `LayoutLMDocumentQAPipeline.from_artifact` re-verifies "
                "the base snapshot, checks the artifact manifest, its digest and its exact tensor set **before** deserialising, "
                "refuses any tensor that is not an adaptable encoder-block or span-head tensor, and overlays the tensors onto "
                "a freshly loaded base — a new object from files, not the in-memory model (VER2). The cell asserts identical "
                "answers on eight test receipts (VER4)."
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "adapted_results = [pipe.answer(question, words=words, boxes=boxes, image_size=image.size) for question in questions]\n"
                "adapted_form = evaluation_report(adapted_results, golds, sample_kind='synthetic')\n"
                "for before, after in zip(results, adapted_results, strict=True):\n"
                "    print({{'question': before['question'], 'frozen': before['answer'], 'adapted': after['answer'], 'gold': golds[questions.index(before['question'])]}})\n"
                "print({{'invoice_after_adaptation': {{m['id']: round(m['value'], 3) for m in adapted_form['metrics']}}, 'verdict': adapted_form['verdict'], 'reason': adapted_form['reason'][:80] + '...'}})\n"
                "with open('outputs/{stem}_answers.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'question', 'frozen_answer', 'adapted_answer', 'adapted_score', 'start', 'end', 'gold'])\n"
                "    for before, after, gold in zip(results, adapted_results, golds, strict=True):\n"
                "        writer.writerow([image_name, after['question'], before['answer'], after['answer'], f\"{{after['score']:.6f}}\", after['start'], after['end'], ' | '.join(gold)])\n"
                "COLOURS = [(200, 30, 30), (0, 140, 0), (40, 90, 220), (200, 120, 0), (130, 0, 160)]\n"
                "annotated = image.convert('RGB').copy()\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "for index, item in enumerate(adapted_results):\n"
                "    if item['start'] is None:\n"
                "        continue\n"
                "    for box in boxes[item['start'] : item['end'] + 1]:\n"
                "        draw.rectangle([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2], outline=COLOURS[index % len(COLOURS)], width=2)\n"
                "annotated.save('outputs/{stem}_annotated.png')\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = LayoutLMDocumentQAPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [pipe.answer(r['question'], words=r['words'], boxes=r['boxes'], image_size=r['image_size'])['answer'] for r in test_records[:8]]\n"
                "after = [reloaded.answer(r['question'], words=r['words'], boxes=r['boxes'], image_size=r['image_size'])['answer'] for r in test_records[:8]]\n"
                "parity = {{'identical_answers': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_answers'] == parity['of']\n\n"
                "weight_entry = next(entry for entry in MANIFEST['files'] if entry['path'] == WEIGHT_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'column': CORPUS_COLUMN, 'files': CORPUS_FILES}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'page': {{'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'n_words': len(words)}}, 'items': [{{k: r[k] for k in ('question', 'answer', 'start', 'end', 'score', 'n_windows', 'seconds')}} for r in results], 'golds': golds, 'frozen_report': frozen_form, 'adapted_report': adapted_form}},\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen model reads most receipt totals already — its test ANLS sits well above both non-neural baselines — and a "
        "bounded fine-tuning of the last four encoder blocks and the span head on 119 receipts lifts held-out ANLS by several "
        "points, concentrated in the fields it had been missing (item names, quantities, cash and change lines), in about a quarter of an "
        "hour on CPU, with a 113 MB adapter that reloads to identical answers. That is the claim: the adaptation contract "
        "works end to end on a real gold-span document corpus whose OCR the model never sees as pixels, and the numbers it "
        "produces are read against two non-neural baselines and the frozen model rather than in isolation.\n\n"
        "The test split is 229 questions over 50 receipts from one seeded split of one corpus, the questions are templated from "
        "field categories rather than written by people, the metrics are two reference-based scores (own implementations of "
        "DocVQA's normalisation, neither a human judgement), and the OCR is CORD's annotation — clean words in reading order "
        "with tight boxes, which no OCR engine on a photographed receipt reproduces. So a gain here says the contract works, "
        "not that the adapted model is better on your documents or your OCR, that it handles other languages, handwriting or "
        "long pages, or that its spans are faithful — a reader returns a plausible wrong span for an unanswerable question, and "
        "the adaptation changes nothing about that. Fine-tuning on a narrow corpus can also erode the model elsewhere; the "
        "invoice re-read in Section 9 is one page of evidence about that, not a measurement.\n\n"
        "Three things to carry to real data. **Baselines first:** the last-number and keyword-lookup baselines and the frozen "
        "model's score on *your* gold spans, with *your* OCR, are the numbers to read before any adapted one, per field. "
        "**Leakage:** keep every question on a page in one split (the contract does this) and split by document or by "
        "vendor when your pages come from few sources, never at random over near-duplicate pages — CORD-v2 itself contains "
        "one duplicated receipt, which the sample drops. **Ceilings:** a question + page over one 512-token window is answered "
        "window by window at inference and dropped from training by the fit check; long-document training is out of scope.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify one column of a real "
        "annotated corpus, validate the demonstrated dataset contract without leakage, execute the inference contract and a "
        "bounded fine-tuning, evaluate against two trivial baselines and the frozen model on a page-disjoint split, and emit "
        "the shown machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark "
        "superiority, document-understanding accuracy on any other domain or OCR, a usable rejection threshold, or production "
        "fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_ENCODER_LAYERS = 2` and compare the "
        "artifact size and the test scores; raise `EPOCHS` and watch the validation ANLS pick the epoch; ask the adapted model "
        "`What is the delivery address?` on the invoice and read the score of a span the page cannot support; or bring your "
        "own annotated pages through BYOD and read the two baselines before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/layoutlm-document-qa-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/layoutlm-document-qa-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/layoutlm-document-qa-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model (Impira, MIT): https://huggingface.co/{MODEL_ID}\n"
        "- Upstream architecture code (LayoutLM, Microsoft): https://github.com/microsoft/unilm/tree/master/layoutlm\n"
        "- LayoutLM: Pre-training of Text and Layout for Document Image Understanding (Xu et al., 2019): https://arxiv.org/abs/1912.13318\n"
        "- DocVQA: A Dataset for VQA on Document Images (Mathew, Karatzas, Jawahar, 2020): https://arxiv.org/abs/2007.00398\n"
        "- Scene Text Visual Question Answering — the ANLS metric (Biten et al., 2019): https://arxiv.org/abs/1905.13648\n"
        "- CORD: A Consolidated Receipt Dataset for Post-OCR Parsing (Park et al., NeurIPS 2019 Document Intelligence Workshop; CORD-v2, CC BY 4.0): https://huggingface.co/datasets/naver-clova-ix/cord-v2\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
