"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "layoutlm_document_qa_pipeline",
    "repo_name": "layoutlm-document-qa-pipeline",
    "stem": "layoutlm_document_qa",
    "notebook_name": "layoutlm_document_qa_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "LayoutLMDocumentQAPipeline",
    "weights_key": "layoutlm-document-qa",
    "runtime_imports": ["torch", "transformers"],
    "title": "LayoutLM Document QA (impira) — DIMER extractive document question answering tutorial (standalone)",
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
    "capability": "extractive document question answering — one question plus the page's OCR words and pixel boxes → the best answer span with its start/end word indices and span score — using the pinned `impira/layoutlm-document-qa` weights",
    "intro": (
        "At inference the LayoutLM (v1) encoder — a 12-layer BERT-style transformer with 2-D position embeddings, "
        "128M parameters, RoBERTa vocabulary, fine-tuned by Impira on SQuAD 2.0 and DocVQA — reads the question and the "
        "page's **words with their boxes** (normalised to a 0–1000 grid) as one token sequence and predicts a start and an "
        "end position; the pipeline scores every span of at most 15 tokens by the product of the start and end softmax "
        "probabilities and returns the best one as a word span. **The model never sees pixels:** OCR is an input provider "
        "outside the model, so the pipeline takes words and boxes from the caller (bring-your-own OCR) and offers an "
        "optional Tesseract adapter that is imported only when called. **No adaptation occurs:** no training, fine-tuning, "
        "in-context conditioning, or preprocessing fitting happens in this notebook — the upstream checkpoint supplies the "
        "weights and tokenizer, and the carried module adds snapshot verification, the input contract (words, one pixel box "
        "per word inside the page, a non-empty question up to 256 characters), windowing for long pages, a fixed output "
        "contract, and the `anls`, `exact_match`, `validate_inputs` and `evaluation_report` helpers. The default sample is "
        "an invoice-style form rendered in code whose renderer records every word's box, with five authored questions and "
        "accepted answers, so ANLS and exact-match are demonstration (plumbing) evidence for one page with perfect OCR, not a "
        "DocVQA benchmark."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, render a synthetic form whose words and boxes are known exactly (or bring your "
        "own page with your own OCR output and questions) and validate it into an input manifest, run the supported task, "
        "read the answers correctly (a word span, its indices, an uncalibrated span score), exercise an optional BYOD path, "
        "produce an evaluation report that is `sample-sanity` with `anls` and `exact_match` only when accepted answers exist "
        "and `not-measurable` otherwise, and export the answers, the annotated page with the answer boxes and provenance."
    ),
    "exclusions": (
        "OCR itself (the notebook installs no Tesseract and ships no OCR model; the default path uses the renderer's own "
        "word boxes and BYOD needs either `pytesseract` with a Tesseract binary already present or a JSON of words and "
        "boxes from your own OCR), PDF or multi-page documents (one page per call), abstention (the model always returns "
        "its best span, even for an unanswerable question), answers that are not a contiguous span of the OCR words, "
        "arithmetic or reasoning, evaluation on the DocVQA benchmark (registration-gated and not bundled), and any "
        "training. Answer quality is bounded by the OCR: a mis-read or mis-ordered word cannot be recovered by the model."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is more than adequate: the repository's model card records 5.1 s to load and 0.11–0.29 s per question on the 73-word rendered form in the Windows venv (Intel Core Ultra 9 275HX). The pinned `torch==2.14.0` install and the 511 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what extractive (span) question answering is; what a start/end softmax product means and why it is not calibrated; what normalised Levenshtein similarity (ANLS) measures.",
        "- **Data:** the default sample is a deterministic 850×1100 invoice-style form rendered in code with Pillow's bundled font, whose renderer records the pixel box of every word it draws (73 words) — perfect OCR by construction — with five authored questions and accepted answers, so nothing is downloaded and no private data is needed. Optional BYOD is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one page image decodable by Pillow plus **either** a `pytesseract`-importable Tesseract installation in the runtime **or** a JSON file `{\"words\": [...], \"boxes\": [[x0, y0, x1, y1], ...]}` in the page's pixel coordinates from your own OCR, plus your questions typed into the form field. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Render the synthetic form (words + boxes) or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own references: an invoice-style form — an `INVOICE` "
                "header, a supplier name and address, five labelled fields, a four-row line-item table, subtotal/VAT/total "
                "lines and a payment-terms sentence — is rendered with Pillow's bundled font at 850×1100, and the renderer "
                "records the pixel box of **every word it draws** (`ImageDraw.textbbox`), which is exactly the `words`/`boxes` "
                "input the model needs. That is perfect OCR by construction — real OCR mis-reads, splits and re-orders words, "
                "and nothing here measures that. Five questions are authored against the page with their accepted answers; "
                "they are the references for the `anls` and `exact_match` sanity checks later, not a labelled dataset. The "
                "image digest is printed for the record.\n\n"
                "BYOD is optional and disabled by default. When enabled, upload one page image; the notebook then tries the "
                "carried `ocr_words_with_tesseract` adapter (which needs `pytesseract` and a Tesseract binary **already present** "
                "in the runtime — the notebook installs neither) and, failing that, asks you to upload a JSON of words and pixel "
                "boxes from your own OCR. Type your questions one per line. No accepted answers exist for them, so the "
                "evaluation report will be `not-measurable`. Nothing is validated in this cell — the next section hands the "
                "words, boxes and questions to the pipeline's own validation stage, which is the only checker."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "byod_questions = 'What is the invoice number?\\nWho is the customer?'  # @param {{type:\"string\"}}\n\n\n"
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
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    try:\n"
                "        words, boxes = ocr_words_with_tesseract(image)\n"
                "        ocr_source = 'tesseract (pytesseract present in this runtime)'\n"
                "    except RuntimeError as exc:\n"
                "        print(f'{{exc}} -- upload a JSON file with \"words\" and pixel \"boxes\" from your own OCR')\n"
                "        ocr_upload = files.upload()\n"
                "        ocr_payload = json.loads(next(iter(ocr_upload.values())).decode('utf-8'))\n"
                "        words, boxes = list(ocr_payload['words']), [list(b) for b in ocr_payload['boxes']]\n"
                "        ocr_source = 'caller-supplied JSON'\n"
                "    questions = [line.strip() for line in byod_questions.splitlines() if line.strip()]\n"
                "    golds = None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic synthetic form: no randomness, so no seed is needed and the digest is stable per Pillow build.\n"
                "    image, words, boxes, qa = synthetic_form()\n"
                "    questions, golds = [q for q, _ in qa], [g for _, g in qa]\n"
                "    image_name = 'synthetic_invoice_850x1100.png'\n"
                "    ocr_source = 'renderer word boxes (perfect OCR by construction)'\n"
                "    sample_kind = 'synthetic'\n\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'size': image.size, 'rgb_sha256': image_sha256, 'n_words': len(words), 'ocr_source': ocr_source, 'n_questions': len(questions), 'has_golds': golds is not None}})\n"
                "print(' '.join(words[:24]) + ' …')"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `answer` applies — "
                "a page size within `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE`, 1..`MAX_WORDS` non-empty words with exactly one "
                "pixel box each lying inside the page, and each question a non-empty string of at most `MAX_QUESTION_CHARS` "
                "characters (whitespace collapsed) — and returns an **input manifest** naming the schema (including the token "
                "encoding, the windowing and the span-scoring rule), the page size and word count, the checked questions and "
                "the verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks "
                "like, the cell also validates a box that lies outside the page and records the pipeline's own error message "
                "as a finding. Inside the pipeline each box is normalised to LayoutLM's 0–1000 grid; the question and the "
                "words are tokenised together and, when they exceed 512 tokens, split into overlapping windows. The pipeline "
                "cannot tell whether the OCR is right or whether the question is answerable: that contract is the caller's."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_WORDS': MAX_WORDS, 'MAX_QUESTION_CHARS': MAX_QUESTION_CHARS, 'MAX_SEQ_LEN': MAX_SEQ_LEN, 'DOC_STRIDE': DOC_STRIDE, 'MAX_ANSWER_TOKENS': MAX_ANSWER_TOKENS, 'BOX_GRID': BOX_GRID}}}})\n"
                "input_manifest = validate_inputs(words, boxes, questions, image_size=image.size, names=[image_name])\n"
                "# Demonstrate rejection on a request that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(words[:1], [[0, 0, image.width + 1, 10]], questions, image_size=image.size)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'box-outside-page-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Answer the questions and read the scores correctly\n\n"
                "`answer` returns, per question, a dict with `answer` (the selected word span joined by spaces), `start` and "
                "`end` (inclusive word indices into your `words`, so the span's boxes are known), `score`, `n_words`, "
                "`n_windows` (how many 512-token windows the page needed), the checked `question` and the model identity. "
                "The `score` is the **product of the start and end softmax probabilities of the chosen span within its "
                "window under the model's own head** — a ranking signal over spans of this page, not a calibrated probability "
                "that the answer is right, and never a signal that the question is answerable: the model always returns its "
                "best span. As recorded in the model card, the repository's CPU smoke on this same form answered all five "
                "authored questions exactly with scores of 0.999–1.000 — and answered `14 Harbour Road,` at 0.28 to \"What "
                "is the delivery address?\", a question the page cannot answer. The lower score is suggestive, not a rule; a "
                "deployment that wants to reject answers must choose its own threshold on its own labelled pages. Inference is "
                "deterministic on a fixed device and dtype (no sampling); CUDA kernel selection can move scores in the third "
                "or fourth decimal place."
            ),
            "code": (
                "import time\n\n"
                "results, seconds = [], []\n"
                "for question in questions:\n"
                "    t0 = time.time()\n"
                "    results.append(pipe.answer(question, words=words, boxes=boxes, image_size=image.size))\n"
                "    seconds.append(round(time.time() - t0, 2))\n"
                "print({{'device': pipe.device, 'seconds_per_question': seconds, 'n_windows': results[0]['n_windows']}})\n"
                "for result in results:\n"
                "    print(f\"Q: {{result['question']}}\\n   A: {{result['answer']!r}}  score {{result['score']:.4f}}  words {{result['start']}}..{{result['end']}}\")"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No accuracy is "
                "reported by default: DocVQA-style accuracy needs labelled question/answer pairs on pages from the deployment "
                "domain **with the deployment's own OCR**, and this repository ships none (the DocVQA benchmark itself is "
                "registration-gated). The repository's metric helpers are `anls` — Average Normalised Levenshtein Similarity, "
                "the benchmark's official metric: `1 − edits / max(len)` over normalised strings, maximised over the accepted "
                "answers, scored 0 below the 0.5 threshold — and `exact_match` after the same normalisation (lower-case, "
                "punctuation removed, whitespace collapsed). When accepted answers are supplied the report carries the mean "
                "`anls`, the `exact_match` rate and one entry per question (with its span score), verdict `sample-sanity`. On "
                "the synthetic path those answers are values **you rendered yourself** and the OCR is perfect by construction, "
                "so a perfect score proves only that the input contract, encoding, forward pass and span decoding round-trip. "
                "On BYOD no accepted answers exist, the verdict is `not-measurable`, and the report states what would make the "
                "task measurable. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(results, golds, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k not in ('metrics', 'per_question')}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    print(f\"{{metric['id']:12}} {{metric['value']:.3f}}  ({{metric['estimation']}})\")\n"
                "for entry in report.get('per_question', []):\n"
                "    print(f\"  anls {{entry['anls']:.2f}}  exact {{str(entry['exact_match']):5}}  score {{entry['score']:.3f}}  {{entry['question']}} -> {{entry['prediction']!r}} (accepted: {{entry['golds']}})\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No accepted answers exist for these questions, so nothing is scored; read the answers against the page yourself.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves every result (question, answer, span indices, score, window count), the "
                "evaluation report, the input manifest, the sample identity, digest, OCR source, words, boxes and accepted "
                "answers, the notebook's source (repository, revision, embedded module digest, generator), the model "
                "identifier, the immutable model revision, the model licence, and the runtime identity (Python, `torch`, "
                "`transformers`, device). The question/answer pairs are also written as CSV with explicit `image`, `question`, "
                "`answer`, `score`, `start`, `end` columns, and an annotated PNG draws the answer span's word boxes on the page "
                "(one colour per question) for visual inspection — a supplement to, not a replacement for, the machine-readable "
                "files. No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "COLOURS = [(200, 30, 30), (0, 140, 0), (40, 90, 220), (200, 120, 0), (130, 0, 160)]\n"
                "annotated = image.convert('RGB').copy()\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "for index, result in enumerate(results):\n"
                "    if result['start'] is None:\n"
                "        continue\n"
                "    colour = COLOURS[index % len(COLOURS)]\n"
                "    for box in boxes[result['start'] : result['end'] + 1]:\n"
                "        draw.rectangle([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2], outline=colour, width=2)\n"
                "annotated.save('outputs/{stem}_annotated.png')\n"
                "payload = {{\n"
                "    'predictions': results,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'ocr_source': ocr_source, 'words': words, 'boxes': boxes, 'questions': questions, 'accepted_answers': golds}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_answers.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'question', 'answer', 'score', 'start', 'end'])\n"
                "    for result in results:\n"
                "        writer.writerow([image_name, result['question'], result['answer'], f\"{{result['score']:.6f}}\", result['start'], result['end']])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The answers are spans of the OCR words the model was given; the model never saw the pixels, so everything rests on "
        "the OCR — on the synthetic form the words and boxes come from the renderer and are perfect, which no real OCR is. "
        "The span score ranks spans within one page under the model's own head and is not calibrated: the smoke run's "
        "unanswerable question still got a span (`14 Harbour Road,` at 0.28), and a deployment must choose and validate its "
        "own rejection threshold on labelled pages. On the synthetic form the `anls` and `exact_match` values in the "
        "evaluation report compare the answers with values you rendered yourself and the verdict is `sample-sanity`, which "
        "proves only that the input contract, encoding, forward pass and span decoding work (the repository's smoke run "
        "scored 5/5 exact on this page); they say nothing about real OCR errors, scans, handwriting, non-Latin scripts, "
        "questions needing arithmetic or reasoning, answers that are not contiguous spans, or pages longer than one window, "
        "and a BYOD result is a single-page observation with the verdict `not-measurable`. **The model returns a span for "
        "every question**, and a page with more than 512 tokens is answered window by window with the best span across "
        "windows. The pipeline provides no OCR, no abstention, no multi-page handling, no benchmark evaluation and no "
        "training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** ask `What is the delivery address?` and compare its score with the five answerable ones; "
        "corrupt one OCR word (`words[12] = 'NW-2O26-O417'`) and watch the answer inherit the error; repeat the words nine "
        "times to exceed one window and read `n_windows`; enable `USE_BYOD` with a page you know and your own OCR JSON, then "
        "pass your own accepted answers to `evaluation_report` to see the verdict switch to `sample-sanity`.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/layoutlm-document-qa-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/layoutlm-document-qa-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/layoutlm-document-qa-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream architecture code (LayoutLM, Microsoft): https://github.com/microsoft/unilm/tree/master/layoutlm\n"
        "- LayoutLM: Pre-training of Text and Layout for Document Image Understanding (Xu et al., 2019): https://arxiv.org/abs/1912.13318\n"
        "- DocVQA: A Dataset for VQA on Document Images (Mathew, Karatzas, Jawahar, 2020): https://arxiv.org/abs/2007.00398\n"
        "- Scene Text Visual Question Answering — the ANLS metric (Biten et al., 2019): https://arxiv.org/abs/1905.13648"
    ),
}
