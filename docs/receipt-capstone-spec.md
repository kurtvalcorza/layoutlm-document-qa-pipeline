# DIMER Capstone 8 Specification
# From Receipts to Records — Small-Business Document Intelligence

**Version:** 1.0  
**Date:** 28 September 2026  
**Document state:** Implementation specification; notebook and execution evidence not yet produced  
**Notebook profile / mode:** `E2E` / `GUIDED`  
**Normative baseline:** DIMER Notebook Specification 2.2  
**Proposed host:** `kurtvalcorza/layoutlm-document-qa-pipeline`  
**Proposed branch:** `feat/small-business-receipt-capstone`  
**Notebook:** `tutorials/DIMER_Small_Business_Receipt_Intelligence_Capstone.ipynb`  
**Intended runtime:** Fresh Google Colab, Python 3.12 isolated environment, NVIDIA T4  
**Default dataset:** CORD v2, public Indonesian receipt sample  
**Initial implementation status:** Candidate, once built; no release qualification implied

---

## 1. Purpose and central experiment

Build a standalone, self-paced notebook that starts with receipt photographs, extracts four financial fields into auditable records, measures errors against published annotations, and demonstrates a validation-selected policy for referring uncertain **total amounts** for human review.

**Central question**

> Can an AI-assisted workflow extract receipt amounts reliably, and how does referring uncertain totals for review change the error rate among the totals left unflagged?

The notebook must distinguish three questions:

1. How accurately does the complete **image → OCR → extraction** workflow recover the annotated amounts?
2. Does receipt-specific, bounded gradient fine-tuning improve on a transparent rule baseline and the unchanged pretrained model?
3. What coverage–error tradeoff does a validation-selected review policy achieve on held-out receipts?

This is a document-processing and experimental-methods capstone, not an accounting application. Its strongest learning outcome is recognizing that readable OCR, a confident-looking answer, and a numerically correct business record are different achievements.

### 1.1 Input → system → output

```text
Receipt image
  → image validation and deterministic orientation handling
  → Tesseract OCR: words, bounding boxes, line identifiers, recognition scores
  → three competing field extractors
      A. fixed keyword-and-number rules
      B. frozen LayoutLM document QA
      C. receipt-adapted LayoutLM document QA
  → conservative amount normalization and evidence checks
  → total-amount review policy + per-field status
  → CSV / JSON records, evaluation evidence, trained adapter and reload verification
```

A separate diagnostic substitutes CORD's annotated words and boxes for OCR. It must never replace the image-based primary experiment or be presented as deployable image-to-record performance.

### 1.2 What this specification fixes

Unless explicitly identified as a build-time resolution, the choices in this document are the version 1 design, not a menu of alternatives. `MUST`, `MUST NOT`, and `SHOULD` describe implementation requirements. Numeric training settings and review-policy targets below are design settings, not measured results or claimed business requirements.

This specification authorizes no repository write, pull request, merge, deployment, external publication, or paid execution by itself.

---

## 2. Source basis and important implementation boundaries

### 2.1 Verified source facts

The following facts were checked while preparing this specification:

| Source | What it establishes | Consequence for this capstone |
|---|---|---|
| DIMER Notebook Specification 2.2 [S1] | Standalone execution, actual default-path adaptation for `E2E`, optional but substantive BYOD, source/hosted-evidence separation, guided teaching requirements | Embed the implementation; do not clone/import DIMER source at runtime or skip training by default. |
| CORD publisher repository [S2] and dataset metadata [S3] | The public v2 sample contains 800 training, 100 validation and 100 test receipts; Indonesian provenance; images and parsing annotations; CC BY 4.0 | Preserve source partition membership and attribution. Do not call it Philippine receipt evidence. |
| Current LayoutLM repository [S4] | The wrapped checkpoint consumes question, words and boxes; it does not consume receipt pixels. It supports bounded last-layer fine-tuning and serialized adapter reconstruction | Add and qualify an actual OCR stage; reuse compatible model semantics rather than implying OCR already exists in the package. |
| Current `samples.py` [S5] | The existing tutorial reads annotations from the **official validation and test shards**, then forms its own 119/30/50 page split | **Do not reuse its sample loader, partitioning helper, trained adapter, or sample-performance claims for this capstone.** |
| Current `pipeline.py` [S6] | Inference selects a document span; the decoder does not expose a learned no-answer decision. Its trainer selects by validation ANLS and includes epoch 0 | Review flags are not calibrated no-answer probabilities. Implement the capstone's different checkpoint-selection rule explicitly. |
| Tesseract documentation [S7–S8] | Tesseract can expose recognized words and geometry; engine, segmentation and language settings are material configuration | Pin the OCR executable, dependencies and language data, not merely a Python wrapper. |

The existing tutorial is a separately scoped experiment. Its source partition choices do not invalidate its documented within-tutorial split, but reusing its fitted state would contaminate this capstone's claimed official holdout.

### 2.2 Source snapshot identifiers

Record the following as preparation-time references, then recheck current repository instructions before implementation:

- Notebook specification Git blob: `7428d5becb8d37133a3ad93a450d08ebf616f411`.
- LayoutLM README Git blob: `b0fd51a71a3a504a411c7e721d9c64dc90e48fc6`.
- LayoutLM `samples.py` Git blob: `033d684ee6ecafda413dd3abd02d3c3269d1d9f5`.
- LayoutLM `pipeline.py` Git blob: `3a553f3699a708040660f74b2e372b2861b020e5`.
- CORD publisher README Git blob: `a2aca8ff89fcd794b1181c897a876ee4e257c61e`.
- Proposed base checkpoint, already pinned by the host: `impira/layoutlm-document-qa`, revision `beed3c4d02d86017ebca5bd0fdf210046b907aa6`.
- Candidate dataset revision inherited from the host's provenance: `naver-clova-ix/cord-v2@7f0115a4b758a71d6473b8d085751692da2fef98`. Independently verify **all three image-bearing partitions** before accepting this pin. The host's annotation-column verification is not image verification.

Git blob IDs are source identities, not whole-notebook execution evidence. Complete model, data and dependency manifests remain build outputs; do not invent their hashes in the specification.

### 2.3 Governing handoff

Preserve the supplied capstone handoff's rules: CPU-only local validation, hosted pretrained/GPU qualification, feature branches, no implicit merge authority, standalone generated notebooks, validation-only selection, safe artifact reconstruction, optional reflection rather than required submissions, and separate build/test/release/publication states [S9].

Do not reinstate the suspended project-wide “Gate 0.” The task-specific data, compatibility and release checks in this specification do not change that project decision.

---

## 3. Scope, exclusions and learner outcomes

### 3.1 Version 1 target fields

| Canonical output | CORD v2 category / parsed path to verify | Fixed question |
|---|---|---|
| `total_amount` | `total.total_price` | What is the total amount? |
| `subtotal_amount` | `sub_total.subtotal_price` | What is the subtotal? |
| `tax_amount` | `sub_total.tax_price` | What is the tax amount? |
| `service_charge` | `sub_total.service_price` | What is the service charge? |

The publisher README uses `subtotal` in its class-description table; the host's v2 loader and question templates use `sub_total`. Version 1 must verify the actual pinned JSON representation and preserve that spelling in the parser. Any compatibility aliases must be explicit, versioned and tested; do not silently guess schema keys [S2, S5].

All four questions are requested on each valid image. Reference annotations must not determine which questions the inference code asks.

**Primary task:** Recover the annotated total amount.  
**Secondary tasks:** Recover subtotal, tax and service charge where a usable source reference exists.  
**Review-policy task:** Route total-amount candidates, not certify the complete accounting record.

### 3.2 Explicit exclusions

Version 1 does not implement full line-item parsing, merchant/customer identification, receipt dates, document-type classification, handwriting recognition, currency conversion, tax compliance, invoice approval, fraud detection, reconciliation against bank records, or automatic posting to a ledger. It does not claim that missing optional annotations establish that no charge was printed or incurred.

No claim of Philippine, unseen-merchant, unseen-template, production, or accounting-grade performance is permitted without a separate appropriate evaluation. A Philippine BYOD exercise is a new evaluation, not an extension of the CORD score.

### 3.3 Audience and observable outcomes

Assume basic Python/Colab familiarity but no prior document-AI experience. By completing the default path, a learner should be able to:

1. Explain the difference between OCR, field extraction, numeric normalization and human verification.
2. Inspect recognized words and boxes and trace an extracted amount back to its image evidence.
3. Compare a rule baseline, frozen model and genuinely fine-tuned model on the same receipts.
4. Explain why an extractive model cannot restore a digit that OCR failed to recognize.
5. Interpret total exact match, annotation-conditional field accuracy, review coverage and selective error.
6. Use validation receipts to change one review-policy setting without changing the held-out experiment.
7. Export a trained adapter and reconstruct matching predictions from serialized files.
8. Identify what additional evidence would be needed before using the workflow on local business records.

Use “notebook” for the executable artifact and “workshop” only for a facilitated event.

---

## 4. Dataset acquisition, provenance and permissions

### 4.1 Canonical sample

Use the complete public CORD v2 sample as the default starting population: 800 official training receipts, 100 official validation receipts and 100 official test receipts [S2–S3]. Do not substitute the existing host tutorial's smaller re-split sample.

The complete source is approximately 2.31 GB according to its published file listing; that is an acquisition estimate, not a measured capstone transfer or memory budget [S3]. Read shards incrementally and release image buffers; do not decode the entire collection into RAM.

A reduced sample may be used in clearly named CPU fixtures. It must not silently become the default capstone experiment. Any future `QUICK` variant needs its own frozen manifest, counts and evidence label.

### 4.2 Acquisition requirements

The build must produce a manifest with dataset ID/revision, source paths, original split, source row/image identifiers, byte counts, publisher-declared digests, independently computed downloaded-file hashes, decoded image-pixel hashes, annotation hashes, and stable receipt IDs.

For full downloaded shards, verify the SHA-256 of the actual complete bytes. For a range-based implementation, separately record the publisher's whole-file declaration and the hashes of bytes/rows actually read; never describe an uncomputed full-file hash as locally verified.

Download only from declared public upstream hosts. Use immutable URLs, bounded retries, explicit timeouts, temporary files and atomic rename after verification. Verify reused caches. Never silently select `main`, a mirror, another dataset version, synthetic receipts, or reference OCR because acquisition fails.

### 4.3 Licensing and privacy

CORD's public data license is CC BY 4.0; keep data attribution separate from code, model and OCR-component licenses [S2–S3]. Record changes such as EXIF orientation, box conversion and amount normalization. Do not relabel source annotations or photos as original DIMER data.

The default result ZIP should omit source receipt photographs and full-page OCR transcripts. Runtime-local images remain available for evidence overlays and reconstruction checks. A trained artifact must not embed receipt images, training examples, personal identifiers or private source paths.

BYOD stays within the chosen hosted notebook runtime; “local notebook execution” does not mean on the participant's own computer. Do not send receipt content to an OCR API, LLM API, analytics service or repository. Downloading ordinary dependencies or pinned weights must not transmit input documents.

Before BYOD selection, state that users must be authorized to process the documents in that environment, should remove unnecessary sensitive content, and must review exports before sharing. Do not promise that model weights are privacy-free merely because raw training examples are excluded.

### 4.4 Source integrity audit before model experiments

Audit all source records without inspecting model predictions. Report partition counts, image decode errors, annotation/schema errors, exact image/annotation duplicates, field frequencies, multiple values, reference ambiguities and source/reference disagreements.

Use byte and canonical pixel hashes for exact duplicates. Record suspected near duplicates separately; automated similarity is not proof of identity. Quarantine an exact duplicate family that crosses official roles from all primary roles, rather than moving records between roles. Within a role, retain a deterministic canonical representative. Record every removed ID and reason. Do not replace removed records to restore attractive sample sizes.

After those source-integrity decisions, freeze the analysis cohort. **OCR failures, missing predicted amounts, long documents, low model scores and wrong predictions must not remove validation or test receipts from that frozen cohort.**

---

## 5. Ground-truth contract and missingness

### 5.1 Separate references from model inputs

Keep image-derived inputs and annotation-derived references in different objects and stage directories. The primary predictors receive only image/OCR data, fixed field questions and frozen configuration. They must not receive `gt_parse`, `valid_line.category`, `is_key`, gold spans, source target values or reference field-presence flags.

Annotations may be used for training supervision, source-quality audit, evaluation and the separately labeled reference-text diagnostic. Record the purpose each time they are accessed.

### 5.2 Reference field states

Each receipt–field pair has one of these reference states:

- `present_usable`: a unique source value and unambiguous canonical amount can be established under the frozen parser.
- `not_annotated`: no corresponding published annotation; **not equivalent to absent or zero**.
- `ambiguous_reference`: multiple incompatible values, conflicting representations, or ambiguous number notation.
- `invalid_reference`: malformed or unusable source structure.

Keep the original source strings and categories in the evaluation-only record. Compare `gt_parse` and the labeled value words where both exist. Do not silently prefer whichever agrees with a model prediction. Explicitly account for discounts, cash tendered and change as different categories; none may substitute for `total_amount`.

If a target occupies more than one annotation line, combine only when the source grouping and reading order unambiguously establish one value. Otherwise mark it ambiguous. Never add several source amounts to manufacture a field reference.

### 5.3 Human audit scope

Before qualification, maintain a deterministic, prediction-blind audit set drawn from training and validation, including all encountered reference-normalization patterns and disputed cases. Record exactly which receipts were visually checked, by whom, when, and what changed. AI-assisted inspection must not be labeled independent human review.

Published annotations remain published references, not newly verified financial truth. A post-evaluation correction requires a new reference-manifest version and a transparently labeled rerun. Retain the original result for comparison; do not silently revise the held-out ground truth.

Absence detection is not a scored claim unless an additional human-verified absence annotation layer is explicitly created. It is outside the default v1 experiment.

---

## 6. Split ownership and leakage prevention

### 6.1 Role assignment

Preserve official training, validation and test membership. Within the official validation partition, assign two disjoint roles by ascending SHA-256 of `"receipt-capstone-v1|42|" + stable_receipt_id`:

| Role | Source and nominal size | Permitted use |
|---|---|---|
| `train` | Official train, 800 before integrity exclusions | Gradient updates, keyword development, supervised OCR alignment, training-derived parsing decisions |
| `validation_model` | First 50 official validation IDs in frozen hash order | Select the trained epoch; inspect development behavior |
| `validation_policy` | Remaining 50 official validation IDs | Select total review thresholds and run the optional threshold exercise |
| `test` | Official test, 100 before integrity exclusions | One frozen evaluation of every predetermined system and diagnostic |

Do the deterministic validation assignment before exclusions; do not move receipts across the two validation roles afterward to improve support. Publish actual counts and the manifest digest.

Validation is development data. Its displayed accuracy is not independent evidence. Separate validation roles reduce reuse but do not make small-sample threshold estimates guarantees.

### 6.2 Prohibited reuse

Do not call the host's `fetch_sample_dataset()` or its sample re-split routine for this experiment. Do not load a previously CORD-adapted adapter from this repository or another capstone. Initialize both learned systems from the same verified upstream base; only the adapted system receives this capstone's training updates [S5].

Read the base model card and document known training provenance. Unknown checkpoint/pretraining overlap must remain unknown; official partition preservation does not prove absence from upstream training.

### 6.3 Freeze record

Before any test scoring, serialize a `selection_record.json` containing:

- cohort and role-manifest hashes;
- model/data/OCR identities;
- preprocessing, numeric grammar, keyword dictionary and question hashes;
- training recipe, chosen trained epoch and adapter digest;
- each system's review threshold, selection support and fallback state;
- metric definitions, failure semantics and diagnostic plan;
- a timestamp and digest of the complete selection record.

Test evaluation must refuse a changed selection record or unqualified upstream stage. Deterministic reruns of an unchanged experiment are allowed and receive a new execution record. They are not new independent test sets. Test-driven revision starts a new experiment and must disclose that the test has already been inspected.

---

## 7. OCR and image-processing contract

### 7.1 Version 1 OCR design

Use one Tesseract configuration for every primary system. Initial design: LSTM engine (`--oem 1`), single-column page segmentation (`--psm 4`), language order `eng+ind`, and TSV word output [S7–S8]. Resolve the exact compatible executable build and `tessdata_fast` revisions/hashes during implementation. The final runtime must assert the expected languages are present.

The Python wrapper is optional; a direct subprocess call is acceptable. Neither `pytesseract` alone nor a floating `apt install tesseract-ocr` constitutes a pinned OCR environment.

The installer must explicitly declare system packages, OCR executable, linked-library provenance, language assets and licenses. Prefer an immutable package snapshot with package versions and archive digests. If the hosted base image prevents reproducible installation, record that unresolved qualification issue rather than silently use a different executable.

No OCR execution or performance is claimed by this specification.

### 7.2 Image handling

Accept single-frame JPEG and PNG images. Validate before decoding expensive content: byte size, dimensions, format, frame count and decompression-bomb limits. Proposed per-image limits: 20 MiB, 20 megapixels, maximum side 10,000 pixels. Confirm these admit the frozen sample; any revision must precede experiment freeze.

Apply EXIF orientation deterministically and retain an invertible coordinate transform to the original image. Use the full image for primary OCR; never crop to a gold ROI or labeled amount box. Do not silently upscale, deskew, binarize or trim receipts. Any preprocessing added during development must be label-independent, explicit, frozen and applied to every primary system.

Use a 60-second per-image OCR timeout. A recoverable image/OCR failure generates a structured failure record and continues through the cohort; infrastructure corruption or a missing pinned OCR engine stops the run.

### 7.3 OCR representation

Emit stable OCR token IDs, exact recognized text, pixel `xyxy` boxes, block/paragraph/line/word identifiers, and raw Tesseract confidence. Keep word-level rows with nonempty text and valid nonzero geometry; record discarded structural rows and invalid words. Do not discard a recognized word solely for low confidence.

Keep the OCR engine's reading order, with a deterministic tie rule. Do not reorder using reference annotations. Preserve exact token-to-box correspondence and the coordinate-system identity.

Empty OCR and inputs beyond the documented 2,000-word model ceiling yield explicit failure/referral, not truncation. Model inference may use overlapping 512-token windows with the host's stride-128 and maximum-answer-token semantics. Report window counts and any cap; never score only the easiest window [S4, S6].

Cache keys must include image-pixel hash, orientation/preprocessing hash, executable/language-data hashes and all OCR settings. A changed key invalidates cached OCR.

---

## 8. Conservative numeric normalization

### 8.1 Raw and normalized values are different artifacts

Every field result must retain `raw_text`, OCR span/token references, `parse_status`, `normalized_amount` as a decimal string or null, and `number_format_policy_id`. JSON numeric floats must not be used as the canonical monetary representation.

Normalization may remove a separately recognized currency marker, normalize Unicode spaces, and interpret separators under a named grammar. It must not replace `O` with `0`, invent a missing digit, strip a minus sign, choose a convenient candidate, sum components into a total, or repair a value to satisfy arithmetic.

### 8.2 Default numeric grammar

Freeze an explicit CORD tutorial grammar after the source audit. The initial grammar supports nonnegative integers, consistent thousands grouping, and zero or two fractional decimal places:

- digit-only strings are integers;
- groups of three digits may use comma or dot consistently;
- a two-digit decimal suffix may use the other separator;
- a single separator followed by exactly two digits is decimal notation;
- a single separator followed by exactly three digits is grouping **under this declared grammar**, not evidence that the source currency has no fractional units;
- inconsistent grouping, unsupported precision, multiple independent amounts, percentage expressions, sign/refund cases and malformed strings are `ambiguous` or `unsupported`, never guessed.

Examples under this grammar: `75,000` and `75.000` normalize to `75000`; `75,000.00` and `75.000,00` also normalize to `75000`. `1,23,456` is rejected. These are grammar examples, not extracted dataset observations.

Apply the same frozen normalizer to prediction strings and source reference strings. Audit the financial interpretation of ambiguous source patterns; raw-text agreement and numeric interpretation are separate evidence. Do not fit the parser using test answers.

### 8.3 Currency and optional fields

Do not infer a per-receipt currency solely from the dataset's country. Record a recognized supported currency marker or an explicitly supplied currency contract; otherwise use `unspecified`. No conversion and no pooled monetary-error claim across unknown currencies.

`not_annotated`, `no_candidate`, `parse_failed`, and the numeric value zero are different states. The implementation must preserve those distinctions in tables, JSON and CSV.

BYOD must declare a supported number-format policy and currency context. Initial alternatives are `dot_decimal_comma_grouping` and `comma_decimal_dot_grouping`; the source-specific mixed CORD grammar must not silently become the Philippine BYOD convention.

---

## 9. Three-system comparison

### 9.1 System A — rules

Run fixed keyword-and-number extraction over the same OCR tokens used by the learned systems. Freeze a field-keyword dictionary developed from training receipts only, with explicit case/Unicode normalization, phrase-boundary matching, field precedence and tie rules.

Initial selection algorithm: find a compatible key phrase; search its line to the right for parseable amount spans; if none exists, inspect the immediately following OCR line within a declared geometry limit; rank by line priority and distance. A tie between distinct equally ranked values becomes `ambiguous`, not an arbitrary pick. `subtotal` must not match the `total` rule. Cash, change, discounts and item prices must not be silently treated as total.

The baseline is not permitted to use source categories or gold coordinates. Keep a count of candidate and ambiguity failures. Record the exact supporting key/value token IDs.

### 9.2 System B — frozen LayoutLM

Use the verified `impira/layoutlm-document-qa` base, fixed questions and the complete validated OCR document. No gradient update, CORD adapter or test-conditioned prompt change is allowed. Keep original answer spans and scores [S4, S6].

### 9.3 System C — receipt-adapted LayoutLM

Use the same base, OCR, questions, normalization, inference windowing and output contract. Train only the declared encoder blocks and span head using training receipts as specified in §10.

Loading systems sequentially is acceptable and preferred when needed for memory. Preserve frozen predictions before adaptation. Never obtain the supposed frozen baseline by running an in-memory model after it was trained.

### 9.4 Common comparison rules

Predict every target field for every valid image. Every frozen receipt gets an output record for every system, including structured failures. Use identical source-scoring masks. Do not compare reference-text rules with OCR-based neural extraction in the primary table.

The comparisons are about these three complete implementations, not universal superiority of rules, LayoutLM or foundation models.

---

## 10. Training-label alignment and bounded adaptation

### 10.1 Label alignment: training only

A span extractor can learn only from a gold value that can be mapped to the recognized document. Build training examples from **actual OCR**, not corrected reference text.

For each `present_usable` training field:

1. Obtain its source value-word region and canonical amount from the training annotation.
2. Enumerate contiguous OCR candidate spans within the supported answer-token limit.
3. Require canonical numeric equality, compatibility with the annotated value region, and a unique match. The initial geometry rule is at least 0.5 intersection-over-area of the OCR candidate box with the source value region after coordinate transformation.
4. Map the matched OCR word span to tokenizer positions. Require a complete span and one training window.
5. Record its supervised OCR indices; never replace the recognized tokens with reference tokens.

If multiple candidates survive, skip that training field as ambiguous. If OCR omitted or corrupted the amount, skip it as `target_not_recoverable_from_ocr`. If the document needs multiple windows, record `training_window_limit`. Do not turn these cases into false no-answer examples: the existing training contract is positive-span supervision [S6].

Record all skips by receipt and field. Report training eligibility and alignment coverage beside the final results. The primary validation/test denominators must not use this training eligibility mask.

Require at least the host-compatible minimum of eight valid training examples from multiple receipts. If that minimum fails, stop the `E2E` training stage with a clear data-contract error; do not synthesize successful supervision. Low but nonzero coverage must be discussed, not concealed.

### 10.2 Frozen recipe

| Setting | Version 1 design |
|---|---|
| Initialization | Fresh verified upstream base; no prior CORD adapter |
| Trainable tensors | Last four encoder blocks plus QA span head |
| Frozen tensors | Embeddings and earlier encoder blocks |
| Optimizer | AdamW |
| Learning rate / weight decay | `3e-5` / `0.01` |
| Epochs | 4, all executed by default |
| Batch size | 8 training examples |
| Precision | float32 for the initial qualified recipe |
| Gradient clipping | Global norm 1.0 |
| Scheduler | None |
| Seed | 42; record Python, NumPy and Torch seeds |
| Loss | Start/end span cross-entropy on aligned OCR tokens |
| Training ceiling | At most four target examples per training receipt; no unbounded augmentation |

These are proposed capstone settings. Measure their real hosted resource usage before describing them as practical or qualified. An out-of-memory result requires an explicit versioned recipe change, not a hidden CPU fallback, smaller cohort, precision change or skipped fine-tune.

Assert positive optimizer-step count, finite training loss, the exact trainable tensor list, and a nonzero change in at least one permitted tensor. Verify that frozen tensors did not change. Report actual trainable and total parameter counts from the loaded checkpoint.

### 10.3 Checkpoint selection differs from the existing helper

Evaluate every completed epoch using raw OCR for all `validation_model` receipts. Select the epoch with highest **total-amount normalized exact match** on its usable source references. Resolve ties by earliest epoch. Optional-field scores and training loss are descriptive, not additional tuning objectives.

The adapted-system candidate is selected from epochs **1–4**, not epoch 0. The unchanged base remains System B. This guarantees that the notebook exports the trained candidate while preserving the possibility that it performs worse than the frozen model.

The host's current `adapt()` selects by ANLS and may retain epoch 0 [S6]. Implement an explicit capstone-owned trainer/selection layer or a clearly tested compatibility extension; do not call the existing helper unchanged and misreport its selection criterion. Do not change the production pipeline's default semantics as a side effect.

A missing usable total reference in `validation_model` prevents the defined selection rule and must be resolved as a source-feasibility failure, not by selecting on the test set.

---

## 11. Reference-text and OCR-recoverability diagnostics

### 11.1 Reference-text comparison

After freezing the systems, repeat extraction on the same evaluation receipts with CORD's annotated words and boxes. Strip categories, `is_key`, gold answer indices and parsed target values before passing inputs to extractors. Preserve the natural page text, including printed labels.

Use the same learned weights and fixed questions. Do not retrain a reference-text-specific model in the default path. Label every result `reference_text_diagnostic`, not `end_to_end`.

The contrast changes recognition, geometry and reading order together. It is an **input-source sensitivity diagnostic**, not a formal causal decomposition and not a guaranteed upper bound. An OCR-trained model may perform worse on reference text; keep that result.

### 11.2 Recoverability diagnostic

For source-usable target values, check whether the correct normalized amount exists in a permitted OCR span and whether it is spatially compatible with the labeled value region. This uses labels and is strictly evaluator-owned. Report:

- OCR-recoverable amount proportion;
- extraction accuracy conditional on recoverability, explicitly secondary;
- nonrecoverable targets and ambiguity counts.

Do not filter the primary score by recoverability. Do not give the predictor this evaluator-owned flag. Avoid treating the full CORD annotation as an exhaustive full-page OCR transcription; removed or unannotated regions make naive whole-page CER/WER potentially misleading.

### 11.3 Failure inspection

Present a small, deterministically selected held-out panel with image overlays, OCR text, prediction span, reference, parse state and failure category. Include success, OCR loss, extraction error, numeric ambiguity and failure/referral where available. Selection occurs after scores are fixed and does not change metrics.

When a category has no example, say so. Never manufacture a real-model failure or success panel.

---

## 12. Evaluation definitions

### 12.1 Primary endpoint

For each system, on frozen test receipts with a `present_usable` total reference:

```text
Total EM = number with exactly matching canonical total / number with a usable total reference
```

OCR failures, missing candidates, unsupported/ambiguous predicted amounts and model failures count as incorrect in that denominator. No rounding tolerance is allowed for the primary amount match. An arbitrary field must not become correct because its text is “close” to the reference.

Report nominal source test count, integrity exclusions, frozen cohort count, usable-reference count and all outcome counts. A receipt with an unusable reference remains in operational/failure counts but not in a correctness denominator requiring unavailable truth.

**Predetermined primary contrast:** adapted LayoutLM minus frozen LayoutLM in total EM. Also report both learned systems against the rule baseline; retain negative or zero differences.

### 12.2 Secondary endpoints

| Measure | Definition / limitation |
|---|---|
| Per-field normalized EM | Exact amount match on that field's usable source references; print each denominator |
| Field-macro EM | Unweighted average across fields with usable references; show which fields contribute |
| All-annotated-fields match | Receipt-level success on every usable, source-annotated target field, for receipts with a usable total; does **not** establish correctness of unannotated fields |
| Text exact match / ANLS | Diagnostic answer-string agreement; never substitute for numeric exactness |
| Candidate and parse coverage | Fractions producing a candidate / supported normalized amount |
| Failure rates | OCR, input limit, model, ambiguity and missing-candidate categories |
| OCR recoverability | Evaluator-only measure in §11.2 |
| Resource use | Acquisition/OCR/train/inference/export times, GPU allocation/reservation peaks, CPU memory observations, output sizes |

Do not assign credit for predicting null on `not_annotated` fields; absence has not been established. Do not assign a false-positive label solely because a field lacks a source annotation. Report predictions on unannotated fields as unscored and unverified.

### 12.3 Uncertainty

Report paired receipt-level bootstrap intervals for system differences using 2,000 seeded resamples. If residual duplicate-family clusters are retained in a diagnostic, resample by cluster there. Preserve system pairing and all fields belonging to a receipt. Do not treat four fields or multiple windows as independent receipts.

For small selective subsets, show accepted counts and a binomial interval for descriptive accuracy/error. These intervals do not account for unidentified merchant/template dependencies, checkpoint contamination, annotation bias or distribution shift. Do not claim statistical significance merely because a point estimate is larger.

---

## 13. Review routing and the controlled learner activity

### 13.1 What the policy can say

The policy marks a total `needs_review` or `total_unflagged`. Neither state means human-verified. Optional-field candidates retain their own evidence and parse status and remain unverified. There is no `accounting_approved`, `safe_to_post` or “complete record verified” state.

The current LayoutLM span score is not calibrated correctness or absence probability [S4, S6]. The rule baseline's OCR-derived score is not calibrated either. Different systems require separate selected thresholds.

### 13.2 Hard failures and ranking scores

A total always needs review when image/OCR/model processing fails, no valid span exists, the amount is unparsable or ambiguous, the result exceeds an explicit input limit, or provenance/integrity checks fail. Integrity failures of the overall run stop execution rather than merely flagging a receipt.

For otherwise eligible totals:

- LayoutLM ranking score: the original returned span score, unchanged.
- Rule ranking score: minimum normalized Tesseract word confidence over the supporting key and amount tokens; missing/invalid confidence makes it ineligible for unflagged routing.

Do not multiply arbitrary confidence sources and call the product a probability. Do not use reference values, recoverability flags or evaluation outcomes as inference-time features.

### 13.3 Threshold selection

For each fixed system, select only on `validation_policy` receipts with usable total references. A candidate is unflagged when no hard failure applies and its score is at least the threshold.

Design target: maximize coverage subject to empirical total accuracy of at least **95%**, at least **50%** coverage of scoreable policy receipts, and at least **25** accepted policy receipts. These are illustrative teaching settings, not verified business-service levels.

Search the finite set of observed eligible scores, plus explicit accept-all-eligible and refer-all states. Resolve ties by lower empirical error, then higher threshold. Store exact thresholds and selected ID sets.

If no candidate meets the conditions, use **refer all** and report `policy_feasible=false`. Do not lower the accuracy target or minimum support automatically. Empty-set selective accuracy is null/undefined, not 100%.

Show selection support, empirical error, accepted count and uncertainty. The selection sample is small; meeting its empirical target does not certify a population error bound.

### 13.4 Held-out reporting

Apply the frozen threshold without adjustment to test receipts. Report:

```text
Scored coverage = unflagged receipts with usable total truth / receipts with usable total truth
Selective error = wrong unflagged totals / unflagged receipts with usable total truth
Operational routing coverage = all unflagged receipts / all frozen receipts
```

Also report unflagged receipts without usable references, total review count, and hard-failure counts. Do not hide unknown correctness inside the denominator. Keep the unfiltered primary EM beside selective metrics.

### 13.5 Change-one-thing activity

The learner changes only the minimum policy accuracy target, for example from 95% to 90%, on cached `validation_policy` predictions. They predict, run the optional activity, observe coverage/error/support, and explain the tradeoff.

Activity outputs go to a separate experiment directory. They must not overwrite the canonical threshold, fitted model, selection record or published test evaluation. The exercise uses no new GPU training and no test-based tuning. Canonical `Run all` executes a complete worked example without requiring an answer or input.

### 13.6 Arithmetic checks

Do not assume `subtotal + tax + service = total` is a universal receipt identity: discounts, tax inclusion, rounding and other adjustments may be omitted from this four-field schema.

Emit a numerical residual only where a separate, explicit input-side reconciliation contract provides the complete applicable formula and all required components. Otherwise emit `arithmetic_check=not_evaluable`. No missing component is silently zero. A pass does not prove correctness, and a failure must not repair any extracted value.

Arithmetic is not used to select the primary threshold in v1. A labeled synthetic unit fixture may demonstrate its mechanics, but its result is not receipt-model evidence.

---

## 14. Inference records and machine-readable outputs

### 14.1 Record schema

Every receipt–system record must include:

```json
{
  "schema": "org.dimer.receipt-intelligence.prediction.v1",
  "receipt_id": "illustrative_receipt_001",
  "image_sha256": "<computed image digest>",
  "system_id": "layoutlm_adapted",
  "input_source": "actual_ocr",
  "fields": {
    "total_amount": {
      "raw_text": "75,000",
      "normalized_amount": "75000",
      "parse_status": "ok",
      "token_ids": [18],
      "span_score": 0.0,
      "currency": "unspecified",
      "evidence_status": "machine_extracted_unverified"
    }
  },
  "total_review": {
    "state": "needs_review",
    "reasons": ["illustrative_record_not_evaluated"],
    "policy_id": "<frozen policy digest>"
  },
  "run_id": "<execution identifier>"
}
```

This is a schema illustration, not a dataset observation or measured confidence. A built notebook must generate actual values, all four field entries, and real hashes; it must contain no such placeholders in executable paths.

References and correctness flags belong in evaluator-owned tables, not the inference record or serving artifact. Include timestamps only as execution metadata, never as invented receipt dates.

### 14.2 Required output files

Produce `predictions.jsonl`, spreadsheet-safe `records.csv`, `field_metrics.json`, `system_comparison.csv`, `review_policy.json`, `review_metrics.json`, `failure_summary.csv`, `training_history.json`, `selection_record.json`, `run_manifest.json` and a compact HTML/Markdown result summary.

Store all primary and diagnostic results with explicit `input_source`. Preserve raw and canonical amount strings separately. Escape formula-like untrusted strings in spreadsheet-facing CSV; retain the exact raw string in JSON with a warning about downstream rendering. CSV quoting alone is not a formula-injection defense.

### 14.3 New-data inference demonstration

After frozen evaluation, pass a deterministic three-receipt subset of held-out images through the input-only predictor and show reconstructed records. Supply no labels to that call. State that these are held-out demonstration replays, not a second independent test set or previously unseen external receipts.

An optional authorized BYOD image is genuinely external to the default sample but has no accuracy claim without checked references.

---

## 15. Trained artifact and reconstruction boundary

### 15.1 Exported object

Export the selected **trained** LayoutLM tensor subset in SafeTensors, with the exact base identity needed to reconstruct it. This is bounded encoder/head fine-tuning, not LoRA, in-context fitting, a full base checkpoint, or OCR-engine training.

Proposed serving-bundle layout:

```text
receipt_intelligence_artifact/
  manifest.json
  adapter.safetensors
  model_config.json
  field_schema.json
  questions.json
  preprocessing.json
  ocr_manifest.json
  number_format_policy.json
  keyword_rules.json
  review_policy.json
  runtime_manifest.json
  RECONSTRUCT.md
  LICENSES_AND_ATTRIBUTION.md
```

The manifest uses a versioned capstone schema, binds every file by size/hash, names tensor shapes/dtypes, declares base/tokenizer revisions, and identifies the training recipe and selected epoch. Executable reconstruction code belongs to the trusted embedded implementation, not to arbitrary uploaded artifacts.

Reuse the host's adapter serialization semantics where compatible. Do not claim the composed OCR/rules/review bundle is already accepted by a DIMER production worker. Any production-compatibility claim requires a separate contract test. A capstone-specific manifest may not be passed blindly to the host's stricter artifact loader.

### 15.2 Fresh-process verification

The same default `Run all` must:

1. Export the artifact and commit the artifact-stage receipt atomically.
2. Release original model objects and launch a new Python process in the isolated environment.
3. Verify the bundle file set, schema, sizes, digests, tensor names/shapes and base identity before loading.
4. Load a fresh verified base and apply only the allowed tensor subset.
5. Reconstruct OCR, questions, normalization, rules and the selected review policy from serialized configuration.
6. Re-run a deterministic receipt subset through the entire image-to-record path, not merely reload saved predictions.
7. Compare outputs with the pre-export predictions.

Require exact OCR token text/boxes, extracted word spans, canonical amounts, field states and review decisions for the pinned runtime. For floating model scores, initial comparison tolerance is `atol=1e-6`, `rtol=1e-5`; any resulting threshold-decision difference still fails. Do not widen tolerances automatically. Record environment differences when testing portability elsewhere.

A changed or missing threshold, number grammar, language file, tokenizer or base checkpoint must fail verification. A replay process must not access training examples, test references or the original in-memory model.

### 15.3 Security

Reject archive traversal, absolute paths, symlinks, unexpected files, oversized expansion and duplicate entries. Reject untrusted pickles, Python modules, executable model objects and unsupported weight formats. Never fetch arbitrary URLs or execute commands supplied by an uploaded manifest. Hash verification detects modification relative to a manifest; it does not make an attacker-controlled manifest trustworthy.

The model artifact and evaluation evidence bundle are separate. Do not embed gold references in the serving bundle. BYOD-generated adapters and evidence remain private unless deliberately shared by the owner.

---

## 16. Notebook architecture and learner experience

The following is the required learner-facing sequence. Infrastructure is collapsed and labeled, but its carried source remains inspectable.

| Section | Learner question / action | Visible output |
|---|---|---|
| 0. Orientation | What will this notebook do, and what does it not establish? | Objectives, prerequisites, status, limits and Input → System → Output |
| 1. How to use | Select T4 and run the complete path | Runtime instructions, roadmap, controls and glossary |
| 2. Setup — infrastructure | Establish the pinned isolated runtime | Version/identity report and setup checks |
| 3. Data card and audit | What counts as a receipt and a reference? | Source counts, split ownership, missingness and field examples |
| 4. OCR inspection | Can the amount be recovered from the recognized words? | Image/word-box overlay, token table and structured OCR failures |
| 5. Rules | What does a simple transparent extractor get right or miss? | Validation predictions and explanation of candidate selection |
| 6. Frozen model | Does layout-aware extraction add value? | Validation comparison and span evidence |
| 7. Alignment and training | What can be learned from noisy OCR? | Eligible/skipped supervision, trainable tensors, loss and epoch selection |
| 8. Review-policy development | What changes when uncertain totals are referred? | Policy-validation coverage/error curve, support and feasibility |
| 9. Freeze and test | Do the fixed systems generalize within this sample? | Unfiltered held-out comparison and separately frozen review metrics |
| 10. Diagnose | Is the failure in OCR, extraction, parsing, or the reference? | Reference-text diagnostic and representative failure panel |
| 11. Infer and export | What record could a downstream process consume? | Input-only replay, CSV/JSON and trained artifact inventory |
| 12. Fresh reload | Can the serialized workflow reproduce its output? | New-process parity report |
| 13. Change one thing | What is gained/lost by a different review target? | Separate validation-only activity results |
| 14. Optional BYOD | Can my authorized data enter the same workflow? | Input schema, positive/negative validation, inference or adaptation |
| 15. Conclude | What does the evidence actually support? | Evidence-based conclusion scaffold and limitations |

Before each main comparison ask for a prediction. After execution explain what to notice without promising that the model wins. Include collapsible worked answers after the question, not before it. Every optional control has a safe default and cannot block `Run all`.

Collect recurring terms in a glossary: OCR, bounding box, extractive span, annotation, fine-tuning, exact match, validation/test, calibration, coverage, selective error, adapter and provenance.

The conclusion scaffold must ask for the task, data population, principal result and baseline, one failure mode, the review tradeoff, uncertainty and evidence needed for transfer. Reflection/completion records are optional; no required submission or certificate claim is added.

Include the standard AI Assistance Disclosure: work is developed under maintainer direction and review; the maintainer validates results and decides release; AI assistance is not independent verification, endorsement or approval [S9].

---

## 17. Standalone runtime and implementation organization

### 17.1 Runtime rules

Create an isolated Python 3.12 environment and execute stages through its interpreter. Hash-lock Python packages and pin system/OCR assets. Set a file-based Matplotlib backend for the isolated process. Do not require a notebook-kernel restart.

The single notebook must carry all DIMER-owned logic through a deterministic source generator. No `git clone`, editable install, `sys.path` injection, DIMER package installation, raw-source download, worker API, secret or account prompt may appear in the canonical path [S1, S9]. Model/data acquisition and documented general-purpose upstream packages are allowed.

At the top, expose documented controls such as `USE_BYOD=False`, `BYOD_MODE`, `BYOD_PATH` and `OUTPUT_DIR`, with equivalent noninteractive inputs. Default `Run all` must never open a file dialog. Do not install a GPU-free substitute when the selected hosted runtime lacks the required accelerator; provide an actionable runtime-selection error.

Engineering targets, not measured claims: one T4, no simultaneous learned-model copies, source acquisition streamed to disk, model-training and inference memory measured per stage, and a complete run practical for a self-paced session. Publish elapsed-time estimates only after a real exact-revision run.

### 17.2 Proposed repository additions

```text
 tools/build_receipt_capstone.py
 tools/receipt_capstone_source.py
 tools/receipt_data.py
 tools/receipt_ocr.py
 tools/receipt_fields.py
 tools/receipt_models.py
 tools/receipt_training.py
 tools/receipt_metrics.py
 tools/receipt_policy.py
 tools/receipt_artifact.py
 tools/receipt_runtime.py
 tools/receipt_byod.py
 tools/receipt-requirements.in
 tools/receipt-requirements.lock
 tutorials/DIMER_Small_Business_Receipt_Intelligence_Capstone.ipynb
 tutorials/receipt_intelligence/
   data_manifest.json
   dataset_audit.json
   split_manifest.json
   model_manifest.json
   ocr_manifest.json
   field_schema.json
   DATA_LICENSE.md
   RECONSTRUCT.md
 docs/receipt-capstone.md
 docs/receipt-annotation-audit.md
 docs/receipt-release-evidence.md
 tests/test_receipt_*.py
```

These are proposed additions, not claims that files exist. Reconcile names with repository instructions. Preserve the original pipeline and tutorials. Register the new notebook in README/tutorial inventory, generator parity checks, release-asset validation and CI without overwriting unrelated work.

### 17.3 Stage graph and rerun ownership

```text
prepare → ocr → baseline/frozen → align/train → select_policy → freeze
                                                           ↓
                                                        evaluate
                                                           ↓
                                                     infer/export → replay
```

A stage receipt records stage ID, run/experiment ID, input hashes, configuration hash, implementation/carrier hash, output hashes and success state. Dependency changes invalidate downstream results. A failed retry must not leave an old success receipt pretending to certify new output. Write stage outputs into a temporary directory and commit them atomically only after validation.

Recoverable receipt failures remain data records; broken global environment, integrity or model-loading conditions are run failures. Do not catch every exception and print success.

---

## 18. BYOD: inference and actual adaptation

An `E2E` notebook needs more than image upload at the final inference cell. Provide both optional modes using the same implementation [S1].

### 18.1 Inference mode

Accept a ZIP containing an input manifest and authorized JPEG/PNG receipts. The manifest provides stable nonidentifying IDs, relative image paths, number-format policy, currency context and permitted processing scope. Process images locally in the hosted runtime, extract four candidates, apply the frozen policy, and export records.

Do not report accuracy without references. A CORD-selected threshold is **not validated for BYOD**; label routing as transferred/exploratory, and default to review-required until a locally validated BYOD policy is supplied. Do not claim that domain shift is detected by a high model score.

### 18.2 Adaptation-and-evaluation mode

Accept explicit, disjoint `train`, `validation_model`, `validation_policy`, `test` and optional inference roles, plus checked value references and gold value boxes for supervised training alignment. Preserve user-provided group IDs; reject receipt or duplicate-group overlap across roles. Do not randomly repartition silently.

Example archive contract:

```text
byod.zip
  manifest.json
  annotations.jsonl
  images/<receipt_id>.png
```

Annotation records must declare field status, raw reference value, canonical value/format policy where appropriate, value boxes in the original image coordinate system, and annotation provenance. Missing optional fields remain unknown unless explicitly verified absent by the user; the default v1 absence task is still not scored.

Run validation → actual OCR → aligned training → fresh-base adaptation → model selection → review-policy selection/fallback → held-out evaluation → inference → export → fresh reconstruction. Apply the same denominator rules. Insufficient policy data can validly produce refer-all; it must not trigger a looser hidden target.

Small BYOD samples may exercise the workflow but must be labeled small-sample evidence. At least eight aligned training examples are required. Both validation roles and a nonempty independent test role are required for the advertised evaluation mode. Missing prerequisites lead to a clear refusal of that mode; an explicit separate inference mode remains available.

### 18.3 Limits and negative tests

Initial archive ceilings: 1,000 images, 2 GiB compressed and 5 GiB expanded, with the per-image limits in §7. Validate all declared paths, duplicates, coordinates, supported number/currency policies, role names and annotation states before training. Include tests for path traversal, ZIP bombs, corrupted images, unsupported formats, duplicate splits, missing labels, invalid boxes and contradictory references.

A hosted positive adaptation/evaluation BYOD run and at least one negative schema/refusal case are qualification requirements. A generated fixture proves mechanics only, not accuracy on Philippine receipts.

---

## 19. Test and review plan

### 19.1 CPU/source tests

Implement deterministic tests for:

- source schema aliases, reference-state handling, exact duplicate policy and stable split assignment;
- no reuse of the existing official-validation/test re-split loader;
- amount parsing, ambiguous separators, percentages, signs, zero versus null, multiple candidates and round-trip decimal strings;
- OCR TSV parsing, coordinate transforms, reading order, timeouts and nonempty low-confidence tokens;
- training-only alignment, repeated equal amounts, missing OCR digits, invalid geometry and training-window rejection;
- field-specific keyword boundaries, cash/change exclusion and ambiguity handling;
- epoch selection, earliest ties, epoch-0 exclusion and retention of an adapted model that loses to the frozen model;
- policy feasibility, ties, insufficient support, refer-all and undefined empty-set selective accuracy;
- denominator preservation for OCR/model failures and missing optional annotations;
- reference-text/actual-OCR labels and prevention of gold-only fields entering prediction;
- immutable selection records, invalidated caches, failed retries and stale stage receipts;
- safe archive loading, tensor allowlists, altered base/tokenizer/OCR/policy hashes and output file schemas;
- generator determinism, carried-source hashes, valid notebook structure and complete guided layer.

Use dependency injection at model/OCR boundaries for CPU fixtures; do not install incomplete fake `torch` modules into global module state. Test doubles must be labeled as such, not described as pretrained evidence.

### 19.2 Gold noninterference tests

Run a predictor fixture with labels absent, then with evaluation reference values/categories shuffled in the evaluator workspace. Its input-only predictions must be identical. A serving artifact must remain sufficient when the reference directory is removed.

Also assert that changing test labels cannot change the trained tensor hashes, keyword/parser configuration, selected checkpoint, threshold or selection digest. A test image may be needed for source-integrity audit and prediction, but its answer must not influence any fit/selection stage.

### 19.3 Real hosted tests

A fresh Colab T4 run must obtain source images and the pinned base, run actual OCR, execute all three systems, perform nonzero gradient updates, evaluate the frozen cohort, produce the artifact, and pass fresh-process image-to-record reconstruction. Record notebook/source/carrier hashes, execution outputs, device/library/OCR versions, resource measurements and all failure counts.

Run a representative BYOD adaptation/evaluation path and a negative case separately. Qualification may reuse verified downloads only where the documented clean-runtime procedure permits it; it must not reuse predictions, a trained adapter, a hidden preprocessed gold dataset or an unrecorded environment.

### 19.4 Learner/content review

Review both code and content. Verify that the notebook teaches what its objectives promise, explains normal failures, displays an interpretable result at each transition, and does not hide decisions in infrastructure. Check that every model claim has an executed comparison and every limitation appears where it affects interpretation.

---

## 20. Acceptance criteria and release states

### 20.1 Design acceptance

The specification is ready to implement when the proposed scope, default dataset, three systems, actual fine-tuning, role ownership, metrics and artifact contract are accepted. This document contains that design; it does not contain empirical results.

### 20.2 Build/source acceptance

The implementation must satisfy all of the following:

- deterministic generated standalone notebook registered in the host repository;
- complete verified asset/dependency manifests with no runtime placeholders;
- new official-partition-aware dataset loader and fresh-base training;
- actual OCR path plus separately labeled reference-text diagnostic;
- complete default training/evaluation/inference/export/replay stages;
- auditable errors, denominator ownership and review-policy fallback;
- substantive inference and adaptation BYOD paths;
- passing applicable source/CPU/contract checks and a content review record.

### 20.3 Hosted qualification

Require exact-revision fresh default execution, real model/OCR evidence, artifact replay, actual resource measurements and BYOD positive/negative evidence. The original LayoutLM tutorial's earlier execution does not qualify this capstone [S1, S9].

No minimum performance gain is required. A losing fine-tuned model, poor OCR, infeasible 95% policy or zero unflagged coverage can be a valid instructional result when honestly executed and explained. Runtime correctness and scientific validity must not be confused with business suitability.

### 20.4 Public claims and state reporting

Track separately:

```text
specified → built → source-checked → CPU-tested → hosted-executed
          → annotation-reviewed (scope stated) → release-qualified → published/merged
```

A successful build or green CI is not hosted qualification. Human audit of a subset is not verification of every reference. Merging does not establish model quality. Keep Candidate until the applicable evidence has been reviewed and an explicit promotion decision is recorded.

---

## 21. Build sequence and unresolved measurements

### 21.1 Implementation order

1. Read workspace/repository instructions and record current source revisions. Make no changes to `NAIRA-SEU` repositories.
2. Resolve the complete source-image dataset, OCR executable/language dependencies and model manifests. Audit source schemas and field support; freeze source-integrity exclusions and roles.
3. Implement and test numeric normalization, reference states, OCR output schema and the rule baseline before any foundation-model comparison.
4. Implement actual OCR and training alignment; measure eligibility without substituting reference text.
5. Add frozen inference and bounded fine-tuning with the specified checkpoint selection; verify unchanged production behavior.
6. Add metrics, reference-text diagnostics, review policy and immutable selection/evaluation boundaries.
7. Add safe artifacts, fresh-process replay and both BYOD modes.
8. Generate the guided notebook; run CPU/source/content checks, then hosted qualification and evidence review.
9. Publish or open a PR only under the user's subsequent explicit instruction. No merge is implied.

### 21.2 Required build-time resolutions

| Item not yet measured/qualified | Required resolution | Forbidden shortcut |
|---|---|---|
| All image-bearing source bytes and exact field support | Verified complete manifests, audit and actual counts | Treat annotation-column pins as image verification |
| OCR system-package and language-data reproducibility | Immutable asset versions/hashes and fresh hosted install | Floating system install or silent English-only fallback |
| Annotation-to-OCR alignment coverage | Measured training-only eligible/skipped counts | Replace OCR with gold text or fabricate spans |
| Numeric conventions and disputed references | Prediction-blind source audit and frozen grammar | Choose per-test formatting to match outputs |
| T4 time and memory envelope | Exact-revision measured default run | Skip training, shrink test data or report estimates as measurements |
| BYOD transfer and threshold validity | Separate authorized evaluation and evidence | Reuse CORD quality claims as local-receipt guarantees |

These are implementation measurements and qualification obligations, not invented evidence. If feasibility requires a different dataset, backbone, field scope, partition design or metric, document the proposed design change rather than silently build a different capstone.

---

## 22. Scholarly grounding and references

Use APA 7 author–date citations for scientific explanations in the eventual notebook, with DOI-bearing references where verified and applicable, following the Notebook Spec 2.2 reference pattern [S1]. Keep engineering sources and dataset/model cards separately identifiable. Do not fabricate a DOI for a dataset workshop paper that does not expose one in the inspected source.

**Suggested scientific anchors**

- Park, S., Shin, S., Lee, B., Lee, J., Surh, J., Seo, M., & Lee, H. (2019). *CORD: A consolidated receipt dataset for post-OCR parsing*. Document Intelligence Workshop at NeurIPS. Publisher-provided citation and paper link: [CORD repository](https://github.com/clovaai/cord). No DOI was verified from the inspected publisher source. Use this for dataset/task provenance, not a claim that this capstone reproduces a published benchmark.
- Xu, Y., Li, M., Cui, L., Huang, S., Wei, F., & Zhou, M. (2020). *LayoutLM: Pre-training of text and layout for document image understanding*. KDD 2020. [Microsoft Research publication page](https://www.microsoft.com/en-us/research/publication/layoutlm-pre-training-of-text-and-layout-for-document-image-understanding/). Use for the text/layout modeling concept. The particular Impira checkpoint and wrapper contracts still require their own source citations [S4, S6, S10].
- Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory, 16*(1), 41–46. [https://doi.org/10.1109/TIT.1970.1054406](https://doi.org/10.1109/TIT.1970.1054406). Use for the concept of error–rejection tradeoffs, not as proof that this capstone's uncalibrated threshold is optimal or guaranteed.

The source review for this specification did not reproduce the cited experiments or independently audit all CORD images. Bibliography checks must distinguish publication provenance from executable qualification.

### Engineering and source register

- **[S1]** Kurt Valcorza. *DIMER Notebook Specification*, v2.2, 26 September 2026. [Source](https://github.com/kurtvalcorza/ml-worker/blob/main/integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md). Relevant sections: 3–5, 10–18, 20, 22–25; inspected Git blob recorded in §2.2.
- **[S2]** NAVER CLOVA. *CORD: A Consolidated Receipt Dataset for Post-OCR Parsing*. [Publisher repository and schema](https://github.com/clovaai/cord). Inspected 28 September 2026.
- **[S3]** NAVER CLOVA Information Extraction. *CORD v2*. [Dataset card](https://huggingface.co/datasets/naver-clova-ix/cord-v2), [dataset metadata](https://huggingface.co/datasets/naver-clova-ix/cord-v2/blob/main/dataset_infos.json), and [source file listing](https://huggingface.co/datasets/naver-clova-ix/cord-v2/tree/main/data). Mutable navigation links are not runtime pins.
- **[S4]** Kurt Valcorza. *LayoutLM document question answering pipeline*. [README](https://github.com/kurtvalcorza/layoutlm-document-qa-pipeline/blob/main/README.md). Inspected 28 September 2026; source blob in §2.2.
- **[S5]** Kurt Valcorza. *LayoutLM CORD sample and split implementation*. [`samples.py`](https://github.com/kurtvalcorza/layoutlm-document-qa-pipeline/blob/main/src/layoutlm_document_qa_pipeline/samples.py). Inspected 28 September 2026; source blob in §2.2.
- **[S6]** Kurt Valcorza. *LayoutLM inference, training and artifact implementation*. [`pipeline.py`](https://github.com/kurtvalcorza/layoutlm-document-qa-pipeline/blob/main/src/layoutlm_document_qa_pipeline/pipeline.py). Inspected 28 September 2026; source blob in §2.2.
- **[S7]** Tesseract contributors. *Command Line Usage*. [Official documentation](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html). Inspected 28 September 2026.
- **[S8]** Tesseract contributors. *Tesseract User Manual* and *tessdata_fast*. [Manual](https://tesseract-ocr.github.io/tessdoc/); [language-data repository](https://github.com/tesseract-ocr/tessdata_fast). Exact executable and language-data pins are build outputs.
- **[S9]** User-supplied *DIMER capstones — agent handoff*, snapshot 28 September 2026, approximately 13:10 Asia/Manila; filename `DIMER_CAPSTONES_AGENT_HANDOFF_2026-09-28.md`. Relevant sections: Operating rules, Shared notebook contract and Later queue. Its six-capstone inventory is historical; this document specifies the subsequently approved eighth concept.
- **[S10]** Impira. *layoutlm-document-qa*. [Upstream model card](https://huggingface.co/impira/layoutlm-document-qa). The immutable checkpoint identity to verify is recorded in §2.2.

---

## 23. Completion statement for the future implementation

The implementation is complete only when a learner can open the single notebook in the documented fresh runtime, execute its default path without edits or prompts, inspect an honest three-system image-based comparison, observe real gradient adaptation, understand the total-review tradeoff, and reconstruct matching records from the exported trained artifact.

Successful completion must remain possible when the scientific result is negative. The notebook succeeds by making the evidence, errors and limits visible—not by ensuring that a foundation model wins.
