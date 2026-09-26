# DIMER Document Question Answering Workshop

## LayoutLM vs Pix2Struct DocVQA

**Proposed filename:** `DIMER_Document_QA_LayoutLM_vs_Pix2Struct_Workshop.ipynb`  
**DIMER Notebook Specification:** `2.1`  
**Profile:** `TASK-INFERENCE`  
**Pedagogical mode:** `WORKSHOP`  
**Comparison scope:** `MULTI-MODEL`  
**Standalone:** `true`  
**Recommended runtime:** Kaggle / Colab Tesla T4  
**Task:** Document Question Answering  
**Canonical dataset:** CORD-v2 receipts  
**Canonical evaluation:** Frozen models; no adaptation  
**Primary metric:** ANLS  
**Secondary metric:** Exact Match  
**Canonical test set:** 50 page-disjoint receipts / approximately 229 QA records

---

# 1. Purpose

This workshop compares two fundamentally different approaches to document question answering:

1. **LayoutLM Document QA**
   - OCR-dependent
   - text + spatial-layout input
   - extractive answer spans

2. **Pix2Struct DocVQA**
   - OCR-free at inference
   - page pixels + question
   - generative answer strings

Both answer the same general question:

```text
page + question
      ↓
document QA model
      ↓
answer
```

but they consume different representations of the document and produce answers with different guarantees.

---

# 2. Core learning question

The workshop should answer:

> What changes when Document QA is formulated as **extracting a span from OCR tokens** versus **generating an answer directly from page pixels**?

This is more important than simply ranking the two models by ANLS.

---

# 3. Learning objectives

By the end of the workshop, participants should be able to:

1. explain Document QA as a page-grounded question-answering task;
2. distinguish OCR-dependent and OCR-free Document QA;
3. describe the role of word boxes in LayoutLM;
4. describe Pix2Struct's pixel-to-text generation mechanism;
5. run both models on the same document and question;
6. interpret ANLS and Exact Match;
7. distinguish extractive answers from generated answers;
8. understand why LayoutLM can localize an answer while Pix2Struct cannot;
9. recognize the effect of OCR quality and layout information;
10. recognize the effect of image quality on pixel-based QA;
11. measure question-paraphrase sensitivity;
12. understand the absence of a reliable abstention mechanism;
13. identify generated answers that do not appear on the page;
14. compare inference cost and model-resource requirements; and
15. evaluate compatible user-supplied document QA data.

---

# 4. Models

## 4.1 LayoutLM Document QA

**Model**

```text
impira/layoutlm-document-qa
```

**Immutable revision**

```text
beed3c4d02d86017ebca5bd0fdf210046b907aa6
```

**License**

```text
MIT
```

**Architecture**

```text
LayoutLMForQuestionAnswering
```

Model structure:

```text
question tokens
+
OCR word tokens
+
2-D word boxes
        ↓
LayoutLM encoder
        ↓
start logits + end logits
        ↓
best OCR span
```

### Snapshot

`model.safetensors`

```text
511,200,628 bytes
SHA-256:
e4bbad3e4a1b5ae50c787b7afd6049a0bfa99fd823b50436e444e092ae2347b9
```

Total verified snapshot:

```text
513,814,827 bytes
```

Parameters:

```text
127,792,898
```

---

# 4.2 Pix2Struct DocVQA

**Model**

```text
google/pix2struct-docvqa-base
```

**Immutable revision**

```text
63f6b3de436e39f75c7a486881a9c2c14a7f4e89
```

**License**

```text
Apache-2.0
```

**Architecture**

```text
Pix2StructForConditionalGeneration
```

Conceptually:

```text
page pixels
+
question rendered as image header
        ↓
Pix2Struct vision encoder
        ↓
text decoder
        ↓
generated answer tokens
```

### Snapshot

`model.safetensors`

```text
1,129,177,976 bytes
SHA-256:
067f7f314d87fa56daa5bcfaf36fa0b33ceebf7b7d4fae6a1e51ab7af64ee0b5
```

Total snapshot:

```text
1,133,308,924 bytes
```

Parameter count should be measured from the loaded model and recorded at runtime.

---

# 5. The architectural contrast

## LayoutLM

```text
OCR
 ↓
words + pixel boxes
 ↓
token embeddings
+
1-D position
+
2-D spatial position
 ↓
transformer encoder
 ↓
start/end span head
```

The model **never sees the original page pixels**.

---

## Pix2Struct

```text
page pixels
 ↓
question rendered above page
 ↓
patch representation
 ↓
vision encoder
 ↓
text decoder
 ↓
generated answer
```

Pix2Struct does **not require external OCR input**.

---

# 6. Critical semantic difference

LayoutLM can only return text represented by the OCR tokens supplied to it.

Therefore:

```text
LayoutLM answer ∈ OCR page text
```

under the canonical inference contract.

Pix2Struct has no such constraint.

Therefore:

```text
Pix2Struct answer
```

may:

- exactly reproduce page text;
- normalize/rephrase page text;
- partially reproduce it;
- or generate something that is not on the page.

This difference must remain visible throughout the notebook.

---

# 7. Why use one common dataset

Do not compare:

```text
LayoutLM on CORD
```

against:

```text
Pix2Struct on a synthetic invoice
```

because dataset differences would dominate interpretation.

Both models must receive:

- the same page;
- the same question;
- the same accepted gold answer.

Only the **representation supplied to the model** changes.

---

# 8. Canonical dataset — CORD-v2

Use the existing pinned DIMER CORD-v2 source.

**Dataset**

```text
CORD-v2
```

**Repository**

```text
naver-clova-ix/cord-v2
```

**Immutable revision**

```text
7f0115a4b758a71d6473b8d085751692da2fef98
```

**License**

```text
CC BY 4.0
```

CORD consists of photographed receipts with:

- page images;
- OCR words;
- word polygons;
- semantic field annotations.

This makes it unusually suitable for comparing the two QA paradigms.

---

# 9. Pinned CORD shards

Reuse the two shards already qualified by the LayoutLM pipeline.

## Test shard

```text
data/test-00000-of-00001-9c204eb3f4e11791.parquet
```

Bytes:

```text
234,202,795
```

SHA-256:

```text
51c65f1788faff392abe2a0b55b023eb23e9be551c509138eaa3a832514224e7
```

Rows:

```text
100
```

---

## Validation shard

```text
data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet
```

Bytes:

```text
242,080,800
```

SHA-256:

```text
0d0f6dac11fdcc549de2746aa9f53136a3bc22a2a1aff2b0b847f7622ad60c15
```

Rows:

```text
100
```

---

# 10. Acquisition strategy

The current LayoutLM carrier reads only:

```text
ground_truth
```

using HTTP range requests.

This workshop needs both:

```text
ground_truth
image
```

from those same immutable parquet files.

Recommended workflow:

### Stage 1

Read only `ground_truth`.

This is inexpensive and sufficient to:

- reconstruct OCR words;
- reconstruct boxes;
- determine supported questions;
- reproduce the deterministic page split.

### Stage 2

Once the 50 held-out pages are known, retrieve the corresponding page images from the pinned parquet shards.

If the parquet layout allows row-selective reads, fetch only the selected rows.

Otherwise reading the image column from the two pinned shards is acceptable, but the notebook must disclose the additional transfer cost.

---

# 11. Common page representation

Each selected page should become:

```python
{
    "page_id": ...,
    "image": PIL.Image,
    "image_size": [width, height],

    "words": [...],
    "boxes": [[x0,y0,x1,y1], ...],

    "lines": ...,
}
```

The image and OCR/layout data must originate from the **same CORD record**.

---

# 12. CORD page integrity

For every selected page, record:

```text
page_id
source shard
source row index

image width
image height
image SHA-256
decoded-pixel SHA-256

number of OCR words
OCR/layout digest
```

The notebook must assert that OCR coordinates fit the corresponding page image.

---

# 13. Question generation

Reuse the exact current LayoutLM DIMER question templates.

```text
total.total_price
→ What is the total amount?

sub_total.subtotal_price
→ What is the subtotal?

sub_total.tax_price
→ What is the tax amount?

sub_total.service_price
→ What is the service charge?

sub_total.discount_price
→ What is the discount amount?

total.cashprice
→ How much cash was paid?

total.changeprice
→ How much change was given?

total.creditcardprice
→ How much was paid by card?

total.menuqty_cnt
→ How many items were bought?

menu.nm
→ What is the name of the first item?
```

---

# 14. Gold answers

A field becomes a QA record only when its value can be represented as a valid contiguous answer span in the CORD OCR.

For every QA record retain:

```text
id
page_id
field
question

answers

answer_start
answer_end
```

`answers[0]` must equal:

```text
" ".join(words[answer_start:answer_end+1])
```

---

# 15. Dataset split

Reuse the existing page-level DIMER split:

```text
train       119 receipts
validation   30 receipts
test         50 receipts
```

Seed:

```text
42
```

The workshop itself performs no model fitting, so only the **test split** is used for model evaluation.

The train split is used only by the simple OCR baselines where required.

---

# 16. Expected evaluation size

Using the current question-generation contract:

```text
test pages ≈ 50
test QA records ≈ 229
```

The notebook should assert the exact generated question count against the pinned sample contract once the image-enabled dataset manifest is finalized.

---

# 17. Split integrity

All questions from one receipt must remain in the same split.

Assert:

```text
no page appears in two splits
no duplicate OCR page content crosses splits
no test question appears in train or validation
```

Page identity—not question identity—is the leakage boundary.

---

# 18. Important evaluation bias

Every canonical gold answer is deliberately constructed as a contiguous OCR span.

Therefore:

```text
gold answer is extractive by construction
```

This structurally aligns with LayoutLM's answer mechanism.

Pix2Struct is being evaluated on an extractive-answer dataset even though it is generative.

The notebook MUST disclose this.

It must not interpret a LayoutLM advantage as proof that extractive architectures are universally better.

---

# 19. Notebook profile

Declare:

```text
profile = TASK-INFERENCE
mode = WORKSHOP
comparison_scope = MULTI-MODEL
```

Neither model is adapted.

The individual carrier notebooks remain the canonical adaptation tutorials.

---

# 20. Runtime

Common pins:

```text
Python 3.12

torch==2.14.0
torchvision==0.29.0
torchaudio==2.11.0

transformers==4.57.6
safetensors==0.8.0

numpy==2.5.3
pillow==11.3.0
huggingface-hub==0.36.2

pyarrow==25.0.1
```

Recommended:

```text
Tesla T4
```

CPU support may work technically, but the full 229-question Pix2Struct evaluation should be documented as GPU-preferred.

---

# 21. Model acquisition

Embed both complete DIMER model manifests.

For each model:

1. fetch only manifest-listed assets;
2. fetch from exact immutable revision;
3. verify bytes;
4. verify SHA-256;
5. fail closed on mismatch;
6. load only from the verified local snapshot;
7. set:

```text
local_files_only=True
trust_remote_code=False
```

No upstream `.bin` fallback.

---

# 22. Model execution order

Do not keep both large models resident simultaneously.

Canonical sequence:

```text
load LayoutLM
→ evaluate all records
→ run LayoutLM-specific experiments
→ retain normalized outputs
→ unload
→ garbage collect
→ clear CUDA cache

load Pix2Struct
→ evaluate all records
→ run Pix2Struct-specific experiments
→ retain normalized outputs
→ unload
```

This reduces peak accelerator memory.

---

# 23. LayoutLM input contract

For each QA record LayoutLM receives:

```text
question
words
boxes
image_size
```

Ceilings:

```text
question <= 256 chars

1..2000 OCR words

one pixel box per word

page size <= 10000 px per side

512-token windows
document stride = 128
max answer span = 15 tokens
```

Long pages are evaluated window-by-window.

---

# 24. LayoutLM output

Normalize to:

```text
answer
normalized_answer

start_word
end_word

span_score
n_words
n_windows

latency
```

The score is uncalibrated.

No universal answer-acceptance threshold is defined.

---

# 25. LayoutLM localization

Because an answer maps to OCR word indices, derive:

```text
answer_boxes
answer_union_box
```

from:

```text
words[start:end+1]
```

This permits visual highlighting of the answer location.

This capability is model-specific and should not be discarded merely because Pix2Struct lacks it.

---

# 26. Pix2Struct input contract

For each QA record Pix2Struct receives:

```text
page image
question
```

Ceilings:

```text
image sides 16..4096
question <= 256 chars
max patches = 2048
```

Default output budget:

```text
max_new_tokens = 32
```

Decoding:

```text
greedy
do_sample = False
```

---

# 27. Pix2Struct question rendering

The question is rendered as a header above the page.

Use the current DIMER workaround:

```text
Pillow bundled Aileron font
```

rather than allowing the upstream processor to download an unpinned Arial font at inference time.

The font bytes and relevant rendering settings must be part of provenance.

---

# 28. Pix2Struct output

Normalize to:

```text
answer
normalized_answer

new_tokens
truncated

latency
```

Pix2Struct produces:

```text
no answer confidence
no answer probability
no page localization
```

The notebook should make these absences explicit rather than inventing substitutes.

---

# 29. Common answer-normalization function

Use one identical answer-normalization implementation for:

- LayoutLM;
- Pix2Struct;
- baselines.

Normalization should match the existing DIMER/DocVQA metric implementation.

No model-specific normalization.

---

# 30. Primary metric — ANLS

Use **Average Normalized Levenshtein Similarity**.

For one prediction:

```text
normalized edit similarity
```

with the standard ANLS threshold:

```text
0.5
```

Large edit-distance mismatches receive zero similarity.

For multiple accepted answers, score against the best accepted answer.

Report:

```text
mean ANLS
```

over all 229 QA records.

---

# 31. Secondary metric — Exact Match

Normalize prediction and accepted answers.

Report:

```text
exact_match_rate
```

Exact Match is stricter than ANLS and useful for:

- numbers;
- dates;
- monetary amounts;
- item names.

---

# 32. Empty-answer rate

Report:

```text
empty_rate
```

for every system.

Interpretation differs:

- LayoutLM normally selects a span;
- Pix2Struct normally generates tokens;
- neither provides reliable abstention.

A low empty rate is not automatically desirable.

---

# 33. Prediction-on-page diagnostic

Using the gold OCR token sequence, calculate:

```text
answer_in_ocr
```

for every model response.

Recommended rule:

1. normalize predicted answer;
2. normalize each contiguous OCR word span up to a reasonable length;
3. determine whether the prediction exactly matches a contiguous OCR span.

Report:

```text
prediction_in_ocr_rate
```

---

# 34. Interpretation of `prediction_in_ocr`

For LayoutLM:

```text
prediction_in_ocr ≈ 100%
```

by architecture.

For Pix2Struct:

a prediction that is not present in OCR may represent:

- useful normalization;
- an OCR disagreement;
- a generative paraphrase;
- or hallucination.

Therefore this is a diagnostic, not a correctness metric.

---

# 35. Common evaluation output

Each answer row should become:

```text
record_id
page_id
field
question
gold_answer

model
answer
normalized_answer

anls
exact_match
empty
answer_in_ocr

latency_seconds
```

Model-specific columns may additionally be present.

---

# 36. LayoutLM-specific columns

```text
start_word
end_word
span_score
n_windows
answer_union_box
```

---

# 37. Pix2Struct-specific columns

```text
new_tokens
truncated
```

Other model's incompatible columns remain empty/null.

---

# 38. Baselines

Use the two baselines already present in the LayoutLM carrier.

## Last-number baseline

Selects a number-like answer from OCR.

Useful primarily as a deliberately weak field-extraction floor.

---

## Keyword-lookup baseline

Uses question/field cues and OCR proximity.

This is a much stronger non-neural reference.

---

# 39. Baseline caveat

Both baselines receive OCR information.

Therefore they are more directly comparable to LayoutLM's information boundary than to Pix2Struct's OCR-free input path.

State this clearly.

They represent:

> What simple extraction rules can achieve if OCR is already available.

---

# 40. Main comparison table

Produce:

| System | Input modality | ANLS | Exact Match | Empty rate | Answer-in-OCR |
|---|---|---:|---:|---:|---:|
| Last-number baseline | OCR | measured | measured | measured | measured |
| Keyword lookup | OCR | measured | measured | measured | measured |
| LayoutLM | OCR + layout | measured | measured | measured | ~1.0 |
| Pix2Struct | page pixels | measured | measured | measured | measured |

Do not declare an overall winner.

---

# 41. Per-field evaluation

For each CORD field report:

```text
support
ANLS
exact_match
empty_rate
```

for both models.

Fields:

```text
total.total_price
sub_total.subtotal_price
sub_total.tax_price
sub_total.service_price
sub_total.discount_price
total.cashprice
total.changeprice
total.creditcardprice
total.menuqty_cnt
menu.nm
```

---

# 42. Field-level comparison table

Example:

| Field | N | LayoutLM ANLS | Pix2Struct ANLS | LayoutLM EM | Pix2Struct EM |
|---|---:|---:|---:|---:|---:|
| total amount | ... | ... | ... | ... | ... |
| subtotal | ... | ... | ... | ... | ... |
| tax | ... | ... | ... | ... | ... |
| first item | ... | ... | ... | ... | ... |

Do not assume numeric fields will behave better.

---

# 43. Agreement analysis

For every QA record determine:

### both exact

Both models exactly match a gold answer.

### LayoutLM only

LayoutLM exact, Pix2Struct not exact.

### Pix2Struct only

Pix2Struct exact, LayoutLM not exact.

### neither exact

Neither exact.

Report counts and percentages.

---

# 44. Similarity-based agreement

Because Exact Match is strict, additionally classify using:

```text
ANLS > 0
```

or a documented acceptance rule based on the ANLS threshold.

Keep this separate from exact-match agreement.

---

# 45. Qualitative error categories

Automatically select examples where available:

1. both correct;
2. LayoutLM correct / Pix2Struct wrong;
3. Pix2Struct correct / LayoutLM wrong;
4. both wrong;
5. Pix2Struct answer not present in OCR;
6. LayoutLM high span score but incorrect.

Selections must follow deterministic sorting rules after metric computation.

---

# 46. Visual result panels

For selected records display:

### Left

Original CORD receipt image.

### Overlay

OCR boxes in light outlines.

### LayoutLM answer

Highlight the returned answer words/boxes.

### Side panel

```text
Question
Gold answer

LayoutLM:
  answer
  ANLS
  span score

Pix2Struct:
  answer
  ANLS
  truncated
```

This is one of the workshop's most important teaching views.

---

# 47. Question-paraphrase sensitivity

Use a deterministic subset of no more than:

```text
20 QA records
```

selected before model inference.

Define one paraphrase per supported field.

Examples:

```text
What is the total amount?
→ How much is the total?

What is the subtotal?
→ How much is the subtotal?

What is the tax amount?
→ What tax amount is shown?

What is the service charge?
→ How much is the service charge?

What is the discount amount?
→ What discount was applied?

How much cash was paid?
→ How much cash did the customer pay?

How much change was given?
→ How much change was returned?

How much was paid by card?
→ How much was charged to the card?

How many items were bought?
→ How many items are on the receipt?

What is the name of the first item?
→ What is the first item listed?
```

---

# 48. Paraphrase metrics

For every selected QA pair record:

```text
canonical_answer
paraphrased_answer

canonical_ANLS
paraphrased_ANLS

answer_stable
```

`answer_stable` means normalized output is identical under both question wordings.

Report per model:

```text
canonical ANLS
paraphrased ANLS
answer stability rate
```

No model is expected to be invariant.

---

# 49. Unanswerable-question probe

Select five predetermined test pages.

Ask:

```text
What is the loyalty membership number?
```

This field is not part of the CORD question schema.

Before execution validate that no annotated field corresponds to such a concept.

Do not assign a fabricated gold answer.

---

# 50. Unanswerable outputs

For each model record:

```text
answer
empty
answer_in_ocr
```

LayoutLM additionally:

```text
span_score
```

Pix2Struct additionally:

```text
new_tokens
truncated
```

---

# 51. Unanswerable interpretation

This is a **behavior probe**, not an accuracy evaluation.

The lesson:

> Neither system has a reliable built-in abstention mechanism.

LayoutLM may confidently select some OCR span.

Pix2Struct may generate plausible text.

No arbitrary score threshold should be presented as solving this problem.

---

# 52. LayoutLM layout ablation

Use the same deterministic 20-record experiment subset.

Canonical LayoutLM:

```text
real OCR words
real word boxes
```

Ablation:

```text
same OCR words
all document word boxes replaced with a constant box
```

For example:

```text
[0, 0, 0, 0]
```

after ensuring the implementation accepts the contract.

Question and text remain unchanged.

---

# 53. Layout ablation output

Report:

```text
canonical ANLS
no-layout ANLS
delta ANLS

canonical exact match
no-layout exact match
```

This demonstrates whether 2-D spatial information contributes to the measured records.

It must not be used to characterize all LayoutLM behavior.

---

# 54. Pix2Struct image-degradation experiment

Use the same predetermined pages.

Variant:

1. downsample image to approximately half its original linear dimensions;
2. re-upsample to original dimensions;
3. pass through the normal Pix2Struct processor.

Question remains unchanged.

---

# 55. Image-degradation output

Report:

```text
canonical ANLS
degraded-image ANLS
delta ANLS

canonical exact match
degraded exact match
```

This is the pixel-modality counterpart to LayoutLM's layout ablation.

---

# 56. Why the robustness experiments are asymmetric

The two systems depend on different upstream evidence.

LayoutLM's vulnerability surface includes:

```text
OCR token errors
OCR ordering
box errors
OCR omissions
```

Pix2Struct's vulnerability surface includes:

```text
resolution
blur
compression
page rendering
visual clutter
```

The notebook should teach these as different system-level dependencies, not force identical corruption operators.

---

# 57. LayoutLM fit/window analysis

For all test records report:

```text
n_windows
```

Summarize:

```text
1-window records
multi-window records
maximum windows
mean windows
```

The canonical CORD adaptation set was designed to fit one window, but inference must preserve the actual measured behavior.

---

# 58. Pix2Struct generation-budget analysis

Report:

```text
new_tokens
truncated
```

Summarize:

```text
mean answer token count
max answer token count
truncation rate
```

Default budget:

```text
32 tokens
```

No record should silently discard the `truncated` flag.

---

# 59. Confidence semantics

## LayoutLM

Provides:

```text
span_score
```

derived from start/end probabilities.

It is not a calibrated probability that the answer is correct.

---

## Pix2Struct

Provides:

```text
no answer confidence
```

in the canonical API.

Do not invent one from decoder token probabilities merely to make the table symmetric.

---

# 60. Runtime/resource comparison

Record for each model:

```text
model parameters
weight bytes

snapshot verification seconds
model load seconds

mean question latency
median question latency
p95 question latency

total evaluation time
peak GPU memory
```

---

# 61. Resource table

| Attribute | LayoutLM | Pix2Struct |
|---|---|---|
| Weight size | ~511 MB | ~1.13 GB |
| Parameter count | 127.8M | runtime measured |
| Requires OCR | Yes | No |
| Uses page pixels | No | Yes |
| Uses layout coordinates | Yes | implicit through image |
| Output | OCR span | generated text |
| Answer localization | Yes | No |
| Answer score | uncalibrated span score | none |
| Windowing / patching | 512 tokens | ≤2048 patches |
| ANLS | measured | measured |
| EM | measured | measured |
| Mean latency | measured | measured |
| Peak memory | measured | measured |

---

# 62. No adaptation

This workshop deliberately does not fine-tune either model.

Reasons:

- LayoutLM already has an E2E CORD adaptation tutorial.
- Pix2Struct has a separate E2E adaptation carrier candidate.
- Different adaptation mechanisms would confound the comparison.
- The workshop is about the **Document QA formulation**, not which fine-tuning recipe is stronger.

---

# 63. Common output directory

Use:

```text
outputs/document_qa_comparison/
```

---

# 64. `predictions.csv`

One row per model/question:

```text
record_id
page_id
field
question
gold_answer

model
answer

anls
exact_match
empty
answer_in_ocr

latency_seconds

start_word
end_word
span_score
n_windows

new_tokens
truncated
```

Model-incompatible fields are empty.

---

# 65. `field_metrics.csv`

```text
model
field
support
mean_anls
exact_match
empty_rate
answer_in_ocr_rate
```

---

# 66. `agreement.csv`

One row per QA record:

```text
record_id
page_id
field
question
gold_answer

layoutlm_answer
layoutlm_anls
layoutlm_exact

pix2struct_answer
pix2struct_anls
pix2struct_exact

agreement_category
```

---

# 67. `paraphrase_experiment.csv`

```text
model
record_id
field

canonical_question
paraphrased_question

canonical_answer
paraphrased_answer

canonical_anls
paraphrased_anls

answer_stable
```

---

# 68. `unanswerable_probe.csv`

```text
model
page_id
question
answer
empty
answer_in_ocr

span_score
new_tokens
truncated
```

---

# 69. `modality_robustness.csv`

```text
model
record_id
experiment

canonical_answer
variant_answer

canonical_anls
variant_anls
delta_anls

canonical_exact
variant_exact
```

Experiments:

```text
LayoutLM:
  no_layout

Pix2Struct:
  low_resolution
```

---

# 70. `resource_metrics.csv`

```text
model
model_id
revision
parameter_count
weight_bytes

load_seconds

mean_latency_s
median_latency_s
p95_latency_s
total_eval_seconds

peak_gpu_memory_bytes
```

---

# 71. `metrics.json`

Contains:

```text
OCR baselines

LayoutLM:
  aggregate
  per-field
  windowing

Pix2Struct:
  aggregate
  per-field
  generation

agreement

paraphrase sensitivity
unanswerable probes
modality robustness

timing
```

---

# 72. `provenance.json`

Record:

```text
notebook_spec
profile
pedagogical_mode

CORD:
  dataset repo
  immutable revision
  licence

parquet shards:
  path
  bytes
  SHA-256

page sample:
  seed
  split sizes
  page ids
  image digests
  OCR/layout digests
  QA sample digest

LayoutLM:
  model ID
  revision
  weight digest
  parameter count

Pix2Struct:
  model ID
  revision
  weight digest
  parameter count
  header font identity

runtime versions
device
dtype

question templates
ANLS implementation/version
```

---

# 73. Input manifest

Export:

```text
input_manifest.json
```

containing:

```text
50 test pages
~229 QA records

image dimension range
OCR word-count range
question-length range
answer-length range

validation verdict

rejection probes
```

---

# 74. Rejection probes

Include non-blocking checks showing refusal of:

### LayoutLM

- mismatched word/box count;
- box outside page;
- empty question.

### Pix2Struct

- empty question;
- image outside dimension ceiling;
- invalid generation budget.

These must not interrupt `Run all`.

---

# 75. BYOD comparison format

For real cross-model comparison, BYOD must provide **both** the page image and OCR/layout data.

Recommended ZIP:

```text
dataset.zip

pages/
  receipt001.png
  receipt002.png

records.jsonl
```

Each JSONL record:

```json
{
  "id": "q001",
  "page_id": "receipt001",
  "file": "receipt001.png",
  "question": "What is the total amount?",
  "answers": ["₱1,250.00"],
  "words": ["TOTAL", "₱1,250.00"],
  "boxes": [[...], [...]]
}
```

---

# 76. BYOD requirements

Recommended limits:

```text
1..100 pages
1..500 QA records

image side <=4096 px
question <=256 chars

1..2000 OCR words/page
one valid pixel box/word
```

Accepted answers may be omitted.

---

# 77. BYOD without answers

If `answers` are absent:

```text
evaluation verdict = not-measurable
```

but both models run and answers are exported.

---

# 78. BYOD without OCR

The common comparison path MUST reject records without:

```text
words
boxes
```

because LayoutLM cannot operate without OCR/layout input.

The notebook may mention that users who need pixel-only inference can use the standalone Pix2Struct tutorial.

Do not silently introduce Tesseract.

---

# 79. OCR boundary

The workshop MUST state:

> LayoutLM does not perform OCR.

The user or an upstream OCR system owns:

```text
text recognition
word ordering
word boxes
```

Errors at that stage propagate into LayoutLM.

---

# 80. Document privacy

Document pages may contain:

- names;
- addresses;
- financial transactions;
- account identifiers;
- receipts;
- tax information;
- medical information;
- signatures.

Do not upload confidential, restricted, personal, or regulated documents into a hosted notebook environment without authorization.

---

# 81. Default parameters

```python
USE_BYOD = False

MAX_NEW_TOKENS = 32

RUN_PARAPHRASE_EXPERIMENT = True
PARAPHRASE_MAX_RECORDS = 20

RUN_UNANSWERABLE_PROBE = True
UNANSWERABLE_PAGES = 5

RUN_MODALITY_ROBUSTNESS = True
ROBUSTNESS_MAX_RECORDS = 20

OUTPUT_DIR = "outputs/document_qa_comparison"
```

---

# 82. Standalone invariant

The notebook MUST NOT:

```text
git clone
pip install -e .
import either DIMER pipeline package
fetch DIMER source at runtime
call a DIMER worker
call a DIMER API
```

Carry notebook-local equivalents of:

- dataset parsing;
- page/question construction;
- ANLS;
- Exact Match;
- LayoutLM input preparation/windowing;
- Pix2Struct question-header handling;
- baselines;
- exports.

General-purpose upstream libraries are allowed.

---

# 83. Notebook cell plan

| # | Type | Section |
|---:|---|---|
| 0 | Markdown | Title and learning objectives |
| 1 | Markdown | Extractive vs generative Document QA |
| 2 | Markdown | Architecture comparison |
| 3 | Code | Form parameters |
| 4 | Markdown | Runtime |
| 5 | Code | Install/check pins |
| 6 | Markdown | Immutable model provenance |
| 7 | Code | Two embedded model manifests |
| 8 | Markdown | CORD-v2 provenance |
| 9 | Code | Verify pinned parquet metadata |
| 10 | Markdown | Read ground_truth and determine split |
| 11 | Code | Build 119/30/50 page split |
| 12 | Markdown | Fetch/verify selected page images |
| 13 | Code | Construct common page records |
| 14 | Markdown | Generate common QA records |
| 15 | Code | Build/validate ~229 held-out questions |
| 16 | Markdown | Common metrics |
| 17 | Code | ANLS / EM / containment helpers |
| 18 | Markdown | OCR baselines |
| 19 | Code | Last-number + keyword lookup |
| 20 | Markdown | LayoutLM semantics |
| 21 | Code | Stage/load LayoutLM |
| 22 | Code | Run held-out LayoutLM QA |
| 23 | Markdown | Layout contribution |
| 24 | Code | No-layout ablation |
| 25 | Code | LayoutLM paraphrase + unanswerable probes |
| 26 | Code | Unload LayoutLM |
| 27 | Markdown | Pix2Struct semantics |
| 28 | Code | Stage/load Pix2Struct |
| 29 | Code | Run held-out Pix2Struct QA |
| 30 | Markdown | Pixel dependence |
| 31 | Code | Low-resolution experiment |
| 32 | Code | Pix2Struct paraphrase + unanswerable probes |
| 33 | Markdown | Main comparison |
| 34 | Code | Aggregate + field metric tables |
| 35 | Markdown | Model agreement |
| 36 | Code | Agreement/error categories |
| 37 | Markdown | Workshop prediction exercise |
| 38 | Code | Selected qualitative comparisons |
| 39 | Markdown | Confidence and abstention |
| 40 | Code | Diagnostic summaries |
| 41 | Markdown | Resource comparison |
| 42 | Code | Latency/memory table |
| 43 | Markdown | Visual page panels |
| 44 | Code | Answer-location overlays |
| 45 | Markdown | Machine-readable exports |
| 46 | Code | CSV/JSON/provenance |
| 47 | Markdown | BYOD |
| 48 | Code | Optional BYOD branch |
| 49 | Markdown | Interpretation and limitations |
| 50 | Code | Terminal summary + assertions |

Every executable cell must have explanatory markdown immediately before it.

---

# 84. Workshop exercise

Before revealing comparative metrics, show one receipt and ask:

1. Which evidence does LayoutLM receive that Pix2Struct does not?
2. Which evidence does Pix2Struct receive that LayoutLM does not?
3. What happens if OCR misses the answer?
4. Can LayoutLM return text that is absent from OCR?
5. Can Pix2Struct?
6. Which model can highlight exactly where its answer came from?
7. Which model can potentially normalize or reformulate the answer?

Then reveal the actual outputs.

---

# 85. Required assertions

Dataset:

```text
CORD revision matches
shard sizes match
shard SHA-256 values match

199 unique supported pages reproduced
119/30/50 split reproduced

test page IDs unique
selected image corresponds to OCR record

test QA count matches pinned contract
gold answer span valid
gold answer equals OCR span
```

---

# 86. LayoutLM assertions

```text
model ID matches
revision matches
snapshot digests match

word count == box count
boxes inside image
answer start/end valid

returned start/end inside OCR words
returned answer equals returned OCR span

scores finite
n_windows >= 1
```

---

# 87. Pix2Struct assertions

```text
model ID matches
revision matches
snapshot digests match

processor is VQA configuration
header font available locally
max patches = 2048

answer is str
new_tokens valid
truncated is boolean
```

Do not assert generated answer appears on the page.

---

# 88. Metric assertions

```text
ANLS in [0,1]
Exact Match in {0,1}

all evaluation rows present
per-field supports sum correctly
```

Do not assert one model must outperform another.

---

# 89. Results that MUST NOT be release invariants

Never assert:

```text
LayoutLM ANLS > Pix2Struct ANLS
Pix2Struct ANLS > LayoutLM ANLS

OCR-free must be better
layout-aware must be better

paraphrasing must hurt
layout removal must hurt
image degradation must hurt

unanswerable questions must produce non-empty text
```

All are empirical findings.

---

# 90. Interpretation requirements

## OCR-free is not dependency-free

Pix2Struct removes the external OCR requirement but depends more directly on page visual quality.

---

## OCR-dependent QA can localize evidence

LayoutLM returns answer spans tied to page words and boxes.

That traceability can matter operationally even when aggregate ANLS is similar.

---

## Generative QA can leave the page

Pix2Struct is not constrained to copy OCR spans.

This provides flexibility but creates hallucination risk.

---

## Same answer metric, different error mechanisms

An ANLS error from LayoutLM may originate from:

```text
OCR error
box error
windowing
span selection
```

An ANLS error from Pix2Struct may originate from:

```text
visual encoding
question rendering
generation
token budget
hallucination
```

---

## The CORD task structurally favors extractive answers

Gold answers are built from OCR spans.

This should remain visible in every comparative interpretation.

---

## ANLS does not establish factual safety

A high mean ANLS does not mean the system should automatically:

- approve payments;
- populate records without review;
- make financial decisions;
- extract sensitive data for consequential actions.

---

# 91. Explicit non-goals

The workshop does not include:

- OCR engine installation;
- Tesseract benchmarking;
- fine-tuning LayoutLM;
- fine-tuning Pix2Struct;
- PDF rendering;
- multi-page document reasoning;
- table-specific QA;
- document-type classification;
- OCR benchmarking;
- handwriting recognition benchmarking;
- question-generation models;
- LLM fallback;
- RAG;
- DIMER service calls.

---

# 92. Recommended repository placement

Because this is a cross-model Document Intelligence workshop:

```text
ml-worker/
  integrations/
    dimer/
      workshops/
        document-qa/
          DIMER_Document_QA_LayoutLM_vs_Pix2Struct_Workshop.ipynb
          README.md
          docs/
            release-verification.md
          tools/
            build_notebook.py
            validate_notebook.py
```

The two carrier repositories should link to the central workshop.

---

# 93. Release verification

Preferred clean runtime:

```text
Kaggle Tesla T4
Python 3.12
```

Qualification must verify:

```text
no repo checkout
no DIMER worker
no credentials

both immutable model snapshots staged
both model digests verified

CORD shard identities verified
50 page images obtained
OCR/image identities aligned
~229 QA records reproduced

OCR baselines run
LayoutLM full evaluation run
Pix2Struct full evaluation run

paraphrase experiment completes
unanswerable probe completes
modality robustness completes

machine-readable outputs written
all cells succeed
```

---

# 94. Release evidence

Record:

```text
notebook commit
notebook blob

runtime
GPU

CORD revision
shard digests
test page IDs
QA sample digest

LayoutLM:
  model revision
  weight digest
  parameters
  ANLS
  EM
  per-field metrics
  latency
  peak memory

Pix2Struct:
  model revision
  weight digest
  parameters
  ANLS
  EM
  per-field metrics
  latency
  peak memory

baselines

paraphrase results
unanswerable outputs
layout ablation
image degradation

cell success count
wall time
output inventory
```

---

# 95. Terminal summary

The final cell should print:

```text
DIMER Document Question Answering Workshop
------------------------------------------

Dataset:
  Test pages: 50
  QA records: ...
  Fields: 10

System                ANLS     Exact Match   Empty
---------------------------------------------------
Last-number            ...        ...         ...
Keyword lookup         ...        ...         ...
LayoutLM               ...        ...         ...
Pix2Struct              ...        ...         ...

LayoutLM
  answer localization: yes
  OCR required: yes
  mean latency: ...

Pix2Struct
  answer localization: no
  OCR required: no
  mean latency: ...
  truncation rate: ...

Outputs:
  outputs/document_qa_comparison/
```

No automatic winner.

---

# 96. Position in the Document Intelligence track

```text
Document Intelligence
│
├── Document-Type Classification
│     └── DiT / RVL-CDIP
│
├── Document Question Answering
│     ├── LayoutLM          ← OCR + layout / extractive
│     └── Pix2Struct        ← pixels / generative
│
├── Table Intelligence
│     ├── Table detection
│     ├── Structure recognition
│     ├── TAPAS
│     └── DePlot
│
└── OCR / Document Extraction
      ├── GOT-OCR
      ├── SmolDocling
      └── other extraction models
```

---

# 97. Workshop learning arc

**Same business question**  
↓  
*Both systems are asked exactly the same thing.*

**Different document representation**  
↓  
*LayoutLM gets OCR words and geometry; Pix2Struct gets pixels.*

**Different answer mechanism**  
↓  
*LayoutLM extracts; Pix2Struct generates.*

**Common answer metric**  
↓  
*ANLS and Exact Match make outputs measurable together.*

**Traceability difference**  
↓  
*LayoutLM can point to answer words; Pix2Struct returns text only.*

**Robustness difference**  
↓  
*OCR/layout corruption affects one system; image quality affects the other.*

**Question sensitivity**  
↓  
*Paraphrasing can change either model's behavior.*

**No reliable abstention**  
↓  
*Both systems can answer when the page does not contain the requested field.*

**Deployment lesson**  
↓  
*Choosing a Document QA architecture is not only about ANLS—it also determines OCR dependency, traceability, hallucination surface, compute cost, and what evidence downstream users can inspect.*