# Receipt capstone — annotation audit record

**State: no human annotation audit has been performed.** Nothing in this repository claims independently verified CORD references.

## What exists

- A prediction-blind **automated** source audit, run at every `prepare` stage (`outputs/reference_audit.json`). It records the reference state of each receipt and field (`present_usable`, `not_annotated`, `ambiguous_reference`, `invalid_reference`), a reason for each non-usable reference, annotation-structure errors and the reference digest. It uses parser `cord-reference-parser-v1`, the same `cord_mixed_v1` grammar that is applied to predictions.
- The frozen freezer audit (`tutorials/receipt_intelligence/dataset_audit.json`): official counts, two within-train exact-pixel duplicate pairs, category frequencies. Near-duplicate review is marked "not performed".

## Required before qualification (specification §5.3)

The audit set is deterministic and prediction-blind, drawn from `train` and `validation_*` only and never from `test`. It covers:

1. every distinct `reason` value reported by `reference_audit.json`, with up to three receipts each, chosen by ascending `sha256("receipt-capstone-v1|audit|" + receipt_id)`;
2. every receipt whose total needed the multi-line rule (`n_lines > 1`);
3. twenty further receipts chosen by the same hash order.

For each audited receipt, record the ID, the fields checked, the reviewer's name, the date, whether the published reference was confirmed, and any disagreement. AI-assisted inspection must be labelled as such and does not count as independent human review. A correction creates a new reference-manifest version and a transparently labelled rerun; the original result is kept for comparison. Held-out ground truth is never silently revised.

| Receipt | Fields | Reviewer | Date | Outcome |
| --- | --- | --- | --- | --- |
| *(none yet)* | | | | |
