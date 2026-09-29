# Receipt Intelligence Capstone — Coding-Session Handoff

**Snapshot:** 28 September 2026  
**Repository:** `kurtvalcorza/layoutlm-document-qa-pipeline`  
**Branch:** `feat/small-business-receipt-capstone`  
**State:** Partial source drafts and asset-preparation work; no runnable capstone notebook; not release-qualified.

## 1. Current task and authority

Kurt approved Capstone 8, **From Receipts to Records — Small-Business Document Intelligence**, and its implementation specification. The build session stopped incomplete. His latest instruction is: **“commit these drafts, open PR with handoff for coding session.”** This change preserves the available drafts and records the continuation work; it is not a completed-build claim.

Read [the approved specification](receipt-capstone-spec.md) in full before implementation. It is the unchanged uploaded `DIMER_SMALL_BUSINESS_RECEIPT_INTELLIGENCE_CAPSTONE_SPEC.md`, version 1.0. The specification, rather than this shorter handoff, governs the detailed data, training, metric, artifact, BYOD and teaching contracts.

No merge, release promotion, deployment, unrelated repository changes or paid execution is authorized by this handoff. Continue on a feature branch and preserve existing work. Every `NAIRA-SEU` repository remains read-only.

## 2. Repository state found during publication

Live reads found `main` at `e2e0f53aa2fbde521f03f4478f753f4ed9786ab5`. The intended feature branch already existed at `d379cc14f4f8c91f6933a6a5c3d67308ab60822e`, three commits ahead of that main and zero behind. There was no open PR for this head branch when checked.

That pre-existing branch contains nine additions, retained without modification:

- `tools/freeze_receipt_assets.py`
- `tools/receipt_prepare.py`
- `tools/receipt-requirements.in`
- `tools/receipt-requirements.lock`
- `tutorials/receipt_intelligence/data_manifest.json`
- `tutorials/receipt_intelligence/dataset_audit.json`
- `tutorials/receipt_intelligence/split_manifest.json`
- `tutorials/receipt_intelligence/model_manifest.json`
- `tutorials/receipt_intelligence/ocr_manifest.json`

The existing head's message is “Freeze verified receipt source, OCR closure and standalone dependency lock.” File existence and commit metadata were checked; those words are the existing commit's claim, not independent re-execution evidence from this publication session. The preparation script is build-only and calls the asset freezer. Do not mistake it for a completed standalone notebook runtime.

Re-read the current branch and PR before acting: concurrent changes may have landed. The new preservation commit is based on that feature head, not a replacement tree based only on main. Existing production code, original tutorials, tests and CI are not changed by the preservation commit.

## 3. Draft inventory and an unresolved upload

Two draft Python modules are included unchanged:

| File | Draft scope | SHA-256 of original attachment |
|---|---|---|
| `tools/receipt_common.py` | Canonical JSON and hashes, pinned downloads, path checks, spreadsheet-facing CSV escaping, atomic stage receipts and downstream invalidation | `b32f13d1c8e37b0bcd0a4273f68fc4b0192be7060ce70faded868ebbc1752a8e` |
| `tools/receipt_data.py` | Official-partition image loader, deterministic validation-role assignment, exact-duplicate quarantine, orientation transforms and evaluator-owned references | `9a7be20f4c2c20675f1e85c23f8b6bd421f04cfe9bba030634e2c23ec2cd375f` |

**`tools/receipt_fields.py` is not included in the preservation commit.** Its upload was blocked twice by the tool with: “we couldn't determine the safety status of the request.” The file was not modified, replaced with a stub, or uploaded through another route. This is a publication limitation, not a code-review verdict.

The missing draft is available as a `receipt_fields.py` attachment in Kurt's originating conversation. It contains conservative amount parsing, keyword/geometry rules and an input-only prediction interface. Its exact attachment identity is:

- Bytes: `13038`
- SHA-256: `bcf62b4fd1103ed071af9c8bdb841e8a5900537d898f40382805fcbc07facbd4`
- Git blob identity: `eb3c698bf17c83f0dd926a0cf4345afa5dd5fca7`

Obtain and inspect that attachment through the normal authorized coding workflow before integrating it. Do not fabricate its contents from this summary. **`receipt_data.py` imports `receipt_fields`; the draft set is therefore not import-complete in this PR.**

The archived specification has SHA-256 `d4c890cbb85294edcbc2f6e8dc042ec26e92df49fe27b87d70df6614d8120153` and Git blob identity `a1d4c3c8e4b604ddd5b344915a8575b11015ad7b`. It is a design document, not test evidence.

## 4. Checks actually performed in the preservation session

- Read all three local draft sources and inspected their boundaries.
- Python **3.13.5** `ast.parse` and `compile(..., 'exec')` succeeded for all three original drafts, including the unpublished fields module. These are syntax-only checks; the specified supported runtime is Python 3.12.
- Scanned the three draft files and the specification for common private-key, GitHub-token, AWS-access-key, literal-secret and signed-download patterns: no matches. This was a limited pattern scan, not a security audit or a scan of all inherited branch files.
- Compared returned Git blob identities for the uploaded common/data modules and specification with identities computed from their original attachment bytes.

**Not performed:** a Python 3.12 dependency installation, draft functional tests, repository pytest/Ruff runs, real CORD preparation with these drafts, actual OCR, pretrained inference, gradient training, GPU execution, artifact replay, BYOD qualification or an independent annotation audit. Do not report syntax compilation as a passing capstone test suite. No model-performance numbers are available from these drafts.

## 5. First integration work

The following are source-inspection priorities, not a completed behavioral review:

1. **Reconcile the two asset/data paths.** The draft `receipt_data.py` has its own `SOURCE_FILES` list, with `bytes=None` for all four train shards, and produces its own runtime manifests. The branch already has frozen asset and split manifests from another preparation path. Compare exact revisions, hashes, byte counts, receipt-ID construction, role assignment, exclusions and manifest schemas before choosing one source of truth. Do not discard the existing manifests or declare the draft's pins independently verified without evidence.
2. **Restore and review the fields draft.** Its `predict` expects a capstone runner with `answer(question, document)`, not the host pipeline's original keyword-argument API. Implement and test an explicit adapter; do not change production defaults incidentally.
3. **Complete receipt-failure handling.** The fields draft validates the document before producing predictions, including a 2,000-word limit. The future orchestrator must distinguish contract/global failures from recoverable receipt failures and keep frozen validation/test receipts in the primary denominator. Never silently drop a difficult receipt.
4. **Check references and geometry.** Test EXIF transforms, malformed boxes, `subtotal`/`sub_total` aliases, `gt_parse` disagreements, multiple annotation lines and `not_annotated` versus absent/zero. The current parser is a draft; reconcile its conservative handling with spec sections 5 and 10 before freezing references.
5. **Harden stage ownership and predictor evidence.** Test failed retries, missing/altered outputs, graph validation, run/experiment identity and cache invalidation. Check that reported model answer strings truly correspond to the exported contiguous OCR indices. Test gold noninterference; do not infer it solely from an allowlist.
6. **Finish dependency/runtime qualification.** Inspect the inherited lock and OCR closure rather than substituting floating packages. A package manifest is not evidence that the composed OCR runtime installs and executes on a fresh supported host.

Keep these two published Python drafts unchanged in the preservation commit. Subsequent repairs should be explicit, tested commits with their own evidence.

## 6. Remaining implementation

The specification names the expected implementation areas. Reconcile names with the branch rather than assuming missing files already exist:

| Area | Required next work |
|---|---|
| Actual OCR | Pinned Tesseract execution, `eng+ind`, OEM 1 / PSM 4, TSV parsing, per-image timeout, cache identity, full-image processing and evidence overlays |
| Model layer | Fresh pinned upstream LayoutLM, complete windowed span inference, three-system outputs with identical actual-OCR inputs |
| Alignment/training | Training-only OCR-to-reference span alignment; four actual float32 epochs; last four encoder blocks plus QA head; optimizer/tensor-change checks |
| Selection | Select trained epochs 1–4 by validation-model total numeric exact match, earliest tie; never reuse the host helper's ANLS/epoch-0 selection unchanged |
| Metrics/diagnostics | Unfiltered primary total EM, secondary field metrics, paired receipt bootstrap, reference-text sensitivity and evaluator-only OCR recoverability |
| Review policy | Separate validation-policy selection; 95% empirical accuracy, 50% coverage and 25 accepted receipts; explicit refer-all when infeasible |
| Runtime/provenance | Stage graph, immutable selection record, controlled reruns, resource measurements and separated input/reference workspaces |
| Artifacts | SafeTensors subset plus serialized OCR/parser/questions/policy configuration; fresh-process image-to-record reconstruction and strict parity checks |
| BYOD | Optional inference and substantive adaptation/evaluation paths; secure ZIP validation, disjoint roles, privacy guidance, transferred thresholds not treated as validated |
| Notebook | Deterministic generator, embedded implementation, all guided sections, validation-only change-one-thing activity, disclosure and evidence-based conclusion |
| Tests/docs/registration | CPU contracts and negative controls, generator parity, README/tutorial registry, release validator and CI integration; exact-revision hosted and BYOD evidence |

There is **no** `DIMER_Small_Business_Receipt_Intelligence_Capstone.ipynb` from this partial build. Do not add an Open in Colab badge until the actual notebook exists on the referenced branch.

## 7. Preserve the approved experimental contract

These decisions come from the approved specification; they are not observed results:

- Target profile/mode: `E2E` / `GUIDED`, Notebook Spec 2.2. The portable notebook carries its own implementation, requires no repository clone/source fetch or DIMER worker, and runs without secrets, mandatory uploads or manual restarts.
- Preserve CORD's official 800/100/100 membership before integrity exclusions; split official validation into two deterministic roles before exclusions. Do not reuse the original tutorial's validation/test re-split sample or an already CORD-adapted artifact.
- Compare fixed rules, frozen LayoutLM and newly receipt-adapted LayoutLM on the same actual OCR. Reference-text runs are separately labeled diagnostics, not substitutes for the image-based result.
- Use the same frozen numeric normalizer for predictions and references. No digit invention, total reconstruction by arithmetic, or missing-charge-as-zero shortcut.
- Primary correctness requires an exactly matching canonical total. OCR/model/parse failures stay incorrect when usable truth exists. Missing annotations do not establish absence.
- Train from the fresh upstream base, not a prior receipt adapter. Select the trained candidate only on `validation_model`; select review policy only on `validation_policy`; freeze before test scoring.
- Review decisions concern total amounts only and do not certify accounting records. Infeasible policy, zero unflagged coverage and a losing fine-tuned model remain valid reportable outcomes.
- Keep the serving artifact free of gold labels/raw receipts. Its manifest is not permission to execute arbitrary uploaded code or fetch arbitrary URLs.
- The sample concerns Indonesian receipts. No Philippine-business, unseen-merchant, production or accounting-grade performance claim follows from its scores.

## 8. Resume in the coding workspace

Read workspace `AGENTS.md`, `.agent/steering/verify-first.md` and applicable repository instructions first. The prior operating handoff identifies the local root as `C:/Users/Kurt Valcorza/Projects`; confirm the current workspace instead of assuming that path exists everywhere.

```powershell
git status --short
git fetch origin
git branch --list feat/small-business-receipt-capstone
git switch feat/small-business-receipt-capstone
git log --oneline -6
git diff --stat origin/main...HEAD
```

If no local branch exists, create a tracking branch from `origin/feat/small-business-receipt-capstone`. If it does exist, reconcile local changes and fast-forward only when safe. Never reset or force-push away another session's work. Read this PR's latest head/comments and the full specification before continuing.

Local validation remains CPU-only. Use clearly labeled dependency-injected fixtures or tiny randomly initialized model-interface tests where appropriate; do not install incomplete fake `torch` modules globally. Pretrained/GPU notebook qualification belongs in hosted Colab/Kaggle, with exact notebook/carrier and runtime identities recorded.

Once the missing attachment is available and the draft imports are repaired, add focused tests before extending the workflow. Run the existing repository baseline separately so pre-existing failures are distinguishable from new ones. Then implement in dependency order: assets/data → OCR/rules → model/alignment/training → metrics/policy/freeze → export/replay → BYOD → guided notebook and registration.

## 9. Completion report expected from the next session

Report separately: source built, source checks, CPU tests, actual hosted execution, annotation-review scope, release qualification and publication state. Include exact commit/notebook identities, commands and observed outcomes; do not infer success from an agent report or a green badge alone.

Keep this PR in draft while the implementation is incomplete. No merge or Release-grade promotion is implied. The first immediate blocker is the missing `receipt_fields.py` source; the main engineering task after that is reconciling the preserved drafts with the branch's existing asset-preparation work and completing the specified standalone workflow.

## 10. Update — capstone build (28 September 2026, later session)

The standalone build is now on this branch. Where it differs from the sections above, this section supersedes them. The sections above are kept as the historical handoff.

- **`receipt_fields.py`:** the missing draft was still unavailable. A new module was written from specification §§8–9 and is labelled as new, not reconstructed. If the original attachment (SHA-256 `bcf62b4f…`) turns up, diff the two before merging.
- **Data path reconciled:** the frozen manifests are the single source of truth. `receipt_data.py` now verifies every shard and row against them and recomputes roles and exclusions, which must match. The draft's own pins and ID scheme were removed.
- **Implemented:** pinned OCR install and contract, rules, frozen and adapted LayoutLM, training-only alignment, epoch 1–4 selection, metrics with paired bootstrap, review policy with refer-all fallback, sealed selection record, diagnostics, SafeTensors bundle, fresh-process replay, both BYOD modes, the generated 16-section guided notebook, tests, CI and registry entries.
- **Evidence:** built, source-checked and CPU-tested with labelled test doubles, plus one chain with a local Tesseract build. **No hosted run, no real-CORD or pretrained execution and no annotation audit exist yet.** See `receipt-release-evidence.md` for commands and results, and `receipt-capstone.md` for the resolved specification details.
- **Next:** one fresh Colab T4 Run all on the PR head, which will first qualify the Linux OCR install. Then BYOD positive and negative runs and the annotation audit. Keep this PR in draft and Candidate.
