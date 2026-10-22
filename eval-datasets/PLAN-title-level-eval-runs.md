# Title-level Mapping Eval — Run Plan (v1, draft)

**Status: Proposed — NOT approved. Nothing executes (no API call, no upload, no service call)
until §0 decision gates are answered.**

Goal: run the three 100-case title-level datasets (`outputs/{sku_cases_100,
brand_in_title_cases_100, titleonly_cases_100}.jsonl`) through the live BirlaNu turnover
product-mapping agent via LangSmith, so results land in the same place + format as the two
existing invoice-level runs and produce a comparable per-case-type readout.

---

## 1. Precedent — the copilot LangSmith eval (what we adapt FROM)

Harness: `~/AgentWork/langsmith-tool-evaluator/` (repo navneetlearns/langsmith-tool-evaluator;
working copies also at `~/AgentWork/eval-dashboard/langsmith-tool-evaluator/` and
`/mnt/f/Langsmith/COPILOT/`). Core at `langsmith-tool-evaluator/langsmith-tool-evaluator/`.

What it does (LangSmith mode):
- Reads runs from LangSmith project `seller-copilot-agent` — SDK `Client.list_runs` (paginated,
  `utils/langsmith_client.py`), `.env` with `LANGSMITH_API_KEY` / `LANGSMITH_ENDPOINT` /
  `LANGSMITH_PROJECT_NAME` (dotenv `override=False` so CLI flags win).
- Scores each run with an external LLM judge (OpenCode/DeepSeek, not LangSmith's judge).
- Writes results BACK to LangSmith: `create_feedback` per run (feedback mode) or
  **experiment mode** (`evaluate_project.py --experiment`):
  1. upsert dataset `tool-selection-evaluator` (read_dataset → clear old examples → create_dataset
     if missing → create_example per row, carrying inputs/outputs/metadata)
  2. `langsmith.evaluate(target_fn, data=ds.id, evaluators=[fn], experiment_prefix=..., client=..., metadata=...)`
     — target is POSITIONAL-ONLY (SDK 0.10.18 in that venv)
  3. experiment appears in LangSmith UI → Datasets & Testing → Experiments
- Evaluator return shape: `{"key": "...", "score": <num>, "comment": <json str>}`.
- Ops discipline (HEART.md): no retry, record failures as results, versioned artifacts.
- Timeline + constraints: key re-attached Aug 7 2026; **LangSmith free trace quota exhausted
  Aug 18 2026 — ingestion died (reads/writes still worked)**; that is why the copilot evals moved
  to the direct-API pipeline (v1–v4 + dashboards). The LangSmith-experiment machinery above is
  the in-house working precedent we reuse; the quota lesson carries over as a risk.

## 2. What changes for the mapping eval (adaptation map)

| Copilot eval | Mapping eval |
|---|---|
| LandSmith project `seller-copilot-agent` (read traces) | LangSmith workspace `21623a89-1bf7-45e9-a86f-0cbfba36e534`, project `agent-playground`; **no traces needed — data comes from our built datasets** |
| Target = stub rescoring historical traces | Target must CALL the mapping service per example (as the platform's own `py_sdk_evaluate` runs did; revision pinned `v8.4.19-2-ga863d2082d4`) |
| Evaluator = external LLM judge (7-step prompt) | **Deterministic evaluators, no judge**: `false_map` / `missed` / `sku_accuracy` / `status_accuracy` / `wrong_sku`, per example — semantics already reverse-engineered and exactly reproducible offline (see Phase 0) |
| Credentials: navneetlearns key for seller-copilot workspace | Needs a key with access to workspace `21623a89` (the F:\Langsmith\COPILOT\.env key got a 404 against a mapping session id in the 10-17 session — workspace access unverified; do NOT re-test without explicit permission) |
| Results: LangSmith UI + feedback per run | Same UI routing; then export CSV per experiment (like the two existing exports) + readout in this repo |
| Dataset upsert per run from traces | Dataset = our 3 built files (static, versioned in git); upload once per version |

## 3. Phases

### Phase 0 — Offline evaluator + replay gate (no access, no permissions needed)
Build in `eval-datasets/code/`:
- `evaluators_mapping.py` — the five deterministic evaluators operating on
  `outputs.line_items` vs `reference expected` (equivalence-aware via `equivalent_skus`;
  confidence not scored). Formulas to reproduce exactly (verified in the 10-17 analysis):
  `status_accuracy = (N − false_map − missed)/N`; `sku_accuracy = (N − false_map − missed − wrong_sku)/N`;
  `false_map` = mapped where GT says UNMAPPED; `missed` = unmapped where GT wants mapped;
  `wrong_sku` = both mapped, SKU not equal/equivalent.
  **Re-verified 2026-10-22** against both exports (150 invoices, 1,842 lines): the three
  count columns and both ratios reproduce exactly for 149 of 150 invoices, `sku_accuracy`
  and `wrong_sku` everywhere (equivalents honoured, e.g. 96001839 ≡ 96001845). The single
  exception shows status is scored as an **exact three-way label match**
  (`MAPPED` / `MAPPED_BRAND_NOT_FOUND` / `UNMAPPED`), not the two-way mapped/unmapped split:
  one line with GT `MAPPED` returned `MAPPED_BRAND_NOT_FOUND`, scoring status 12/13 while
  `false_map` and `missed` both stay 0. Quote the three-way definition when reporting
  status accuracy; treat `(N − false_map − missed)/N` as an approximation.
  No `evaluators_mapping.py` exists in this repo — the replay was run ad hoc from the CSVs;
  the Phase 0 files below are still to be built if the runner is ever wired up.
  **Acceptance = replay gate:** recompute all five columns from the two existing exports
  (`/mnt/f/Langsmith/BRAND NAME/*.csv`, 71 + 79 rows) and require EXACT equality with the
  published evaluator columns. This pins evaluator semantics with zero API.
- `mapping_eval.py` — runner skeleton: `--replay <csv>` (the gate), `--dry-run` (print upload +
  run plan), `--run` (Phase 2, wired path TBD by §0.1). Runner follows the copilot harness
  patterns: dataset upsert or batch `Client.create_examples`, `evaluate()` positional target,
  `{"key","score","comment"}` evaluator returns, per-example timeout, failures recorded
  (HEART: no retry), `max_concurrency` modest, `num_repetitions=1`.

### Phase 1 — Smoke run (needs §0.1/§0.2 access)
1. Upload the 3 datasets to the mapping workspace (names/split per §0.4), or hand files over.
2. Wire the target:
   - **Route A (platform):** deliver the datasets to whoever ran the two existing experiments;
     point their existing `py_sdk_evaluate` runner (target = mapping pipeline) at the new datasets.
   - **Route B (self):** caller hits the mapping service endpoint directly — needs endpoint URL +
     auth token from the platform team, plugged into our runner as the target function.
   - **Hybrid:** they supply only the target wrapper; we run `evaluate()` from here.
3. Smoke: 3–5 examples (one per set) end-to-end; verify row-shaped outputs match the export
   schema (`line_items[].{sku_code,status,mapped_product_name,matched_brand,mapping_confidence_score}`).

### Phase 2 — Full runs + export
- Run each dataset (3×100) with the smoke-validated path; pin both the dataset version and the
  mapping revision (§0.3) in experiment metadata.
- Per-example timeout ~300s; async hang precedent ("event loop closed", 62 min) ⇒ keep
  concurrency low, record timeouts as results.
- Export each experiment to CSV in the same shape as the existing exports (columns: id, inputs,
  reference_outputs, session_name, repetition, outputs, run, status, error, latency, tokens,
  total_cost, false_map, missed, sku_accuracy, status_accuracy, wrong_sku, note).

### Phase 3 — Readout + docs
- `eval-datasets/readouts/EVAL_READOUT_v1.md`: per-set results, headline comparable to §6 of the
  tracker (status/sku accuracy, error families per case type), model-vs-GT status-rule check,
  confidence-band table per set.
- Update `BirlaNu_ActionTracker_and_Mapping_Analysis.md` (new §7) + PROJECTS.md line.
- Optional (separate go): static dashboard in the eval-dashboard repo style — not part of this plan.

## 4. §0 Decision gates (answer before any execution)

1. **Runner route:** A = platform team runs our datasets with their harness; B = we run from here
   with endpoint + token provided; Hybrid = they provide the target wrapper, we run.
2. **LangSmith access:** which API key sees workspace `21623a89`; explicit permission to use it
   (standing rule: no API/credential use without explicit permission). Dataset visibility confirm.
3. **Mapping revision pin:** keep `v8.4.19-2-ga863d2082d4` (comparability with the two existing
   runs) or run current revision (measures latest behavior; not directly comparable).
4. **Dataset names + split** in the workspace (proposal:
   `turnover-product-mapping-titlelevel-{sku|brand|titleonly}`, split `base`).
5. **Results home/format:** readout-only, or readout + CSV exports copied to
   `F:\Langsmith\BRAND NAME\` (current export drop folder), or also a dashboard.

## 5. Verification checklist (after execution)

1. Phase 0 replay gate: all 5 columns exact on all 71 + 79 rows.
2. `verify_datasets.py` still passes (300/300 lineage) on the exact files that were uploaded.
3. Smoke run: 3/3 examples return scorable line_items; CSV export parses with the same column set.
4. Full runs: 300 examples accounted (success/error), no silent drops; failures recorded per HEART.
5. Readout numbers recomputed from the exported CSVs, not restated from memory.
6. Docs updated: eval-datasets README + tracker §7 + PROJECTS.md; committed locally (push = separate ask).

## 6. Risks / pitfalls

- **LangSmith quota:** the free-tier trace-limit exhaustion (2026-08-18) froze an entire project —
  check quota/plan status for the mapping workspace BEFORE the full run; prefer dataset upload +
  experiment (writes) over relying on ingestion.
- **Credentials:** never use any key (even read-only) without explicit permission — the exact
  failure that happened on 10-17 with the F: COPILOT .env; that key is unverified for this workspace.
- **Target fidelity:** a prompt-only local replica is NOT comparable to the live pipeline (no
  retrieval, no pre/post-processing); only counts if the real mapping service is exercised.
- **Timing:** per-title calls should be fast (invoice-level run: 3.7–190s per 13-line invoice), but
  set per-example timeouts + concurrency caps; record timeouts, don't retry.
- **Evaluator drift:** the replay gate (Phase 0) is the defense — do not skip it.
