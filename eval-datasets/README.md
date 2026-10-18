# Eval datasets — title-level mapping sets (built 2026-10-17)

Three 100-case LangSmith-ready datasets for the BirlaNu turnover product-mapping eval
(map_catalog_products / revision v8.4.19-2 family), built from the two experiment exports
in `F:\Langsmith\BRAND NAME\`. One test case = ONE product title (not an invoice).

## The three sets (`outputs/`)

| file | case type | chosen mix (expected_status) | picked from |
|---|---|---|---|
| `sku_cases_100.jsonl` | title carries an embedded 8-digit SKU code | MAPPED 100 | 176 titles / 200 occurrences |
| `brand_in_title_cases_100.jsonl` | no code, title carries the birla/hil token | MAPPED 63 / UNMAPPED 37 | 495 titles / 1,039 occurrences |
| `titleonly_cases_100.jsonl` | no code, no brand token (pure fuzzy matching) | MAPPED 34 / MBNF 32 / UNMAPPED 34 | 405 titles / 603 occurrences |

The three buckets are mutually exclusive and disjoint (300 unique titles total, verified).
`titleonly_cases_100.jsonl` includes 2 examples with an empty candidate set (abstention cases).

## Example shape (one JSONL line)

```
{"inputs": {"catalog_results": {<raw title>: [candidates...]},
            "distributor_exclusive": false,
            "line_items": [{hsn_sac_code, product_name, serial_no: 1, unit_rate_excl_gst}],
            "product_mapping_guidance": <12,494-char current prompt text>,
            "workspace_id": "21623a89-1bf7-45e9-a86f-0cbfba36e534"},
 "outputs": {"expected": [{equivalent_skus, expected_sku, expected_status, product_name}]},
 "metadata": {category, source_experiment, source_invoice_id, source_serial_no,
              normalized_title, title_occurrences_in_pool, n_candidates}}
```

Pipeline-shaped 1-line "invoice" — nothing on the consuming side needs restructuring.

## Selection method (deterministic, no RNG)

1. Bucket every line occurrence from both exports by title text (code → sku; else brand token → brand; else titleonly).
2. One example per distinct normalized title; representative variant = largest candidate set (ties hashed).
3. Stratified by expected_status, quotas proportional to the bucket's title mix (largest-remainder rounding to exactly 100).
4. Within a stratum: ordered by sha1(normalized title).

Rebuild twice → byte-identical outputs (md5-verified).

## Data notes

- `product_mapping_guidance` was standardized: 4 of 151 source rows carried an older
  10,668-char prompt draft (missing the brand-literal-text paragraph, mm↔inch table,
  SKU-fallback notes). All examples now carry the current 12,494-char text.
- Provenance kept per example (source experiment, invoice id, serial, occurrence count).
- Known edge included as-is: 17 titles disagree on GT across occurrences (GT is
  candidate-set-conditional); representative rule picks the largest candidate set.
- Candidate sets are invoice-time retrieval results — these sets test mapping, not retrieval.

## Rebuild

```
cd code && python3 build_datasets.py     # reads the two F: exports, rewrites ../outputs/
```

Change `quotas(...)` target / the per-bucket title cap in `build_datasets.py` to cut
different sizes or re-sample different titles (availability per bucket above).

## Catalogue check (VLOOKUP workbook, 2026-10-18)

`outputs/catalogue_check/` — the brand-name eval run's INPUT products vs the current
product catalogue export (`F:\Product\zotok\Product Mapping\Eval Related Product Mapping\products-18-October-2026-10-52-16.csv`,
2,335 products), cleaned for MANUAL one-to-one comparison of the two name columns:

- `eval_inputs_vs_catalogue_vlookup.xlsx` — sheets: `How to use` | `Input products`
  (the 100 unique products sent into run `turnover-product-mapping-c4cce19d`; the 2
  hang+rerun duplicates collapsed to one entry) | `Catalogue` (Sku Code | Product Name) |
  `Catalogue A-Z` (Product Name sorted A→Z | Sku Code — for VLOOKUP approximate lookups,
  which require an ascending-sorted first column).
- `eval_run_input_products.csv`, `catalogue_cleaned.csv` — the two sides as CSV.

Rebuild / refresh for a new run or a new catalogue export:

```
python3 code/build_catalogue_check.py [--run <run.csv>] [--catalogue <cat.csv>] [--copy-to <dir>]
```

Cleaning (both sides, same rules): whitespace/nbsp collapse, zero-width removal,
mojibake repair, duplicate `Description` column dropped, sorted by SKU; all SKU
columns written as TEXT. None of the 100 input titles matches a catalogue name
exactly — comparison is by key words / fuzzy lookup, not bare VLOOKUP.
NOTE: the catalogue export carries only 9000/9600/9800/9900-prefix SKUs — the
`412xxx` family seen live in eval candidates is absent, so such rows read as
"not in catalogue" regardless.

### Phase 2 — Power Query run + verification (2026-10-18)

The comparison was then done as a Power Query **fuzzy merge** inside the workbook
(Merge Queries in the `Input Products` query — `Table.FuzzyNestedJoin` on
`Input Product (from eval run)` ↔ `Product Name`, LeftOuter, threshold **0.70**,
1 match per row, similarity score shown; result expanded and loaded to a new sheet
`Input Products (2)` — wiring verified from the saved file's stored M code + queryTable
parts).

Output summary (manual verification pending):
- **48/100** inputs matched above threshold (scores 0.70–1.00, median 0.86; 24 of them ≥ 0.85).
- 52 below threshold — but **10 of those carry an embedded SKU that resolves in the
  catalogue**; fuzzy matching missed them because brand + code tokens dilute the score.
  "Blank" ≠ "absent" — check the code column first.
- Eyeball notes: a few CPVC inputs matched UPVC catalogue items (family slips), and one
  product's coded title came out blank while its uncoded twin matched (similarity
  ignores codes).

`fuzzy_match_offline_preview.csv` / `.xlsx` — OFFLINE STAND-IN computed the same day while
the in-workbook load was still failing (approx token-Jaccard with fraction normalization +
embedded-code pinning; top-3 candidates per input; CSV byte-reproducible from the script).
Superseded by the PQ run above for official numbers; kept as fallback + alternate-candidate
source. Regenerate (deterministic):

```
python3 code/build_fuzzy_candidates.py   # -> outputs/catalogue_check/fuzzy_match_offline_preview.{csv,xlsx}
```

`below_threshold_candidates.csv` / `.xlsx` — top-3 proxy candidates (best-first, with SKUs) for
the 42 test cases the PQ run left blank (the non-code blanks) — the scan aid for the manual pass.

## Files

- `code/build_datasets.py` — the builder (single source of truth for the sets)
- `code/verify_datasets.py` — independent lineage verifier (300/300 pass)
- `code/upload_to_langsmith.py` — SDK uploader for the three sets (the LangSmith UI
  file-import fails on this data: its schema inference can't handle the dynamic-key
  `catalog_results` or mixed null/number fields — see the run error "Couldn't modify
  dataset. failed input schema validation")
- `code/make_ui_variant.py` — UI-import-friendly variants of the three sets (recursively omits
  null-valued dict fields; canonical files untouched; also a 3-row starter) → `outputs/ui_upload/`
- `outputs/ui_upload/` — the variants (`*_cases_100_ui.jsonl`) + `test_3rows.jsonl`
- `outputs/*.jsonl` — the three datasets
- `outputs/selection_summary.csv` — flat list of all 300 chosen cases for eyeballing
- `code/build_catalogue_check.py` — manual-verification workbook builder (Catalogue check section above)
- `code/build_fuzzy_candidates.py` — offline fuzzy-match preview generator (Phase 2 section above)
- `outputs/catalogue_check/` — manual-verification workbook + cleaned CSVs + `fuzzy_match_offline_preview.{csv,xlsx}`
- `code/build_linewise_payload.py` — product-line-wise rebuild of a downloaded experiment
  export → upload payload (one row per product line; recorded results + the 5 house scores
  recomputed per line — totals reproduce the source exports exactly)
- `code/upload_experiment.py` — fires a payload at POST /datasets/upload-experiment
  (`--dry-run` = offline validation; needs `LANGSMITH_API_KEY` to send). This endpoint is
  the only route to make recorded results appear as an EXPERIMENT in LangSmith
- `outputs/linewise/` — line-wise artifacts per experiment: `<slug>_linewise_review.csv`
  (flat, Excel-friendly), `<slug>_linewise_rows.jsonl` (lean records),
  `<slug>_upload_payload.json` (the exact POST body)
- `PLAN-title-level-eval-runs.md` — proposal for running these via LangSmith experiments
  (adapted from the copilot eval harness); §0 decision gates pending user answers
