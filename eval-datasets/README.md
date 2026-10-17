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

## Files

- `code/build_datasets.py` — the builder (single source of truth for the sets)
- `code/verify_datasets.py` — independent lineage verifier (300/300 pass)
- `code/upload_to_langsmith.py` — SDK uploader for the three sets (the LangSmith UI
  file-import fails on this data: its schema inference can't handle the dynamic-key
  `catalog_results` or mixed null/number fields — see the run error "Couldn't modify
  dataset. failed input schema validation")
- `outputs/*.jsonl` — the three datasets
- `outputs/selection_summary.csv` — flat list of all 300 chosen cases for eyeballing
- `PLAN-title-level-eval-runs.md` — proposal for running these via LangSmith experiments
  (adapted from the copilot eval harness); §0 decision gates pending user answers
