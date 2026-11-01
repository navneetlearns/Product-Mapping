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
- 52 below threshold — of which **10 carry an embedded SKU that resolves in the catalogue**
  (fuzzy missed them: brand + code tokens dilute the score). 16 titles carry codes in total,
  and **6 of those the fuzzy run caught anyway** — so the running tally is **58/100
  matched-or-code-confirmed, 42 still open** (48 + 10; adding all 16 would double-count the 6).
  "Blank" ≠ "absent" — always check the code column first.
- Eyeball notes: a few CPVC inputs matched UPVC catalogue items (family slips), and one
  product's coded title came out blank while its uncoded twin matched (similarity
  ignores codes).
- Input-set census: every one of the 100 titles carries a brand token — 98 Birla/Birlanu,
  2 HIL-only ("HIL PVC PIPE…"); a brand scan looking for birla/birlanu alone would miss
  exactly those 2.

`fuzzy_match_offline_preview.csv` / `.xlsx` — OFFLINE STAND-IN computed the same day while
the in-workbook load was still failing (approx token-Jaccard with fraction normalization +
embedded-code pinning; top-3 candidates per input; CSV byte-reproducible from the script).
Superseded by the PQ run above for official numbers; kept as fallback + alternate-candidate
source. Regenerate (deterministic):

```
python3 code/build_fuzzy_candidates.py   # -> outputs/catalogue_check/fuzzy_match_offline_preview.{csv,xlsx}
```

`below_threshold_candidates.csv` / `.xlsx` — top-3 proxy candidates (best-first, with SKUs) for
the 42 test cases the PQ run left blank (the non-code blanks) — the scan aid for the manual pass
(a copy sits in the F: drop folder next to the workbook).

### Resolving the 42 — resolver BUILT (2026-10-19)

`code/build_open_case_resolution.py` (deterministic; `cd code && python3 build_open_case_resolution.py`)
resolves the 42 blank, code-less cases from TITLE TEXT ONLY: normalization v2
(quotes/fractions/spelling), synonym + class tables (COUPLER→SOCKET, REDUCER→Reducing
Socket/Elbow/Bush, AGRI / pipe-kg→Pressure Pipe CLASS-n/LW, SHOE→45-deg elbow, COLUMN
PIPE→submersible, ROTTO→BIRLA HIL Water Tank, S/F→Solfit, END CAP→"Plain end cap", …),
and an mm↔inch size canon (CPVC/CTS vs pressure-pipe conventions). Candidate gates =
type + size; class/length/material are scored and flagged; variant gaps are reported on
the dimensions the title leaves open. Strictness: exact-variant "found" bar (user
decision, 2026-10-19) — GT-lead column included, export-evidence close-out.

- Outputs: `open_42_resolution.{xlsx,csv}` (sheets: How to use | Resolve 42 |
  All candidates | Not-found evidence; the VERDICT column ships EMPTY for the user's
  manual pass) + `open_42_candidates_long.csv` + `open_cases_42.csv` (manifest).
  A copy of the workbook sits in the F: folder next to the PQ workbook.
- Read of the 42: **21 GT-LEAD / 4 EXACT candidate / 17 AMBIG-VARIANT** (one with an
  unrecognized type word: HALFSET, flagged with token evidence). GT leads all verify
  present in the 18-Oct export.
- QA: GT unit test 19/21 leads in top-3 — both misses are genuinely under-specified
  titles (75mm door tee: 7 candidates; "4\" TYPE-A" pipe: 24 variants); rebuilds are
  byte-identical (md5-verified); F: copy md5-identical to the repo build.

## Sentinel lane datasets (2026-10-23) — see `SENTINEL-lanes.md`

Three 100-row sentinel lane exports — brand (`brand-001…100`), title-only (`plain-001…100`), SKU
(`wsku-001…100`) — were received; the first two were cleaned, compared and corrected, the SKU lane
arrived user-verified and is used input-only for scoring (its 100 titles just feed the title
sheet; every one carries an 8-9 digit SKU code by design). A later brand audit of that lane found
19 token titles (BIRLANU 10 / BN 6 / BIRLA 3) and the token-only fix was applied too — 19 rows
rewritten, 81 untouched, code and mapped SKU kept. Full record, numbers, artifacts and repro commands live
in [`SENTINEL-lanes.md`](SENTINEL-lanes.md); the short version:

- Same 11-column schema, **not the same dataset** (brand vs title-only): 0 shared ids, 0 shared
  titles, but 20 shared mapped SKUs (~23%) and 32 shared invoices — two disjoint lanes cut from one
  invoice pool.
- `status`/`sku` in these files are the **expected** side, not results (brand 100/100 MAPPED
  against 10 recorded fails; title-only 78 MAPPED / 22 MAPPED_BRAND_NOT_FOUND against 10 fails).
  The `…Eval Related Product Mapping\sentiel run eval dataset for product mapping.csv` tally
  (300 items, 280 pass, seven empty detail columns) stays **NOT YET RECONCILED** with
  `../birlanu-eval-deck/code/numbers.json` (plain 94, MRP 95); its 20 failing ids now resolve to
  titles + expected SKUs, but no mapper output exists on disk to score.
- Brand lane: 20 of 100 titles carried an embedded SKU code — stripped and flagged; all 20 codes
  equalled the row's mapped SKU.
- Title-only lane: **0 titles carry a SKU code** (so nothing to strip) but **54 of 100 carried a
  Birla/BIRLANU/HIL token**, which the lane must not have. Those 54 were replaced from the same
  invoice universe with catalogue-backed titles (all 54 expected SKUs exist in the 2,335-SKU
  catalogue export), preserving the 78/22 status mix → `02_cleaned/title only sentinel tarun -
  corrected.csv` (54 rewritten, 46 byte-identical), then the two abbreviation leftovers
  (`plain-036 … - B`, `plain-066 BN …`) were removed token-only with
  `code/apply_titleonly_token_fix.py` — 2 more rows, SKU/status untouched. The lane is now **100/100
  brand-free and 100/100 code-free** (56 change rows in the workbook). The F: folder is
  organised `01_source/` (as delivered, incl. the GST-492 invoice and the run/trace JSONs) ·
  `02_cleaned/` (expanded/corrected CSVs + one-column title sheets, now three lanes incl.
  `sku_only`) · `03_review/` (comparison and replacement workbooks), with a `README.md` folder guide.
- Open: the 8 duplicate replacement SKUs and 2 same-HSN fallbacks are listed for a tighter pass.
  (`plain-036` / `plain-066` were resolved 2026-10-23 — abbreviation tokens removed.)

## Mapping issue list — classifier (2026-10-31)

A second, different input arrived from the BirlaNu team: `Mapping issue list shared by amit.xlsx`
(29-Oct-2026; three sheets — Distributor not mapped / retailer not mapped / SKU not mapped), not
an eval set. `code/classify_mapping_issues.py` buckets it against the 18-Oct catalogue
(2,335 rows) by component — family → type → size → qualifier — and writes a review workbook.
The written analysis is `../Mapping_Inconsistency_Analysis.md`; the tracker record is
`../BirlaNu_ActionTracker_and_Mapping_Analysis.md` §9.

Profile: 69 distributor rows / 41 names / 73 Inv / ₹29.6 L; 517 retailer rows / 363 names / 519 Inv
/ ₹2.58 Cr; 3,216 SKU rows / 2,913 descriptions / 4,775 Inv / ₹2.48 Cr.

SKU result: exact catalogue title 1,020 rows (31.7%) / ₹1.22 Cr (49.5%); 100 of those rows (3.1% /
17.0%) are exact titles blocked only by **19 of the catalogue's 49 duplicate-SKU pairs** (old 9600 vs new
9800/9800-10xx). Uniquely resolvable today (exact one-SKU + one-SKU-after-normalisation) = 1,239
rows (38.5%) / ₹94.9 L (38.4%). A further 1,122 rows (34.9%) / ₹72.2 L (29.2%) are catalogue
products with 2+ fitting SKUs because the invoice omits SDR/SCH/CLASS/length/Type A-B. Gaps/
unparsed 592 rows (18.4%) / ₹29.6 L; other brands 163 (5.1%) / ₹8.7 L. 554 rows (21.7% of value)
appear verbatim in the human mapping corpus, all with current SKUs.

Root-cause lead: the source invoice line has `status` but **no `sku_code` / `product_variant_id`**;
the SKU is on the turnover-invoice line. In trace `01a15db1` (Vaibhav GST-492, 18 lines) only the
two `MAPPED_BRAND_NOT_FOUND` lines appear in the issue list, the 15 `MAPPED` lines do not — the tab
is likely a status/report artifact, not a mapping gap. Confirm with one source-vs-turnover SQL
sample before any pipeline change.

```
python3 code/classify_mapping_issues.py \
  --workbook "F:/Product/zotok/Product Mapping/Mapping Inconsistency/01_source/Mapping issue list shared by amit.xlsx" \
  --catalogue "F:/Product/zotok/Product Mapping/Eval Related Product Mapping/products-18-October-2026-10-52-16.csv" \
  --out "F:/Product/zotok/Product Mapping/Mapping Inconsistency/03_review/Mapping issue analysis.xlsx"
```

## Mapping truth — Turn Over Report coverage baseline (2026-11-01)

The user designated `01_source/Turn Over Report.xlsx` (secondary invoice report, 2026-01-01 → 2026-11-01)
as the **source of truth for how many distributors, retailers and products are mapped**, priority
retailer/distributor first. `code/build_turnover_truth.py` reads the 16.8 MB xlsx once into a cached
CSV (`outputs/turnover-truth/turnover-truth-<date>.csv.gz`), then computes every number from the cache
(~35 s per rebuild). Plan: `../PLAN-turnover-report-truth-mapping.md`; readout
`../TRUTH-MAPPING-BASELINE.md`; workbook
`F:/…/Mapping Inconsistency/03_review/Truth - mapping coverage.xlsx` (11 sheets).

Headline (deduplicated population: 9,749 `Duplicate Invoice` lines dropped from 53,187 → 43,438 lines,
₹15.97 Cr):

| Axis | Rule | Lines mapped | Value | Entities fully mapped |
|---|---|---:|---:|---:|
| Retailer | `Retailer Status = Identified` | 38,699 (89.1%) | ₹14.23 Cr (89.1%) | 964 / 1,233 (78.2%) |
| Distributor | `Distributor Status = Identified` | 41,787 (96.2%) | ₹14.86 Cr (93.0%) | 271 / 332 (81.6%) |
| Product | `Parsing Status ∈ {Mapped, MBNF}` | 34,190 (78.7%) | ₹12.55 Cr (78.6%) | 9,350 / 14,148 titles |

- MBNF is **mapped** (it carries a `Material No`): Mapped 28,166 + MBNF 6,024 = 34,190; Unmapped 9,178;
  blank 70. Counting `status <> 'MAPPED'` as "not mapped" would report 15,202 lines as failures.
- **`Distributor Status` / `Retailer Status` are invoice-level**, stamped on every line of that invoice —
  they are not per-line-item decisions. 3,287 of 3,299 invoice+distributor groups carry one status; the 12
  exceptions are the same invoice present twice with the split exactly in half (5/5, 4/4, 12/12), a
  duplicate ingest where one copy got mapped. So a party's coverage is a ratio of *invoices*, and "partly
  mapped" = some invoices identified, some not. Two conventions in use: any-unidentified-invoice (the
  workbook headline, 269 retailers / 61 distributors "not fully mapped") and **strict, partial folded into
  mapped** (252 retailers / 45 distributors unmapped).
- Retailer failures are master coverage: 207 of 1,010 GST-bearing retailers are not fully matched,
  44 have neither GST nor phone. Distributor failures concentrate: 61 accounts, 10 making 80% of value.
- Unmapped product rows (5,165 desc+distributor): 42.1% are catalogue products (A/A2/B/C, 1,607
  ambiguous), 57.9% gaps/other brand/unparsed; 45.7% of unmapped lines carry a brand token.
- **Amit's list does not reconcile:** closest definition (MBNF / desc+distributor / created ≥ 2026-08-01)
  gives 2,948 rows / 4,580 Inv / ₹1.98 Cr vs his 3,216 / 4,775 / ₹2.48 Cr. His `Value` is not a line
  amount — confirm his export filter before using his tabs as a KPI.
- **Handover pack:** `--out-onboarding` also writes `Onboarding - retailers and distributors.xlsx` —
  every unmatched retailer and distributor with its identifiers, ordered by value, `In 80% cut` flagged.
  QA the rendered result with `code/qa_render.py` (Playwright; 0 clipped cells on the presentation
  sheets). LibreOffice is not installed, so the PDF render gate cannot run here.
- **Rank the onboarding cut on `gap_value`, never on the account's total.** Sorting by total value puts a
  big account with one bad line above a genuinely broken one and overstates the gap (it read ₹2.97 Cr
  against a true ₹1.11 Cr for distributors). `with_cum` now sorts and cumulates on the value sitting on
  unidentified lines only.
- **BirlaNu's own six GSTINs hold 61.6% of the distributor gap** (₹68.59 L of ₹1.11 Cr) — primary-channel
  records, flagged in the `Note` column, not onboarding targets. Real distributors: ₹42.85 L.

```
python3 code/build_turnover_truth.py \
  --report "F:/Product/zotok/Product Mapping/Mapping Inconsistency/01_source/Turn Over Report.xlsx" \
  --catalogue "F:/Product/zotok/Product Mapping/Eval Related Product Mapping/products-18-October-2026-10-52-16.csv" \
  --amit "F:/Product/zotok/Product Mapping/Mapping Inconsistency/01_source/Mapping issue list shared by amit.xlsx" \
  --out-workbook "F:/Product/zotok/Product Mapping/Mapping Inconsistency/03_review/Truth - mapping coverage.xlsx" \
  --out-md ../TRUTH-MAPPING-BASELINE.md --out-json outputs/turnover-truth/summary.json
```

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
- `outputs/catalogue_check/` — manual-verification workbook + cleaned CSVs + `fuzzy_match_offline_preview.{csv,xlsx}` + `below_threshold_candidates.{csv,xlsx}`
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
- `PLAN-resolve-42-open-cases.md` — approach for resolving the 42 open catalogue-check
  cases from title only (gates answered 2026-10-19; resolver built same day)
- `code/build_open_case_resolution.py` — the title-only resolver (synonym/class tables +
  mm↔inch canon + GT-lead cross-check; deterministic)
- `outputs/catalogue_check/open_42_resolution.{xlsx,csv}`, `open_42_candidates_long.csv`,
  `open_cases_42.csv` — resolver outputs (workbook copy in the F: folder; verdicts pending)
- `SENTINEL-lanes.md` — the sentinel lane record (cleaning, comparison, title-only correction,
  artifacts, repro commands, open items)
- `code/clean_sentinel_titles.py` — sentinel lane cleaner: expands `line_item` JSON into columns,
  strips an embedded SKU code from the title, flags the row → `<stem> - cleaned.{xlsx,csv}`
- `code/compare_sentinel_lanes.py` — brand vs title-only lane comparison + bucket check +
  fail-lead join → `Sentinel lanes - comparison.xlsx`
- `code/titles_only_sheets.py` — one-column-per-lane title sheets → `Sentinel input titles.xlsx`
- `code/build_titleonly_fix.py` — replacement plan for brand-named titles inside the title-only
  lane (catalogue-backed candidates only, status mix preserved, verdict column empty)
- `code/plan_sku_brand_fix.py` — audits the SKU lane for brand tokens (19 titles: BIRLANU 10, BN 6,
  BIRLA 3) and writes `03_review/SKU lane - brand-name fixes.xlsx` with a token-only fix per row
  (the code and the row's SKU stay put), each checked for: one code left == mapped SKU, token gone,
  unique in lane, SKU in the catalogue.
- `code/apply_sku_brand_fix.py` — applies that plan: rewrites only `line_item.product_name` (raw
  string swap, so key order and every other value stay byte-identical) → `02_cleaned/sku only
  tarun sentinel - corrected.{csv,xlsx}` (19 rewritten, 81 untouched) + `Changes` sheet + an
  `Applied (final set)` sheet in the review workbook.
- `code/apply_titleonly_token_fix.py` — removes the two brand abbreviations the token rule missed
  in the corrected title-only lane (`plain-036` trailing ` - B`, `plain-066` leading `BN `); guarded
  to exactly those ids, swaps only `product_name`, keeps SKU/status/invoice → lane 100/100 brand-free
  and code-free, 56 change rows in the corrected workbook.
- `code/apply_titleonly_fix.py` — applies the approved plan from the workbook: rewrites the 54
  rows (product fields from the replacement's own source record, ids/order kept) →
  `title only sentinel tarun - corrected.{csv,xlsx}` + `Applied (final set)` sheet
- `code/classify_mapping_issues.py` — buckets Amit's 3-sheet issue list (29-Oct-2026) against the
  18-Oct catalogue: SKU buckets A–I, duplicate-SKU pairs, distributor/retailer root-cause/action
  columns, trace evidence → `03_review/Mapping issue analysis.xlsx` in the F: folder
- `code/build_turnover_truth.py` — mapping-truth builder for the Turn Over Report (distributor /
  retailer / product coverage under the agreed definitions; caches the xlsx read to
  `outputs/turnover-truth/`; also emits the readout + `summary.json`; `--out-onboarding` adds the
  handover pack)
- `code/qa_render.py` — renders workbook sheets to HTML/PNG via Playwright and measures text clipping
  per cell (the no-Excel visual gate; exits non-zero on any clipped cell)
- `code/amit_overlap.py` — Amit's 29-Oct list vs the report at **account** level: collapses his invoice
  rows to parties (GST, else name) and answers how many report-unmapped accounts his sheet already names,
  and how many of his parties still hold up. Current: 269 unmapped retailers / 61 distributors in the
  report, of which he names 227 / 38; strict convention (partial = mapped) 252 / 45 unmapped, he names
  217 / 26, leaving 35 / 19 missing. His 385 retailers split 214 wholly unmapped / 8 partly / 124 already
  mapped / 39 absent from the report, his 46 distributors 22 / 10 / 6 / 8. Writes
  `outputs/turnover-truth/amit-overlap.json` (both conventions). Counts only — his `Value` is not
  comparable to report money.
- `SENTINEL-lanes.md` §10 / `../Mapping_Inconsistency_Analysis.md` — the issue-list record and the
  full solution/verification proposal (tracker §9 is the short version)

## Consumed by

- `../birlanu-eval-deck/` — the 4-slide BirlaNu client deck. Its three LangSmith lanes use
  these sets (SKU in title / brand in title / plain title, 100 each, hits of 100 shown on
  the slide); the MRP lane's figure comes from the eval report, not from these files.
  Sample titles quoted on the deck's "What was tested" slide are taken verbatim from
  `outputs/*.jsonl`.

## Related workstreams

- `../product-context/PLAN-practical-product-context.md` (2026-10-23) — proposal for a
  practical product-context doc derived from the catalogue. Its audit explains why a
  correct record can sit in `catalog_results` and still be rejected: Step 2a.3 of the
  production prompt requires the fitting token on both sides (so COUPLER cannot match
  SOCKET), Step 1b has no SOC/SHOE entries, and Step 3 resolves variant ties by list
  order. Same 300 cases used for validation.
