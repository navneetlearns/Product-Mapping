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

## Presence check — Amit sheet vs catalogue / workspace (2026-11-02)

Input: `F:/…/Mapping Inconsistency/01_source/Mapping issues - presence check in catalogue.xlsx` —
Amit's three tabs with the platform's own presence flags (`In Workspace` for distributor/retailer,
`In Catalog` + `Detail` for SKU, plus the workspace identity columns). `code/presence_check_summary.py`
buckets it and crosses it with the unique-party status; output `outputs/turnover-truth/presence-check.json`.

Products — 3,216 rows / 2,455 distinct descriptions:

| In Catalog | rows | descriptions | Detail |
|---|---:|---:|---|
| Yes | 2,611 | 1,903 | matched (different wording) 1,115 / same name 849 / product type present but invoice lacks detail 647 |
| No | 605 | 564 | not in catalog 383 / 355 · competitor brand 222 / 209 |

So the genuine BirlaNu catalogue gap is 383 rows / 355 descriptions; the 222 competitor-brand rows are
not missing catalogue. Of the 2,611 that exist, 647 are ambiguous variant (the product type is in the
catalogue, the invoice does not state SDR/class/length) and 1,115 need wording normalisation.

Parties, crossed with the unique-party status (`annotate_amit_status.py`):

| Side | unique | not mapped | present but unmapped | absent → onboarding |
|---|---:|---:|---:|---:|
| Distributors | 40 | 19 | 3 | 16 |
| Retailers | 358 | 196 | 51 | 144 |

The 3 distributors are GST-matched (Sri Lakshmi Electrical & Hardware, Jeevish Sales, Nepolite
Concrete). The 51 retailers are name-matched (50) or phone-matched (1); those workspace records carry
no GST and one is a clearly wrong name match (Tatsam Enterprises → Manish Choudhary), so 51 is a
candidate count. The presence check keys on the GST/name in his sheet, so it has false negatives:
6 distributors and 12 retailers that the report has fully mapped read `No` because his GST has typos
(Kamal Sanitary, Future Trade Com, Jagdamba, Maa Sherawalis, S.K. Enterprises, Satyam Home Decore;
12 retailers).

## Platform status note checked (2026-11-02)

The platform side's summary — 69 distributors / 518 retailers / 3,217 products, with 9 / 98 / 21
marked unmapped — was checked with `code/verify_platform_note.py`. Presence rows tie exactly
(36 / 295 / 2,611), but they are rows, not parties (16+1 / 196+2 / 1,903 descriptions). The record
counts 518 and 3,217 are the tabs' `max_row` including the header; the data rows are 517 and 3,216,
and the product tab holds 2,359 unique descriptions. The report has only Identified / Not Identified,
so 9 and 98 are not party counts — strict join gives 32/69 and 280/517 rows, 11/36 and 89/295 present
rows, and 3 / 51 present unique names. The product 21 is close to the 22 descriptions present in the
master with every line Unmapped, but ignores the 605 rows / 564 descriptions absent from the master
(383 genuine gap + 222 competitor brand) and 2,240 MBNF descriptions.

## 4-Nov report refresh and the 51-retailer matching-fix queue (2026-11-04)

The Turn Over Report at `01_source/Turn Over Report.xlsx` was re-exported on 4-Nov (filters 2026-01-01 →
2026-11-04). Line grain: 54,903 lines (was 53,187), 10,005 duplicate-flagged → 44,898 deduplicated;
retailer 40,092 identified (89.3%, was 38,699/89.1%), distributor 43,787 (97.5%, was 41,787/96.2%),
product 28,834 Mapped / 6,337 MBNF / 9,657 Unmapped / 70 NA. Retailer entities recomputed with the §10
rule: 1,271 = 992 fully / 28 partly / 251 not identified (was 1,233 = 964/17/252). Distributor and
product entity counts were NOT recomputed.

Amit's retailer tab at party grain, 4-Nov: 358 parties → 194 not mapped / 17 partly / 145 fully /
2 no party match; 198 present in the master (295 is the ROW count) and 160 absent. Cross: 51 present &
unmapped, 13 present & partly, 133 present & fully, 1 present & unmatched. Builder
`code/build_master_match_queue.py` turns the present & unmapped cell into
`03_review/Retailers present but unmapped - 51.xlsx` + `.csv` — his GST/phone, the workspace record the
presence check returned, the match basis (49 exact name / 2 phone / 0 GST) and `GST relation`
(6 near-typo / 11 different GST / 34 no GST on the workspace record). Tracker §14 has the full table.

The 4-Nov file also changed the export preamble: it adds a leading `Column1…` row, moving the header
from row 3 to row 4. `load_report` used a fixed `header=2`, which read the filter line as the header and
silently cached a garbage frame for every downstream join; it now locates the header row (first cell ==
`Invoice Number`). Re-tested on the 4-Nov file: 54,903 × 57.

## Amit's unique products vs the report (2026-11-05)

The `SKU not mapped` tab (3,216 rows) dedupes to **2,359 unique products** under `cmi.norm_text`
(2,913 exact strings, 2,737 case/space). Joined to the 4-Nov report at description grain, product
status is — export as delivered: **1,903 fully mapped / 400 partly mapped / 56 never mapped**; on the
deduplicated population: 1,945 / 350 / 64. "Never mapped" = no mapped-tier line anywhere; 456 products
still carry at least one Unmapped line. No product on his list is absent from the report (32 appear
only on duplicate-invoice rows). Builder `code/amit_products_vs_report.py`, per-product CSV
`outputs/turnover-truth/amit-products-vs-report-2026-11-04.csv`; tracker §15.

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
- `code/annotate_amit_status.py` — unique party view of his issue list: collapses his invoice rows by
  normalised name (69 distributor rows → 40 shops, 517 retailer rows → 358), keeps every GST in one
  cell, and gives each party its Turn Over Report status, report party, report GST and match key.
  Match inside the invoice is GST, then GST-near + name, then phone, then name/fuzzy; non-exact
  matches are flagged. Adds `Distributor master check` (the 45 report-unmapped distributors with the
  user's 02-Nov manual verdict: 34 not in master, 1 in master but unmapped — Jeevish, 3 BirlaNu
  GSTINs, 7 no-GST) → `03_review/Mapping issue list shared by amit - unique parties.xlsx`.
- `code/presence_check_summary.py` — buckets the platform's presence-check workbook (products by
  `In Catalog`/`Detail`, parties by `In Workspace`/`Matched By`) and crosses it with the unique-party
  status → `outputs/turnover-truth/presence-check.json`; numbers in the Presence check section above.
- `code/verify_platform_note.py` — checks the platform team's status note line by line: rows vs
  unique names, presence rows vs names, strict row / present-row / party / invoice cuts, and product
  descriptions wholly vs partly Unmapped; numbers in the Platform status note section above.
- `code/build_master_match_queue.py` — the matching-fix queue behind the retailer tab: the parties that
  are `Not mapped` on every invoice in his sheet AND `In Workspace = Yes` in the platform presence
  check (51 of 358 at the 4-Nov export). One row per party with his GST/phone, the workspace record the
  presence check returned, how it matched (49 exact name / 2 phone; none GST) and a `GST relation`
  column (same / near ≤2 chars / different / no GST in workspace — 6 / 11 / 34 here) so dev can tell a
  typo fix from a possible wrong name match → `03_review/Retailers present but unmapped - 51.xlsx` +
  `.csv`. Reuses `annotate_amit_status`'s party logic, so the 51 equal that workbook's present/not-mapped
  cell.
- `code/amit_products_vs_report.py` — product-grain comparison of Amit's SKU tab (3,216 rows → 2,359
  unique) against a report snapshot: per-product fully / partly / never mapped from `Parsing Status`,
  reported on both the deduplicated population and all rows (duplicate-invoice rows flip a handful of
  products). Writes `outputs/turnover-truth/amit-products-vs-report-<snapshot>.csv`; tracker §15.
- `code/build_never_mapped_products.py` — the never-mapped fix list for the dev team: one row per
  product plus one row per unmapped report line with distributor, retailer, invoice number/date and
  value. Writes the workbook + CSV to the F: `03_review/` folder (party identifiers — not committed
  to this public repo); tracker §15.
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
