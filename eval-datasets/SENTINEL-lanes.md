# Sentinel lane datasets — cleaning, comparison, title-only fix (2026-10-23)

Two 100-row sentinel lane exports arrived as CSVs in
`F:\Product\zotok\Product Mapping\Mapping Inconsistency\`:

| file | lane (`case`) | ids | source_dataset |
|---|---|---|---|
| `Tarun Sentinel Dataset.csv` | `brand_named` | `brand-001…100` | `turnover-product-mapping-eval-brand-named-in-catalog` |
| `title only sentinel tarun.csv` | `title_only` | `plain-001…100` | `turnover-product-mapping-eval-title-only-in-catalog` |

Both use the same 11-column schema:
`distributor_exclusive, line_item, equivalent_skus, sku, status, case, dataset_split, sentinel_eval,
sentinel_item_id, source_dataset, source_invoice`, where `line_item` is a JSON string
`{serial_no, hsn_sac_code, product_name, unit_rate_excl_gst}`.
This doc records what was checked, what was found, what was corrected, and where every artifact
lives. The scripts referenced here are in `code/`; the data artifacts are in the F: folder above.

---

## 1. The run results file these lanes belong to — NOT YET RECONCILED

`F:\…\Eval Related Product Mapping\sentiel run eval dataset for product mapping.csv`
(filename spelling as delivered). 300 rows, shape
`item,item_result,kind,typed,sku,expected_qty,got_qty,ok,reason`:

- `item` = `wsku-001…100`, `plain-001…100`, `brand-001…100` — 100 per lane, no MRP lane.
- `item_result` = pass on 280, fail on 20: wsku **100/100**, plain **90/100**, brand **90/100**.
- The seven detail columns are empty in all 300 rows, so no per-item reason is recoverable.
- Failing items: plain 014, 035, 036, 037, 043, 051, 052, 077, 084, 098; brand 027, 043, 053,
  056, 062, 063, 064, 076, 097, 100.

Why it is not reconciled: it disagrees with the recorded run — `../birlanu-eval-deck/code/numbers.json`
carries the plain lane at 94/100 and an MRP lane at 95/100 that this file does not contain — and
the `wsku-001`-style ids exist nowhere on disk. The lane CSVs above now supply the input side for
`brand-*` and `plain-*` (all 20 failing ids resolve to a title + expected SKU), but no mapper
output is present in any of these files, so pass/fail still cannot be recomputed.

## 2. Cleaning pass (`code/clean_sentinel_titles.py`)

Expands `line_item` into columns and cleans the title with the same `clean()` the eval builds use
(mojibake repair, zero-width/C1 strip, NBSP/whitespace collapse), then removes an embedded SKU code
from the title — parenthesised `… Birlanu (96000045)` or bare leading `96000754 …` — and flags the
row. Writes `<stem> - cleaned.{xlsx,csv}` next to the source. Sheets: `Input products` (expanded,
flagged rows shaded, autofilter) | `Flagged titles` (omitted when nothing is flagged) |
`Same SKU, many titles` | `Summary` | `Raw line_item`.

Brand lane (`Tarun Sentinel Dataset.csv`) — 100 rows, 56 invoices, 88 distinct SKUs:

- **20 titles carry a SKU code** (16 parenthesised, 4 bare-leading) and all 20 codes equal the
  row's mapped SKU — 0 conflicts. Nothing else needed cleaning (no mojibake, no NBSP, no doubled
  spaces; the five U+2019 characters in three titles are left as delivered — `’` vs `’’` is the
  feet/inch distinction).
- 10 SKUs are reached from more than one title (22 rows), e.g. `96000014` ×3, `98000874` ×3.
- HSN: 77 eight-digit, 22 family-only `3917`, 1 missing (`brand-074`). 18 rows
  `distributor_exclusive=true`, 13 rows carry `equivalent_skus`.

Title-only lane, as delivered — 100 rows, 62 invoices, 86 distinct SKUs:

- **0 titles carry a SKU code** (the cleaning step was a no-op beyond whitespace).
- 12 SKUs reached from >1 title (26 rows), HSN 79 eight-digit / 18 `3917` / 3 missing
  (`plain-004`, `plain-010`, `plain-075`), 21 exclusive, 13 with equivalents, 2 rows with
  `serial_no` null (`plain-019`, `plain-073`).
- **54 of 100 titles carry a Birla/BIRLANU/HIL token** — the defect corrected in §5.

Status in both files is the **expected** side, not a result: brand reads 100/100 `MAPPED` against
10 recorded fails, title-only carries 78 `MAPPED` + 22 `MAPPED_BRAND_NOT_FOUND` against 10 fails.
Never read pass/fail off these files.

## 3. Are the two files the same dataset? No — two disjoint lanes of one run

`code/compare_sentinel_lanes.py --brand … --title-only … --results …` →
`Sentinel lanes - comparison.xlsx` (`Lanes at a glance` / `Overlap` / `Bucket check` / `Fail leads`).

- Identical schema, same `sentinel_eval`, same `dataset_split=["base"]`; different `case`,
  `source_dataset`, id namespace.
- **0 shared sentinel ids, 0 shared product titles** (raw or cleaned).
- Not independent either: **20 of 88/86 mapped SKUs are shared (~23%)**, and **32 invoices** are
  shared (56 vs 62 distinct) — both lanes were sampled from the same invoice pool.
- **Lane labels do not match the recorded bucket rule** (`build_datasets.py`: 8-digit code → sku
  bucket, `birla|hil` → brand bucket, else title_only). Under that rule the brand lane splits
  20 code / 80 brand and the title-only lane 0 code / **54 brand** / 46 title_only. Either the
  sentinel set was cut with a different definition ("title-only" = no embedded code, brand words
  allowed) or 54 rows were mislabelled. User resolved this: the lane must carry no brand name.
- `Fail leads` joins the 20 failing ids to titles + expected SKUs from the two lane files (all 20
  resolved); 5 of the 10 plain fails (014, 035, 036, 037, 084) are `MAPPED_BRAND_NOT_FOUND` by
  expected status.

## 4. Titles-only workbook (`code/titles_only_sheets.py`)

`Sentinel input titles.xlsx` — one sheet per lane, one column (`product_title`, 100 rows each),
nothing else: no ids, SKUs, HSN, rates, invoices or flags. Source apostrophes are preserved on
purpose (`’` vs `’’` = feet vs inches; folding them to ASCII would lose a distinction the mapping
parser depends on). Rebuilt after the correction so `title_only` carries the corrected titles.

## 5. Correcting the title-only lane (`code/build_titleonly_fix.py`)

**54 of 100 title-only titles carry a Birla/BIRLANU/HIL token** (BIRLA 40, BIRLANU 13, HIL 1; 20
lead with it). Two more are suspected abbreviations, flagged `confirm` and left alone:
`plain-036 … Pushfit S/S - B`, `plain-066 BN UPVC PIPES SCH 40 3MTR 1"`.

Second lane rule checked on the same sheet: **no SKU/code in a title-only title — 0 of 100** (no
8-9 digit run, in fact no 4+ digit run, and no catalogue-style code; the only alphanumeric hit is
`160X110 MM`, a size). `codey()` implements this: a 4+ digit run counts as a code only when no unit
follows it (MM/MTR/LTR/KG/ML/GM/HP…), plus dotted codes like `P.CPS-154`; `SDR11`, `3MTR`,
`1000 Ltr` are specs.

Replacement plan — `Title only - brand-name replacements.xlsx`:

- Universe = the same invoice exports (1,076 distinct titles, `F:\Langsmith\BRAND NAME\`):
  405 carry no code and no Birla/HIL token → −40 already used by a lane → −50 another brand
  (PRINCE, ASTRAL, VECTUS, WATERTEC, BONDTITE, KAJU, Dr. Fixit…) or promo item (pens, key rings,
  merchandize kit) → **315 usable, 12 code-bearing (flag-red, never auto-selected), 303 selectable,
  117 with expected status MAPPED**.
- Rules: same expected status (all 54 rows are MAPPED, so the 78/22 mix is preserved) → same
  product family → same HSN family → sha1 tie-break, each title used once. Result: 52 same-family,
  2 same-HSN fallbacks (only 3 MAPPED UNION titles exist for 5 UNION rows: `plain-080`,
  `plain-092` — hand-pick if tighter is wanted).
- **All 54 proposed expected SKUs exist in the BirlaNu catalogue export**
  (`products-18-October-2026-10-52-16.csv`, 2,335 SKUs): 54 in, 0 missing — the user's rule, since
  only catalogue products are usable. The sheet prints the catalogue product name per pick
  (mean token similarity to the invoice title 0.91); 8 of the 54 proposed SKUs duplicate a SKU
  already in the lane (finite pool); none reuses its own row's SKU.
- `spec only in catalogue name` column: 11 of the 54 titles are silent on a spec the catalogue
  product carries (class 4kg vs 6kg, `LONG HANDLE`, `Solfit`, `TRD`, `ISI`) — legal eval cases,
  but the title can't justify the variant, so they are the first to swap for a tighter set.
- The candidate pool also holds 96 titles with no expected SKU (UNMAPPED cases) — not catalogue
  products; marked `NO`, greyed, never auto-selected. Catalogue-backed pool = 219 (117 MAPPED).
- `approved? (Y/N)` column ships empty — the user's pass. It was approved as proposed.

## 6. Applied (`code/apply_titleonly_fix.py`)

Reads the approved plan out of the workbook's `Replacements (proposed)` sheet, so what lands is
exactly what was signed off. Row identity is kept (`plain-XXX` slot, same order, same 11-column
schema); every product field comes from the replacement's own source record.

Outputs:

- `title only sentinel tarun - corrected.csv` — 54 rows rewritten, 46 byte-identical. Fields that
  changed: `line_item` 54, `sku` 54, `source_invoice` 54, `distributor_exclusive` 43,
  `equivalent_skus` 10. `status` unchanged on all 100 (all 54 old and new rows were MAPPED).
- `title only sentinel tarun - corrected.xlsx` — `Corrected set` (all 100, replaced rows shaded) +
  `Changes` (old/new title, status, SKU, invoice per row).
- `Title only - brand-name replacements.xlsx` gained `Applied (final set)` (100 rows, `changed?`).
- `Sentinel input titles.xlsx` rebuilt with the corrected titles.

Verified after write: 100 rows, 100 distinct titles, ids/order/schema unchanged, **0 brand tokens,
0 codes (not even a 4-digit run), every SKU present in the catalogue**, no empty titles or SKUs,
status mix still 78 MAPPED / 22 MAPPED_BRAND_NOT_FOUND.

`source_invoice` is the one field whose *namespace* changes: original rows carried the sentinel
run's trace ids (`01a0f7…`), the 54 replacements carry the source export's row ids (`e22533b0…`).
Nothing else identifies those invoices on disk, so this is the honest provenance — if a downstream
tool opens the invoice from that field, the 54 new rows point at the export record.

## 7. Artifacts

F: `…\Mapping Inconsistency\`

| file | what it is |
|---|---|
| `Tarun Sentinel Dataset.csv` | brand lane, as delivered |
| `Tarun Sentinel Dataset - cleaned.xlsx` / `.csv` | brand lane, line_item expanded + 20 codes stripped/flagged |
| `title only sentinel tarun.csv` | title-only lane, as delivered |
| `title only sentinel tarun - cleaned.xlsx` / `.csv` | title-only lane, expanded |
| `title only sentinel tarun - corrected.csv` / `.xlsx` | **the fixed lane** (54 titles replaced) + `Changes` |
| `Sentinel lanes - comparison.xlsx` | lane comparison + bucket check + 20 fail leads |
| `Title only - brand-name replacements.xlsx` | replacement plan, lane check, candidate pool, applied set |
| `Sentinel input titles.xlsx` | one-column title sheets for both lanes (title_only = corrected) |

Repo `eval-datasets/code/`: `clean_sentinel_titles.py`, `compare_sentinel_lanes.py`,
`titles_only_sheets.py`, `build_titleonly_fix.py`, `apply_titleonly_fix.py`.

## 8. Reproduction

```
python3 code/clean_sentinel_titles.py  "<F:>/Tarun Sentinel Dataset.csv"
python3 code/clean_sentinel_titles.py  "<F:>/title only sentinel tarun.csv"
python3 code/compare_sentinel_lanes.py --brand "<F:>/Tarun Sentinel Dataset.csv" \
    --title-only "<F:>/title only sentinel tarun.csv" \
    --results "<F:>/Eval Related Product Mapping/sentiel run eval dataset for product mapping.csv" \
    --out "<F:>/Sentinel lanes - comparison.xlsx"
python3 code/titles_only_sheets.py --out "<F:>/Sentinel input titles.xlsx" \
    "brand_named=<F:>/Tarun Sentinel Dataset.csv" \
    "title_only=<F:>/title only sentinel tarun - corrected.csv"
python3 code/build_titleonly_fix.py --title-only "<F:>/title only sentinel tarun.csv" \
    --brand "<F:>/Tarun Sentinel Dataset.csv" \
    --export "/mnt/f/Langsmith/BRAND NAME/turnover-product-mapping-136d0d1e.csv" \
    --export "/mnt/f/Langsmith/BRAND NAME/producttitle only.csv" \
    --catalogue "<F:>/Eval Related Product Mapping/products-18-October-2026-10-52-16.csv" \
    --out "<F:>/Title only - brand-name replacements.xlsx"
python3 code/apply_titleonly_fix.py --title-only "<F:>/title only sentinel tarun.csv" \
    --plan "<F:>/Title only - brand-name replacements.xlsx" \
    --export "/mnt/f/Langsmith/BRAND NAME/turnover-product-mapping-136d0d1e.csv" \
    --export "/mnt/f/Langsmith/BRAND NAME/producttitle only.csv"
```

Selection is deterministic (sha1 tie-break), so a rebuild reproduces the same 54 picks. The join
trap worth remembering: exports can hold several line items per row, so `reference_outputs.expected`
must be matched **by product_name**, never by position — a positional join handed one SKU to
unrelated titles (caught by the catalogue-family check).

## 9. Open items

- `plain-036` (`… Pushfit S/S - B`) and `plain-066` (`BN UPVC PIPES …`) — brand abbreviations the
  token rule does not catch; both `MAPPED_BRAND_NOT_FOUND`, left as-is pending the user's call.
- The 8 duplicate replacement SKUs and the 2 same-HSN fallbacks (`plain-080`, `plain-092`) are
  acceptable but listed for a tighter pass.
- The 20 recorded fails can now be named (title + expected SKU) but not scored: no mapper output
  exists on disk for these lanes.
