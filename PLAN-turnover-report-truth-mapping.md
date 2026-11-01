# Plan — Turn Over Report as the mapping truth baseline

**Status: EXECUTED 2026-11-01.** Gates D1–D8 were answered by the user's go-ahead on the recommendations below; the build, readout and doc trail exist (see §8). User priority, stated at approval: **retailer/distributor mapping first, products second** — the workbook and readout are ordered that way.
Date: 2026-11-01
Source of truth: `F:\Product\zotok\Product Mapping\Mapping Inconsistency\01_source\Turn Over Report.xlsx`
(downloaded 2026-11-01 15:00; sheet `Secondary Invoice Report`; filters applied 2026-01-01 → 2026-11-01)
Question it answers: **how many distributors, retailers and products are actually mapped.**

Supersedes the Oct-31 open item "what does the SKU-not-mapped tab query?" — this file carries the answer
on every line (`Parsing Status` + `Material No`), so the report-semantics hypothesis can be tested directly
instead of from a trace sample.

---

## 0. Decision gates — answer before any counting

Nothing below §0 gets built until these are answered. Each has a recommendation; none is adopted silently.

| # | Gate | Options | Recommendation | Resolved 2026-11-01 |
|---|---|---|---|---|
| D1 | Population: keep rows flagged `Duplicate Invoice` in `Validation Status`? | (a) all 53,187 lines as delivered · (b) exclude duplicate-flagged (8,699 + combo rows) · (c) both | **(c)** — headline on deduplicated, all-lines number beside it as sensitivity | **RESOLVED — (c)**, headline deduplicated (43,438 lines), all-lines beside it |
| D2 | Is `Mapped - Branding Not Found` counted as **mapped**? | (a) yes, split out · (b) no, count as a failure | **(a)** — it carries a `Material No`; the split stays visible because this is the crux of Amit's list | **RESOLVED — (a)**, MBNF mapped and split (6,024 lines) |
| D3 | KPI unit | line · invoice · distinct entity · ₹ value | **all four**, headline = line share **and** value share | **RESOLVED — all four** |
| D4 | Entity identity | GST-first · SAP code (distributor) · phone (retailer) · matched name | **GST first**, then SAP/phone, name is display only | **RESOLVED — GST → SAP/phone → name+pin** |
| D5 | Scope window | invoice date 2026-01-01→11-01 as delivered · only ingested data (`Created At` ≥ 2026-07-07) | **as delivered**, ingested-only as a sensitivity line | **RESOLVED — as delivered**; sensitivity uses created ≥ 2026-09-01 (all rows were created after 7-Jul, so that window was identical to the whole file) |
| D6 | Unmapped-product depth | counts only · counts + catalogue-join buckets + candidates | **catalogue join** (reuse the existing A–I classifier) | **RESOLVED — catalogue join**, buckets A–I + candidates |
| D7 | Reconcile against Amit's 3-sheet list in this deliverable? | yes · separate note | **yes**, one sheet — he shared it and the numbers will be compared anyway | **RESOLVED — in scope**, sheet `Reconciliation vs Amit` |
| D8 | Delivery form | workbook + md readout in `03_review` · client deck | **workbook + readout**; no deck (a deck is a separate, separately-approved job) | **RESOLVED — workbook + readout**, no deck |

---

## 1. What the file is — verified profile

Measured on this download, before any dedupe or definition decision. These are the raw denominators;
they are **not** yet the report numbers.

```
shape            53,187 line rows · 57 columns · header on row 3 · one sheet
invoices         3,135 distinct invoice numbers · 4,206 distinct (invoice, sequence) pairs
                 → invoice numbers are NOT unique keys (multi-page pages share a number,
                   and numbers repeat across distributors, e.g. "89", "112")
value            ₹19,79,51,674 total line amount excl. tax (≈ ₹19.80 Cr)
dates            invoice date 2026-01 → 2026-10 (164 blank, a stray tail back to 2006)
                 created at 2026-07-07 → 2026-11-01 (platform ingestion window)
distributors     408 distinct (name, GST) pairs · 74 have ≥1 not-identified row
retailers        1,395 distinct (name, GST) pairs · 325 have ≥1 not-identified row
material codes   1,336 distinct `Material No`; 42,422 of 42,492 coded lines carry a code that exists
                 in the 18-Oct catalogue export — the only code outside it is `-` (70 lines)
```

Status distributions (line counts, and the value for the product axis):

```
Distributor Status   51,465 Identified (96.8%)    1,722 Not Identified (3.2%)
Retailer Status      47,444 Identified (89.2%)    5,743 Not Identified (10.8%)
  both identified    46,153 lines (86.8%)

Parsing Status       35,272 Mapped (66.3%)          ₹12.92 Cr
                      7,026 Mapped - Branding N/F (13.2%)  ₹2.81 Cr
                     10,819 Unmapped (20.3%)        ₹4.07 Cr
                         70 NA (0.1%)               ₹0.00 Cr
  mapped incl. MBNF  42,298 lines (79.5%)
```

Traps already visible in the data — each becomes an explicit rule in §2 rather than a silent filter:

- **124 lines are `Unmapped` but carry a `Material No`** (e.g. `98000139 UPVC PIPE CLAMP (Metal) 1 1/4"`).
  Status and artifact disagree; a "has SKU" KPI and a "status" KPI differ by these.
- **70 lines have `Parsing Status = NA`**, blank `Material Description in Secondary Invoice` and
  `Material No = -` — parse failures, not products.
- **8,699 lines are flagged `Duplicate Invoice`** (plus combination flags) — 16.4% of the file.
- **Retailer `Not Identified` while carrying a valid GST** (e.g. `09ALPPA1559E1Z5`) — the GST exists but
  the master has no match. That is master coverage, not identifier capture, and the retailer sheet will
  separate the two.
- **`Not Identified` distributors carry no SAP code** (`NA`) and `Matched Distributor Name = NA`;
  identity fields and status are consistent, so the two are cross-checkable.
- Top unmapped descriptions are dominated by **brand-token titles**: `MERCHANDIZE KIT` 69,
  `BIRLA UPVC RED.TEE 1*1/2"` 42, `3" PIPE SWR BIRLANU` 37, `25MM UPVC PIPE 'BIRLA'` 36,
  `CK BIRLA SWR 110mm Bend` 34 — the same vocabulary gap the eval lanes studied.

### 1.1 What this does to the Oct-31 hypothesis

Amit's "SKU not mapped" tab is 3,216 rows / ₹2.48 Cr. This file holds 10,819 truly-unmapped lines
(₹4.07 Cr, 5,586 distinct description+distributor pairs) and 7,026 MBNF lines (₹2.81 Cr, 3,891 pairs).
`Unmapped + MBNF` aggregated by description+distributor = 9,122 rows, ₹6.88 Cr. **No single obvious
definition lands on 3,216 / ₹2.48 Cr / 4,775 Inv**, so his filter is narrower than "status <> MAPPED" —
finding it is Phase P5, not an assumption.

---

## 2. Definitions to fix before counting

The counting is trivial once these are written down; the disputes are all in the definitions.

**Mapping dictionary (per line).**
`distributor_mapped` = `Distributor Status = Identified` (equivalently `Matched Distributor Name` populated
and `Distributor Code (SAP)` not `NA`). `retailer_mapped` = `Retailer Status = Identified`.
`product_mapped` = `Parsing Status ∈ {Mapped, Mapped - Branding Not Found}` **and** `Material No` is a real
code; the residual 124 status/artifact mismatches are reported as their own bucket, never folded in.

**Population ledger.** Four populations, all reported so no denominator is hidden:
`all` (53,187) · `deduped` (drop lines whose `Validation Status` contains `Duplicate Invoice`) ·
`identified-pair` (distributor and retailer both identified) · `ingested` (`Created At ≥ 2026-07-07`).

**Denominators per subject.** Distributor = distinct entity (GST-normalised, name as display).
Retailer = distinct entity (GST, else phone, else name+pin). Product = distinct
`Material Description in Secondary Invoice` (17,085) for title-level coverage, and distinct `Material No`
(1,336) for SKU-level coverage. Invoice = distinct (invoice number, distributor, invoice date).

**Roll-ups that get reported, not chosen between.** line %, invoice %, distinct-entity %, ₹ %.
A product counts "mapped" at title level only under a stated rule: all lines mapped · any line mapped ·
majority mapped — the plan reports all three and headlines **all lines mapped** (strict).

---

## 3. Deliverables

1. `eval-datasets/code/build_turnover_truth.py` — deterministic builder: reads the xlsx once, caches to
   `eval-datasets/outputs/turnover-truth/<download-date>.parquet` (+ `.csv` fallback), then computes every
   number from the cache. Full xlsx read costs ~80 s, so the cache is not optional.
2. `F:\...\Mapping Inconsistency\03_review\Truth - mapping coverage.xlsx`, sheets:
   `Read me` (definitions + the answered gates) · `Headline` (3 subjects × 4 units) ·
   `Distributor coverage` (408 rows) · `Retailer coverage` (1,395) · `Product coverage` (17,085 titles) ·
   `Unmapped titles A–I` (5,586 desc+dist rows, buckets + candidate SKUs) ·
   `Anomalies` (124 + 70 + duplicate-flagged) · `Reconciliation vs Amit` · `Count reconciliation`
   (every headline number traced to its filter and formula).
3. `TRUTH-MAPPING-BASELINE.md` — two-page readout: what is mapped, what is not, by subject and value, with
   the definitions used and the population ledger.
4. Doc trail in the same pass: `BirlaNu_ActionTracker_and_Mapping_Analysis.md` §10, `eval-datasets/README.md`,
   `SENTINEL-lanes.md` §11, the F: folder `README.md`, `~/AgentWork/PROJECTS.md`.

No client deck, no push, unless asked separately.

---

## 4. Phases

**P0 — ingest and cache.** Read the xlsx with `openpyxl` `read_only`, normalise `NA`/blank, write the parquet
cache and a row-count manifest. Gate: cache row count == 53,187 and column set == the 57 headers.

**P1 — population ledger.** Apply the duplicate / NA / anomaly rules; print the four populations with
line, invoice, entity and ₹ counts. Gate: the four populations are nested and each reconciles to the next
by exactly the excluded rows.

**P2 — three subject counts.** Under the D1–D5 answers, produce the headline table: distributor, retailer,
product, each as line / invoice / entity / ₹. Include the MBNF split and the status-vs-artifact residual.

**P3 — entity lists.** The 74 distributors and 325 retailers carrying unmapped lines, with rows, ₹, dominant
`Validation Status` reason and identifier coverage (GST / phone / SAP code presence). Top-N that make 80% of
the unmapped value = the onboarding list.

**P4 — product-level analysis.** For the unmapped titles: join to the 18-Oct catalogue and bucket with the
existing component classifier (`import classify_mapping_issues` — `norm_text`, `family`, `ptype`,
`parse_sizes`, `parse_variants`, `Classifier`), so the buckets match the Oct-31 analysis and are comparable
line-for-line. Output: how much of the unmapped population is exact-title / duplicate-SKU / ambiguous
variant / true gap / other brand, weighted by the truth file's own value instead of Amit's sheet's.

**P5 — reconcile Amit's list.** Search the definition space {status filter (Unmapped · MBNF · both) × date
window (month ranges) × aggregation key (desc+dist · desc · +month) × dedupe on/off} and score each
candidate against his three numbers (3,216 rows · 4,775 Inv · ₹2.48 Cr). Report the closest definition and
its residual; if none reproduces him within ±1%, say so and show the three nearest.

**P6 — write-up.** Readout md + doc trail updates + commit locally.

Commands (from `~/AgentWork/product-mapping`):

```
python3 eval-datasets/code/build_turnover_truth.py \
  --report "F:/Product/zotok/Product Mapping/Mapping Inconsistency/01_source/Turn Over Report.xlsx" \
  --catalogue "F:/Product/zotok/Product Mapping/Eval Related Product Mapping/products-18-October-2026-10-52-16.csv" \
  --out-workbook "F:/Product/zotok/Product Mapping/Mapping Inconsistency/03_review/Truth - mapping coverage.xlsx" \
  --out-json   eval-datasets/outputs/turnover-truth/summary.json
```

---

## 5. Verification checklist (the deliverable is not done until all pass)

1. Per-axis status counts each sum to the population size (53,187 all / N deduped) — asserted in code, printed.
2. The four populations are nested and the deltas equal the excluded-row counts.
3. Entity counts hold under GST normalisation (strip spaces, upper, dedupe) and every `Identified` entity has
   a non-`NA` matched name; the converse checked and reported if violated.
4. Every `Material No` is in the 18-Oct catalogue export except `-`; count printed, not assumed.
5. Independent recount: one headline cell reproduced from a filtered copy in Excel/LibreOffice (pivot on
   `Parsing Status`) and quoted beside the script's number.
6. Reconciliation sheet reproduces or explains Amit's 3,216 / ₹2.48 Cr / 4,775 Inv.
7. Workbook QA without hand-opening: LibreOffice headless → PDF → `pdftotext -layout`; grep for `###`
   (narrow numeric columns) and assert each headline number and every sheet name appears.
8. Rebuild is deterministic: run twice, `md5sum` the JSON summary — identical.

---

## 6. Risks and caveats to state in the readout

- `Invoice Status = PENDING` covers 95% of lines; it is a workflow state, not a mapping state. Reporting
  "pending" as unmapped would be wrong, and the readout should say so before anyone else does.
- The report is a **snapshot at download time**; parsing status can change on reprocess. Every number is
  dated to this download — file mtime 2026-11-01 15:00.
- Live-only `412xxx` SKUs are absent from the 18-Oct catalogue export; "not in catalogue" means "not in this
  export" (same caveat as the Oct-31 analysis).
- Value is `Total line Amount (Excluding tax)`; using the incl-tax column moves every ₹ share. State which.
- A "mapped" product can still be the wrong SKU (MBNF is a name-path label, not a correctness proof). This
  file measures coverage, not accuracy; accuracy needs the eval sets.

---

## 7. Acceptance

The plan is executed when `Truth - mapping coverage.xlsx` exists in `03_review`, every headline number in
`TRUTH-MAPPING-BASELINE.md` is reproduced by one command, the verification checklist passes, and the doc
trail (§3.4) is committed locally with the gates' answers recorded in `Read me`.

---

## 8. Build record

Built 2026-11-01 (builder `eval-datasets/code/build_turnover_truth.py`, ~35 s from cache).

| Output | Path |
|---|---|
| Workbook (11 sheets) | `F:\…\Mapping Inconsistency\03_review\Truth - mapping coverage.xlsx` |
| Onboarding handover pack | `F:\…\Mapping Inconsistency\03_review\Onboarding - retailers and distributors.xlsx` |
| Readout | `TRUTH-MAPPING-BASELINE.md` (+ copy in `03_review/Truth - mapping baseline.md`) |
| Machine summary | `eval-datasets/outputs/turnover-truth/summary.json` |
| Cache | `eval-datasets/outputs/turnover-truth/turnover-truth-2026-11-01.csv.gz` (gitignored) |

Headline (deduplicated population, 43,438 lines / ₹15.97 Cr): retailers 89.1% of lines, 78.2% of 1,233
entities fully identified; distributors 96.2% / 81.6% of 332; products 78.7% of lines, 66.1% of 14,148
titles fully mapped.

Verification actually run:

- **Independent recount** — a separate raw-openpyxl pass (no pandas, no cache) reproduced every axis
  exactly: 53,187 rows, 9,749 duplicate-flagged, 43,438 deduped, ₹19,79,51,674.04 / ₹15,97,03,352.41,
  parsing 28,166 / 9,178 / 6,024 / 70, status 41,787 / 1,651 and 38,699 / 4,739, entity keys 332 / 1,233.
- **Determinism** — two builds produce identical `summary.json` modulo the `built_at` timestamp.
- **Workbook audit** — all 11 sheets present; 0 cells whose rendered number exceeds its column width
  (no `###`); every headline value present in `Headline`; row counts match the tables.
- **Rendered check** — `eval-datasets/code/qa_render.py` lays each presentation sheet out the way Excel
  does (fixed layout, Excel column widths), screenshots it through Chromium and measures text clipping
  per cell: **0 clipped cells** on Read me, Headline, Onboarding 80%, Reconciliation vs Amit,
  Distributor coverage, and on every sheet of the onboarding pack. Screenshots land in `/tmp/qa_render_*`
  (scratch, not a deliverable). LibreOffice is still not installed here, so this stands in for the
  PDF→`pdftotext` gate; it catches the same defect class (text/numbers too wide for their column).

Deviations from the plan as written:

- The "ingested ≥ 2026-07-07" sensitivity was useless: every row was created on/after that date, so the
  population equalled the whole file. It was replaced with `created ≥ 2026-09-01`.
- The reconciliation's first scoring ignored the ₹ axis and ranked a slice that matched rows but was 33%
  off on value; scoring now weighs rows, Inv **and** ₹, and the verdict names which axes match.
- Amit's list does **not** reconcile: closest is MBNF / desc+distributor / created ≥ 2026-08-01
  (2,948 rows / 4,580 Inv / ₹1.98 Cr vs his 3,216 / 4,775 / ₹2.48 Cr), and his `Value` is not any line
  amount in this report. This is now an open question for him, not a build defect.
- **The onboarding cut was ranked on the wrong value and is fixed.** `with_cum` originally sorted
  accounts by their TOTAL value, so a large account with one bad line outranked a genuinely broken one
  and the pack claimed ₹1.99 Cr (retailer) / ₹2.97 Cr (distributor) against a true gap of ₹1.74 Cr /
  ₹1.11 Cr. Ranking and cumulative share now use `gap_value` (value on unidentified lines only);
  counts moved 75 → 78 retailers and 10 → 9 distributors in the cut.
- **BirlaNu's own GSTINs top the distributor list.** ₹68.59 L (61.6%) of the ₹1.11 Cr distributor gap
  sits on six BirlaNu/HIL GSTINs — primary-channel records, not distributors to onboard. Flagged in a
  `Note` column in both workbooks and stated in the readout; real distributors account for ₹42.85 L.
- **A render QA gate was added** (`code/qa_render.py`): 0 clipped cells across the presentation sheets
  and the handout pack, measured per cell in Chromium at Excel column widths.

---

## 9. What is still open

1. **File swap on F: — DONE 2026-11-01 16:26 UTC.** Excel held the workbook open at 16:25 (`~$` lock,
   `Permission denied`); after it closed, the corrected 16:21 build was copied over and verified by md5
   (`8e323ebf…`, identical to the staged copy in `eval-datasets/outputs/turnover-truth/`). Close the
   workbook before any future rebuild.
2. Amit's export filter and `Value` definition (the reconciliation sheet lists the 12 closest definitions).
3. Anomaly follow-ups: 95 lines Unmapped-yet-coded, 70 blank-status rows, 391 distributor + 610 retailer
   lines "Identified" with a blank matched name, 4,592 lines whose invoice number is used by more than one
   distributor, 636 lines dated outside 2026.
4. The 207 GST-bearing retailers that are not fully matched = the master-coverage onboarding list.
5. The six BirlaNu/HIL GSTINs need a rule (primary channel), not onboarding — owner to decide whether
   they are excluded from the distributor KPI or mapped to a synthetic master record.

