# Mapping truth baseline — Turn Over Report

**Status: built 2026-11-01 16:21 · source `Turn Over Report.xlsx` (downloaded 2026-11-01 15:00, filters 2026-01-01 → 2026-11-01)**

**Superseded on disk (2026-11-04):** the `01_source/Turn Over Report.xlsx` path now holds a 4-Nov export
(2026-01-01 → 2026-11-04; 54,903 lines, 44,898 deduplicated). Everything below describes the 1-Nov
snapshot and is reproducible from `turnover-truth-2026-11-01.csv.gz`. 4-Nov line coverage: retailer
40,092 / 44,898 (89.3%), distributor 43,787 / 44,898 (97.5%), product 28,834 mapped / 6,337 MBNF /
9,657 unmapped / 70 NA; retailer entities 1,271 = 992 fully / 28 partially / 251 not identified. The
workbook `Truth - mapping coverage.xlsx` has NOT been rebuilt on the 4-Nov export — the two numbers
above are a line-level recount plus the retailer entity rule, nothing else. Details: tracker §14.

The platform's own line-level secondary-invoice report, used as the truth for mapping coverage. Definitions are the plan's §0 gates (D1–D8): headline = deduplicated population (Duplicate-Invoice lines dropped), MBNF counted as mapped but always split, value = line amount excluding tax.

Population: 43,438 lines · 3,322 invoices · ₹15.97 Cr (as delivered: 53,187 lines, ₹19.80 Cr).

## 1. Retailers and distributors (priority)

**Retailers** — 1,233 distinct entities

- 964 fully identified (78.2%), 17 partially, 252 not identified
- lines: 38,699 of 43,438 (89.1%) · value: ₹14.23 Cr of ₹15.97 Cr (89.1%)
- invoices fully identified: 2,974 of 3,322 (89.5%)

**Distributors** — 332 distinct entities

- 271 fully identified (81.6%), 16 partially, 45 not identified
- lines: 41,787 of 43,438 (96.2%) · value: ₹14.86 Cr of ₹15.97 Cr (93.0%)
- invoices fully identified: 3,210 of 3,322 (96.6%)

Retailer identifier coverage (all entities, by what the invoice gives us):

| Identifier | Entities | Fully identified | Not identified |
|---|---:|---:|---:|
| GST+phone | 56 | 8 | 46 |
| GST only | 953 | 794 | 146 |
| phone only | 23 | 7 | 16 |
| none | 201 | 155 | 44 |

Of the 1,010 retailers that carry a valid GST, 207 are not fully matched — that is master coverage, not identifier capture. 44 retailers give us neither GST nor phone and are not matched, which is a hard limit in the invoice itself.

Unmapped accounts: 61 distributors and 269 retailers carry at least one unidentified line. The 80%-of-value cut (₹1.11 Cr distributor gap + ₹1.74 Cr retailer gap) is 9 distributors and 78 retailers (sheet `Onboarding 80%`, and the handover pack `Onboarding - retailers and distributors.xlsx` lists every account with its identifiers and flags that cut).

Of the ₹1.11 Cr distributor gap, ₹68.59 L (61.6%) sits on BirlaNu's own GSTINs (6 records) — primary-channel lines, not distributors to onboard; real distributors account for ₹42.85 L.

Top distributors by unmapped value:

1. BirlaNu Limited (formerly HIL Limited) · 21AAACH2676Q1ZC [BirlaNu's own GSTIN] — ₹32.41 L on 22 unidentified lines
2. BirlaNu Limited (formerly HIL Limited) · 24AAACH2676Q1Z6 [BirlaNu's own GSTIN] — ₹16.90 L on 185 unidentified lines
3. BirlaNu Limited (formerly HIL Limited) · 06AAACH2676Q1Z4 [BirlaNu's own GSTIN] — ₹11.09 L on 92 unidentified lines
4. Jeevish Sales Corporation · 09AHMPG9281E1Z0 — ₹8.53 L on 85 unidentified lines
5. BirlaNu Limited (formerly HIL Limited) · 36AAACH2676Q1Z1 [BirlaNu's own GSTIN] — ₹7.63 L on 7 unidentified lines

Top retailers by unmapped value:

1. Kamal Trading — ₹12.16 L on 30 unidentified lines
2. Maa Vindhyawasini Hardware · 10DYJPP8353K1Z1 — ₹10.52 L on 125 unidentified lines
3. Bajaj Tiles & Sanitary — ₹7.45 L on 102 unidentified lines
4. M S Construction · 18ALLPM1452G5ZY — ₹6.40 L on 96 unidentified lines
5. SP ENTERPRISES · 18DKXPG3759J1ZA — ₹6.18 L on 95 unidentified lines

## 2. Products

- lines mapped: 34,190 of 43,438 (78.7%), ₹12.55 Cr (78.6%)
- split: Mapped 28,166 · Mapped-Branding Not Found 6,024 · Unmapped 9,178 · blank 70
- titles: 14,148 distinct; 9,350 fully mapped (66.1%), 1,108 mixed, 3,690 never mapped
- SKUs: 1,325 distinct `Material No`; 1,257 carry only mapped lines

Unmapped titles bucketed against the 18-Oct catalogue (5,165 description+distributor rows, 9,178 lines, ₹3.42 Cr):

| Bucket | Rows | Lines | ₹ | Share |
|---|---:|---:|---:|---:|
| C · 2+ SKUs fit | 1,607 | 3,138 | ₹1.25 Cr | 31.1% |
| I · unparsed | 1,254 | 1,955 | ₹92.09 L | 24.3% |
| B · one SKU after normalisation | 438 | 864 | ₹37.15 L | 8.5% |
| D · qualifier conflict | 453 | 886 | ₹31.04 L | 8.8% |
| E · size not in export | 770 | 1,328 | ₹26.68 L | 14.9% |
| F · no size stated | 354 | 558 | ₹17.88 L | 6.9% |
| A · exact catalogue title, one SKU | 124 | 207 | ₹5.98 L | 2.4% |
| H · other brand | 150 | 220 | ₹4.84 L | 2.9% |
| A2 · exact title, catalogue has two SKUs | 7 | 10 | ₹53,084 | 0.1% |
| G · type not in export | 8 | 12 | ₹24,996 | 0.2% |

In-catalogue (A + A2 + B + C): 2,176 rows (42.1%), ₹1.69 Cr (49.4%) — of which 1,607 rows are ambiguous (2+ SKUs fit). Genuine gaps / other brand / unparsed: 2,989 rows (57.9%).

45.7% of the unmapped lines carry a brand token (BIRLA / BIRLANU / HIL / BN) in the description — 4,197 of 9,178 lines (₹1.67 Cr).

## 3. What is not mapped, and why

Distributor identification is high (96.2% of lines) but concentrated: 61 accounts carry every unidentified line and 9 of them make 80% of that value. Retailers are the weaker axis (89.1% of lines, 78.2% of retailers fully identified); the split that matters is master coverage versus no key — 207 of the 1,010 retailers that carry a valid GST are not fully matched, while 44 entities give us neither GST nor phone. Products behave differently again: 78.7% of lines carry a SKU, and of the unmapped description+distributor rows 42.1% are catalogue products (A/A2/B/C — 1,607 of them ambiguous) while 57.9% are size gaps, qualifier conflicts, competitor brands or unparsed text. 45.7% of unmapped lines carry a brand token, so the next product gain is variant/vocabulary rules, not new catalogue entries.

## 4. Reconciliation with Amit's list

No definition reproduces all three of Amit's numbers. Closest: MBNF / desc+distributor / created ≥ 2026-08-01 → rows 2,948 vs his 3,216 (Δ -268), Inv 4,580 vs 4,775, ₹19,764,578 vs ₹24,752,005. Matches on Inv; fails rows, value. For this slice the report's other money columns read ₹3,717,435 (incl. tax) and ₹19,764,578 (excl. tax) — neither is his ₹24,752,005, so his Value column is not the line amount. Confirm the export filter (date window) and what Value sums before treating his list as a KPI.

| Tab | Rows | Names | Inv | ₹ |
|---|---:|---:|---:|---:|
| SKU | 3,216 | 2,913 | 4,775 | ₹2.48 Cr |
| Distributor | 69 | 41 | 73 | ₹29.57 L |
| Retailer | 517 | 363 | 519 | ₹2.58 Cr |

## 5. Method and reproducibility

- builder `eval-datasets/code/build_turnover_truth.py`; cache `turnover-truth-2026-11-01.csv.gz` (a full xlsx read costs ~80 s, the cache makes rebuilds cheap)
- populations: all lines 53,187 · deduplicated (headline) 43,438 · distributor+retailer identified 37,454 · ingested last two months (created ≥ 2026-09-01) 31,823
- values use `Total line Amount (Excluding tax)`; the incl-tax column is never mixed in
- workbook `F:\...\Mapping Inconsistency\03_review\Truth - mapping coverage.xlsx` carries the per-entity and per-title detail
