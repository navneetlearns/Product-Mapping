# BirlaNu — "Mapping issue list shared by Amit" (29-Oct-2026)

Analysis of the three-sheet issue list shared by the BirlaNu team, checked against the
18-Oct-2026 catalogue export and the 21-Oct production trace in the same folder.

Last updated: 2026-10-31
Workbook analysed: `F:\Product\zotok\Product Mapping\Mapping Inconsistency\01_source\Mapping issue list shared by amit.xlsx`
Catalogue: `F:\Product\zotok\Product Mapping\Eval Related Product Mapping\products-18-October-2026-10-52-16.csv` (2,335 rows)
Trace: `01_source\trace-01a15db1-41cb-73e2-a1b3-bd1cc5283723.json` (Vaibhav Enterprises, invoice GST-492, 21-Oct-2026)
Review workbook: `03_review\Mapping issue analysis.xlsx`
Reproduce: `python3 eval-datasets/code/classify_mapping_issues.py --workbook … --catalogue … --out …`

---

## 0. The one-line answer

**The list is not a catalogue-absence problem.** Of the 3,216 SKU rows (₹2.47 crore),
**41.6% of rows / 55.4% of value are the exact catalogue title or resolve to one SKU**
after case/space/quote/plural/brand-suffix normalisation. A further **34.9% / 29.2% are
catalogue products whose invoice title omits the variant** (SDR/SCH/CLASS, length, Type
A/B, Pushfit/Solfit), so 2–5 legitimate SKUs fit. Only **5.1% / 3.5% are other brands**
(Astral, Supreme, JSW, Nerolac, Prince, Paras, Ashirvad, …) and 18.4% / 11.9% are real
catalogue/qualifier gaps or unparsed items.

The first fix is not better matching. It is (a) the report/status definition behind the
"SKU not mapped" tab and (b) 19 catalogue names that carry two SKUs. Those two account
for most of the exact-title 49.5% of value.

---

## 1. What the three sheets contain

| Sheet | Rows | Unique entities | Invoices (Inv sum) | Value |
|---|---:|---:|---:|---:|
| Distributor not mapped | 69 | 41 names / 44 GSTs | 73 | ₹29.6 L |
| Retailer not mapped | 517 | 363 names / 312 GSTs | 519 | ₹2.58 Cr |
| SKU not mapped | 3,216 | 2,913 material descriptions | 4,775 | ₹2.48 Cr |

`Inv` is read as the number of invoices a row aggregates; the workbook does not define it.
The retailer and SKU sheets cover a similar value (~₹2.5 Cr), and 30 of the 62 distributor
invoices also appear in the retailer sheet — the three tabs are three join stages of the
same secondary-invoice population, not three disjoint problems.

The SKU sheet is the main event: 3,216 rows, 2,913 distinct descriptions, 4,775 invoice-line
occurrences. A long tail: the median description is 5 words, 2,913 distinct titles for
4,775 occurrences.

---

## 2. SKU sheet — catalogue verification

Classification is component-based: family (CPVC/UPVC/SWR/PRESSURE/…), type
(PIPE/ELBOW/TEE/…), diameter sizes (mm, inch, mixed fractions, family-specific conversion)
and qualifiers (SCH/SDR/CLASS/kg, Type A/B, Pushfit/Solfit, S/S-D/S, length, colour,
volume/capacity). A description is matched only inside its family+type cell; no raw
fuzzy-string matching.

| Bucket | Rows | Row share | Value | Value share |
|---|---:|---:|---:|---:|
| A. Exact catalogue title, one SKU | 920 | 28.6% | ₹80.3 L | 32.4% |
| A2. Exact title, catalogue has **two** SKUs | 100 | 3.1% | ₹42.2 L | **17.0%** |
| B. One SKU after normalisation (format/vocabulary only) | 319 | 9.9% | ₹14.7 L | 5.9% |
| C. In catalogue, **2+ SKUs fit** (variant/length not stated) | 1,122 | 34.9% | ₹72.2 L | 29.2% |
| D. Qualifier conflict (CLASS/SDR/colour/kg) | 150 | 4.7% | ₹14.8 L | 6.0% |
| E. Family/type exists, size absent | 225 | 7.0% | ₹6.6 L | 2.7% |
| F. Family/type exists, no size stated | 94 | 2.9% | ₹3.5 L | 1.4% |
| G. Type/family not in export | 2 | 0.1% | ₹0.01 L | 0.0% |
| H. Other brand / non-BirlaNu | 163 | 5.1% | ₹8.7 L | 3.5% |
| I. Unparsed / non-standard item | 121 | 3.8% | ₹4.7 L | 1.9% |
| **Total** | **3,216** | **100%** | **₹2.48 Cr** | **100%** |

Headline aggregates:

- **In catalogue and resolvable (A + A2 + B): 1,339 rows (41.6%), ₹1.37 Cr (55.4%).**
- **Exact title alone (A + A2): 1,020 rows (31.7%), ₹1.22 Cr (49.5%).**
- In catalogue but ambiguous (C): 1,122 rows (34.9%), ₹72.2 L (29.2%).
- Catalogue gap / qualifier / unparsed (D+E+F+G+I): 592 rows (18.4%), ₹29.6 L (11.9%).
- Other brand (H): 163 rows (5.1%), ₹8.7 L (3.5%).

### 2.1 The top-value rows are exact catalogue titles

| Value | Bucket | Catalogue SKU(s) | Description |
|---:|---|---|---|
| ₹5.68 L | A2 | 96000676 / 98000874 | SWR Pipe Type A Pushfit S/S 3m 110 mm |
| ₹3.86 L | A | 96000271 | CPVC ISI Pipe SDR-13.5 (5-mtr) 1" |
| ₹3.52 L | A2 | 96000674 / 98000873 | SWR PIPE TYPE A PUSHFIT S/S 3M 75MM |
| ₹3.38 L | A2 | 96000676 / 98000874 | SWR Pipe Type A Pushfit S/S 3M 110 MM |
| ₹3.21 L | C | 96000002 / 96000265 | CPVC ISI PIPE SDR 11 1" |
| ₹2.98 L | A2 | 96000676 / 98000874 | SWR PIPE TYPE A PUSHFIT S/S 3M 110MM |
| ₹2.75 L | A2 | 96000676 / 98000874 | SWR Pipe 110mm 3m Type A Pushfit S/S - B |
| ₹2.61 L | A | 96000014 | UPVC Pipe 40 (3 Mtr) 1'' |

### 2.2 Nineteen catalogue names carry two SKUs — ₹42.2 L blocked

The catalogue export has 48 normalised names with two SKUs (96 rows). Nineteen of those
pairs carry rows in this issue list: 100 rows, ₹42.2 L (17.0% of the list). The pattern is
an old and a new code for the same article (9600xxxx vs 9800xxxx / 9800-10xx), for example:

| Issue rows | Value | Name | SKUs |
|---:|---:|---|---|
| 22 | ₹24.4 L | SWR Pipe Type A Pushfit S/S 3m 110 mm | 96000676 \| 98000874 |
| 17 | ₹8.4 L | SWR Pipe Type A Pushfit S/S 3m 75 mm | 96000674 \| 98000873 |
| 8 | ₹3.2 L | SWR Pipe Type A Pushfit S/S 3m 160 mm | 96000677 \| 98000817 |
| 10 | ₹2.8 L | CPVC ISI PIPE SDR 11 (3 Mtr) 3/4" | 96000001 \| 98001060 |
| 8 | ₹0.7 L | Pressure Pipe CLASS-3 3m 50mm | 96000872 \| 98000891 |

Even a perfect title match returns two SKUs here. If the mapper treats the tie as
unmappable (or the report counts the line as ambiguous), a product that is literally in
the catalogue still shows as "not mapped". This is a catalogue-hygiene fix, not a model fix.

### 2.3 A fifth of the value was mapped verbatim by humans before

The historical human mapping corpus (`BirlaNu Products Mapping(Mapped).csv` +
`(New Records).csv`, 3,746 rows) contains **554 of the issue rows verbatim** (21.7% of the
value), and every one of those carries a SKU still present in the current catalogue. These
are not new or unresolvable products. In the same corpus, the title `SWR Pipe Type A
Pushfit S/S 3m 110 mm` → 96000676 is recorded as MAPPED.

### 2.4 Family split — SWR is a duplicate-SKU story, CPVC is an ambiguity story

| Family | Rows | Value | Exact share | Ambiguous share |
|---|---:|---:|---:|---:|
| SWR | 528 | ₹83.3 L (33.7%) | 78.0% | 13.3% |
| CPVC | 944 | ₹72.8 L (29.4%) | 35.5% | 51.0% |
| UPVC | 701 | ₹36.3 L (14.7%) | 55.5% | 22.7% |
| Pressure | 204 | ₹16.2 L (6.6%) | 41.5% | 21.9% |
| Other brand | 163 | ₹8.7 L (3.5%) | — | — |

SWR is mostly exact titles, with the 110/75/160 mm 3 m Type A Pushfit S/S pairs duplicated
in the catalogue. CPVC is the opposite: the invoice says `CPVC PIPE 3/4"` or `1" CPVC PIPE
13.5 SDR` and the catalogue has SDR 11 vs 13.5, 3 m vs 5 m vs 300 mm, and TRD duplicates —
so five to seven SKUs fit and the title alone cannot choose. Price is the missing signal
(see §5).

### 2.5 The C bucket is genuinely ambiguous, not just parser noise

Sample candidate sets from the review workbook:

- `CPVC PIPE 3/4"` → 6-7 SKUs: SDR 11 vs 13.5 × 3 m / 5 m / 300 mm / TRD.
- `CPVC ISI PIPE SDR 11 1"` → 96000002 (3 m) vs 96000265 (5 m).
- `SWR pipe solefit S/S 110mm` → 20 SKUs: Type A/B × 3 m / 6 m / 12 ft / 2-6 ft × S/S-D/S.
- `UPVC PIPE SCH 40 1/2"` → 96000012 (3 m) vs 96000018 (6 m).
- `BALL VALVE 3/4" CPVC` → 9 SKUs: plain vs MFG vs compact vs brass, one/two piece.

These are the invoices' under-specification meeting the catalogue's variant density. The
answer is usually in the line's unit price or in the distributor's prior purchase history;
neither is in the spreadsheet (no unit rate column), but both exist in the production line
item (`unit_rate_excl_gst`, `product_match_context.price_matched`, `catalog_price`).

### 2.6 What the trace says about why exact titles are on the list

Trace `01a15db1-…` is the production `turnover_schemes` run for Vaibhav Enterprises invoice
GST-492 (18 line items, 21-Oct-2026). Facts observed:

1. `fetch_distributor_catalog` returned **`[]` for all 18 line items**. The distributor-
   specific catalogue is empty for this CFA.
2. The fallback `fetch_catalog_text` returned candidates for all 18, and
   `map_catalog_products` mapped **17 of 18** (one line lacked the second dimension and
   scored 65 → UNMAPPED).
3. `create_turnover_invoice` persisted **16 MAPPED, 1 MAPPED_BRAND_NOT_FOUND, 1 UNMAPPED**,
   with `sku_code` + `product_variant_id` on the mapped lines.
4. The **source invoice** record (`fetch_invoice`) has line items with `status` but **no
   `sku_code` and no `product_variant_id`** fields at all. The SKU exists only on the
   turnover-invoice line.
5. Of this invoice's 18 descriptions, only **two** appear in Amit's workbook — and those two
   are exactly the two `MAPPED_BRAND_NOT_FOUND` lines (`UPVC Reducing Bush 1 1/4"x1"`,
   `UPVC Union 1 1/2"`). The 15 MAPPED lines do not appear.

That pattern points at **status/report semantics, not catalogue absence**:

- `MAPPED_BRAND_NOT_FOUND` means "a catalogue SKU was found, but the title did not contain
  the literal words birla/hil". It is a mapped tier, not an unmapped one (the skill's own
  rule; the eval runs carry it as a separate GT label). A report that filters
  `status <> 'MAPPED'`, or that reads the source invoice table (which has no SKU column),
  will list these lines as "SKU not mapped" while the SKU is present on the turnover line.
- The 92% of SKU-sheet rows whose title contains no brand token are structurally MBNF if
  they map at all. This is exactly the population in the list.

**Hypothesis to confirm (one query, not a model rerun):** for a sample of invoice numbers
from the SKU sheet, join the source invoice line to the turnover invoice line and select
`source.status`, `turnover.sku_code`, `turnover.matched_brand`, `turnover.mapping_confidence`.
If the list is MBNF-only, the result is: status = MAPPED_BRAND_NOT_FOUND, sku_code non-null.
If the list is true UNMAPPED, sku_code is null. The trace suggests the former.

This is cheaper and more decisive than reprocessing invoices in LangSmith. The free
LangSmith plan keeps ~2 weeks; the workbook is 29-Oct, so the window is still open for
invoices from ~15-Oct onward.

---

## 3. Distributor sheet — 41 names, but 7 variant clusters and 14 unlinked invoices

69 rows, 62 invoice numbers, 73 `Inv`, ₹29.6 L. 49 of the 69 rows belong to distributors
that also appear as sellers in the retailer sheet — the entity exists elsewhere in the same
data, so this is largely a link/identity problem, not a missing-party problem.

| Root cause | Rows | Value | Share |
|---|---:|---:|---:|
| Invoice-to-master link gap (distributor trades elsewhere, this invoice unlinked) | 14 | ₹10.9 L | 36.8% |
| Entity resolution — name/GST variants and duplicate records | 37 | ₹10.7 L | 36.2% |
| Not in platform master — onboarding/coverage | 14 | ₹6.1 L | 20.6% |
| GST missing/invalid | 4 | ₹1.9 L | 6.4% |

Concrete defects:

- **7 name-variant clusters (15 names → 7 entities)**: `Future Trade Com` / `Future Trade
  Com (2026-27)`, `B K Enterprises` / `B K Enterprises (2026-27)`, `G M ENTERPRISE` /
  `G. M. Enterprise`, `Kamal Sanitary & Hardware Store` / `M/S KAMAL…`, `MAA SHERAWALIS
  MATABHOOMI` / `MATRABHOOM` / `MATRABHOOMI PIPES`, `MAA VINDHYAWASHINI ENTERPRISES 25-26`
  / without suffix, `Satyam Home Decor` / `Decore`.
- **Same GST under two names**: 09AASPL3614M1Z5 (B K Enterprises ± year), 09BBFPG1458M2ZD
  (Gupta Agency / M/S Gupta Agency).
- **Same name under multiple GSTs**: M/S Kamal Sanitary (2), G M Enterprise (3), Maa
  Vindhyawashini (5 — likely distinct firms or data errors).
- **3 rows with GST = NA plus 1 malformed GST** (4 rows, ₹1.9 L).
- The trace shows the duplicate-master problem in production: `fetch_cfa` returned **two
  seller records for Vaibhav Enterprises** — id `366a06fa…` code `0001121653` and id
  `ff09993d…` code `1121653`. The catalog lookup used the first and returned empty.

The top distributor is Jeevish Sales Corporation (₹8.5 L, 29% of the sheet) and it is
flagged "seller elsewhere in retailer sheet" — it is known to the invoice population but
this invoice is not linked to its master record.

---

## 4. Retailer sheet — 90.7% carry a GST; the gap is master coverage, not missing identity

517 rows, 474 invoice numbers, 519 `Inv`, ₹2.58 Cr, 363 names. Identifier coverage:

| Identifier strength | Rows | Row share | Value | Value share |
|---|---:|---:|---:|---:|
| GST only | 241 | 46.6% | ₹1.18 Cr | 45.7% |
| GST + phone | 178 | 34.4% | ₹63.9 L | 24.7% |
| Phone only | 48 | 9.3% | ₹26.6 L | 10.3% |
| No GST and no phone | 50 | 9.7% | ₹49.7 L | 19.3% |

So **90.7% of rows have a GST** and 43.7% have a phone. If GST matching worked, most of
this sheet should clear. It does not, which points at the customer master (the retailer is
not registered / the match is phone-first) rather than at identifier capture. The 50
no-identifier rows (₹49.7 L, 19.3% of value) are a hard limit — no key exists in the
invoice.

| Root cause | Rows | Value | Share |
|---|---:|---:|---:|
| Not in customer master (or match is phone-only); GST present | 215 | ₹1.09 Cr | 42.0% |
| Both identifiers present but not in master — coverage | 159 | ₹54.2 L | 21.0% |
| No GST and no phone — cannot be keyed | 50 | ₹49.7 L | 19.3% |
| Not in master; only a phone | 40 | ₹25.4 L | 9.8% |
| Name variants on the same GST | 53 | ₹20.5 L | 7.9% |

Other defects: 291 rows have no phone (₹1.68 Cr, 65% of value), 98 have no GST (₹76.3 L),
31 rows have no distributor SAP code (₹7.5 L), 15 GSTs are malformed (₹5.0 L), 24 rows sit
on a GST that appears under two names (₹7.1 L). 106 retailers account for 80% of the
value — a bulk-onboarding list, not a long tail.

---

## 5. Root causes, ranked

1. **Report/status semantics (largest, cheapest to fix).** The source invoice line table
   stores `status` but not `sku_code`; the SKU is on the turnover line. A "SKU not mapped"
   view built from the source table, or filtered on `status <> 'MAPPED'`, reports mapped
   MBNF lines as unmapped. Trace: 2/2 MBNF lines present, 15/15 MAPPED lines absent. This
   explains why exact catalogue titles appear on the list at all. Fix: report from the
   turnover line; show `sku_code`; separate MAPPED / MAPPED_BRAND_NOT_FOUND / UNMAPPED;
   do not call MBNF "not mapped".
2. **Catalogue duplicate SKUs (₹42.2 L, 17.0%).** 19 names carry an old and a new SKU.
   Even exact titles return two answers. Fix: one canonical SKU per name (retire/hide the
   old), or rank a primary and return one candidate.
3. **Variant under-specification in the invoice (C, ₹72.2 L, 29.2%).** CPVC SDR/length,
   SWR Type A/B × Pushfit/Solfit × length, UPVC 3 m/6 m, pressure CLASS/length. The
   candidate set contains the answer but the title does not choose. Fix: use unit price vs
   catalogue price/MRP (already in `product_match_context.price_matched`), distributor
   history, then a per-family priority rule; unresolved → status `AMBIGUOUS` with top-2,
   not `UNMAPPED`.
4. **Normalisation/alias gaps (B + D, ₹29.5 L, 11.9%).** mm↔inch, `13.5sdr`, `Pipe 40` =
   SCH 40, kg↔CLASS, coupler↔socket (CPVC/UPVC), SOFIT/SELFFIT, `R/Tee`, `C`/`U` prefixes,
   `160M`, `118mi`, trailing price/pack tokens (`419/-`, `20PC`), plural/word order.
   Deterministic pre-processing, not a model change.
5. **Master-data coverage/identity (distributors ₹29.6 L; retailers ₹2.58 Cr).** Distributor:
   name variants, 2 GSTs under one name, 2 CFAs under one GST (Vaibhav trace), 14 invoices
   unlinked despite the entity trading elsewhere. Retailer: 90.7% have GST but are not in
   the master; 50 rows have no key at all. Fix: canonical name + GST-first matching,
   duplicate-CFA merge, bulk onboarding of the top 11 distributors / 106 retailers.
6. **Other-brand lines are counted as mapping failures (₹8.7 L, 3.5%).** Astral, Supreme,
   JSW, Nerolac, Prince, Paras, Ashirvad, Watertec, Prakash, Flowton, Kisan, Indiano,
   Adcolac, paints/putty. These should be labelled `NOT_BIRLANU` and excluded from the
   mapping KPI, not chased.

---

## 6. Proposed solution, phased

**P0 — this week, no model change**

1. Fix the report: source = turnover invoice line, columns `status`, `sku_code`,
   `matched_brand`, `mapping_confidence`; separate `MAPPED_BRAND_NOT_FOUND` from
   `UNMAPPED`; the KPI is "SKU found", not "status = MAPPED". If the panel cannot change,
   export the turnover line and rebuild the three tabs.
2. Dedupe the 19 catalogue pairs: pick the primary SKU (the 9600/9800 suffix rule needs a
   catalogue owner decision), hide the other from search, or return one candidate.
3. Tag other-brand lines (`NOT_BIRLANU`) and exclude them from the mapping KPI.

**P1 — 2–4 weeks**

4. One normalisation/alias layer in the retrieval path: quotes/fractions, mm↔inch, plural,
   word order, brand suffix (`-B`, `BN`, `TRD`), price/pack tokens, `R/Tee`, `13.5sdr`,
   `160M`, `118mi`, `Pipe 40` → `SCH 40`, kg↔CLASS, coupler↔socket per family.
5. Variant resolution: candidate set + unit price / catalogue price / MRP + distributor
   history; if still >1, emit `AMBIGUOUS` with top-2 suggestions. Never let a tie become
   `UNMAPPED`.
6. Master data: canonical name (strip year suffix, punctuation, M/s), GST as primary key,
   phone as fallback, duplicate CFA merge (Vaibhav is the test case), bulk onboarding of the
   top 11 distributors and 106 retailers, capture GST/mobile for the 50 no-identifier
   retailers.

**P2 — ongoing**

7. Catalogue hygiene: one name = one SKU; add the missing sizes/variants surfaced in E/D;
   include live-only `412xxx` in the export used for mapping (today's export has only
   9000/9600/9800/9900 prefixes).
8. Regression set: the 554 verbatim corpus rows + 100 exact-title rows + the 19 duplicate
   pairs; run after every retrieval/prompt change.

---

## 7. Verification plan (the second path)

Cheapest to most expensive:

1. **SQL/panel query (minutes).** For 10 invoice numbers from the SKU sheet, pull source
   line `status` + turnover line `sku_code`/`matched_brand`/`confidence`. Decision rule:
   if MBNF with sku → report semantics confirmed; if UNMAPPED with null → true mapping gap.
2. **LangSmith trace sample (1–2 hours, within the free 2-week window).** Filter
   `service-agent-prod` runs for those invoice numbers/dates; read
   `fetch_distributor_catalog` → `fetch_catalog_text` → `map_catalog_products` →
   `create_turnover_invoice`.
3. **Reprocess 2–3 invoices (half a day).** Re-run the WhatsApp flow for invoices outside
   retention and inspect the same nodes. Do this only after (1) because (1) is decisive
   and does not consume the window.

Acceptance for this analysis: on a 100-line sample, ≥80% should fall into the same buckets
(A/A2/B/C/H). If the MBNF hypothesis holds, the dominant bucket is A/A2 with a non-null
SKU on the turnover line.

---

## 8. Open questions for the user / platform team

1. **What exactly does Amit's "SKU not mapped" tab query?** Source `invoice_line_items`
   (status only) or turnover `line_items` (sku_code)? This decides whether P0 is a report
   fix or a mapping fix.
2. **Can we run the SQL/trace sample?** If yes, one hour resolves the MBNF hypothesis.
3. **Catalogue ownership:** who retires the 19 duplicate SKUs (old 9600 vs new 9800)?
4. **Price signal:** the workbook has no unit rate. Can the mapping export include
   `unit_rate_excl_gst` and `catalog_price`? That would let us resolve C offline.
5. **Other-brand policy:** should competitor lines be excluded from the BirlaNu secondary
   sales KPI or kept as "non-BirlaNu"?

---

## 9. Caveats

- The catalogue is the 18-Oct-2026 export (2,335 rows). Live-only `412xxx` SKUs are out of
  scope; "not in catalogue" means "not in this export".
- The workbook has no unit price/quantity; C is classified by title/variant only. The
  production line has the price signal, so C is not necessarily unmappable in production.
- "Inv" is assumed to be invoice count; the workbook does not define it.
- The classifier is deterministic and parser-limited. Buckets are a review aid; the
  workbook carries the per-row reason and candidate SKUs so a human can verify. The exact
  title buckets (A/A2) are string-exact after normalisation and are the most reliable.
- The MBNF explanation is a hypothesis supported by one production trace, not a proven
  conclusion. Confirm with §7.1 before changing the pipeline.
