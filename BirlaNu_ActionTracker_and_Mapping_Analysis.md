# BirlaNu — Action Tracker & Invoice Mapping Analysis

Last updated: 2026-11-05
Context: BirlaNu (seller "Birlanu Limited") invoice digitisation — action tracker review + a live mapping-pipeline trace post-mortem + LangSmith turnover-mapping eval analysis (§6) + the sentinel lane exports cleaned and corrected (§7) + turnover OCR-setup benchmark arm verified (§8) + the BirlaNu "mapping issue" list (Amit, 29-Oct) verified against the catalogue (§9) + the Turn Over Report designated as the mapping truth, with distributor/retailer/product coverage measured from it (§10) + Amit's own tabs annotated with each party's report status (§10) + the platform's presence check bucketed against the catalogue and workspace (§11) + the platform team's status note checked line by line (§12) + the turnover DB DDL mapped to the product-mapping and turnover-report paths (§13) + the sqlit/SSM access path verified and documented (runbook in db-schemas) + the 4-Nov report refresh and the 51-retailer matching-fix queue (§14) + Amit's unique SKU-tab products compared with the 4-Nov report at product grain (§15).

## Source artifacts
- Action Tracker (PDF, exported from the live workbook):
  `/mnt/f/Product/zotok/Invoice/Fake Invoices/20260714_BirlaNu_Action_Tracker_ZOTOK.pdf`
- Cadence-meet deck (3-slide PDF, built 2026-10-09):
  `/home/sumit/AgentWork/product-mapping/birlanu-cadence-deck/outputs/BirlaNu-Cadence-Deck.pdf`
  (source + render/QA scripts + README in the same folder; see §5.)
- Live pipeline trace (intent `scheme_invoice`, graph `turnover_schemes`, node `map_catalog_products`, model gpt-5.4-mini):
  invoice 495/26-27, distributor MAA SHERAWALIS MATRABHOOMI PIPES (GST 27AFPPJ3948E1Z6),
  buyer RAMDEV HARDWARE & ELECTRICALS (GST 27CEXPR0237R1ZF), dated 2026-08-23.
- LangSmith eval export — turnover product-mapping (dataset example version 2026-10-16,
  mapping revision v8.4.19-2):
  `/mnt/f/Langsmith/BRAND NAME/turnover-product-mapping-136d0d1e.csv` — see §6.
- Turnover OCR-setup benchmark (2026-10-30): `/mnt/f/Product/zotok/Product Mapping/CU READ EVAL OUTPUT/` —
  `cu-read-text-only-856a0156.csv` (40 runs) + `turnover_ocr_setups.html` (5 setups × 20 invoices); see §8.
- BirlaNu issue list (2026-10-29): `/mnt/f/Product/zotok/Product Mapping/Mapping Inconsistency/01_source/Mapping issue list shared by amit.xlsx`
  (69 distributor / 517 retailer / 3,216 SKU rows); full analysis in
  `Mapping_Inconsistency_Analysis.md`, per-row review workbook
  `Mapping Inconsistency/03_review/Mapping issue analysis.xlsx`, classifier
  `eval-datasets/code/classify_mapping_issues.py` — see §9.
- Turn Over Report (mapping truth — the path now holds the 4-Nov re-export, §14; the 1-Nov snapshot
  behind §10 is preserved as the cached CSV):
  `/mnt/f/Product/zotok/Product Mapping/Mapping Inconsistency/01_source/Turn Over Report.xlsx`
  (secondary invoice report, 2026-01-01 → 2026-11-04; 54,903 lines / 3,431 invoice+date keys / 44,898
  deduplicated; the 1-Nov snapshot read 53,187 lines / 3,135 invoice numbers / ₹19.80 Cr);
  coverage workbook `Mapping Inconsistency/03_review/Truth - mapping coverage.xlsx` (1-Nov build),
  readout `TRUTH-MAPPING-BASELINE.md`, builder `eval-datasets/code/build_turnover_truth.py` — see §10
  and §14.
- Turnover DB DDL (2026-11-03): `F:\reports\Dbeaver\turnoverinvoice.sql` (full `public` schema,
  80,946 bytes); analysis `~/AgentWork/vault/zotok/db-schemas/turnover-invoice-ddl-analysis.md`,
  inventory `turnover-invoice-schema.md` — see §13.

---

## 1. Action Tracker — Open Items (grouped)

Dashboard banner: TOTAL 33 · OPEN 13 · IN PROGRESS 0 · RESOLVED 14 · ON HOLD 2 · HIGH-still-open 5 · 42% resolved.

OPEN (13), by bucket:
- **Mapping — Retailer/Product/SKU (8):** AP-09 (inconsistent retailer ID; GSTIN "Identified" vs "Not Identified") [HIGH], AP-14 (PAN-based mapping not happening), AP-23 (product code in description but SKU not mapped) [HIGH], AP-24 (filter shows only entries with that exact status/tag) [HIGH], AP-25 ("BIRLANU" in SKU but item not mapped, 4 of 20) [HIGH], AP-32 (reprocessing → mapped item becomes unmapped), AP-33 (reprocessing flips score 100%↔98%), AP-34 (same invoice scanned differently → mapping not found).
- **Validation / Data Accuracy (1):** AP-13 (no validation sum of line items = total invoice value) [MEDIUM].
- **Reporting (2):** AP-15 (invoices in system missing from downloaded report), AP-17 (report vs panel — status/tax-status columns missing).
- **Upload / Multi-page (2, both HIGH):** AP-19 (2 pages uploaded, only 1 shown + wrongly marked fraudulent), AP-20 (2 pages uploaded properly but system errors — edge-detection too strict).
- Also open (non-High): AP-22 (BirlaNu invoice 2040003504, mismatched distributor/retailer names), AP-30 (multi-page upload — if retailer doesn't reply "no", invoices collapse into one; see §4).

ON HOLD (2): AP-07 (Primary vs Secondary report — blocked on BirlaNu primary-order data), AP-11 (retailer uploading other retailers' invoices — deferred to next sprint).

---

## 2. Invoice 495/26-27 — Line 6 Mapping Failure

### Result: 6 of 7 lines mapped; line 6 UNMAPPED (confidence 38).
Mapped lines all carried sku_code + product_variant_id. Line 6 had both = null.

### Line 6
- Invoice text: `Agri 6kg(Class3)3Meter Pipe 50MM 1 1/2"code96000872` (HSN 39172390, price 41200).
- Extraction artifact: SKU came out as `code96000872` (lowercase, no parens, run-together) vs the clean `(96000012)` form on the other 6 lines.
- Candidates returned by the Search API for this line:
  1. UPVC Pipe 21/2" (SCH 80) (3meter)   SKU 96000406
  2. UPVC Pipe 4"   (SCH 80) (3meter)   SKU 96000408
  3. UPVC Pipe 6"   (SCH 80) (3meter)   SKU 96000986
  4. CPVC Pipe 21/2" (SCH 40) (3meter)   SKU 96000397
  5. CPVC Pipe 21/2" (SCH 80) (3meter)   SKU 96000403
- All share HSN 39172390 → retrieval was HSN-filtered (top-N). The real SKU 96000872
  (Agri 6kg 50MM 1 1/2" 3Mtr pipe, also HSN 39172390, confirmed present in the
  BirlaNu mapping workbooks) was NOT in the candidate set.
- LLM reasoning (verbatim): "No candidate SKU 96000872 was present; remaining
  candidates are different pipe sizes/materials and do not match." → returned None.

### Root cause
1. RETRIEVAL MISS (primary): candidate Search API returned HSN-top-N, dropped the
   correct SKU 96000872. The code WAS present in the description but the search
   didn't surface it. → this is AP-23 in production.
2. SKU-STRING IRREGULARITY (secondary): extraction produced `code96000872` instead
   of a clean parenthesised code, so even the LLM's Step-0 SKU detection couldn't
   force the direct match path.

### Not at fault
The LLM. Given the 5 wrong candidates, returning None was spec-compliant
(Step 4: confidence ≤80 → sku_code None).

---

## 3. Proposed Fix — SKU-only Retrieval Branch

Agreed approach (user, 2026-09-04): in the retrieval/Search node, branch on
embedded SKU detection:
- **if** product title contains an embedded SKU code → call Search API with SKU only
  (expect the exact catalog row, e.g. 96000872, returned at confidence 100).
- **else** → call Search API with full product title (fuzzy/semantic).

### Conditions for this to work (must hold before coding)
1. SKU detection must be robust: normalize `code`/`Code`/`CODE`, require 8-digit
   token, parentheses optional. The if-condition and the search payload share one
   helper. (Current extraction gave `code96000872` — fails a naive check.)
2. The "SKU-only" call must be an exact primary-key / getProductBySku lookup, NOT
   another ranked search — otherwise 96000872 can be dropped from top-N again.
3. Add a fallback: if SKU-lookup returns nothing, fall back to full-title search
   rather than returning None (don't let one missed row kill the line).
4. Keep the LLM's Step-0 SKU-priority path as safety net (already forces conf 100
   on a found SKU).

### Scope note
The else-branch (full title) retains the latent AP-23/AP-25 exposure for lines
without an embedded SKU. Most BirlaNu invoices carry codes, so this is an
acceptable split — but not a full fix for the no-SKU case.

---

## 4. AP-30 — Multi-page Upload: User Doesn't Reply "no" to Continue Prompt

Status: Open (Threads bucket). Tracked 2026-09-07 by user as a work item.

### Problem
The multi-page invoice upload flow assumes an explicit "no" is the ONLY terminal
signal. Flow intent (per AP-06 design: one invoice = one record, multi-page
uploads merged into a single invoice):
- user uploads a page → system confirms → agent asks "any more pages?"
- on "no" → finalize the current invoice and advance to the next stage.

Failure mode: when the agent asks "do you have more pages?", the user does NOT
reply "no" (no reply at all, or keeps uploading the next invoice's pages before
answering). Because the flow only advances on an explicit "no", every subsequent
page keeps accumulating into the SAME invoice record → **multiple distinct
invoices collapse into one upload** (also strands that data as partial per the
tracker note, and drives re-uploads / AP-35 duplicate noise).

### Why it matters
- Inverse companion of the (fixed) AP-06 bug: AP-06 was one invoice → two records;
  this is two invoices → one record. Both break the "one invoice = one record"
  invariant AP-06 was fixed to enforce.
- Feeds AP-35 (duplicates: mis-batched record looks like noise, real invoices get
  re-uploaded separately).
- Compound risk with AP-32/AP-33 (reprocessing flips mapping scores) — a
  mis-batched multi-invoice upload is likely to be reprocessed.

### Core design flaw
Trusting an explicit negative ("no") as the only terminal signal is fragile on
WhatsApp: users go silent, upload the next invoice before answering, or answer
ambiguously ("ok", "yes done", a photo) that may not classify as "no".

### Recommended approach (assessment — not yet spec'd/approved)
1. **Timeout-based finalization (cheapest).** After the last confirmed page, if no
   user message arrives for N minutes, auto-finalize the current invoice and
   advance to the next stage. Silence becomes the terminal signal.
2. **Page-gating / new-invoice boundary.** Once "no" (or finalize) has fired, treat
   any new image upload as the START of a new invoice, never an extension of the
   last one — prevents two invoices merging.
3. **Image-level invoice boundary heuristic (robust, needs OCR).** If a newly
   uploaded image carries its own GSTIN / invoice-number / date differing from the
   record being assembled, split it into a new invoice (reuses AP-06 merge logic
   in reverse).
4. **Prompt affordances.** Make the confirmation unambiguous (button-style options,
   re-prompt on ambiguous reply) so "no" is actually detected more often.

### Open decision
Confirm terminal signal semantics with the platform team: timeout duration + what
happens to an abandoned partial invoice (auto-finalize vs hold). Tracker currently
notes "these invoices will be partial."

---

## 5. Cadence Deck — Delivered 2026-10-09 (deliverable, not analysis)

Built a 3-slide cadence-meet deck for the BirlaNu team (canvas-design philosophy + claude-design
deck rules, house style inherited from `zochief-demo-docs/sales-deck`, Playwright Chromium → PDF).

- **PDF:** `birlanu-cadence-deck/outputs/BirlaNu-Cadence-Deck.pdf` — 3 pages, 16:9 (1440×810 pt).
- **Slides (3, condensed from an earlier 6-slide cut):** cover · cadence counts (all five counts +
  September mark-matrix + approval bar on one page) · enhancements implemented.
- **Figures reported for this cadence (supplied, not re-derived from the workbook):**
  2,323 invoices parsed till date · 30,063 line items till date · 957 parsed in September ·
  181 approved · 36 issues resolved. Only additions are arithmetic from those: 41.2% = 957/2,323,
  12.9 lines/invoice = 30,063/2,323, 7.8% = 181/2,323, 1,366 = 2,323−957.
- **Source of the numbers:** cadence input from the team — NOT read from the live workbook or
  dashboard; freshness/accuracy unverified (the tracker banner in §1 is a separate 2026-07-14 snapshot).
- **Commits (local only, not pushed):** `de9e5ac` initial 5-slide deck, `f907ca2` enhancements slide,
  plus the 3-slide condensation commit in `git log`.
- **Rebuild/QA:** see `birlanu-cadence-deck/README.md` (`render_deck.py` + QA/inspect scripts).

### Enhancements reported as implemented (as-reported; not verified against the panel)
1. **Invoice reprocessing (Man-in-the-Middle, ZoTok panel)** — single or bulk. Relates to
   AP-32/AP-33 (reprocessing flips mapped→unmapped / 100↔98): reprocessing now being used in
   volume makes those score-instability items more exposed, not less — keep them prioritised.
2. **Automatic rejection of unclear/illegible images on WhatsApp** — quality rejection at upload.
   Adjacent to AP-19/AP-20 (multi-page upload + strict edge detection): confirm the new rejection
   path doesn't reclassify a legitimately cropped second page as "unclear".
3. **Original-invoice link for duplicate detection** — duplicates link back to the first submission.
   Direct mitigation for AP-35 (duplicate invoices surfaced to a retailer).
4. **Page display limit 40 → 100** (ZoTok panel pagination) — pure UX; no AP impact.

Status of 1–4: reported in the 2026-10-09 cadence input as implemented. Not yet exercised or
confirmed in the panel by us — treat as claims pending a smoke test before citing them as fixed.

## 6. LangSmith Eval Export — turnover-product-mapping-136d0d1e (71 invoices / 899 lines)

Analyzed the export at `/mnt/f/Langsmith/BRAND NAME/turnover-product-mapping-136d0d1e.csv`
(LangSmith project agent-playground; mapping revision v8.4.19-2-ga863d2082d4; dataset split
"base"; example version 2026-10-16). All 71 runs success; avg 21.2s / 8,624 tokens / $0.0015
per invoice, $0.10 for the whole eval. All numbers reconciled against the export's own
evaluator columns.

### Headline
- Model mapped 661/899 lines (73.5%); expected/GT 607/899 (67.5%).
- Status correct 835/899 (92.9% = 899 − 59 false maps − 5 missed); SKU correct 815/899 (90.7%)
  after equivalence-aware scoring (20 wrong SKUs; 32 if equivalence ignored).
- 30/71 invoices carry ≥1 wrong line; 41 clean. Invoice level: fully mapped 25 (GT wants 20),
  partial 42, fully unmapped 4 (those 4 hold 8 lines total).
- Confidence bands (mapped / wrong): 80–84: 43/30 · 85–89: 103/39 · 90–94: 198/10 · 95–100: 317/0.
  Nothing maps at ≤79; the 5 should-have-mapped died under the ≤80 gate at conf 65–76 (correct SKU
  was in the candidate set for all 5). The 80–89 zone holds 69 of the 79 mapping errors.

### Brand census
- 709/899 lines (79%) carry a Birla/HIL brand token in the title; 190 don't (~15 of those carry
  another brand: PRINCE 10, ASTRAL 3, VECTUS 1, SUPREME 1).
- Invoice level: all 71 invoices have ≥1 branded line; 34/71 fully branded.
- Candidate side: only 74/899 candidate sets (8%) contain any brand token; of the 709 branded
  lines just 64 (9%) had a same-brand candidate (645 had none). No non-Birla brand token appears
  in candidates at all (260/4,450 candidate entries branded; 69 of 1,031 distinct candidates).
- Brand is a label only: detected to fill matched_brand, then stripped from both sides before
  matching (title tokens + dimensions only). 621 of the 825 lines with zero branded candidates
  still mapped.

### MBNF semantics (user-confirmed) + full status rule
MAPPED_BRAND_NOT_FOUND = "mapped, but the Birla brand keyword (birla/hil) was not found in the
line title". Complete rule, reproduces 893/899 statuses:
- brand in title + mapped → MAPPED
- no brand + exact match (embedded SKU equal to a candidate SKU, or exact normalized title) →
  MAPPED (both GT and model apply this exception)
- no brand + fuzzy name match → MBNF
- no match → UNMAPPED
Counts (model / GT): MAPPED-brand 535/489 · MAPPED-exact 69/69 · MBNF 57/49 · UNMAPPED 238/292.
Model vs GT on MBNF: same 49 + 8 over-maps GT wants unmapped (ASTM ELBOW SCH80 25MM ×2, AGRI PVC
Solvent 500 ML, Upvc Elbow 1", ASTRAL CPVC BRASS TEE 1", 40x25 Reducer, 160mm Elbow, PRINCE UPVC
SCH 80 UNION 3/4"). Non-Birla brand tokens (ASTRAL/PRINCE/ASHIRVAD/SUPREME…) count as "brand not
found" → MBNF. 'sku_matched' never surfaces in outputs; code-carrying lines show matched_brand
None (138) or birla (62) — not a defect.

### Embedded-SKU lines
200/899 lines carry an 8-digit code that matches a candidate SKU (137 branded / 63 unbranded);
all 200 mapped via the exact path. Contrast §2: the line-6 failure was the irregular
`code96000872` extraction, not the code path itself.

### False maps (59) — families
51 of 59 sit on Birla-branded lines. 39 elbow/bend lines → an angle or series sibling (45° /
Trans picks, or the "Pressure (kg/cm2)" series in place of CPVC); 4 SELFFIT/Solfit lines →
Pushfit/SilentFit series; 5 CPVC coupler/brass-tee → Pressure/Reducing picks; 11 other (SWR
tee/coupler, reducer, solvent, pipe, end cap; incl. the ASTRAL/PRINCE picks).

### Data limits
The export's candidates (~top-5 per line; 1,031 distinct products) are the mapper's actual
universe — "exists in master" is testable only against them; the full catalog file on disk
(`/mnt/f/Product/zotok/Product Mapping/Map Product Zotok Catalogue(Sheet1).csv`, ~2.3k rows)
is available for a true master pass. The status rule above is inferred from labels + outputs
(893/899 fit), not read from pipeline source.

---

## 7. Sentinel lane datasets — cleaned, compared, corrected (2026-10-23)

Three 100-row sentinel lane exports were delivered in
`F:\Product\zotok\Product Mapping\Mapping Inconsistency\` (now organised `01_source/` ·
`02_cleaned/` · `03_review/`, with a folder README), alongside the invoice behind the production
trace (`invoice-GST-492-Vaibhav-Enterprises-2026-09-22.pdf`) and its LangSmith run/trace JSONs.
Full record, artifacts and repro commands: `eval-datasets/SENTINEL-lanes.md`.

- **Brand lane** (`Tarun Sentinel Dataset.csv`, `brand-001…100`): 100 rows / 56 invoices / 88 SKUs.
  20 titles carried an embedded SKU code — stripped and flagged; all 20 codes equalled the row's
  mapped SKU. 10 SKUs sit on more than one title (22 rows); HSN 77 eight-digit / 22 family `3917` /
  1 missing.
- **Title-only lane** (`title only sentinel tarun.csv`, `plain-001…100`): 100 rows / 62 invoices /
  86 SKUs, **0 titles with a SKU code**, but **54 titles carried a Birla/BIRLANU/HIL token** — the
  defect. The lane's own rule is "no brand name and no SKU", so those 54 were replaced with
  catalogue-backed titles from the same invoice universe (1,076 titles → 405 no-code/no-brand →
  315 usable → 117 MAPPED), preserving the status mix (78 MAPPED / 22 MAPPED_BRAND_NOT_FOUND),
  52 same-product-family picks + 2 same-HSN fallbacks. All 54 replacement expected SKUs exist in
  the 2,335-SKU catalogue export. Result: `02_cleaned/title only sentinel tarun - corrected.csv`
  (54 rows rewritten, 46 byte-identical).
- **SKU lane** (`sku only tarun sentinel.csv`, `wsku-001…100`, `case=with_sku`): arrived
  user-verified, so it is not cleaned or scored; its 100 titles feed the third sheet of the title
  workbook. A brand-token pass was applied on request: 19 titles carried BIRLA / BIRLANU / the
  `BN` prefix, fixed by deleting the token only (the code and the row's mapped SKU stay) → 19 rows
  rewritten, 81 byte-identical, 0 brand tokens, `02_cleaned/sku only tarun sentinel -
  corrected.csv`; `sku_only` sheet rebuilt from it. 100 distinct titles, **all 100 carry an 8-9 digit
  SKU code by design**, 81 distinct SKUs (12 SKUs on >1 title, 31 rows), all `MAPPED`, 62 invoices,
  48 `distributor_exclusive=true`, 1 row with `equivalent_skus`, HSN 75 `39174000` / 20 `39172390` /
  3 `35061000` / 2 `3917`. Three source quirks left as delivered: `wsku-047` carries 96002445 in the title
  but is mapped to 98000441 (the only code-vs-SKU mismatch in the lane), and `wsku-048` /
  `wsku-062` repeat their code and phrase twice in the title.
- **The brand and title-only files are not the same dataset**: 0 shared ids, 0 shared titles, but 20
  shared mapped SKUs (~23%) and 32 shared invoices — disjoint lanes cut from one invoice pool.
- **`status`/`sku` in these files are the expected side, not results** (brand 100/100 MAPPED against
  10 recorded fails; title-only 78 MAPPED + 22 MBNF against 10 fails). The 300-item results file
  stays unreconciled with `birlanu-eval-deck/code/numbers.json`; its 20 failing ids now resolve to a
  title + expected SKU, but no mapper output exists on disk to score them.
- **Lane labels do not match the recorded bucket rule** (`build_datasets.py`: code → sku, brand
  token → brand, else title-only): the brand lane splits 20 code / 80 brand, the title-only lane
  0 code / 54 brand / 46 title-only. Settled with the user as "the lane must carry no brand".
- `plain-036` (`… Pushfit S/S - B`) and `plain-066` (`BN UPVC PIPES …`) were the two brand
  abbreviations left after the 54 replacements; both were resolved 2026-10-23 by removing the token
  only (`apply_titleonly_token_fix.py`) — SKU, status and invoice untouched — so the lane is now
  100/100 brand-free and 100/100 code-free.
- Still open: 8 duplicate replacement SKUs; 2 same-HSN fallbacks (`plain-080`, `plain-092`).

---

## 8. Turnover OCR setups — cu-read text-only arm verified (2026-10-30)

Benchmark folder: `F:\Product\zotok\Product Mapping\CU READ EVAL OUTPUT\` — the LangSmith export
`cu-read-text-only-856a0156.csv` plus the generated report `turnover_ocr_setups.html` (the folder
README is the file guide, with findings and rules). The report compares 5 setups on the same 20
invoices (10 photos from prod `turnover_schemes` traces, 10 PDFs from `turnover-extraction-eval`,
299 lines), 2 runs each, gpt-5.4 on the prod prompt `v8.3.5-1081-gd7fbdc1b8af`, `InvoiceData`
through `extract_data_node`.

Verified: the CSV is the export behind the report's `read_only` column (cu-read `prebuilt-read`,
text only, no file) and reproduces it exactly — eval .694, header .939, GSTIN 74/80, line-item
fields 466/3,588 wrong (.870), photos .652 / PDFs .735, $0.0562 per run.

- **The photo loss is row binding, not characters.** `prod-image-08` returns 11 rows for 10, first
  row name-only, every number one row down (.469; the cu-layout-only arm scores 1.000 on the same
  photo); `prod-image-02` the same (9 for 8, .500 vs 1.000); `prod-image-07` binds amounts to the
  wrong rows (row 0's own amount 1,067,850 lands on row 2; row 0 carries 534,925, absent from the
  reference).
- **GSTIN 74/80: all six misses are two photos** — `prod-image-05` (buyer null; distributor
  `03BQYPT5080J1Z3` read as `03BQYPTS080J1Z3`) and `prod-image-08` (buyer null). PDFs are 40/40.
- **Four invoices are pinned by label convention, not extraction**: `pdf-03` / `pdf-05` / `pdf-07`
  at ≤.5 in every setup, `prod-image-01` at .5 in every setup. `pdf-05`'s 32 amounts, quantities
  and rates all match; only the incl/excl line-total column disagrees with the labels.
- **8 of the 22 header-field errors are formatting only** (phone spacing, `& electricals`, dropped
  "M/s"), so .939 understates alignment (~96.1% modulo format).
- **Setup ranking (eval, all 20):** cur .640 · cu-read text .694 · cu-layout text .754 ·
  cu-read + file .753 · cu-layout + file .783. On photos the layout text arm is .803 vs .652; on
  PDFs cu-read text .735 ≈ cu-layout .705 — the photo lane needs cu-layout, PDFs do not.

---

## 9. "Mapping issue list shared by Amit" — catalogue verification (2026-10-31)

Input: `F:\Product\zotok\Product Mapping\Mapping Inconsistency\01_source\Mapping issue list shared by amit.xlsx`
(created 29-Oct by Amit Kumar; three sheets: Distributor not mapped / retailer not mapped /
SKU not mapped). Catalogue = 18-Oct export (2,335 rows). Full analysis + per-row review workbook:
`Mapping_Inconsistency_Analysis.md` and `03_review/Mapping issue analysis.xlsx` in the same F: folder;
classifier `eval-datasets/code/classify_mapping_issues.py`.

Sheet profile: Distributor 69 rows / 41 names / 73 Inv / ₹29.6 L; Retailer 517 / 363 / 519 /
₹2.58 Cr; SKU 3,216 / 2,913 descriptions / 4,775 Inv / ₹2.48 Cr. 30 of 62 distributor invoices
also appear in the retailer sheet — three join stages of the same secondary-invoice population.

**SKU sheet — the list is not a catalogue-absence problem.** Component-based bucketing
(family → type → size → qualifier) gives: exact catalogue title 920 rows / ₹80.3 L (A) +
exact title but catalogue carries two SKUs 100 / ₹42.2 L (A2) + one SKU after normalisation
319 / ₹14.7 L (B). Exact title = 31.7% of rows / 49.5% of value; uniquely resolvable today
(A+B) = 38.5% / 38.4%; if the 19 duplicate pairs are retired the in-catalogue answer rises to
41.6% / 55.4%. Then 1,122 rows / ₹72.2 L (34.9% / 29.2%) are catalogue products with
2+ fitting SKUs because the invoice omits SDR/SCH/CLASS/length/Type A-B (C); qualifier/size/
unparsed gaps 592 rows / ₹29.6 L (18.4% / 11.9%); other brands 163 / ₹8.7 L (5.1% / 3.5%).
Family split: SWR ₹83.3 L (78% exact — a duplicate-SKU story), CPVC ₹72.8 L (51% ambiguous —
SDR/length), UPVC ₹36.3 L (55.5% exact), Pressure ₹16.2 L.

- **19 of the catalogue's 49 duplicate-SKU pairs block issue rows** (old 9600 vs new 9800/9800-10xx), 100 issue
  rows, ₹42.2 L (17.0%). Top pair: `SWR Pipe Type A Pushfit S/S 3m 110 mm` = 96000676 | 98000874,
  22 rows / ₹24.4 L. Even an exact title returns two answers.
- **554 issue rows (21.7% of value) appear verbatim in the human mapping corpus** and every one
  carries a SKU still in the current catalogue — these are not new products.
- **Trace 01a15db1 (Vaibhav GST-492, 18 lines):** `fetch_distributor_catalog` returned `[]` for
  all 18; fallback `fetch_catalog_text` mapped 17/18; turnover invoice persisted 16 MAPPED /
  1 MBNF / 1 UNMAPPED. The source invoice line schema has `status` but **no `sku_code` /
  `product_variant_id`** — the SKU lives only on the turnover line. Of the 18 descriptions, only
  the two **MAPPED_BRAND_NOT_FOUND** lines appear in Amit's workbook; the 15 MAPPED lines do not.
  Leading hypothesis: the "SKU not mapped" tab is a status/report artifact (MBNF or a source-table
  join), not a mapping gap. Confirm with one source-vs-turnover SQL sample (§7 of the analysis doc)
  before changing the pipeline.

  **Update 2026-11-01 — see §10.** The Turn Over Report answers the artifact half of this at line
  level: `Mapped - Branding Not Found` lines carry a `Material No`, so any tab filtered on
  `status <> 'MAPPED'` reports 15,202 lines as "not mapped" of which 6,024 actually have a SKU. Amit's
  exact filter still does not reconcile with the report (his 3,216 rows / ₹2.48 Cr vs the closest
  definition's 2,948 / ₹1.98 Cr), so the remaining question is his export filter, not the pipeline.

**Distributor sheet:** 49 of 69 rows belong to distributors that also sell in the retailer sheet —
link/identity problem, not missing party. Root causes: invoice-to-master link 14 rows / ₹10.9 L
(36.8%); entity resolution (7 name-variant clusters, same GST under 2 names, same name under 2-5
GSTs) 37 / ₹10.7 L (36.2%); not in master 14 / ₹6.1 L (20.6%); GST missing/invalid 4 / ₹1.9 L.
Trace shows the duplicate-CFA pattern: `fetch_cfa` returned two Vaibhav seller records
(`366a06fa…` code 0001121653 and `ff09993d…` code 1121653); the catalog lookup used the first and
returned empty.

**Retailer sheet:** 90.7% of rows carry a GST (241 GST-only, 178 GST+phone) and 43.7% a phone;
only 50 rows (₹49.7 L, 19.3% of value) have neither. Root causes: not in customer master / phone-
only match 215 / ₹1.09 Cr (42.0%); both identifiers but not in master 159 / ₹54.2 L (21.0%); no
key 50 / ₹49.7 L (19.3%); phone only 40 / ₹25.4 L (9.8%); name variants 53 / ₹20.5 L (7.9%).
106 retailers = 80% of value → bulk onboarding list, not a long tail.

**Actions (P0):** (1) report from the turnover line, show `sku_code`, split MAPPED / MBNF /
UNMAPPED — do not label MBNF "not mapped"; (2) retire/hide one of each of the 19 duplicate SKUs;
(3) tag other-brand lines `NOT_BIRLANU`. **P1:** normalisation/alias layer (mm↔inch, `Pipe 40`→
SCH 40, kg↔CLASS, coupler↔socket, typos/price/pack), price/history variant resolution with
`AMBIGUOUS` status, master-data canonicalisation + duplicate-CFA merge + top-11 distributor /
106 retailer onboarding. **P2:** one name = one SKU, missing variants, full `412xxx` export,
regression set (554 corpus + 100 exact + 19 pairs).

---

## 10. Turn Over Report — mapping-truth baseline (2026-11-01)

Input: `F:\Product\zotok\Product Mapping\Mapping Inconsistency\01_source\Turn Over Report.xlsx`
(downloaded 2026-11-01 15:00; sheet `Secondary Invoice Report`; filters 2026-01-01 → 2026-11-01).
**Superseded on disk by the 4-Nov re-export — see §14; this section is the 1-Nov snapshot.**
The user designated this file as the source of truth for how many distributors, retailers and products
are mapped, with **retailer/distributor as the top priority and products second**.

Shape: 53,187 line rows · 3,135 invoice numbers (3,384 invoice+distributor+date keys — numbers repeat
across pages and sellers) · ₹19.80 Cr excluding tax · 57 columns carrying `Distributor Status`,
`Retailer Status`, `Parsing Status` and the mapped `Material No`. Headline population deduplicates
9,749 lines flagged `Duplicate Invoice` → 43,438 lines, ₹15.97 Cr.

| Axis | Rule | Lines mapped | Value mapped | Entities fully mapped |
|---|---|---:|---:|---:|
| Retailer | `Retailer Status = Identified` | 38,699 / 43,438 (89.1%) | ₹14.23 Cr (89.1%) | 964 / 1,233 (78.2%) |
| Distributor | `Distributor Status = Identified` | 41,787 / 43,438 (96.2%) | ₹14.86 Cr (93.0%) | 271 / 332 (81.6%) |
| Product | `Parsing Status ∈ {Mapped, MBNF}` | 34,190 / 43,438 (78.7%) | ₹12.55 Cr (78.6%) | 9,350 / 14,148 titles (66.1%) |

- **Retailer failures are master coverage, not identifier capture:** 207 of the 1,010 retailers carrying
  a valid GST are not fully matched; 44 give neither GST nor phone. Failures cluster — 269 retailers
  carry an unidentified line and **78** make 80% of that ₹1.74 Cr gap.
- **Distributors:** 61 accounts carry every unidentified line and **9** make 80% of the ₹1.11 Cr gap —
  but **₹68.59 L of it (61.6%) sits on BirlaNu's own six GSTINs** (formerly HIL Limited), i.e.
  primary-channel records that should be resolved by rule, not onboarded as distributors. Real
  distributors account for ₹42.85 L. The cut is computed on the value sitting on unidentified lines,
  not the account's total: ranking on total value put big accounts first and read ₹2.97 Cr; ranking on
  the gap gives ₹1.11 Cr and moves the cut from 10 accounts to 9.
- **Products:** Mapped 28,166 · Mapped-Branding-Not-Found 6,024 · Unmapped 9,178 · blank 70. Of the 5,165
  unmapped description+distributor rows, 42.1% are catalogue products (A/A2/B/C — 1,607 ambiguous),
  57.9% are size gaps, qualifier conflicts, competitor brands or unparsed text; 45.7% of unmapped lines
  carry a brand token (BIRLA / BIRLANU / HIL / BN).
- **Reconciliation with Amit's list:** no definition reproduces all three of his numbers. Closest is
  MBNF / desc+distributor / created ≥ 2026-08-01 → 2,948 rows (his 3,216), 4,580 Inv (4,775),
  ₹1.98 Cr (₹2.48 Cr). His `Value` column is not a line amount in this report (that slice reads
  ₹1.98 Cr excl-tax, ₹37.17 L incl-tax), so his export filter and Value definition need confirming
  before his tabs are treated as a KPI.

Deliverables: `03_review/Truth - mapping coverage.xlsx` (11 sheets: Read me, Headline, Distributor
coverage, Retailer coverage, Onboarding 80%, Product coverage, Unmapped buckets, Unmapped titles,
Anomalies, Reconciliation vs Amit, Count reconciliation), handover pack
`03_review/Onboarding - retailers and distributors.xlsx` (269 retailers / 61 distributors with their
identifiers, 80%-of-value cut flagged), readout `TRUTH-MAPPING-BASELINE.md`, builder
`eval-datasets/code/build_turnover_truth.py` (cache in `eval-datasets/outputs/turnover-truth/`), render
QA `eval-datasets/code/qa_render.py`, plan `PLAN-turnover-report-truth-mapping.md`. Verified by an
independent raw-openpyxl recount (identical on every axis: 53,187 rows, 9,749 duplicates, 43,438
deduped, parsing 28,166/9,178/6,024/70, entity keys 332/1,233), a deterministic rebuild (summary.json
identical minus its timestamp), and a Playwright render check (0 cells clipped on the presentation
sheets and on the onboarding pack).

**File state (2026-11-01 16:26 UTC):** swapped — the F: copy of `Truth - mapping coverage.xlsx` is now
the gap-ranked build (md5 `8e323ebf…`, same bytes as the staged copy in
`eval-datasets/outputs/turnover-truth/`). The onboarding pack and the markdown were already current. The
earlier `Permission denied` was Excel holding a `~$` lock; close the workbook before any rebuild.

### Amit's issue list at account level (2026-11-01)

His tabs are invoice-level rows, so they are not comparable to the entity counts above until they are
collapsed to parties (GST where he gives one, else normalised name): **385 retailers, 46 distributors**.
Everything below is counts, not value.

`Distributor Status` / `Retailer Status` is decided **per invoice** and stamped on every line of that
invoice — not per line item. Of 3,299 invoice+distributor groups, 3,287 carry a single status; the 12
exceptions are the same invoice present twice with the status split exactly in half (5/5, 4/4, 12/12),
i.e. a duplicate ingest where one copy was mapped and the other was not. "Partly mapped" below therefore
means *some invoices of that party are identified and others are not*, and the two blocks of the table are
the two reporting conventions: any-unidentified-invoice (used by the workbook) and strict (partial folded
into mapped).

| | retailers | distributors |
|---|---:|---:|
| not fully mapped in the report (any unidentified invoice) | 269 | 61 |
| … named somewhere in his list | 227 | 38 |
| … missing from his list | 42 | 23 |
| strictly unmapped (partial counted as mapped) | 252 | 45 |
| … named somewhere in his list | 217 | 26 |
| … missing from his list | 35 | 19 |
| parties in his list | 385 | 46 |
| … wholly unmapped | 214 | 22 |
| … partly mapped | 8 | 10 |
| … fully mapped already | 124 | 6 |
| … not in the report at all | 39 | 8 |
| … with at least one unmapped row | 222 | 32 |

Two consequences. First, his list is not a subset of the report: 39 retailer parties and 8 distributor
parties in it do not exist in the 01-Nov report, and 124 + 6 are already fully mapped — so the difference
between his 29-Oct list and this report **cannot** be read as a mapping rate over three days (it is 32%
of his retailers and 13% of his distributors mapped, not ~50%). Measuring that needs the report exported
on 29-Oct too. Second, "unmapped" for him means *any* bad invoice: 10 of his 46 distributors and 8 of his
385 retailers have a mix of resolved and unresolved invoices, which is why his 22 wholly-unmapped
distributors and his 32 rows are different numbers.

Reproduce: `eval-datasets/code/amit_overlap.py` (reads the same cache) → `outputs/turnover-truth/amit-overlap.json`.

**Status annotation (2026-11-02):** `03_review/Mapping issue list shared by amit - unique parties.xlsx`
is the unique party view of his list — his 69 distributor / 517 retailer invoice rows collapse by
normalised name to **40 distributors / 358 retailers**, with every GST kept in one cell — each carrying
its report status, report party, report GST and match key. Matching inside the invoice is GST, then
GST-near + name, then phone, then name / fuzzy; every non-exact match is flagged in `Matched on`.
Party status: distributors 19 not mapped / 11 partly / 10 fully, retailers 196 / 16 / 144 / 2 no match
(the exact-key GST-party table above is the strict view). A `Distributor master check` sheet carries the
user's 02-Nov manual verification of the 45 report-unmapped distributors: 34 not in the master, 1 in the
master but unmapped (Jeevish Sales — its report lines are split across two typo'd GSTs that were matched
via SAP 1118617), 3 BirlaNu own GSTINs, 7 with no GST. Builder
`eval-datasets/code/annotate_amit_status.py`.

## 11. Presence check in catalogue / workspace — Amit sheet (2026-11-02)

Input: `01_source/Mapping issues - presence check in catalogue.xlsx` — Amit's three tabs with the
platform's own presence flags: `In Workspace` (distributor / retailer, with the workspace identity
columns) and `In Catalog` + `Detail` (SKU). Bucketed by `eval-datasets/code/presence_check_summary.py`
→ `eval-datasets/outputs/turnover-truth/presence-check.json`; readout `03_review/Presence check - brief.md`.

**Products — 3,216 rows / 2,455 distinct descriptions**

| In Catalog | rows | descriptions | Detail |
|---|---:|---:|---|
| Yes | 2,611 | 1,903 | different wording 1,115 / same name 849 / product type present but invoice lacks detail 647 |
| No | 605 | 564 | not in catalog 383 / 355 · competitor brand 222 / 209 |

The genuine BirlaNu catalogue gap is **383 rows / 355 descriptions** (11.9% of the list); the 222
competitor-brand rows are not missing catalogue. Of the 2,611 that exist, 647 are ambiguous variant —
the product type is in the catalogue, the invoice does not state SDR/class/length — and 1,115 only
differ in wording. The product problem is variant detail and vocabulary, not missing products.

**Parties, crossed with the unique-party status (§10)**

| Side | unique | not mapped | present but unmapped | absent → onboarding |
|---|---:|---:|---:|---:|
| Distributors | 40 | 19 | 3 | 16 |
| Retailers | 358 | 196 | 51 | 144 |

- The **3 distributors** are GST-matched to the workspace: Sri Lakshmi Electrical & Hardware, Jeevish
  Sales Corporation, Nepolite Concrete Solutions — present on the platform, never mapped.
- The **51 retailers** are exact-name (50) or phone (1) matches. Those workspace records carry no GST
  and one is a clearly wrong name match (Tatsam Enterprises → Manish Choudhary), so 51 is a candidate
  list, not a verified count. They need identity confirmation before being treated as a matching fix.
- The other **16 distributors / 144 retailers** are absent from the workspace — master onboarding, not
  a matching fix. This is the queue the 02-Nov manual master check (34 not in master) points at.
- **Presence-check noise:** 6 distributors and 12 retailers that the report has fully mapped read `No`
  because the check joins on the GST in his sheet and his GST has typos (Kamal Sanitary, Future Trade
  Com, Jagdamba, Maa Sherawalis, S.K. Enterprises, Satyam Home Decore; 12 retailers). So its raw No
  counts (23 distributors / 160 retailers) overstate absence; the intersection with our status above
  is the defensible number.

---

## 12. Platform status note checked (2026-11-02)

The ZoTok side circulated a summary — 69 distributors / 518 retailers / 3,217 products, of which
9 / 98 / 21 are "marked unmapped", and all 9 distributors are already mapped on the invoice list
page. Checked line by line against the 1-Nov report and the presence-check sheet with
`eval-datasets/code/verify_platform_note.py`.

| Claim | Verdict | What the sources say |
|---|---|---|
| 69 distributor records | correct | 69 rows; 40 unique names |
| 36 present in catalogue | correct as rows | 16 names all-present + 1 mixed; 23 absent; "catalogue" = workspace |
| 9 marked unmapped | not reproducible | report has only Identified / Not Identified; strict join: 32/69 rows, 11/36 present rows, 3 names wholly unmapped, 10 distinct invoices; closest cut is 8 invoices all-unmapped |
| all 9 already mapped on the invoice page | unverifiable here | if true, the discrepancy is 3 parties / 11 rows, not 9 parties |
| 518 retailer records | wrong | 517 data rows; 518 is the tab's `max_row` including the header |
| 295 present in Customer Master | correct as rows | 196 names all-present + 2 mixed; 160 absent |
| 98 marked unmapped | not reproducible | strict join: 280/517 rows, 89/295 present rows, 51 names present, 85 present invoices; closest invoice cut 91 |
| 3,217 product descriptions | wrong | 3,216 line rows; only 2,359 unique descriptions (2,455 with the looser normaliser) — rows are not descriptions |
| 2,611 present in Product Master | correct as rows | 1,903 unique descriptions |
| 21 marked unmapped | roughly right | 22 unique descriptions present in the master have every line Unmapped; 236 have at least one Unmapped line |
| "only 21 products flagged" | misleading | 605 rows / 564 descriptions are not in the master at all (383 genuine gap + 222 competitor brand); 2,240 descriptions carry MBNF |

The note's three presence counts tie exactly to the presence sheet at row level. Its record counts are
row counts and two of them include the header. Its unmapped counts are not party counts — the
party-level residue is **3 distributors / 51 retailers**, and 21 is only the present-in-master-and-
wholly-Unmapped product slice.

---

## 13. Turnover DB DDL — where mapping and report truth live (2026-11-03)

Input: `F:\reports\Dbeaver\turnoverinvoice.sql` (full `public` schema, 80,946 bytes). Full analysis:
`~/AgentWork/vault/zotok/db-schemas/turnover-invoice-ddl-analysis.md`; table inventory:
`turnover-invoice-schema.md`.

- **Access:** sqlit 1.6.5 → `127.0.0.1:5434` (SSM port-forward) → Postgres; setup/troubleshooting in
  `~/AgentWork/vault/zotok/db-schemas/sqlit-ssm-connection-runbook.md`. The connection is verified
  (2026-11-03); the findings below remain DDL-derived until row-level checks are run.
- 32 tables: 19 business/domain + 11 LangGraph runtime + 2 infra. The product master is **not** here;
  it lives in commerceservice (`product_variant`, `ProductVariantAttributes`, `product_mapping`,
  `POFileLine`).
- Product-mapping result is on `turnoverinvoice_line`: `product_name` (raw),
  `sku_code`/`product_variant_id`/`mapped_product_name` (mapped), `product_match_context` (JSON evidence),
  `mapping_confidence_score`, `status`, `issues`. The source invoice line has `status` but no SKU; the
  turnover line is authoritative for the mapped SKU.
- Distributor-specific catalog path: `turnoverinvoice.cfa_id` → commerceservice
  `product_mapping.cfaId` (+ `channelId`). Empty `product_mapping` for the CFA explains
  `fetch_distributor_catalog` returning `[]`; it is a coverage/CFA identity issue, not necessarily a
  product-name issue.
- Turnover report: `Parsing Status` → `turnoverinvoice_line.status`; `Material No` → likely
  `sku_code`; `Distributor/Retailer Status` → likely `turnoverinvoice_details.matched_*` (there is no
  explicit status column); `Invoice Status` → `turnoverinvoice.status`; amounts → line
  `total_amount`/`taxable_value` vs header `invoice_amount` (header discounts/round-off/freight explain
  residuals); duplicate-invoice dedupe is a report rule, not a DB constraint.
- No FKs on the line-scheme bridge, `turnover_scheme_actions.scheme_id`, `turnover_customer.invoice_id`,
  or the history tables. `turnover_customer.invoice_id` is varchar vs uuid; workspace IDs are
  uuid/varchar/text across tables. These are the join hazards to control.
- Status: DDL-only findings; the sqlit/SSM connection is verified, but exact status strings, the
  `Material No` column, JSON shapes, and the report formula still need one live sample.

---

## 14. Report refresh (4-Nov export) and the 51-retailer matching-fix queue (2026-11-04)

Input: the same `01_source/Turn Over Report.xlsx` path, re-exported 2026-11-04 06:07 with filters
2026-01-01 → 2026-11-04. Nothing had processed it; the numbers in §10 and `TRUTH-MAPPING-BASELINE.md`
describe the 1-Nov snapshot.

Shape: 54,903 line rows (was 53,187) · 3,431 invoice+date keys · 10,005 Duplicate-Invoice lines →
44,898 deduplicated (was 43,438).

| Axis | 1-Nov | 4-Nov |
|---|---|---|
| Retailer lines identified | 38,699 / 43,438 (89.1%) | 40,092 / 44,898 (89.3%) |
| Distributor lines identified | 41,787 / 43,438 (96.2%) | 43,787 / 44,898 (97.5%) |
| Product lines (Mapped / MBNF / Unmapped / NA) | 28,166 / 6,024 / 9,178 / 70 | 28,834 / 6,337 / 9,657 / 70 |
| Retailer entities | 1,233 = 964 full / 17 partial / 252 none | 1,271 = 992 / 28 / 251 |

Retailer entities were recomputed with the §10 identity rule (GST → phone → name+pin) and the dedup
population; distributor and product ENTITY counts were not recomputed, so quote the line figures only.

**Export-format defect.** The 4-Nov file adds a leading `Column1…` row, so the header moved from row 3
to row 4. `build_turnover_truth.load_report`'s fixed `header=2` read the filter line as the header and
silently cached a garbage frame (every downstream join then returned "no match"). The loader now sniffs
the header row (first cell == `Invoice Number`) and was re-tested on the 4-Nov file: 54,903 × 57. The
1-Nov cache (`turnover-truth-2026-11-01.csv.gz`) remains valid.

**Amit's retailer tab against the 4-Nov report** (party grain, reusing `annotate_amit_status.py`):
517 rows / 474 invoices / 363 spellings → 358 parties; status 194 not mapped · 17 partly · 145 fully ·
2 no party match (1-Nov: 196 / 16 / 144 / 2). Presence: **198 parties in the master, 160 absent** —
295 is the ROW count inside 517 (222 rows read No). The cross:

| | present | absent |
|---|---:|---:|
| not mapped | 51 | 143 |
| partly mapped | 13 | 4 |
| fully mapped | 133 | 12 |
| no party match | 1 | 1 |

Row grain inside the 295 present rows: 200 mapped · 3 partly · 89 not mapped · 3 no party match.
(19 rows flagged `No` by the presence check are mapped in the report — the known GST-typo false
negatives.) "358 → 295 → 51" therefore mixes grains: it is 358 parties → 198 present → 51 unmapped.

**Matching-fix queue.** `03_review/Retailers present but unmapped - 51.xlsx` + `.csv`: one row per
party with his GST/phone, the workspace record the presence check returned, its match basis (49 exact
name · 2 phone · 0 GST) and `GST relation` (6 near-typo ≤2 chars · 11 different GST · 34 no GST on the
workspace record). Builder `eval-datasets/code/build_master_match_queue.py`; renders clean (0 clipped
cells). Read: only the 6 are clean GST-typo fixes; the 11 need identity confirmation because the exact
name match carries a different GST; the 34 can only be matched on name/phone.

**Name cross-check, held back:** against the user's 63-name list, 46 entries match the 51. The six of
the 51 not on the list are ASHOK ENTERPRISES, BALA JI ENTERPRISES, MAHADEV ELECTRICALS, RAJESHWAR
ELECTRICALS, SHIVAM ENTERPRISE and one party whose name is literally `NA` (GST 29BPWPNG683Q1ZA). Most
of the list entries outside the 51 are already fully mapped in the report (Riddhi Siddhi 224/224,
Shiv Tara 109/109, BDC PROJECT 278/278; Kishan Traders Thana Chappur 57/71 partly), so the list reads as
a master list rather than an unmapped list.

Still stale, deliberately: the delivered `Mapping issue list shared by amit - unique parties.xlsx` and
`Truth - mapping coverage.xlsx` carry the 1-Nov statuses — the 4-Nov run wrote to `/tmp` for this check.
Refresh them when the client needs current status columns.

---

## 15. Amit's unique products vs the 4-Nov report (2026-11-05)

Input: the `SKU not mapped` tab of `01_source/Mapping issue list shared by amit.xlsx` — 3,216 data rows,
one description column. Report: the 4-Nov export already described in §14 (54,903 lines / 44,898 after
dropping duplicate-invoice rows).

**Uniqueness.** `cmi.norm_text` dedupes case, quote/inch spellings, unit spacing and the known typo
pairs (`pusfit`/`push fit`, `socket`/`soc`, `sofit`/`solfit`): 3,216 rows → 2,913 exact strings →
2,737 case/space → **2,359 unique products**. MBNF counts as mapped; product status is `Parsing
Status`, which is decided per line and distributor-specific, so "partly mapped" is a real state.

| Product status | dedup population | all rows as delivered |
|---|---:|---:|
| Fully mapped | 1,945 | 1,903 |
| Partially mapped | 350 | 400 |
| Never mapped | 64 | 56 |
| total | 2,359 | 2,359 |

**Answer at product grain: 56 of the 2,359 unique products (2.4%) never appear on a mapped line
anywhere in the 4-Nov report; 400 are partly mapped and 1,903 fully mapped.** On the deduplicated
population (the §10 baseline convention) the split is 64 / 350 / 1,945. If the question is instead
"still carries at least one Unmapped line", the count is **456** (414 dedup) — that is the wider
"not clean" set, not "unmapped".

**Why the two populations differ.** The export carries 10,005 duplicate-invoice lines whose `Parsing
Status` can differ from the non-duplicate copy: 11 of the dedup never-mapped 64 have a mapped duplicate
line, and 32 products appear only on duplicate rows. **No product on his list is absent from the report
entirely** — all 2,359 appear somewhere.

**The never-mapped set** (dedup: 64 products / 107 lines / 57 invoices / 32 distributors / ₹4.51 L).
Four carry a clear competitor brand and are unmapped by design — NEROLAC WALL PUTTY _20K, ROTO TANK
1000 LITRE 3PLY, 20MM UPVC UNION SUPREME, ASTRAL UPVC STEP OVER BEND 40SCH 1 1/2". The rest are
BirlaNu descriptions with wording/qualifier defects; largest: `Cpvc Pipe 11/2 13.5 SDR` (13 lines /
7 invoices), the U-Pipe SCH40 sizes 1/2"–1 1/4" (13 lines), `CPVC9600037- CPVC ELBOW 90° 3/4` (4
lines), `CPVC Elbow 45D 40mm - B` (4 lines).

**Repro:** `eval-datasets/code/amit_products_vs_report.py` (default paths, `--refresh` to re-read the
xlsx); per-product CSV `outputs/turnover-truth/amit-products-vs-report-2026-11-04.csv`. Caveat: exact
join after normalisation — a product reworded beyond the normaliser would read never-mapped, but since
all 2,359 appear in the report that class is empty; 70 report lines carry no parsing status, and MBNF
is counted as mapped.

---

## Related AP refs
AP-23, AP-25 (SKU-in-description-not-mapped) — directly demonstrated by line 6.
AP-15 (invoice "not found" vs success caption) — also visible in this trace.
AP-09/AP-14 (retailer mapped by GSTIN; PAN fallback not exercised).
AP-01 (encoding bug: "UPVC ELBO 90Â°" mojibake in mapped names).
AP-30 (multi-page upload: user doesn't reply "no" to continue prompt) — see §4.
AP-06 (multi-page merge / duplicate processing) — the merge invariant the AP-30
fix must preserve; AP-30 is its inverse failure.
AP-35 (duplicate invoices surfaced to a retailer) — downstream symptom fed by
mis-batched multi-invoice uploads. Mitigated per §5 (original-invoice link now reported shipped).
AP-23, AP-25 (SKU-in-description) — §6: 200/899 turnover lines carried an embedded code and all
mapped (clean codes work; §2's failure was the irregular `code96000872` extraction).
