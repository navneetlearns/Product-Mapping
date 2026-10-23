# BirlaNu — Action Tracker & Invoice Mapping Analysis

Last updated: 2026-10-23
Context: BirlaNu (seller "Birlanu Limited") invoice digitisation — action tracker review + a live mapping-pipeline trace post-mortem + LangSmith turnover-mapping eval analysis (§6) + the sentinel lane exports cleaned and corrected (§7).

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
- Open: `plain-036` (`… Pushfit S/S - B`) and `plain-066` (`BN UPVC PIPES …`) are suspected brand
  abbreviations, left as-is pending a decision; 8 duplicate replacement SKUs; 2 same-HSN fallbacks.

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
