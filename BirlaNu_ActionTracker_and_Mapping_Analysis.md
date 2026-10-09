# BirlaNu — Action Tracker & Invoice Mapping Analysis

Last updated: 2026-10-09
Context: BirlaNu (seller "Birlanu Limited") invoice digitisation — action tracker review + a live mapping-pipeline trace post-mortem.

## Source artifacts
- Action Tracker (PDF, exported from the live workbook):
  `/mnt/f/Product/zotok/Invoice/Fake Invoices/20260714_BirlaNu_Action_Tracker_ZOTOK.pdf`
- Cadence-meet deck (6-slide PDF, built 2026-10-09):
  `/home/sumit/AgentWork/product-mapping/birlanu-cadence-deck/outputs/BirlaNu-Cadence-Deck.pdf`
  (source + render/QA scripts + README in the same folder; see §5.)
- Live pipeline trace (intent `scheme_invoice`, graph `turnover_schemes`, node `map_catalog_products`, model gpt-5.4-mini):
  invoice 495/26-27, distributor MAA SHERAWALIS MATRABHOOMI PIPES (GST 27AFPPJ3948E1Z6),
  buyer RAMDEV HARDWARE & ELECTRICALS (GST 27CEXPR0237R1ZF), dated 2026-08-23.

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

Built a 6-slide cadence-meet deck for the BirlaNu team (canvas-design philosophy + claude-design
deck rules, house style inherited from `zochief-demo-docs/sales-deck`, Playwright Chromium → PDF).

- **PDF:** `birlanu-cadence-deck/outputs/BirlaNu-Cadence-Deck.pdf` — 6 pages, 16:9 (1440×810 pt).
- **Slides:** cover · parse volume till date · August · approvals & issues · enhancements implemented · summary.
- **Figures reported for this cadence (supplied, not re-derived from the workbook):**
  2,323 invoices parsed till date · 30,063 line items till date · 957 parsed in August ·
  181 approved · 36 issues resolved. Only additions are arithmetic from those: 41.2% = 957/2,323,
  12.9 lines/invoice = 30,063/2,323, 7.8% = 181/2,323, 1,366 = 2,323−957.
- **Source of the numbers:** cadence input from the team — NOT read from the live workbook or
  dashboard; freshness/accuracy unverified (the tracker banner in §1 is a separate 2026-07-14 snapshot).
- **Commits (local only, not pushed):** `de9e5ac` initial 5-slide deck, `f907ca2` enhancements slide.
- **Rebuild/QA:** see `birlanu-cadence-deck/README.md` (`render_deck.py` + 4 QA scripts).

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
