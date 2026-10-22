# Plan — BirlaNu product-mapping eval deck (5 slides)

**Status: Proposed — NOT approved.** (2026-10-22, revised from the 10-slide cut)

Scope: 5 slides. Catalogue-coverage story only (the 899-line accuracy study is out).
**All figures are supplied by the user** — the deck is authored number-agnostic, with
named slots, and nothing is quantified until those numbers arrive.

Audience: BirlaNu team. Deliverable: 5-slide 16:9 PDF, HTML→Playwright, house
"Quiet Ledger" style inherited from `birlanu-cadence-deck/`.

Working folder: `~/AgentWork/product-mapping/birlanu-eval-deck/`.

---

## Number slots (user supplies; nothing is invented)

| Slot | Meaning | Used on |
|---|---|---|
| `INPUTS` | unique input products from the eval run | 1, 2, 3, 5 |
| `CATALOGUE` | products in the current catalogue export | 2, 3 |
| `FUZZY` | located on name similarity alone | 3 |
| `CODE` | located by embedded SKU code (missed by fuzzy) | 3 |
| `LOCATED` | total located (should equal FUZZY + CODE) | 1, 3, 5 |
| `OPEN` | remaining open rows (should equal INPUTS − LOCATED) | 3, 5 |
| `EXPORT_SCOPE` | what SKU families the export covers (e.g. only 9xxx prefixes) | 5 |

Derived percentages (coverage rate, share by path) are computed from these and
labelled as arithmetic. `LOCATED = FUZZY + CODE` and `OPEN = INPUTS − LOCATED` are
asserted before the deck builds; a mismatch stops the build.

---

## Slide map

**1 — Cover**
Kicker: BIRLANU · PRODUCT MAPPING · OCTOBER 2026
Title: "Does the catalogue hold what the invoices describe?" (serif italic on *hold*)
Sub: Input-product coverage against the current catalogue export.
Figures: `INPUTS` input products · `CATALOGUE` catalogue products · `LOCATED` located
Layout: dark pine full-bleed, mono kicker, three figures in a band.

**2 — What we tested**
Headline: "Every product we sent in, checked against the catalogue."
- `INPUTS` unique products from the eval run — the titles the mapping agent actually receives.
- `CATALOGUE` products in the current catalogue export; both sides cleaned identically (whitespace, encoding, duplicates) so names compare like-for-like.
- Two passes: fuzzy name match, then an embedded-SKU rescue for titles carrying a code.
- Every call then verified by hand; nothing is auto-decided.
Takeaway line: this measures catalogue coverage, not the agent's mapping accuracy.
Layout: ledger rows, basis column.

**3 — What we found**
Headline: "`LOCATED` of `INPUTS` products exist in the catalogue."
- `FUZZY` matched on name similarity alone.
- `CODE` rescued by an embedded SKU code the fuzzy pass could not see.
- `OPEN` remain open, carried into the manual pass.
Takeaway line: the misses are not random — they follow two patterns.
Layout: count strip + a mark matrix of `INPUTS` with `LOCATED` filled.

**4 — Why the misses happen**
Headline: "The catalogue speaks a different dialect — and some titles leave a dimension open."
Left panel — vocabulary: COUPLER → SOCKET · SHOE → 45 Elbow · ROTTO → BIRLA HIL Water Tank · END CAP → "Plain end cap"; sizes 50 mm = 2", 110 mm = 4"; pipe sold by weight → Pressure Pipe CLASS-n.
Right panel — variant ambiguity: the title pins type and size but not series, length or colour, so several catalogue SKUs are equally valid on the evidence.
Takeaway line: most "not found" rows are translation problems, not missing products.
Layout: two panels, hairline between.

**5 — What we recommend**
Headline: "Put the vocabulary in front of the matcher, and close the open rows with your product team."
1. Synonym and size canon ahead of matching — the single biggest lever.
2. Brand-aware candidate retrieval — branded titles often arrive with unbranded candidate sets.
3. Confidence as a gate — mid-band scores get review, not auto-apply.
4. Catalogue export scope — `EXPORT_SCOPE`; items outside it read as absent.
5. Close the open rows — product-team verdict on the workbook, then re-run and report final coverage.
Close line: "Coverage as measured: `LOCATED` of `INPUTS`. Open: `OPEN`, each with a named lead candidate."
Layout: five numbered rows, mono index.

---

## Build phases (after numbers + approval)

1. `code/copy.md` — this slide map with the numbers substituted; user approves wording.
2. `code/deck_template.html` + `code/build_deck.py` — inject values, render 1920×1080 PDF.
3. QA — `pdfinfo` (5 pages, 1440×810 pts); slot arithmetic assertions; `pdftotext`
   round-trip (every supplied number present, none invented); identifier grep
   (LangSmith, run ids, revision, cost) returns zero; per-slide pixel overflow check;
   the open rows never described as resolved.
4. Delivery — PDF in `outputs/`, copy beside the workbooks on F:, README + PROJECTS.md
   pointer, commit locally (push only on request).

## Out of scope

- The 899-line accuracy study (71 invoices), confidence-band analysis, failure families.
- Filling the open rows' verdicts (manual by design).
- Any live-API catalogue check.
