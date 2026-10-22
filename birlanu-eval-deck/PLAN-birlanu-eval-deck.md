# Plan — BirlaNu product-mapping eval deck (4 slides)

**Status: Proposed — NOT approved.** (2026-10-22, third cut)

Scope: 4 slides — cover, data provenance, test method, findings. No recommendations.
No "why the misses happen" section. No mention of the Excel workbook, VLOOKUP, Power
Query or any manual-verification file: the deck describes how the check was run, not
the tooling it was run in.

All figures supplied by the user. The deck is authored number-agnostic with named
slots; nothing is quantified until the numbers arrive.

Audience: BirlaNu team. Deliverable: 4-slide 16:9 PDF, HTML→Playwright, house
"Quiet Ledger" style inherited from `birlanu-cadence-deck/`.

Working folder: `~/AgentWork/product-mapping/birlanu-eval-deck/`.

---

## Number slots (user supplies)

| Slot | Meaning | Used on |
|---|---|---|
| `INPUTS` | unique input products from the eval run | 1, 2, 4 |
| `CATALOGUE` | products in the catalogue export | 2, 4 |
| `CATALOGUE_DATE` | date of the catalogue export | 2, 4 |
| `FUZZY` | located on name similarity alone | 4 |
| `CODE` | located by embedded SKU code | 4 |
| `LOCATED` | total located = FUZZY + CODE | 1, 4 |
| `OPEN` | not located = INPUTS − LOCATED | 4 |

Derived percentages are computed from these and labelled as arithmetic.
`LOCATED = FUZZY + CODE` and `OPEN = INPUTS − LOCATED` are asserted before the build;
a mismatch stops it.

---

## Slide map

**1 — Cover**
Kicker: BIRLANU · PRODUCT MAPPING · OCTOBER 2026
Title: "Does the catalogue hold what the invoices describe?" (serif italic on *hold*)
Sub: Input-product coverage against the catalogue export.
Figures: `INPUTS` input products · `CATALOGUE` catalogue products · `LOCATED` located

**2 — Where the data came from**
Headline: "Two exports, one question."
- Input side: the eval run's product titles — `INPUTS` unique products, exactly as the mapping agent receives them (name text only; no codes added).
- Catalogue side: the product master export dated `CATALOGUE_DATE` — `CATALOGUE` products, SKU and product name.
- Both are exports, not live look-ups: the count reflects the catalogue as it stood on that date.
Layout: two-column source table — source | rows | what was used.

**3 — How we tested**
Headline: "One title at a time, against the whole catalogue."
- Both sides cleaned identically — whitespace, encoding, duplicates — so names compare like-for-like.
- Pass 1: every input title matched on name against all `CATALOGUE` catalogue names.
- Pass 2: any title carrying an embedded SKU code re-checked against that SKU.
- Each candidate match then checked individually; no row counted automatically.
- A row counts as located only when it is the same item, not merely a similar name.
Layout: numbered method rows, mono index.

**4 — What we found**
Headline: "`LOCATED` of `INPUTS` input products exist in the catalogue."
- `FUZZY` located on name alone.
- `CODE` located by embedded SKU code.
- `OPEN` not located.
- Coverage rate computed from the two figures above and labelled as arithmetic.
Layout: count strip + mark matrix of `INPUTS` with `LOCATED` filled; export date in the footer.

Deck ends on the finding. No recommendations, no next steps, no causes.

---

## Build phases (after numbers + approval)

1. `code/copy.md` — this map with numbers substituted; user approves wording.
2. `code/deck_template.html` + `code/build_deck.py` — inject values, render 1920×1080 PDF.
3. QA — `pdfinfo` (4 pages, 1440×810 pts); slot arithmetic assertions; `pdftotext`
   round-trip (every supplied number present, none invented); zero hits for
   `Excel`, `VLOOKUP`, `Power Query`, `workbook`, `LangSmith`, run ids, revision, cost;
   per-slide pixel overflow check.
4. Delivery — PDF in `outputs/`, copy beside the source CSVs, README + PROJECTS.md
   pointer, commit locally (push only on request).

## Out of scope

- Recommendations, next steps, or causes of misses.
- The 899-line accuracy study, confidence bands, failure families.
- The manual-verification workbook and its verdict column.
- Any live-API catalogue check.
