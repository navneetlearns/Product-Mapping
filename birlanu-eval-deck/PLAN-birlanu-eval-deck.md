# Plan — BirlaNu product-mapping eval deck (LangSmith eval, 4 slides)

**Status: Proposed — NOT approved.** (2026-10-22, fourth cut)

Scope: the product-mapping eval run on LangSmith. NOT the local catalogue-check work
(cleaning, matching passes, verification files) — that stays out of the deck entirely.
No recommendations, no "why the misses happen" section, no workbook/tooling mentions.

All figures supplied by the user. Authored number-agnostic with named slots.

Audience: BirlaNu team. Deliverable: 4-slide 16:9 PDF, HTML→Playwright, house
"Quiet Ledger" style inherited from `birlanu-cadence-deck/`.

Working folder: `~/AgentWork/product-mapping/birlanu-eval-deck/`.

---

## Number slots (user supplies)

| Slot | Meaning | Used on |
|---|---|---|
| `RUN_DATE` | date the eval ran | 1, 4 |
| `SET_DESC` | what the eval set is (e.g. "71 invoices, 899 product lines" or "100 product titles") | 2, 4 |
| `LINES` | product lines (or cases) evaluated | 1, 2, 4 |
| `MAPPED` | lines the model mapped | 4 |
| `STATUS_OK` | lines whose status matched the recorded expected result | 1, 4 |
| `SKU_OK` | lines whose chosen SKU was correct | 4 |
| `FALSE_MAPS` | wrong mappings | 4 |
| `MISSED` | lines that should have mapped but did not | 4 |

Derived rates (status accuracy, SKU accuracy) computed from the counts and labelled as
arithmetic. Optional 5th slide if the breakdown numbers exist: counts per status.

---

## Slide map

**1 — Cover**
Kicker: BIRLANU · PRODUCT MAPPING EVAL · `RUN_DATE`
Title: "Mapping, measured." (serif italic accent)
Sub: The product-mapping agent scored against recorded results in the eval set.
Figures: `LINES` product lines evaluated · `STATUS_OK` status-accurate · `SKU_OK` SKU-accurate

**2 — What was tested**
Headline: "Every product line the agent was given, scored against its recorded result."
- The eval set: `SET_DESC`, each line a product title exactly as the agent receives it.
- Each line carries its expected result, recorded in the set before the run.
- One run, one revision, same conditions for every line; nothing adjusted after the fact.
Layout: source table — what it is | rows | what was used.

**3 — How the scoring works**
Headline: "Two questions per line: was the status right, and was the SKU right."
- Status: mapped / mapped-with-brand-not-found / unmapped — compared to the recorded result.
- SKU: the chosen product compared to the expected one, with equivalent catalogue items counted as correct.
- Every line scored the same way; totals come from the run's own evaluator output.
Layout: numbered method rows, mono index.

**4 — What we found**
Headline: "`STATUS_OK` of `LINES` lines status-accurate."
- `MAPPED` mapped · `FALSE_MAPS` wrongly mapped · `MISSED` should-have-mapped lines missed.
- SKU-accurate: `SKU_OK` of `LINES`.
- Rates computed from these counts, labelled as arithmetic.
Layout: count strip + horizontal bars; run date in the footer.

Deck ends on the finding. No recommendations, no next steps, no causes, no local tooling.

---

## Build phases (after numbers + approval)

1. `code/copy.md` — this map with numbers substituted; user approves wording.
2. `code/deck_template.html` + `code/build_deck.py` — inject values, render 1920×1080 PDF.
3. QA — `pdfinfo` (4 pages, 1440×810 pts); slot arithmetic assertions (STATUS_OK +
   FALSE_MAPS + MISSED = LINES where that identity holds for the supplied numbers);
   `pdftotext` round-trip (every supplied number present, none invented); zero hits for
   `Excel`, `VLOOKUP`, `Power Query`, `workbook`, `catalogue check`, cost figures;
   per-slide pixel overflow check.
4. Delivery — PDF in `outputs/`, README + PROJECTS.md pointer, commit locally (push only
   on request).

## Out of scope

- The local catalogue-coverage work (cleaning, matching passes, verification files).
- Recommendations, next steps, causes of misses.
- Any live-API checks.
