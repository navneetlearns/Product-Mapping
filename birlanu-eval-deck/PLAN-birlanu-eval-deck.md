# Plan / decision record — BirlaNu mapping-eval deck

**Status: BUILT with numbers (22 Oct 2026).** Run setup settled; ready to share once the
stale source report is kept out of the client pack.

Deliverable: `outputs/BirlaNu-Mapping-Eval-Deck.pdf` — 5 slides, 16:9.

## What it says

Four mapping logics, each on its own 100-title set, hits shown as x / 100:

| Lane | Hits |
|---|---|
| SKU-based (code in the title) | 100 / 100 |
| Brand-based (brand token, no code) | 90 / 100 |
| Plain title (no code, no brand) | 94 / 100 |
| MRP-based (title + price) | 95 / 100 |

Slides: cover · what was tested · how the sets were built · how it was scored · what we found.
The two method slides were added on request (2026-10-22): pool counts per bucket come from
`eval-datasets/README.md`, the scoring definitions from `PLAN-title-level-eval-runs.md`.
Slide 5 carries one footnote: the brand lane scores the name path only, so 90 is its floor;
with the code-based path, brand-named lines are expected around 95 (a projection, not a run).
Copy decisions applied: "plain title" naming; brand-097 label note left out;
no recommendations; no causes-of-misses slide; no QA/Sentinel wording anywhere.

## Source and what was taken

Source: `birlanu-product-mapping-eval.html` (the eval report dropped in the folder).

- Taken: the 300 prod lines, the three 100-title case sets and their scores
  (1.00 / 0.94 / 0.90), the scoring rules, and the MRP result 95 / 100.
- Left out: every QA-chain and Sentinel figure (95/100 status mix, 98 mapped, 3 wrong,
  2 missed, top-5 100, #1 85, MRR, progression, by-case, MRP split, failure table,
  catalogue MRP defects), the engineering status table, and the next-steps list.

## Run setup — settled 2026-10-22

All four lanes ran live search against the real production data; traces are in LangSmith
and the CSV/Excel exports are shareable. The source report's "mapping on saved search
results" wording for stage 01 is stale and must not be circulated as-is. The earlier
per-lane caveat (MRP lane on a separate product set) was removed from the deck.

## Note carried on slide 5

"The brand lane measures the name path on its own. This eval covers only part of the full
invoice pipeline — a brand-named line also runs the code-based mapping, so 90 is the floor
for this lane; with both paths combined the brand-named logic is expected to land around 95."

This is the client's own framing of the brand figure; it is an expectation, not a measured
result, so the copy says "expected to land" and the number never becomes a bar.

## Scoring wording — corrected after replay

Slide 4 states the counting rules in words, not as algebra. Reason: replaying the planned
formulas over both prod exports (150 invoices, 1,842 lines) reproduces every published
evaluator column except one line, where GT `MAPPED` came back `MAPPED_BRAND_NOT_FOUND` —
status scores 12/13 while `false_map` and `missed` both read 0, so
`(N − false_map − missed)/N` overstates status accuracy in that case. Status is an exact
three-way label match. Detail is in `eval-datasets/PLAN-title-level-eval-runs.md`.

## Build

```
cd code && python3 build_deck.py
cd .. && python3 code/qa_deck.py
```

Figures live in `code/numbers.json`; bars derive from hits / titles.
