# Plan / decision record — BirlaNu mapping-eval deck

**Status: BUILT with numbers (22 Oct 2026).** One open factual question before it can
be shared (see below).

Deliverable: `outputs/BirlaNu-Mapping-Eval-Deck.pdf` — 4 slides, 16:9.

## What it says

Four mapping logics, each on its own 100-title set, hits shown as x / 100:

| Lane | Hits |
|---|---|
| SKU-based (code in the title) | 100 / 100 |
| Brand-based (brand token, no code) | 90 / 100 |
| Plain title (no code, no brand) | 94 / 100 |
| MRP-based (title + price) | 95 / 100 |

Slides: cover · what was tested · how it was run and scored · what we found.
Copy decisions applied: "plain title" naming; brand-097 label note left out;
no recommendations; no causes-of-misses slide; no QA/Sentinel wording anywhere.

## Source and what was taken

Source: `birlanu-product-mapping-eval.html` (the eval report dropped in the folder).

- Taken: the 300 prod lines, the three 100-title case sets and their scores
  (1.00 / 0.94 / 0.90), the scoring rules, and the MRP result 95 / 100.
- Left out: every QA-chain and Sentinel figure (95/100 status mix, 98 mapped, 3 wrong,
  2 missed, top-5 100, #1 85, MRR, progression, by-case, MRP split, failure table,
  catalogue MRP defects), the engineering status table, and the next-steps list.

## Open question — resolve before sharing

The deck's method slide says the runs searched the **live catalogue**. The source report
says the opposite for the LangSmith stage: stage 01 is titled "Mapping on saved search
results", and its summary describes the move to live search as happening at the Sentinel
stage. If the 1.00 / 0.94 / 0.90 figures come from the saved-candidate run, the deck's
method line is wrong and must be reworded; if they were re-run live, the report is the
stale document and should be corrected instead.

## Note carried on slide 4

"The MRP lane ran on the live search chain against the product set loaded for that
test; the other lanes ran against the full production catalogue." This keeps the 95/100
from being read as directly comparable to 90–94; it names no environment.

## Build

```
cd code && python3 build_deck.py
cd .. && python3 code/qa_deck.py
```

Figures live in `code/numbers.json`; bars derive from hits / titles.
