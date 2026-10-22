# Plan / decision record — BirlaNu mapping-eval deck

**Status: BUILT — numbers pending.** (2026-10-22)

Deliverable: `outputs/BirlaNu-Mapping-Eval-Deck.pdf`, 4 slides, 16:9, placeholder
build. Nothing in it is quantified yet; the user supplies the figures.

## Scope, as settled

- 4 slides. Cover · what was tested · how it was run and scored · what we found.
- Subject: the eval run for the four mapping logics — SKU-based, Brand-based, Normal
  product title, MRP-based — each on its own dataset (100 product titles per logic).
- Out: the local catalogue-coverage work and its tooling; recommendations; a
  causes-of-misses slide; any figure not supplied by the user.

## Numbers

`code/numbers.json` holds six slots — `RUN_DATE`, `TITLES`, `ACC_SKU`,
`ACC_BRAND`, `ACC_TITLE`, `ACC_MRP`. Null slots render as visible placeholder
chips. Build refuses accuracies outside 0–100.

## Build

```
cd code && python3 build_deck.py     # placeholder or filled, same code path
cd .. && python3 code/qa_deck.py
```

QA: page count/size, text round-trip of slots, forbidden-token grep (Excel, VLOOKUP,
Power Query, workbook, LangSmith, csv, cost fields), edge-band and density pixel
checks, DOM overflow/tight-text report. Placeholder build and a numbers-filled test
build both pass.

## Open questions for the user

1. One-line description of the MRP-based logic — the slide currently reads
   "Titles matched with price in the picture — the price-led path, tested on its own."
2. Confirmation that each set is 100 titles (the `TITLES` slot is on slide 2 only).
3. Whether the results slide should carry a count per lane (correct out of 100) in
   addition to the percentage.
