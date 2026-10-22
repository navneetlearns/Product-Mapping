# BirlaNu mapping-eval deck (4 slides)

Client-facing update for the BirlaNu team on the product-mapping eval: four mapping
logics — SKU-based, Brand-based, Normal product title, MRP-based — each run on its
own set of product titles (100 per logic, per the working scope).

**Numbers are intentionally absent.** Every figure is a named slot in
`code/numbers.json`; until a slot is filled the slide shows a visible placeholder
chip (`⟦ACC_SKU⟧` etc.), so the deck can be reviewed and approved first and the
figures dropped in afterwards.

## Deliverable

- `outputs/BirlaNu-Mapping-Eval-Deck.pdf` — 4 slides, 16:9 (1440×810 pt = 1920×1080 px),
  Chromium-rendered. Placeholder build; no numbers in it.
- `code/deck.html` — the generated source for that build.

## Slides

1. Cover — title, one-line scope statement, four lane figures (one per mapping logic).
2. What was tested — one dataset per logic, what each set is built to exercise.
3. How it was run and scored — the sets, the runs, the scoring.
4. What we found — one row per logic with an accuracy bar and value.

No recommendations, no causes-of-misses slide, no local tooling mentioned: the deck
describes the eval and its result, nothing about how the files were prepared.

## Filling the numbers

`code/numbers.json` slots:

| Slot | Meaning |
|---|---|
| `RUN_DATE` | date the eval ran |
| `TITLES` | product titles per set (one dataset per logic) |
| `ACC_SKU` | SKU-based logic accuracy (%) |
| `ACC_BRAND` | Brand-based logic accuracy (%) |
| `ACC_TITLE` | Normal-product-title logic accuracy (%) |
| `ACC_MRP` | MRP-based logic accuracy (%) |

```
cd code
# edit numbers.json, then
python3 build_deck.py
cd .. && python3 code/qa_deck.py
```

A slot left `null` stays a placeholder, so partial numbers are fine. Accuracies must
be 0–100 or the build stops.

## Design

House "Quiet Ledger" system inherited from `birlanu-cadence-deck/`: pine cover, paper
`#f4f6f5`, ink `#111c16`, green `#1f5c3d`, clay `#b0812c`; Instrument Sans (text),
Instrument Serif italic (one accent per headline), JetBrains Mono (labels, footers).

## QA (run before handing over)

```
python3 code/qa_deck.py
```

Checks: 4 pages at 1440×810 pt; every slot placeholder (or supplied value) present in
the PDF text; no forbidden tokens (Excel, VLOOKUP, Power Query, workbook, LangSmith,
csv, cost fields); no ink in the outer 3 px frame or right strip of the light slides;
plausible ink density; no long empty band. The build also reports DOM overflow and
any text block that overflows its own box.

Last verified: placeholder build and a numbers-filled test build (sample values) both
pass, 4 pages, no overflow, no clipped text.

## Rebuild chain

```
python3 code/build_deck.py          # -> outputs/BirlaNu-Mapping-Eval-Deck.pdf + outputs/qa/slide_*.png
python3 code/qa_deck.py             # -> PASS/FAIL report
```
