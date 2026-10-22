# BirlaNu mapping-eval deck (4 slides)

Client-facing update for the BirlaNu team on the product-mapping eval: four mapping
logics — SKU-based, Brand-based, Plain title, MRP-based — each run on its own set of
100 product titles.

## Deliverable

- `outputs/BirlaNu-Mapping-Eval-Deck.pdf` — 4 slides, 16:9 (1440×810 pt = 1920×1080 px),
  Chromium-rendered, numbers filled.
- `code/deck.html` — the generated source for that build.
- `birlanu-product-mapping-eval.html` (folder root) — the source report the figures were
  taken from. It also contains QA/Sentinel material that is deliberately NOT in the deck.

## Slides

1. Cover — title, scope line, four lane scores (hits / total).
2. What was tested — one dataset per logic, what each set exercises, and two real
   sample titles per lane (from `eval-datasets/outputs/*.jsonl`). The MRP lane samples
   carry no price figure: none of the available exports has an MRP column, so the row
   states the signal in words ("matched on the invoice rate against the catalogue
   MRP") instead of inventing a value.
3. How it was run and scored — the sets, one live run per logic, scoring against the
   expected SKU recorded before the run.
4. What we found — one row per logic with a bar and hits / total, plus a two-line note
   on how the MRP lane was run (live search chain, product set loaded for that test)
   versus the other lanes (full production catalogue).

No recommendations, no causes-of-misses slide, no QA/Sentinel content, no brand-097
label note, no local tooling.

## Numbers

`code/numbers.json`:

| Slot | Value | Meaning |
|---|---|---|
| `RUN_DATE` | October 2026 | date line on the cover |
| `TITLES` | 100 | product titles per set |
| `HITS_SKU` | 100 | SKU-based, correct of 100 |
| `HITS_BRAND` | 90 | Brand-based, correct of 100 |
| `HITS_PLAIN` | 94 | Plain title, correct of 100 |
| `HITS_MRP` | 95 | MRP-based, correct of 100 |

Bars are computed from hits / titles; a null slot falls back to a visible placeholder
chip, so partial numbers still render. Hits above `TITLES` stop the build.

```
cd code && python3 build_deck.py
cd .. && python3 code/qa_deck.py
```

## Design

House "Quiet Ledger" system inherited from `birlanu-cadence-deck/`: pine cover, paper
`#f4f6f5`, ink `#111c16`, green `#1f5c3d`, clay `#b0812c`; Instrument Sans,
Instrument Serif italic (one accent per headline), JetBrains Mono (labels, footers).

## QA

`code/qa_deck.py` checks: 4 pages at 1440×810 pt; every supplied figure present in the
PDF text; no forbidden tokens (Excel, VLOOKUP, Power Query, workbook, LangSmith, csv,
cost fields); no ink in the outer 3 px frame or right strip; plausible density; no long
empty band. The build also reports DOM overflow, off-canvas text and text that
overflows its own box.

Last verified: 4 pages, no overflow, no clipped text, no edge-band ink.
