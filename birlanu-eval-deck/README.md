# BirlaNu mapping-eval deck (5 slides)

Client-facing update for the BirlaNu team on the product-mapping eval: four mapping
logics — SKU-based, Brand-based, Plain title, MRP-based — each run on its own set of
100 product titles.

## Deliverable

- `outputs/BirlaNu-Mapping-Eval-Deck.pdf` — 5 slides, 16:9 (1440×810 pt = 1920×1080 px),
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
3. How the sets were built — pool sizes per bucket (176 / 495 / 405 = 1,076 distinct
   titles), one case per title, stratified cut to exactly 100 per logic, rebuilds
   byte-identical, every case traced to its invoice line.
4. How it was scored — status rule, SKU + equivalence, and the error definitions, with
   the two accuracy formulas printed on the slide.
5. What we found — one row per logic with a bar and hits / total. All four runs searched
   the live production catalogue; there is no per-lane setup caveat.

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
| `POOL_SKU` | 176 | distinct titles in the code-in-title pool |
| `POOL_BRAND` | 495 | distinct titles in the brand-token pool |
| `POOL_PLAIN` | 405 | distinct titles in the plain pool (the slide sums them: 1,076) |

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

`code/qa_deck.py` checks: page count matches the built HTML, 1440×810 pt; every supplied
figure present in the PDF text; no forbidden tokens (Excel, VLOOKUP, Power Query,
workbook, LangSmith, csv, cost fields); no ink in the outer 3 px frame or right strip;
plausible density; no long empty band.

The build itself now FAILS (non-zero exit) on DOM overflow, off-canvas text or text
overflowing its own box, so a clipped slide cannot be rendered silently — that gate
caught a 120 px overflow on slide 2 when the sample lines were added.

Last verified: 5 pages, no overflow, no clipped text, no edge-band ink.

## Open items

1. **Run setup — settled.** All four lanes ran live search against the real production
   data; traces sit in LangSmith and the exports can be shared with the client. The
   source report's "mapping on saved search results" wording for the first stage is
   therefore stale — do not circulate that file as-is alongside the deck.
2. **Source report is deliberately untracked.** `birlanu-product-mapping-eval.html`
   carries internal PR numbers, commit hashes, branch names, workspace ids and
   "uncommitted / not deployed" status. This repo is public, so the file is left out of
   git (`.gitignore`) and must not be attached to a client pack as-is.
3. **MRP lane has no price shown.** No export available here has an MRP column, so the
   lane states the signal in words. If real rates are wanted on the slide, supply the
   QA catalogue CSV or two rate values.
