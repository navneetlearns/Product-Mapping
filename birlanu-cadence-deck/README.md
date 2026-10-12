# BirlaNu Cadence Deck (3 slides, PDF)

Cadence-meet update deck for the BirlaNu team — parsing / approvals / issue-resolution counts
plus the four enhancements implemented this cycle. Condensed from an earlier 6-slide cut to 3.

## Deliverable
- `outputs/BirlaNu-Cadence-Deck.pdf` — 3 slides, 16:9 (1440×810 pt = 1920×1080 px), Chromium-rendered.

## Slides
1. Cover — title + the four headline counts.
2. Cadence counts — all five counts on one page: a 5-cell count strip (2,323 · 30,063 · 957 · 181 · 36,
   each with its derived sub-figure), a 2,323-mark matrix with 957 filled = September (41.2%, 1,366 before
   September), the approval bar (7.8% of invoices parsed) and a 36-mark matrix for issues resolved.
3. Enhancements implemented — invoice reprocessing (MITM, ZoTok panel), automatic rejection of unclear
   images on WhatsApp, original-invoice link for duplicate detection, page display limit 40 → 100.

All figures are the numbers supplied for this cadence; the only additions are arithmetic derived from them
(41.2% = 957/2,323 · 12.9 = 30,063/2,323 · 7.8% = 181/2,323 · 1,366 = 2,323−957).

## Design
- Philosophy: `code/design-philosophy.md` ("Quiet Ledger" — canvas-design skill flow).
- House style inherited from `zochief-demo-docs/sales-deck`: 1920×1080 slides, pine/paper palette,
  Instrument Sans + Instrument Serif italic + JetBrains Mono (fonts copied to `code/assets/fonts/`).

## Rebuild
```
cd birlanu-cadence-deck
python3 code/render_deck.py     # writes outputs/BirlaNu-Cadence-Deck.pdf + outputs/qa/slide_*.png
python3 code/qa_pil.py          # ink band profile (no empty full-width bands above footbar)
python3 code/qa_pixels.py       # verifies the dot matrices / bar count to the real numbers
python3 code/inspect_bar.py     # DOM rects for the bar + mark boxes (feeds qa_pixels constants)
python3 code/inspect_rows.py    # row/title/desc boxes for list-layout slides
python3 code/qa_pdf.py          # re-renders the PDF and diffs against the screen slides
```
QA status: 3 pages, no overflow, no text outside canvas, no console errors; matrices count exactly
957 filled / 1,366 empty marks and 36 clay marks; approval bar measures 7.83% of its track (181/2,323);
text contrast >=4.9:1; PDF-screen diff 3.0-6.4% (resampling only).
