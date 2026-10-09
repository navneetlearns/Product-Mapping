# BirlaNu Cadence Deck (5 slides, PDF)

Cadence-meet update deck for the BirlaNu team — parsing / approvals / issue-resolution counts.

## Deliverable
- `outputs/BirlaNu-Cadence-Deck.pdf` — 5 slides, 16:9 (1440×810 pt = 1920×1080 px), Chromium-rendered.

## Slides
1. Cover — title + the four headline counts.
2. Parse volume till date — 2,323 invoices / 30,063 line items, bars on a shared scale, derived 12.9 line items per invoice.
3. August — 957 invoices parsed in August; one mark per invoice (2,323 marks, 957 filled = 41.2%), 1,366 parsed before August.
4. Approvals & issues — 181 approved (7.8% of invoices parsed), 36 issues resolved, drawn to scale.
5. Summary — the five reported counts in ledger form.

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
python3 code/qa_pixels.py       # verifies bars/dot-matrices count to the real numbers
python3 code/qa_pdf.py          # re-renders the PDF and diffs against the screen slides
```
QA status: 5 pages, no overflow, no text outside canvas, no console errors, dot matrices count
exactly 2,323/957/36 marks, bar ratios 7.7% and 100%, text contrast ≥4.9:1.
