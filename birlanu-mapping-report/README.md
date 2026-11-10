# BirlaNu invoice-mapping coverage report

Client-facing weekly coverage report for the BirlaNu team, built from the Secondary Invoice
Report export. Three A4 pages in the "Quiet Ledger" house style (pine/paper, Instrument Sans +
Instrument Serif Italic + JetBrains Mono), delivered as **one self-contained HTML file** so it
can be emailed or opened on any machine with no assets folder.

## Deliverable

`outputs/BirlaNu-Mapping-Coverage-2026-11-08.html` — 605 KB, fonts embedded, prints to 3 A4 pages.
A PDF twin is written by the QA pass: `outputs/BirlaNu-Mapping-Coverage-2026-11-08.pdf`.

Snapshot — 2–8 November 2026:

| | Unique | Fully | Partly | Nothing |
|---|---|---|---|---|
| Distributors | 60 | 56 | 1 | 3 |
| Retailers | 168 | 137 | 13 | 18 |
| Products | 2,187 | 1,462 | 49 | 676 |

4,007 lines · 326 duplicate-flagged · 3,681 counted · 279 invoices. Line split: 1,902 Mapped
(51.7%), 629 branding not found (17.1%), 1,150 Unmapped (31.2%). 1,511 descriptions mapped at
least once, on 580 distinct SKUs.

## The pages

1. **Cover and at a glance** — three ratios (56/60, 137/168, 1,511/2,187), the parties ledgers with
   full/partly/nothing bars, and the line split as a stacked bar.
2. **The week in counts** — the full table, how the four counting rules work, and the scope caveats.
3. **Why these did not map** — the reason table the client asked for: five product reasons
   (203 descriptions written differently, 278 outside the catalogue extract, 177 size/grade not
   listed, 14 with no size in the title, 4 exact-but-unmapped), then retailers and distributors by
   the identifier the invoice carried.

## What the report says, and what it refuses to say

- **Counts only.** Identification and mapping coverage; line value / ₹ is out of scope by rule.
- **No internals.** No tooling names (Excel, workbook, VLOOKUP, Power Query), no repo paths, no
  script names, no run ids, no eval terms. The QA greps the rendered PDF for all of them.
- **No party names.** Aggregate counts only, so the file is safe to send outside.
- **Scope caveats travel with the numbers:** one week only (not comparable with a longer period);
  the export we were given lists 9000/9600/9800/9900-series SKUs only, so absence here is not
  evidence a product is missing from the catalogue; mapping runs per distributor catalogue.
- **MBNF counts as mapped.** Branding-not-found lines carry a SKU — the report says so on the page.
- **Reasons are read, not asserted.** Every reason is a statement about the check that was run
  against the catalogue extract, not a judgement about the world. "Not in the catalogue extract"
  means no catalogue name shares a product word with the title; "written differently" means one
  does and the title's own sizes appear in it.
- **No priors, no trend, no recommendations.** Different windows are different populations; a
  delta between them is not a progress rate.

## Files

| Path | What it is |
|---|---|
| `code/report.template.html` | The authored page — edit copy and layout here |
| `code/numbers.json` | Every figure, single source. The builder refuses if they don't add up |
| `code/build_report.py` | Embeds fonts, substitutes tokens, asserts identities, writes the HTML |
| `code/unmapped_reasons.py` | Reads the reasons off each line; `--update-numbers` writes page 3's figures |
| `code/qa_report.py` | Playwright + PIL + pdftotext QA; writes the PDF and page PNGs into `outputs/qa/` |
| `assets/fonts/` | Instrument Sans / Instrument Serif Italic / JetBrains Mono |
| `outputs/` | The HTML deliverable, its PDF twin, and QA renders |

## Rebuild

```
cd ~/AgentWork/product-mapping/birlanu-mapping-report
python3 code/build_report.py     # numbers.json -> outputs/...html
python3 code/qa_report.py        # asserts, then writes the PDF twin
```

## Refreshing for a new week

1. Read the new export's own filter line and mtime — it lands at the same path and is overwritten
   in place. Never reuse a window label from the previous run.
2. Recount with the calibrated tools:
   `python3 ~/.hermes/skills/zotok/zotok-product-mapping/scripts/turnover_entity_counts.py "<xlsx>"`
   and `python3 ~/AgentWork/product-mapping/eval-datasets/code/turnover_product_counts.py "<xlsx>"`.
   The entity script is proven only after `--calibrate` reproduces a published triple.
3. Refresh page 3's figures from the export itself:
   `python3 code/unmapped_reasons.py "<xlsx>" "<catalogue csv>" --update-numbers code/numbers.json`.
   Pass the current catalogue export — the reasons are only as good as its date, which the page states.
4. Update `window` + `prepared` in `code/numbers.json`, bump the output filename, rebuild.
5. Run the QA — it fails on a broken identity, a missing figure, a clipped page, or any internal term.

## Definition notes the page carries

- A product is the printed invoice title, normalised with the classifier's `norm_text` (case, quote
  marks, unit spacing, known spelling variants) — descriptions, not catalogue items.
- A description is fully mapped when every line it appears on carries a SKU, partly when some do,
  never when none do.
- Unique parties are keyed GST → distributor SAP code / retailer mobile → name + pin code.
- Entity status is decided per invoice and stamped on each line, so full / partly / none are real
  states within one week.

## Reason groups, in the words the check used

| Group | What it establishes |
|---|---|
| Catalogue product, title written differently | a catalogue name shares the title's product words **and** carries all of its sizes |
| Not in the catalogue extract | no catalogue name shares a product word; 84 name another maker's brand |
| Catalogue family, this size or grade not listed | a catalogue name shares a product word, but none carries these sizes |
| Catalogue family, no size in the title | a catalogue name shares a product word; the title has no size to resolve |
| Exactly as the catalogue, still unmapped | the title is a catalogue name word for word and the line still failed |

Parties: a valid 15-character GSTIN that did not resolve (`no matching account`), a PAN or short
string in the GST field, a phone with no GSTIN, or nothing usable. BirlaNu's own GSTIN is its own
group.

Version: 2 — 10 November 2026 (export window 2–8 November 2026; reasons added).
