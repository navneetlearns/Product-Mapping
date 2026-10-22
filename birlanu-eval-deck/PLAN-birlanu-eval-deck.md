# Plan — BirlaNu product-mapping eval deck

**Status: Proposed — NOT approved.** (2026-10-22)

Audience: the BirlaNu team (client-facing). Deliverable: a 10-slide 16:9 PDF built
HTML→PDF with Playwright, plus the editable source and render script, in the house
"Quiet Ledger" style already used for `birlanu-cadence-deck/`.

Working folder for the build: `~/AgentWork/product-mapping/birlanu-eval-deck/`
(sibling of the cadence deck; no new root folder).

---

## 0. Decision gates — answer these before anything is built

| # | Gate | Options | Default if you reply "go" |
|---|---|---|---|
| G1 | Scope | (a) two studies: mapping accuracy (71 invoices / 899 lines) + catalogue coverage (100 inputs) — 10 slides · (b) catalogue coverage only — 5 slides | (a) |
| G2 | Candour | (a) show accuracy numbers and the three named failure families · (b) qualitative only ("strong, with known gaps") | (a) |
| G3 | The 42 open cases | (a) show as "open — manual verification running" · (b) hold the deck until the verdict column is filled · (c) omit the 42 entirely | (a) — verdicts are 0/42 today; presenting it as a result would be false |
| G4 | Internal identifiers | (a) strip LangSmith, run ids, mapping revision, token/cost figures from every slide · (b) keep them in a backup slide | (a), with one methodology line ("scored against recorded ground truth") |
| G5 | Format | (a) PDF only · (b) PDF + editable PPTX | (a) |
| G6 | Design & branding | inherit the cadence deck's Quiet Ledger system; do you have a BirlaNu logo file to place, or use a text wordmark? | inherit + text wordmark |

G3 is the one that changes the deck's honesty posture: the catalogue work is
mid-flight. Everything else is presentation.

---

## 1. What the deck argues

One sentence, carried by the cover and repeated in the close:

> The mapping agent is right on about nine lines in ten; where it misses, the cause
> is vocabulary and variant ambiguity — on both sides of the match — not products
> missing from the catalogue.

Two studies support that, and they must never share a denominator:

- **Study A — mapping accuracy.** 71 invoices / 899 product lines, recorded ground
  truth per line. 92.9% status-correct, 90.7% SKU-correct after equivalence scoring.
- **Study B — catalogue coverage.** The 100 unique products fed into the brand-name
  run, checked one-to-one against the 2,335-product catalogue export. 58 located,
  42 open, of which 21 already carry a ground-truth SKU.

---

## 2. Slide map (10 slides)

Every number quoted below exists in the fact base (§3). Layout notes assume the
1920×1080 Quiet Ledger grid.

| # | Slide | Content | Numbers on the slide | Layout |
|---|---|---|---|---|
| 1 | Cover | "Product mapping, measured." kicker: BIRLANU · PRODUCT MAPPING EVAL · OCT 2026. One-line thesis (§1). | Three figures only: 899 lines tested · 92.9% status correct · 58 of 100 catalogue matches | Dark pine full-bleed, mono kicker, one serif italic accent word |
| 2 | The question | Mapping quality is two questions, not one: does the agent choose right, and does the catalogue hold the product at all. We tested them separately. | — | Two-column split, hairline rule between |
| 3 | Study A — how it was tested | 71 invoices, 899 product lines, one line = one title with its candidate set; ground truth recorded per line; identical scoring rules applied to GT and model. | 71 · 899 | Ledger row list with basis column (as cadence deck slide 3) |
| 4 | Study A — accuracy | Headline rates + what the confidence score is worth. | 73.5% mapped (GT 67.5%) · 92.9% status correct (835/899) · 90.7% SKU correct (815/899) · 59 false maps · 5 missed | Count strip + horizontal bar for status/SKU |
| 5 | Study A — confidence is honest at the top, thin in the middle | Nothing wrong above 95; 69 of 79 mapping errors sit in the 80–89 band; the 5 should-have-mapped lines died under the ≤80 gate with the right SKU already in the candidate set. | bands 80–84: 43 mapped / 30 wrong · 85–89: 103/39 · 90–94: 198/10 · 95–100: 317/0 | Four-segment stacked band, mono labels |
| 6 | Study A — where it fails | Three families, in plain product language: elbow/bend angle or series siblings (39 of 59), Solfit→Pushfit series (4), CPVC coupler/brass-tee→Pressure picks (5), 11 other. Plus the retrieval gap: 79% of titles carry a brand token, only 8% of candidate sets did. | 39 · 4 · 5 · 11 · 79% vs 8% | Family rows + a small paired bar |
| 7 | Study B — the catalogue question | 100 unique eval inputs vs 2,335 catalogue products; all 100 titles carry a brand token (98 Birla/Birlanu, 2 HIL-only). 48 matched above threshold, 10 more rescued by an embedded code = 58 located. | 100 · 2,335 · 48 · 10 · 58 · 42 open | Count strip + a 100-mark matrix with 58 filled |
| 8 | Study B — why titles miss, and what is still open | Vocabulary translation is the mechanism: COUPLER→SOCKET, SHOE→45 Elbow, ROTTO→HIL Water Tank, END CAP→"Plain end cap"; plus size conventions (50 mm = 2", 110 mm = 4") and the catalogue's own class naming (Pressure Pipe CLASS-n). Of the 42 open: 21 carry a GT lead SKU (all verified present), 4 exact, 17 variant-ambiguous, 1 unknown term (HALFSET). | 21 · 4 · 17 · 1 · 0 of 42 verdicts closed | Two-panel: translation table left, state-of-42 right. Must read "open", not "solved" |
| 9 | What this means | Four concrete implications: (1) put a synonym/size canon in front of matching; (2) retrieve brand-aware candidates; (3) treat 80–89 confidence as review, not auto-apply; (4) widen the catalogue export — it carries only 9000/9600/9800/9900-prefix SKUs, so live-only 412xxx items read as absent. | — | Four numbered rows, mono index |
| 10 | Next steps + asks of BirlaNu | Confirm catalogue export completeness; product-team pass on the 42; re-run with the vocabulary layer; then the 300-case title-level sets (already built, not yet run). | 300 cases ready | Ledger rows with owner column (ZoTok / BirlaNu) |

Backup slide (only if G4 asks for it): method, data limits, run identifiers.

---

## 3. Fact base — every number, with its provenance

Built as `code/facts.py` (deterministic; recomputes from the two F: CSVs and
asserts each value). The deck HTML is generated from this module so no number is
hand-typed.

**Study A** — `turnover-product-mapping-136d0d1e.csv` (71 runs, all success),
analysed in `BirlaNu_ActionTracker_and_Mapping_Analysis.md` §6:

| Metric | Value | Derivation |
|---|---|---|
| invoices / lines | 71 / 899 | row count; sum of line_items |
| mapped by model / GT | 661 (73.5%) / 607 (67.5%) | status=MAPPED count |
| status correct | 835/899 = 92.9% | 899 − 59 false maps − 5 missed |
| SKU correct (equivalence-aware) | 815/899 = 90.7% | 20 wrong SKUs |
| confidence bands (mapped/wrong) | 80–84: 43/30 · 85–89: 103/39 · 90–94: 198/10 · 95–100: 317/0 | §6 |
| false-map families | 39 elbow/bend · 4 Solfit · 5 CPVC coupler/brass-tee · 11 other | §6 |
| brand tokens | 709/899 titles (79%); 74/899 candidate sets (8%); 645/709 branded lines with no same-brand candidate | §6 |
| embedded-SKU lines | 200/899, all mapped | §6 |
| cost / latency | 21.2 s, 8,624 tokens, $0.0015 per invoice; $0.10 total | §6 — do NOT print if G4 = strip |

**Study B** — `turnover-product-mapping-c4cce19d.csv` (102 rows → 100 unique
inputs) vs `products-18-October-2026-10-52-16.csv` (2,335 products):

| Metric | Value | Derivation |
|---|---|---|
| unique inputs | 100 | 102 rows − 2 hang/re-run duplicates |
| brand-carrying titles | 100 (98 Birla/Birlanu, 2 HIL-only) | census in README Phase 2 |
| fuzzy matches ≥0.70 | 48 (median 0.86; 24 ≥0.85) | user's Power Query run, verified from stored M code |
| code-rescued blanks | 10 | 16 titles carry codes; 6 of them already inside the 48 |
| located / open | 58 / 42 | 48 + 10, no double count |
| open-case split | 21 GT-lead · 4 exact · 17 ambig-variant · 1 unknown type | `open_42_resolution.xlsx` |
| GT leads verified in export | 21 of 21 | join on expected_sku |
| verdicts closed | 0 of 42 | VERDICT column read 2026-10-22 |
| catalogue export scope | SKUs 9000/9600/9800/9900 only | prefix census |
| 100-case run status-correct (optional) | 89/100 | export's own false_map=9, missed=2 columns — do NOT quote a re-derived 85% without labelling it "status+SKU exact" |

Rule: Study A and Study B denominators (899 lines vs 100 titles) are never added,
averaged, or presented in one figure.

---

## 4. Build phases

**P1 — fact base (≈30 min).**
`code/facts.py` recomputes every row of §3, prints the table, exits non-zero on any
drift. Acceptance: `python3 code/facts.py --check` exits 0 and its output matches §3
line for line. No design work until this passes.

**P2 — copy deck (≈30 min).**
`code/copy.md`: slide-by-slide final wording, written before any HTML. Acceptance:
you read it and approve the wording — the cheapest place to change the story.

**P3 — design & build (≈2 h).**
`code/build_deck.py` imports `facts.py`, injects values into
`code/deck_template.html`, writes `outputs/BirlaNu-Mapping-Eval-Deck.html`, renders
with Playwright at 1920×1080 (`page.pdf(width="1920px", height="1080px",
print_background=True)`, `@page{size:1920px 1080px; margin:0}`).
CSS and fonts copied from `birlanu-cadence-deck/code/` (Instrument Sans, Instrument
Serif Italic, JetBrains Mono; paper `#f4f6f5`, ink `#111c16`, green `#1f5c3d`,
pine `#14402a`, clay `#b0812c`).

**P4 — QA (≈40 min).**
See §6. Frozen: no slide ships until the number check and the identifier grep pass.

**P5 — delivery & docs (≈20 min).**
PDF to `outputs/`, copy to the F: eval folder next to the workbooks, README for the
folder, PROJECTS.md pointer, commit locally. Push only if you ask.

---

## 5. File structure

```
birlanu-eval-deck/
  README.md                      index: what the deck is, slide list, QA status
  PLAN-birlanu-eval-deck.md      this plan
  code/
    facts.py                     recomputes every number; --check mode
    copy.md                      approved slide wording
    deck_template.html           source, values injected at build
    build_deck.py                inject + render
    assets/fonts/*.ttf           copied from the cadence deck
    qa_pil.py qa_pixels.py       adapted from the cadence deck
  outputs/
    BirlaNu-Mapping-Eval-Deck.pdf
    qa/slide_*.png               per-slide captures
```

---

## 6. Verification checklist

1. `pdfinfo` → `Pages: 10`, `Page size: 1440 x 810 pts`.
2. `python3 code/facts.py --check` → exit 0.
3. Number round-trip: every value in facts.json appears in `pdftotext -layout` output; script asserts set membership both ways for the headline figures.
4. Identifier grep on the PDF text returns zero hits for: `LangSmith`, `langsmith`, `turnover-product-mapping`, `c4cce19d`, `136d0d1e`, `v8.4.19`, `false_map`, `total_cost`, `\$0.`, `eval-dashboard`.
5. Pixel QA per slide: no ink in the outer 3 px band; footer zone sparse; ink density 0.04–0.3 (dark cover exempt).
6. Denominator audit: no slide places an 899-line figure next to a 100-title figure without an explicit label; the 58/100 pair never appears as a percentage of 899.
7. Wording audit: the 42 appear only as open/pending; no cell, caption or footnote implies they are resolved.
8. Screenshot-free deck: no embedded UI captures (nothing to redact); if a workbook capture is added later, re-run the redaction pass.
9. Read the rendered text end-to-end once (pdftotext) as the client would.

---

## 7. Risks and client-facing pitfalls

- **The 42 read as a failure.** Frame them as verification work in progress with named leads; the deck's claim is method + coverage, not a verdict.
- **Mixing denominators.** The two studies answer different questions; a combined "accuracy" number would be indefensible in the room.
- **Over-claiming the fix.** "Synonym layer" is a proposed direction, not a shipped change — say "recommend", not "implemented".
- **Catalogue criticism.** The 412xxx gap is an export-scope fact, not a statement that their catalogue is wrong. Phrase as "the export we were given".
- **Confidence bands as a boast.** 92.9% is strong but the 80–89 zone is the actionable finding; lead with the finding, not the score.
- **Client identity.** No other client, account or internal project names anywhere in the deck.

## 8. Out of scope

- Running the 300-case title-level sets (gates in `PLAN-title-level-eval-runs.md` still open).
- Filling the 42 verdicts (yours, manual by design).
- Any live-API catalogue check; export-only, as before.
- PPTX twin unless G5 says otherwise.
