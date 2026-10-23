# Practical product context for the product mapper — proposal (2026-10-23)

Status: PROPOSAL. No build started. Five decision gates in §7 are unanswered.
Author's evidence base: the production prompt, the catalogue export, one run export,
and the two historical mapping files (paths and counts in §2).

## 1. The problem, stated precisely

The mapper can receive the correct record in `catalog_results` and still not select it,
because nothing in the loop says that two different names denote the same article.
Concrete case: the catalogue names the CPVC straight joint `CPVC SOCKET <size>`, while
invoices and plumbers call it a `COUPLER`. The prompt's matching rule requires the
fitting token to be present on both sides, so the correct candidate is rejected by rule.

This is a vocabulary/context gap, not a retrieval gap. It is the same class of failure
the catalogue-check workstream found in Phase 2 (fuzzy merge: 52 of 100 below threshold
on vocabulary and format, not on substance), but the fix has to live where the mapping
happens, not in a manual workbook.

## 2. What exists today — audit

### 2.1 The production prompt (373 lines, 12.5 KB)

Same text as `product_mapping_guidance` in every eval case (`eval-datasets/outputs/*.jsonl`)
and in the run export. It contains, in full: brand detection/stripping (Step 1a),
an abbreviation table (Step 1b), dimension and fraction normalisation (Step 1c),
an mm↔inch table (Step 1d), bag-of-words matching (Step 2a), strict dimension checks
(Step 2b), HSN tiebreak (Step 2c), ambiguity resolution (Step 3), confidence threshold
(Step 4), output schema, and six worked examples.

It contains no statement of what any article is, where it is used, or which catalogue
names are other names for the same article. The only product-level vocabulary is
orthographic (SELFFIT→Solfit, VALUE→Valve, BUSH→Bushing, R.TEE→Reducing Tee,
D/S→Double Socket, INCHI→inch) plus units.

Three rules actively produce misses:

| Line | Rule | Effect |
|---|---|---|
| Step 2a.3 | "Core product tokens (material: CPVC/UPVC/SWR, fitting: Tee/Elbow/Coupler) must be present in both" | `CPVC COUPLER 3/4"` cannot match `CPVC SOCKET 3/4"`, which is the only CPVC straight joint in the catalogue at that size |
| Step 1b | abbreviation table has no `SOC`→Socket, no `SHOE`→45° Elbow, no `TRANS TEE`→Reducing Tee, no `PLAIN/FAB` handling | those misses have no recovery path |
| Step 3 + Example 6 | "If multiple candidates still tie … pick the first one" (the example ties SWR Single Y against Silent Fit Single Y) | right family, wrong variant, at 84 confidence — a false map by construction |

### 2.2 The catalogue export (2,335 rows)

`F:\Product\zotok\Product Mapping\Eval Related Product Mapping\products-18-October-2026-10-52-16.csv`

- Columns: `Product Name | Description | Sku Code` (plus unused empties).
- `Description` is byte-identical to `Product Name` in 2,334 of 2,335 rows — no metadata.
- No category, segment, usage, UOM, size or class field.
- SKU prefix does not encode family: 9600-prefix spans PRESSURE (362), SWR (344),
  UPVC (311), CPVC (293), UGD (61), SILENT (57), FOAM (53).
- 13 top-level segments by leading token: PRESSURE 387, UPVC 383, SWR 375, CPVC 344,
  BIRLA HIL 342, NEWKOR 133, OTHER 129, UGD 64, SILENT 57, FOAM 53, CASING 39, PVC 18,
  AGRI 11. 55 distinct first tokens; 2,335 distinct names.
- Data defects that affect matching: 71 names carry `BIRLA HILÂ` (mojibake `Â`), which
  the prompt's ignore list (`BIRLA HIL`) does not cover; double spaces and `11/2`-style
  glued fractions are common.

### 2.3 What the model actually sees per line item

From the run export `turnover-product-mapping-c4cce19d.csv` (102 rows = 102 line items,
100 unique titles, exactly 5 candidates each):

- candidate object fields: `product_name`, `sku_code`, `hsn_code`, `price`, `MRP`,
  `product_variant_id` (MRP present on 360 of 510 candidate rows, PTR on 1)
- line item fields: `product_name`, `serial_no`, `hsn_sac_code`, `unit_rate_excl_gst`

No description, no category, no family, no UOM, no size field, no sibling names.

### 2.4 Latent context that does exist

Not in the loop, but on disk, and the only place trade→house naming was ever recorded:

`F:\Product\zotok\Product Mapping\BirlaNu Products Mapping(Mapped).csv` (1,223 rows) and
`BirlaNu Products Mapping(New Records).csv` (2,523 rows) = 3,589 rows →
1,369 distinct (invoice name → mapped name → SKU) pairs, 1,071 distinct invoice names,
570 distinct SKUs.

Observed type-token transformations, with support (distinct pairs):

| n | invoice tokens | catalogue tokens | example |
|---|---|---|---|
| 21 | ELBOW | ELBOW+TRANS | `Cpvc Elbow 3/4''` → `CPVC Trans Elbow 3/4"` |
| 19 | BEND | ELBOW | `SWR SELFFIT BEND 45D 90MM` → `SWR 45 Elbow Solfit 90mm` |
| 12 | TEE | REDUCING+TEE | `CPVC RED.TEE 1.50"X1"` → `CPVC Reducing Tee …` |
| 10 | COUPLER | SOCKET | `CK BIRLA CPVC 1 1/2X1" Coupler` → `CPVC SOCKET 1 1/2"` |
| 9 | ELBOW | ELBOW+REDUCING | `CPVC ELBOW 90 D (BRASS) 1"X1/2"` → `Cpvc Reducing Elbow …` |
| 6 | TEE | TEE+TRANS | `CPVC TEE 3/4"` → `CPVC Trans Tee 3/4"` |
| 5 | BEND+SHOE | ELBOW | `SWR 110mm 45 Shoe Bend` → elbow family |
| 4 | TEE | CROSS+TEE | `Cpvc Tee 3/4''` → `CPVC Cross Tee 3/4"` |

The corpus is noisy and must never be ingested as truth. Same file contains
`CK BIRLA SWR 110mm Bend(Door)` → `SWR Pipe Type A Pushfit S/S 3m 110mm-TRD` (a bend
mapped to a pipe) and `SWR 110mm 45 Shoe Bend` → `CPVC Elbow 45 D 1"` (SWR 4" mapped to
CPVC 1"). Treat it as alias evidence with a support count, gated by hand verification.

### 2.5 The catalogue's own naming convention (the rule nobody wrote down)

| segment | COUPLER rows | SOCKET/SOC rows |
|---|---|---|
| CPVC | 1 (`CPVC Coupler 6" (SCH 80)`) | 29 (1/2"–4") |
| UPVC | 1 (`UPVC Coupler 6" (SCH 80)`) | 63 |
| Pressure | 24 | 5 |
| SWR | 16 | 4 |
| UGD | 5 | 0 |
| Silent | 4 | 0 |
| PVC | 2 | 0 |

So the same trade word resolves to a different house word by material family — and flips
back for 6" CPVC/UPVC. This is exactly the practical context the mapper lacks, and it is
recoverable from the export without asking anyone.

### 2.6 Variant axes the invoice omits and the catalogue splits

Counts of catalogue names carrying each axis: Pushfit 222 / Solfit 234 ·
S/S 176 / D/S 144 · SCH 80 94 / SCH 40 34 · Fab 85 / Plain 49 ·
Type A 97 / Type B 92 · 3Layer 13 / 4LayerFoam 15.

These are the axes that produce right-family/wrong-SKU false maps, currently resolved by
"pick the first candidate". No amount of synonym work fixes them; they need an explicit
disambiguation order per family.

### 2.7 Verdict

There is no product context in the loop. There is recoverable context in the catalogue's
naming behaviour and in the historical mapping pairs, and there is the user's trade
knowledge, which is the only authority on functional equivalence.

## 3. What the deliverable is

One source of truth, three artefacts, all generated from the same YAML:

| artefact | consumer | size target |
|---|---|---|
| `product_context.yaml` | build source | ~60–90 family records, 60–100 KB |
| `product_context.md` | BirlaNu team / humans | generated, readable end to end |
| `guidance_block.md` | the mapper prompt | 3–5 KB, paste-ready diff |

Record schema (every field mandatory before a record may be used in `guidance_block.md`):

```yaml
- id: cpvc_socket
  segment: CPVC
  house_pattern: '^CPVC\s+(REDUCING\s+)?SOC(KET)?\b'   # selects SKUs from the export
  house_sku_count: 29
  what_it_is: "Straight push-fit joint joining two CPVC pipe ends of equal size."
  where_used: "Hot/cold potable water lines; concealed and exposed."
  trade_names: [COUPLER, COUPLING, JOINT, SOC]          # each with source + status
  house_names: [SOCKET, Soc, Reducing Socket]
  sizes: {unit: inch, ladder: ['1/2','3/4','1','1 1/4','1 1/2','2','2 1/2 (SCH80)','3 (SCH80)','4 (SCH80)']}
  varies_on: [schedule, reducing]
  same_as: [cpvc_coupler_6in]                            # 6" ships as COUPLER
  not_same_as:
    - {family: cpvc_reducing_socket, why: "different bore each end; wrong when invoice gives one size"}
    - {family: cpvc_union, why: "threaded demountable; not a solvent/push joint"}
  if_invoice_silent: "prefer SCH 40 up to 2\"; report SCH 80 sizes 2 1/2\"+ explicitly"
  evidence: {catalogue_pattern_matches: 29, history_pairs: 10, verified_by: pending}
```

## 4. Build phases

Phase 0 — catalogue spine (deterministic, ~0.5 day)
  Clean the export (mojibake `Â`, double spaces, quote/½/¼ glyphs, glued fractions) and
  split all 2,335 SKUs into family records by name pattern. Emit `sku_family.csv` +
  `family_census.csv`.
  Gate: every SKU assigned to exactly one record, or on an unclassified list with a
  reason. Target ≥97% assigned. The unclassified list goes to the user, not into the doc.

Phase 1 — alias evidence (deterministic, ~1 day)
  Mine the 1,369 historical pairs for token transformations with support counts; keep
  n≥3, drop contradictions, verify the mapped name still exists in the current export.
  Add the catalogue-internal splits (§2.5, §2.6). Emit `alias_evidence.csv`: trade term,
  family, house term, support count, example invoice line, status.
  Gate: every proposed alias traceable to a real invoice line; no verdicts.

Phase 2 — author the family records (2–3 days, user in the loop)
  Fill the schema in §3, 15–20 records per review round. The user corrects; the build
  re-renders. Unverified records are excluded from `guidance_block.md`.
  Gate: user sign-off per record.

Phase 3 — render and wire (~1 day)
  Produce `guidance_block.md` as a paste-ready diff against the current 373 lines:
  trade→house equivalence table keyed by family, must-not-equate list, variant-priority
  rules, confidence cap when a variant is missing, 4–6 worked examples from real misses.
  Proposed prompt edits: Step 2a.3 permits equivalence substitution; Step 3 defers to
  the variant rule instead of list order.
  Gate: the user approves the wording before any run uses it.

Phase 4 — prove it (~1 day offline; live only with permission)
  a. Coverage/uniqueness checks on the YAML.
  b. Offline recall replay on the 300 eval cases: with equivalence expansion, is the
     correct family/SKU now in the candidate set? No API calls.
  c. Live A/B on the same 300, with and without the block: pass rate, false maps,
     confidence drift. Requires explicit user permission to call the API.
  d. User hand pass; verdicts are the user's, never filled by the builder.
  Acceptance: wsku lane stays 100/100 (no precision leak), plain and brand lanes rise,
  false maps do not rise, every alias traceable.

Phase 5 — ship and maintain (~0.5 day)
  `product-context/` is the source of truth; a copy of the generated `.md` goes to the F:
  folder with the other deliverables. `code/refresh_context.py` diffs a new catalogue
  export against the YAML and lists new names with no family home. Changelog per record.

Total: 5–6 working days of build, Phase 2 gated on user review.

## 5. Non-goals

- No auto-decided matches and no filled verdicts — same contract as the catalogue check.
- No API calls, no prompt deployment, no LangSmith changes without explicit permission.
- No ingestion of the historical pairs as ground truth (they contain known-wrong rows).
- No change to the catalogue itself; the doc annotates, it does not correct.
- Newkor boards (133), Birla HIL tanks (342) and Casing (39) are out of the first pass.

## 6. Risks

| risk | mitigation |
|---|---|
| Synonym expansion creates false maps | equivalence only within a family; variant gaps capped in confidence; A/B must show no rise in false maps |
| Historical pairs encode human errors | support count ≥3 + user verification per record |
| The doc becomes too big to inject | hard 5 KB cap on `guidance_block.md`, checked in the build |
| Catalogue naming is itself inconsistent (6" flip) | record the exception explicitly; do not normalise it away |
| Sentiment that the effort is "just a glossary" | acceptance is measured on the 300 cases, not on document completeness |

## 7. Decision gates (unanswered)

1. Authority: when trade knowledge and catalogue naming conflict, does the doc state the
   rule or only flag the SKU pair?
2. Prompt change: may `guidance_block.md` change Step 2a/Step 3 behaviour, or must it be
   read-only reference? Without the 2a change, COUPLER→SOCKET stays blocked by rule.
3. Scope: plumbing only (CPVC/UPVC/SWR/UGD/Pressure/Silent/Casing/Agri), Newkor and tanks
   deferred?
4. Variant policy when the invoice is silent (Plain/Fab, S/S vs D/S, Pushfit vs Solfit,
   SCH 40/80): low-confidence flag, or pick by per-family frequency?
5. Permission for the live A/B; and are transliterated Hindi/Gujarati trade words in scope?

## 8. Reproduction — the numbers in §2

Catalogue census (§2.2, §2.5, §2.6):

```python
import csv, re, collections
p = r"F:\Product\zotok\Product Mapping\Eval Related Product Mapping\products-18-October-2026-10-52-16.csv"
rows = list(csv.DictReader(open(p, newline="", encoding="utf-8-sig")))
names = [r["Product Name"].strip() for r in rows]
U = [n.upper() for n in names]
print(len(rows), len({n for n in names}))
print(sum(1 for r in rows if (r["Description"] or "").strip() == (r["Product Name"] or "").strip()))
print(sum(1 for n in U if re.search(r"\bCOUPLER\b", n)),
      sum(1 for n in U if re.search(r"\bSOCKET\b|\bSOC\b", n) and "NIPPLE" not in n))
print(sum(1 for n in U if "BIRLA HILÂ" in n))
for lab, pat in [("Plain", r"\bPLAIN\b"), ("Fab", r"\bFAB\b"), ("SCH40", r"SCH\s*40"),
                 ("SCH80", r"SCH\s*80"), ("S/S", r"\bS/S\b"), ("D/S", r"\bD/S\b"),
                 ("Pushfit", r"PUSHFIT"), ("Solfit", r"SOLFIT|SELFFIT"),
                 ("TypeA", r"\bTYPE\s*A\b"), ("TypeB", r"\bTYPE\s*B\b")]:
    print(lab, sum(1 for n in U if re.search(pat, n)))
```

Historical pairs (§2.4): read both `BirlaNu Products Mapping*.csv`, dedupe on
(Product Name, Mapped Product Name, SKU Code), then compare the set of type tokens on
each side; counts above are distinct pairs per token-transformation signature.

Prompt audit (§2.1): the 373-line text is `product_mapping_guidance` in any
`eval-datasets/outputs/*.jsonl` case, or
`json.loads(row["inputs"])["product_mapping_guidance"]` from the run export.
