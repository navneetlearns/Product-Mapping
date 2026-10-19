# Plan — resolve the 42 open cases (eval title → catalogue)

**Status: Proposed — NOT approved.** (2026-10-19)

Scope: the 42 non-code "blank" input products left open by the catalogue check
(58/42 tally, Phase 2). Goal: for each of the 42, establish — using the product
TITLE ONLY (no SKU codes exist in these titles) — whether an equivalent product
exists in the current catalogue export, and which SKU it is.

## Context

- Inputs: the 100 input products of run `turnover-product-mapping-c4cce19d`
  (brand-name set), vs catalogue export `products-18-October-2026-10-52-16.csv`
  (2,335 products). Sources = ONLY the two CSVs in the F: folder.
- Located so far: 48 fuzzy matches (≥0.70) + 10 code-rescued blanks = **58/100**.
- Open: **42** — no embedded code, fuzzy score below 0.70 (or no match at all).
- First-scan aid already exists: `below_threshold_candidates.xlsx` (top-3 proxy
  candidates, best-first). The resolver below supersedes it for these rows.

## What the data already tells us (probed 2026-10-19)

1. **21 of the 42 carry a recorded GT SKU in the run export** (`reference_outputs`
   → MAPPED with `expected_sku`) — and **all 21 SKUs verify present in the
   18-Oct export**. Strong leads (subject to the user's eyeball, not auto-decided).
2. **GT = UNMAPPED is NOT evidence of absence.** Of the other 21, several already
   resolve under the catalogue's own vocabulary:
   - `UPVC COUPLER 3/4"` → `UPVC SOCKET 3/4"` (96000131) — catalogue says SOCKET
   - `BIRLA SWR PIPE TYPE A S/F (6MTR) 75MM` → `SWR Pipe Type A Solfit S/S 6m 75 mm` (96000606)
   - `Pvc Shoe 160mm` → `SWR 45 Elbow Pushfit 160 mm` (96000748) family
   - `BIRLANU ROTTO TANK 3L 1000LITRE` → `BIRLA HIL Water Tank-… 3Layer 1000L` (98000507 family; ROTTO→HIL Water Tank)
   - `11/4"x25kg Birla Column Pipe` → `UPVC soc submersible Pipe` family (96000553–96000567, V4 pipes)
3. Two systematic reasons the fuzzy merge missed them:
   a. **Vocabulary**: REDUCER→Reducing Socket/Elbow/Bush; COUPLER→SOCKET/Soc;
      AGRI PIPE / PVC-pipe-kg→`Pressure Pipe CLASS-N` (2.5kg=CLASS-1, 4kg=LW 4Kg —
      both GT-verified; remaining kg↔CLASS pairs seeded then validated);
      SHOE→"45 Elbow" families; COLUMN PIPE→"UPVC soc submersible Pipe";
      ROTTO TANK→"BIRLA HIL Water Tank"; END CAP→"Plain end cap"; S/F→Solfit;
      MTA/FTA→"TRANS/PLASTICS MTA/FTA" (token BRASS appears 0× in the export).
   b. **Size/format**: 50MM=2", 25MM=1", 40x32MM=1½"x1¼"; 32x25MM=1¼"x1";
      quote/fraction variants ('' vs ", 11/4 vs 1¼); and class/length tokens that
      are separate dimensions (SCH40/80, SDR 13.5, 3MTR/6MTR/6ft/300mm).
4. Expected genuine-absent tail (to be confirmed by family-absence evidence):
   e.g. a BRASS FTA (no brass-token items in the export), `PVC HALFSET` (no
   SET/HALFSET token; needs the 90mm-6kg family eyeball). Likely small (~3–6).

## §0 — Decision gates (answer before the build)

- **G1 — Bar for "found".** (A, recommended) family+type+size match counts as
  found; variant doubts (length/SCH/colour) are flagged not blocking — verdict
  options EXACT / AMBIGUOUS-VARIANT / NOT FOUND. (B) strict: only an exact
  variant counts as found.
- **G2 — GT-lead column.** (A, recommended) show the run's recorded GT SKU as a
  "lead" column (21 of 42 have one, all verified in-export — roughly halves the
  manual work). (B) omit — pure title-only output.
- **G3 — Not-found handling.** (A, recommended) close with export evidence +
  a family-absence log, keep the items on a watch list. (B) escalate to the live
  catalogue via the copilot agent API — needs explicit permission + credentials,
  separate task.
- **G4 — Outputs.** Default (follows established pattern): workbook
  `open_42_resolution.xlsx` in the F: Eval folder + repo artifacts in
  `outputs/catalogue_check/`. Adjust name/location on request.

## Approach

### Phase 1 — resolver (offline, deterministic): `code/build_open_case_resolution.py`

- Normalization v2 (both sides): quote chars unified, fractions (11/4→1¼),
  spelling fixes (degre→degree, INCHI→inch, RED.TEE→REDUCING TEE), brand tokens
  stripped (BIRLA/BIRLANU/HIL/CK), size canon (mm↔inch table incl. 40x32, 3/4x1/2
  dual sizes), length tokens, class tokens.
- Synonym/class tables seeded from evidence (section above); **unit-test them
  against the 21 GT pairs** (each GT SKU must land top-3 for its title).
- Scoring rubric per input×candidate: family anchor + type synonym + size-set
  compatibility + class + length → per-component flags (type✓ size✓ class✓),
  ranked, top-5 kept.
- Outputs: `open_42_resolution.xlsx` + CSV sidecars; copy to the F: folder.

### Phase 2 — verification workbook

- Sheet `How to use` — flags + verdict instructions (2-min read).
- Sheet `Resolve 42` — one row per input (grouped by family bucket): input title |
  GT lead SKU + name | best candidate pair + flags | up to top-3 | **empty
  verdict column for the user** (EXACT / AMBIG-VARIANT / NOT FOUND).
- Sheet `All candidates` — long format (input × rank) for a full sweep.
- Sheet `Not-found evidence` — family counts + example SKUs backing each
  "absent from export" call.

### Phase 3 — build QA gates

- Unit tests: ≥18/21 GT leads ranked #1–#3 (below that, fix tables before ship).
- Determinism: rebuild → byte-identical CSV (house convention).
- No auto-decisions: workbook ships with the verdict column EMPTY.

### Phase 4 — user's manual pass → close-out

- User marks verdicts (target ~30–60 min given the pre-flagged GT leads).
- Final tally = 58 + EXACT(+AMBIG if he counts those) — recorded in README /
  PROJECTS.md; commit locally; push on request.

## Out of scope / guardrails

- Don't touch the Power Query workbook; resolver output is a separate file.
- Sources = only the two CSVs in the F: folder (+ the run export's own GT
  column, same file).
- No live-API calls without the G3 answer and explicit permission.
- The 412xxx caveat stands: export-carried SKUs only.

## Expected shape (to verify at build)

- 21 rows pre-flagged via GT leads; of the remaining 21, most expected
  resolvable by synonym+size matching (examples in section 2); genuine-absent
  tail ~3–6 pending user verdicts.
