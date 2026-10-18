#!/usr/bin/env python3
"""Build a VLOOKUP-ready workbook: eval test cases (sku / brand / titleonly)
vs the current product catalogue.

Purpose: let the user MANUALLY verify, one product at a time, how many of the
eval test-case products exist in the catalogue (products-18-October-2026
export), per mapping logic. This script only CLEANS and lays out the data —
it computes no coverage counts.

Inputs (defaults):
  - catalogue : /mnt/f/Product/zotok/Product Mapping/Eval Related Product Mapping/products-18-October-2026-10-52-16.csv
  - case sets : ../outputs/{sku_cases_100,brand_in_title_cases_100,titleonly_cases_100}.jsonl

Output:
  - ../outputs/catalogue_check/eval_testcases_vs_catalogue_vlookup.xlsx
  - ../outputs/catalogue_check/catalogue_cleaned.csv (same cleaned catalogue)

Sheets: How to use | Catalogue | SKU cases | Brand cases | Titleonly cases | Candidates
All SKU / code columns are written as TEXT so VLOOKUPs stay type-consistent.
"""
import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "outputs" / "catalogue_check"

DEF_CAT = Path("/mnt/f/Product/zotok/Product Mapping/Eval Related Product Mapping/products-18-October-2026-10-52-16.csv")
DEF_SETS = {
    "SKU cases": OUTDIR.parent / "sku_cases_100.jsonl",
    "Brand cases": OUTDIR.parent / "brand_in_title_cases_100.jsonl",
    "Titleonly cases": OUTDIR.parent / "titleonly_cases_100.jsonl",
}
LOGIC_TAG = {"SKU cases": "sku", "Brand cases": "brand", "Titleonly cases": "titleonly"}

CODE_RE = re.compile(r"\d{8}")


def _to_bytes(s: str) -> bytes:
    """Re-encode a mojibake string back to its original bytes.

    cp1252 for everything it defines (so '€' -> 0x80), raw ord() for the
    C1 range it leaves undefined (0x80-0x9F like '\\x9d'), which is exactly
    where utf-8 continuation bytes land in this class of mojibake."""
    out = bytearray()
    for ch in s:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) <= 0xFF:
                out.append(ord(ch))
            else:
                raise
    return bytes(out)


def clean(s: str) -> str:
    """Repair mojibake, strip zero-widths/C1 controls, collapse whitespace."""
    if s is None:
        return ""
    s = str(s)
    # 1) mojibake repair: byte round trip -> utf-8 when it cleanly decodes
    if any(ch in s for ch in ("Â", "â", "€", "Ã")):
        try:
            fixed = _to_bytes(s).decode("utf-8")
            s = fixed
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    # 2) drop zero-width & C1 control chars, nbsp -> space
    s = "".join(ch for ch in s if unicodedata.category(ch) not in ("Cf",) and not (0x80 <= ord(ch) <= 0x9F))
    s = s.replace("\xa0", " ").replace("\u200b", "")
    # 3) collapse whitespace
    s = " ".join(s.split())
    return s


def sku_sort_key(s):
    return (len(s), s)  # numeric-ish ordering by length then digits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=str(DEF_CAT))
    ap.add_argument("--copy-to")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ---------------- catalogue ----------------
    raw = list(csv.DictReader(open(args.catalogue, newline="", encoding="utf-8")))
    cat_rows = []
    desc_ne_name = []
    for r in raw:
        name = clean(r.get("Product Name"))
        desc = clean(r.get("Description"))
        sku = clean(r.get("Sku Code"))
        if desc and desc != name:
            desc_ne_name.append((sku, name, desc, r.get("Description")))
        cat_rows.append((sku, name))
    # dedupe check (keep first occurrence, report)
    seen, dedup = {}, []
    for sku, name in cat_rows:
        if sku in seen:
            print(f"WARN duplicate sku {sku}: {name!r} vs {seen[sku]!r} -> keeping first")
            continue
        seen[sku] = name
        dedup.append((sku, name))
    cat_rows = sorted(dedup, key=lambda t: sku_sort_key(t[0]))
    print(f"catalogue: {len(raw)} raw rows -> {len(cat_rows)} clean rows; "
          f"desc!=name: {len(desc_ne_name)}")
    for d in desc_ne_name[:3]:
        print("   desc oddity:", d)
    bad = [s for s, _ in cat_rows if not re.fullmatch(r"\d+", s)]
    print("   non-digit skus after clean:", len(bad), bad[:5])
    moji_raw = [r.get("Product Name") for r in raw
                if any(ch in (r.get("Product Name") or "") for ch in ("Â", "â", "Ã", "€"))]
    still = [(s, n) for s, n in cat_rows if any(ch in n for ch in ("Â", "â", "Ã", "€"))]
    print(f"   mojibake in raw names: {len(moji_raw)} | remaining after clean: {len(still)}")
    for n in moji_raw:
        print("      fixed:", repr(n), "->", repr(clean(n)))

    # ---------------- case sets ----------------
    case_sheets = {}
    candidates_rows = []
    for sheet, path in DEF_SETS.items():
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        tag = LOGIC_TAG[sheet]
        out = []
        for i, ex in enumerate(rows, 1):
            li = ex["inputs"]["line_items"][0]
            exp = ex["outputs"]["expected"][0]
            md = ex.get("metadata", {})
            title = clean(li.get("product_name"))
            m = CODE_RE.search(title)
            code = m.group(0) if m else ""
            cr = (ex["inputs"].get("catalog_results") or {})
            cands = list(cr.values())[0] if cr else []
            cand_txt = "; ".join(f"{c.get('sku_code')} | {clean(c.get('product_name'))}" for c in cands)
            cid = f"{tag.upper() if tag!='sku' else 'SKU'}-{i:03d}"
            out.append([
                cid, title, code, clean(li.get("hsn_sac_code")),
                exp["expected_status"], str(exp["expected_sku"] or ""),
                cand_txt, "", "", "", "",
                md.get("source_experiment", ""), md.get("source_invoice_id", ""),
                md.get("source_serial_no", ""), len(cands),
            ])
            for c in cands:
                candidates_rows.append([
                    cid, tag, title, str(c.get("sku_code") or ""),
                    clean(c.get("product_name")), clean(c.get("hsn_code")),
                ])
        case_sheets[sheet] = out
        print(f"{sheet}: {len(out)} cases, {sum(1 for r in out if r[2])} with code, "
              f"candidates rows {len(candidates_rows)} so far")

    # ---------------- workbook ----------------
    wb = openpyxl.Workbook()
    HEAD_FILL = PatternFill("solid", fgColor="D9D9D9")
    FILL_FILL = PatternFill("solid", fgColor="FFF2CC")  # manual-fill columns
    bold = Font(bold=True)

    def write_sheet(ws, headers, rows, widths, fill_cols=(), freeze="A2", filt=True):
        for j, h in enumerate(headers, 1):
            c = ws.cell(row=1, column=j, value=h)
            c.font = bold
            c.fill = FILL_FILL if j in fill_cols else HEAD_FILL
        for i, row in enumerate(rows, 2):
            for j, v in enumerate(row, 1):
                if v == "":
                    v = None
                ws.cell(row=i, column=j, value=v)
        for j, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(j)].width = w
        if freeze:
            ws.freeze_panes = freeze
        if filt:
            ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(rows)+1}"

    # How to use
    ws = wb.active
    ws.title = "How to use"
    lines = [
        ("Eval test cases vs Catalogue — VLOOKUP check", True),
        ("", False),
        ("What: the 300 eval test cases we built (100 per mapping logic) + the current product catalogue,", False),
        ("cleaned so each test-case product can be mapped one-to-one to a catalogue product by hand.", False),
        ("", False),
        ("Test case sets (built 2026-10-17, source: ~/AgentWork/product-mapping/eval-datasets):", False),
        ("  SKU cases       — title carries an embedded 8-digit SKU code (exact-code path); 100 cases.", False),
        ("  Brand cases     — brand name in the title, no code (brand + fuzzy name path); 100 cases.", False),
        ("  Titleonly cases — plain title, no code, no brand (pure name/dimension path); 100 cases.", False),
        ("", False),
        ("Catalogue: products-18-October-2026-10-52-16.csv, exported 2026-10-18 — 2,335 products.", False),
        ("  Cleaned: whitespace / zero-width / mojibake fixed; duplicate 'Description' column dropped", False),
        ("  (it simply repeated Product Name); no duplicate SKUs; sorted by Sku Code.", False),
        ("", False),
        ("How to check a case: find its product in the 'Catalogue' sheet, then fill the yellow columns:", False),
        ("  In Catalogue? (Y/N) | Catalogue SKU | Catalogue Product Name (+ Notes).", False),
        ("  VLOOKUP example:  =VLOOKUP(C2, Catalogue!$A:$B, 2, FALSE)   (C2 = embedded_code)", False),
        ("", False),
        ("Column notes:", False),
        ("  embedded_code = 8-digit code inside the title (SKU cases only; equals expected SKU).", False),
        ("  gt_status / gt_sku = ground truth for that case on the original eval.", False),
        ("  candidates = the 5 retrieval candidates the eval model saw (sku | product name).", False),
        ("  'Candidates' sheet = same candidates exploded one-per-row, for VLOOKUP against the Catalogue.", False),
        ("  ALL SKU/code columns are stored as TEXT — keep them text in your lookups.", False),
    ]
    for i, (t, b) in enumerate(lines, 1):
        c = ws.cell(row=i, column=1, value=t)
        if b:
            c.font = bold
    ws.column_dimensions["A"].width = 110

    # Catalogue
    ws = wb.create_sheet("Catalogue")
    write_sheet(ws, ["Sku Code", "Product Name"], [[s, n] for s, n in cat_rows], [14, 70])
    # force text on sku col
    for i in range(2, len(cat_rows) + 2):
        ws.cell(row=i, column=1).number_format = "@"

    # case sheets
    case_headers = [
        "case_id", "input_title", "embedded_code", "hsn", "gt_status", "gt_sku", "candidates",
        "In Catalogue? (Y/N)", "Catalogue SKU", "Catalogue Product Name", "Notes",
        "src_experiment", "src_invoice_id", "src_serial", "n_candidates",
    ]
    widths = [11, 55, 14, 11, 22, 12, 75, 15, 13, 38, 25, 11, 38, 9, 11]
    for sheet in ("SKU cases", "Brand cases", "Titleonly cases"):
        ws = wb.create_sheet(sheet)
        write_sheet(ws, case_headers, case_sheets[sheet], widths, fill_cols=(8, 9, 10, 11))
        for i in range(2, len(case_sheets[sheet]) + 2):
            for col in (3, 6):
                ws.cell(row=i, column=col).number_format = "@"

    # candidates
    ws = wb.create_sheet("Candidates")
    write_sheet(ws, ["case_id", "logic", "input_title", "candidate_sku", "candidate_name", "candidate_hsn"],
                candidates_rows, [11, 9, 48, 13, 60, 12])
    for i in range(2, len(candidates_rows) + 2):
        ws.cell(row=i, column=4).number_format = "@"

    xlsx = OUTDIR / "eval_testcases_vs_catalogue_vlookup.xlsx"
    wb.save(xlsx)
    print("saved:", xlsx)

    # csv sidecar of cleaned catalogue
    csv_path = OUTDIR / "catalogue_cleaned.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Sku Code", "Product Name"])
        w.writerows(cat_rows)
    print("saved:", csv_path)

    if args.copy_to:
        import shutil
        dest = Path(args.copy_to) / xlsx.name
        shutil.copy2(xlsx, dest)
        print("copied to:", dest)

    # ---------------- QA: reload and count ----------------
    wb2 = openpyxl.load_workbook(xlsx, read_only=True)
    for name in wb2.sheetnames:
        ws2 = wb2[name]
        print(f"QA {name}: {ws2.max_row - 1 if name not in ('How to use',) else ws2.max_row} rows")
    # spot checks
    ws2 = wb2["SKU cases"]
    print("QA spot SKU case 2:", [ws2.cell(row=3, column=j).value for j in (1, 3, 6)])
    ws2 = wb2["Catalogue"]
    print("QA spot catalogue row 2:", [ws2.cell(row=2, column=j).value for j in (1, 2)])
    print("QA done")


if __name__ == "__main__":
    main()
