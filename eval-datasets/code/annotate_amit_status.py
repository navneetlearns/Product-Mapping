#!/usr/bin/env python3
"""Annotate Amit's issue list with the Turn Over Report status of each retailer / distributor.

His workbook is invoice-level rows. Each row gets the status of that party on that invoice in the
report (deduplicated population), the report's own name for the party, and the key the match used.
A party-level sheet collapses the rows to one status per retailer / distributor. The SKU tab is
copied untouched.

    python3 code/annotate_amit_status.py \
        --amit "<Mapping issue list shared by amit.xlsx>" \
        --out "<03_review/Mapping issue list shared by amit - with report status.xlsx>"

Matching is GST first, then phone (retailers), then normalised name, inside the report rows carrying
the same invoice number. The fallback matters: his GSTs carry typos and the report's own GST column
carries truncated values, so a GST-only join marks mapped parties as absent. The `Matched on` column
says which key was used and flags `GST differs` when his GST is not the report's.
"""
from __future__ import annotations

import argparse
import difflib
import sys
from collections import Counter
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amit_overlap as A  # noqa: E402
import build_turnover_truth as B  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
F_DIR = Path("/mnt/f/Product/zotok/Product Mapping/Mapping Inconsistency")
DEFAULT_AMIT = F_DIR / "01_source" / "Mapping issue list shared by amit.xlsx"
DEFAULT_OUT = F_DIR / "03_review" / "Mapping issue list shared by amit - with report status.xlsx"

ROW_ORDER = ["Not mapped", "Partly mapped", "Mapped", "Invoice not in report", "No match on invoice"]
PARTY_ORDER = ["Not mapped", "Partly mapped", "Fully mapped", "No match found"]

FILL = {
    "Not mapped": ("FFC7CE", "9C0006"),
    "Partly mapped": ("FFEB9C", "9C6500"),
    "Mapped": ("C6EFCE", "006100"),
    "Fully mapped": ("C6EFCE", "006100"),
    "Invoice not in report": ("EDEDED", "595959"),
    "No match on invoice": ("EDEDED", "595959"),
    "No match found": ("EDEDED", "595959"),
}
THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
HEAD_FILL = PatternFill("solid", start_color="14402A")


def fill_cell(cell, status: str) -> None:
    bg, fg = FILL.get(status, ("FFFFFF", "000000"))
    cell.fill = PatternFill("solid", start_color=bg)
    cell.font = Font(bold=True, color=fg)
    cell.alignment = Alignment(horizontal="left", vertical="center")
    cell.border = BORDER


def head(cell) -> None:
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = HEAD_FILL
    cell.alignment = Alignment(horizontal="left", vertical="center")
    cell.border = BORDER


def report_name(row, side: str) -> str:
    matched = row.get("Matched Distributor Name" if side == "dist" else "Matched Retailer Name")
    base = row.get("Distributor Name" if side == "dist" else "Retailer Name")
    for v in (matched, base):
        if v is not None and str(v).strip() and str(v).strip().lower() != "nan":
            return str(v).strip()
    return ""


def gst_near(a: str, b: str) -> int:
    """Hamming distance between two 15-char GSTs, or 99 if not comparable."""
    if not a or not b or len(a) != 15 or len(b) != 15:
        return 99
    return sum(x != y for x, y in zip(a, b))


def fuzzy_name(cand, nm: str, spec: dict):
    """Closest report party on the invoice, only when the names are genuinely close."""
    if not nm:
        return None
    best, best_score = None, 0.0
    for _, row in cand.iterrows():
        rn = row[spec["ncol"]]
        if not rn:
            continue
        ratio = difflib.SequenceMatcher(None, nm, rn).ratio()
        a, b = set(nm.split()), set(rn.split())
        contain = (a <= b or b <= a) and min(len(a), len(b)) >= 2
        score = max(ratio, 1.0 if contain else 0.0)
        if score > best_score:
            best, best_score = row, score
    return best if best is not None and best_score >= 0.85 else None


def name_score(a: str, b: str) -> tuple[float, int]:
    """(difflib ratio, shared significant tokens) between two normalised names."""
    if not a or not b:
        return 0.0, 0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    return ratio, len(ta & tb)


def match_row(rec: dict, cand, spec: dict, side: str) -> tuple[str, str, str, str]:
    """(status, report party name, matched on, report GST) for one of his rows."""
    m = cand[cand[spec["gcol"]] == rec["_gst"]] if rec["_gst"] else cand.iloc[0:0]
    basis = "GST"
    if len(m) == 0 and rec["_gst"]:
        dist = cand[spec["gcol"]].map(lambda g: gst_near(rec["_gst"], str(g) if g else ""))
        near = cand[dist <= 2]
        if len(near):
            # a 1-2 char GST difference is only evidence when the name agrees: the same invoice can
            # carry several near-identical GSTs (e.g. two shops whose PANs differ by one digit)
            scored = [(name_score(rec["_nm"], str(r[spec["ncol"]] or "")), i) for i, r in near.iterrows()]
            (ratio, shared), idx = max(scored, key=lambda x: (x[0][1], x[0][0]))
            if shared >= 1 or ratio >= 0.5:
                m, basis = cand.loc[[idx]], "GST near + name"
    if len(m) == 0 and rec["_phone"] and spec["phone_col"]:
        m = cand[cand[spec["phone_col"]] == rec["_phone"]]
        basis = "Phone"
    if len(m) == 0 and rec["_nm"]:
        m = cand[cand[spec["ncol"]] == rec["_nm"]]
        basis = "Name"
    if len(m) == 0:
        row = fuzzy_name(cand, rec["_nm"], spec)
        if row is not None:
            m, basis = cand.loc[[row.name]], "Name (fuzzy)"
    if len(m) == 0:
        return "No match on invoice", "", "", ""

    matched_gsts = {str(g) for g in m[spec["gcol"]] if g and str(g).lower() != "nan"}
    if basis in ("Phone", "Name") and rec["_gst"] and rec["_gst"] not in matched_gsts:
        basis += " (GST differs)"
    got, tot = int(m[spec["flag"]].sum()), len(m)
    status = "Mapped" if got == tot else ("Not mapped" if got == 0 else "Partly mapped")
    rgst = next((str(g) for g in m[spec["gcol"]] if g and str(g).lower() != "nan"), "")
    return status, report_name(m.iloc[0], side), basis, rgst


def annotate_tab(ws, spec: dict, head_by_inv: dict, side: str) -> tuple[list[dict], dict]:
    hdr = [str(c.value).strip() if c.value is not None else "" for c in ws[1]]
    ncols = len(hdr)
    new_cols = ["Turn Over Report status", "Report party name", "Matched on"]
    for j, h in enumerate(new_cols, start=ncols + 1):
        head(ws.cell(row=1, column=j, value=h))
    ws.column_dimensions[get_column_letter(ncols + 1)].width = 24
    ws.column_dimensions[get_column_letter(ncols + 2)].width = 38
    ws.column_dimensions[get_column_letter(ncols + 3)].width = 22

    records = []
    for r in range(2, ws.max_row + 1):
        raw = {h: ws.cell(row=r, column=i + 1).value for i, h in enumerate(hdr)}
        if all(v is None for v in raw.values()):
            continue
        rec = dict(raw)
        rec["_gst"] = A.gst(raw.get(spec["gst_hdr"]))
        rec["_nm"] = A.nm(raw.get(spec["name_hdr"]))
        rec["_phone"] = A.phone(raw.get(spec["phone_hdr"])) if spec["phone_hdr"] else ""
        rec["_party"] = rec["_gst"] or ("NAME:" + rec["_nm"])
        rec["_inv"] = A.inv(raw.get("Invoice Number"))

        cand = head_by_inv.get(rec["_inv"])
        if cand is None:
            status, rname, basis, rgst = "Invoice not in report", "", "", ""
        else:
            status, rname, basis, rgst = match_row(rec, cand, spec, side)
        rec.update(_status=status, _rname=rname, _basis=basis, _rgst=rgst)
        records.append(rec)

        fill_cell(ws.cell(row=r, column=ncols + 1, value=status), status)
        for j, val in ((ncols + 2, rname or None), (ncols + 3, basis or None)):
            c = ws.cell(row=r, column=j, value=val)
            c.border = BORDER
            c.alignment = Alignment(horizontal="left", vertical="center")

    return records, dict(Counter(r["_status"] for r in records))


def party_verdict(verdicts: list[str]) -> str:
    matched = [v for v in verdicts if v in ("Mapped", "Not mapped", "Partly mapped")]
    if not matched:
        return "No match found"
    if all(v == "Not mapped" for v in matched):
        return "Not mapped"
    if all(v == "Mapped" for v in matched):
        return "Fully mapped"
    return "Partly mapped"


def add_party_sheet(wb, sides: dict) -> dict:
    ws = wb.create_sheet("Party status")
    ws.append(["Side", "Party (as in Amit's sheet)", "GST in his sheet", "Rows",
               "Turn Over Report status", "Report party name", "GST in report", "Matched on"])
    parties = {}
    for side_label, (spec, records) in sides.items():
        groups: dict[str, list[dict]] = {}
        for rec in records:
            groups.setdefault(rec["_party"], []).append(rec)
        for key, recs in groups.items():
            pname = next((str(r.get(spec["name_hdr"])).strip() for r in recs
                          if r.get(spec["name_hdr"]) not in (None, "")), key)
            gst = next((r["_gst"] for r in recs if r["_gst"]), "")
            status = party_verdict([r["_status"] for r in recs])
            rname = next((r["_rname"] for r in recs if r["_rname"]), "")
            rgst = next((r["_rgst"] for r in recs if r["_rgst"]), "")
            bases = [r["_basis"] for r in recs if r["_basis"]]
            basis = " / ".join(sorted(set(bases)))
            parties[(side_label, key)] = dict(side=side_label, name=pname, gst=gst, rows=len(recs),
                                              status=status, rname=rname, rgst=rgst, basis=basis)

    order = {"Not mapped": 0, "Partly mapped": 1, "No match found": 2, "Fully mapped": 3}
    rows = sorted(parties.values(), key=lambda p: (p["side"] != "Distributor", order[p["status"]], -p["rows"]))
    for p in rows:
        ws.append([p["side"], p["name"], p["gst"] or None, p["rows"], p["status"],
                   p["rname"] or None, p["rgst"] or None, p["basis"] or None])
        fill_cell(ws.cell(row=ws.max_row, column=5), p["status"])

    for i, w in enumerate([13, 46, 20, 8, 24, 40, 20, 22], start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for c in ws[1]:
        head(c)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:H{ws.max_row}"
    return parties


def add_summary(wb, sides: dict, parties: dict, source: str) -> None:
    ws = wb.create_sheet("Status summary")
    ws["A1"] = "Amit's issue list — status in the Turn Over Report"
    ws["A1"].font = Font(bold=True, size=14, color="14402A")
    ws["A2"] = f"Report: {source}"
    ws["A3"] = ("Status is the party's mapping state on the invoice named in his row, from the "
                "deduplicated report population. Match: GST, then phone (retailers), then normalised "
                "name, inside the same invoice number.")
    ws["A4"] = ("'Matched on' shows the key used. '(GST differs)' = his GST is not the report's for "
                "that party — the party is in the report under a different/typo'd GST.")
    ws["A5"] = ("Row status = that invoice; party status = all of that party's rows in his sheet. "
                "SKU tab unchanged. His Value column is untouched and not used.")
    for r in (2, 3, 4, 5):
        ws.cell(row=r, column=1).font = Font(italic=True, size=10, color="595959")

    r0 = 7
    ws.cell(row=r0, column=1, value="Rows in his sheet").font = Font(bold=True, size=11)
    headers = ["Tab", "Rows"] + ROW_ORDER
    for j, h in enumerate(headers, start=1):
        head(ws.cell(row=r0 + 1, column=j, value=h))
    row = r0 + 2
    for side_label, (spec, records) in sides.items():
        counts = Counter(r["_status"] for r in records)
        ws.cell(row=row, column=1, value=side_label)
        ws.cell(row=row, column=2, value=len(records))
        for j, h in enumerate(ROW_ORDER, start=3):
            ws.cell(row=row, column=j, value=counts.get(h, 0))
        row += 1

    r1 = row + 2
    ws.cell(row=r1, column=1, value="Parties (one row per retailer / distributor)").font = Font(bold=True, size=11)
    headers2 = ["Side", "Parties"] + PARTY_ORDER
    for j, h in enumerate(headers2, start=1):
        head(ws.cell(row=r1 + 1, column=j, value=h))
    row = r1 + 2
    for side_label in sides:
        vals = [p for p in parties.values() if p["side"] == side_label]
        counts = Counter(p["status"] for p in vals)
        ws.cell(row=row, column=1, value=side_label)
        ws.cell(row=row, column=2, value=len(vals))
        for j, h in enumerate(PARTY_ORDER, start=3):
            ws.cell(row=row, column=j, value=counts.get(h, 0))
        row += 1

    ws.cell(row=row + 2, column=1,
            value="Legend: Mapped / Fully mapped = every report line identified · Partly mapped = "
                  "some identified · Not mapped = none identified · grey = no match found on that "
                  "invoice (check by hand).").font = Font(italic=True, size=10, color="595959")
    ws.cell(row=row + 3, column=1,
            value="Manual master check (user, 02-Nov): of the 45 strictly unmapped distributors, none "
                  "appear in the distributor master except Jeevish Sales Corporation.").font = Font(
        italic=True, size=10, color="595959")
    for i, w in enumerate([16, 10, 16, 16, 14, 22, 22], start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amit", default=str(DEFAULT_AMIT))
    ap.add_argument("--report", default=str(A.DEFAULT_REPORT))
    ap.add_argument("--cache-dir", default=str(HERE / "outputs" / "turnover-truth"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()

    df, _ = B.load_report(Path(a.report), Path(a.cache_dir), refresh=False)
    df = B.normalise(df)
    head = df[~df["dup"]].copy()
    head["inv_n"] = head["Invoice Number"].map(A.inv)
    head["ret_phone"] = head["Retailer Mobile No"].map(A.phone)
    head_by_inv = {k: v for k, v in head.groupby("inv_n")}

    wb = openpyxl.load_workbook(a.amit)
    sides = {}
    for side_label, sheet, spec in (
        ("Distributor", "Distributor not mapped", A.SIDES["distributors"]),
        ("Retailer", "retailer not mapped", A.SIDES["retailers"]),
    ):
        records, counts = annotate_tab(wb[sheet], spec, head_by_inv,
                                       "dist" if side_label == "Distributor" else "ret")
        sides[side_label] = (spec, records)
        print(f"{side_label}: {len(records)} rows | " + " · ".join(f"{k} {v}" for k, v in counts.items()))
        print("  matched on:", dict(Counter(r["_basis"] for r in records)))

    parties = add_party_sheet(wb, sides)
    add_summary(wb, sides, parties, Path(a.report).name)
    wb.active = wb.sheetnames.index("Distributor not mapped")
    wb.save(a.out)
    print(f"parties: {len(parties)} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
