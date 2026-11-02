#!/usr/bin/env python3
"""Unique retailer / distributor list from Amit's issue list, with Turn Over Report status.

One row per party (grouped by normalised name), not one row per invoice. Each party carries its
report status, the report party it matched, the match key, and — for distributors — the manual
master check the user reported on 02-Nov. The SKU tab is not part of this view; the source workbook
stays untouched.

    python3 code/annotate_amit_status.py \
        --amit "<Mapping issue list shared by amit.xlsx>" \
        --out "<03_review/Mapping issue list shared by amit - with report status.xlsx>"

Matching inside the invoice: GST, then GST-near + name, then phone (retailers), then normalised
name, then a conservative name fuzzy. Every non-exact match is named in `Matched on`.
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

STATUS_ORDER = {"Not mapped": 0, "Partly mapped": 1, "No match found": 2, "Fully mapped": 3}
FILL = {
    "Not mapped": ("FFC7CE", "9C0006"),
    "Partly mapped": ("FFEB9C", "9C6500"),
    "Fully mapped": ("C6EFCE", "006100"),
    "No match found": ("EDEDED", "595959"),
}
THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
HEAD_FILL = PatternFill("solid", start_color="14402A")


def head(cell) -> None:
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = HEAD_FILL
    cell.alignment = Alignment(horizontal="left", vertical="center")
    cell.border = BORDER


def fill_cell(cell, status: str) -> None:
    bg, fg = FILL.get(status, ("FFFFFF", "000000"))
    cell.fill = PatternFill("solid", start_color=bg)
    cell.font = Font(bold=True, color=fg)
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
    if not a or not b or len(a) != 15 or len(b) != 15:
        return 99
    return sum(x != y for x, y in zip(a, b))


def name_score(a: str, b: str) -> tuple[float, int]:
    if not a or not b:
        return 0.0, 0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    return ratio, len(ta & tb)


def fuzzy_name(cand, nm: str, spec: dict):
    if not nm:
        return None
    best, best_score = None, 0.0
    for _, row in cand.iterrows():
        rn = row[spec["ncol"]]
        if not rn:
            continue
        ratio, shared = name_score(nm, rn)
        contain = shared >= 2 and (set(nm.split()) <= set(rn.split()) or set(rn.split()) <= set(nm.split()))
        score = max(ratio, 1.0 if contain else 0.0)
        if score > best_score:
            best, best_score = row, score
    return best if best is not None and best_score >= 0.85 else None


def match_row(rec: dict, cand, spec: dict, side: str) -> tuple[str, str, str, str]:
    """(status, report party name, matched on, report GST) for one of his rows."""
    m = cand[cand[spec["gcol"]] == rec["_gst"]] if rec["_gst"] else cand.iloc[0:0]
    basis = "GST"
    if len(m) == 0 and rec["_gst"]:
        dist = cand[spec["gcol"]].map(lambda g: gst_near(rec["_gst"], str(g) if g else ""))
        near = cand[dist <= 2]
        if len(near):
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


def party_verdict(verdicts: list[str]) -> str:
    matched = [v for v in verdicts if v in ("Mapped", "Not mapped", "Partly mapped")]
    if not matched:
        return "No match found"
    if all(v == "Not mapped" for v in matched):
        return "Not mapped"
    if all(v == "Mapped" for v in matched):
        return "Fully mapped"
    return "Partly mapped"


def read_party_rows(wb, sheet: str, spec: dict, side: str, head_by_inv: dict) -> list[dict]:
    ws = wb[sheet]
    hdr = [str(c.value).strip() if c.value is not None else "" for c in ws[1]]
    out = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if all(v is None for v in row):
            continue
        raw = dict(zip(hdr, row))
        rec = dict(raw)
        rec["_gst"] = A.gst(raw.get(spec["gst_hdr"]))
        rec["_nm"] = A.nm(raw.get(spec["name_hdr"]))
        rec["_phone"] = A.phone(raw.get(spec["phone_hdr"])) if spec["phone_hdr"] else ""
        rec["_inv"] = A.inv(raw.get("Invoice Number"))
        cand = head_by_inv.get(rec["_inv"])
        if cand is None:
            rec.update(_status="Invoice not in report", _rname="", _basis="", _rgst="")
        else:
            status, rname, basis, rgst = match_row(rec, cand, spec, side)
            rec.update(_status=status, _rname=rname, _basis=basis, _rgst=rgst)
        out.append(rec)
    return out


def unique_parties(records: list[dict], name_key: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for rec in records:
        groups.setdefault(rec["_nm"], []).append(rec)
    parties = []
    for key, recs in groups.items():
        gsts = sorted({r["_gst"] for r in recs if r["_gst"]})
        name = next((str(r.get(name_key)).strip() for r in recs
                     if r.get(name_key) not in (None, "")), key)
        rname = next((r["_rname"] for r in recs if r["_rname"]), "")
        rgst = next((r["_rgst"] for r in recs if r["_rgst"]), "")
        bases = [r["_basis"] for r in recs if r["_basis"]]
        parties.append(dict(
            key=key, name=name, gsts=gsts, rows=len(recs),
            status=party_verdict([r["_status"] for r in recs]),
            rname=rname, rgst=rgst, basis=" / ".join(sorted(set(bases))),
        ))
    return sorted(parties, key=lambda p: (STATUS_ORDER[p["status"]], -p["rows"], p["name"]))


def strict_unmapped_distributors(df) -> list[dict]:
    head = df[~df["dup"]].copy()
    per = head.groupby("dist_key")["dist_ok"].agg(["sum", "count"])
    keys = set(per[per["sum"] == 0].index)
    sub = head[head.dist_key.isin(keys)]
    out = []
    for key, g in sub.groupby("dist_key"):
        out.append(dict(
            key=key, name=str(g["Distributor Name"].iloc[0]).strip(),
            gst=str(g["gst_ok"].iloc[0]) if str(g["gst_ok"].iloc[0]) != "nan" else "",
            lines=len(g),
        ))
    return sorted(out, key=lambda d: (-d["lines"], d["name"]))


def master_check(party: dict, strict: list[dict]) -> str:
    """The user's 02-Nov manual check, applied to the party it covers."""
    by_name = [s for s in strict if A.nm(s["name"]) == party["key"]]
    by_gst = [s for s in strict if s["gst"] and s["gst"] in party["gsts"]]
    hit = by_gst[0] if by_gst else (by_name[0] if by_name else None)
    if hit is None:
        return "Not in the 45 checked"
    if "AAACH2676Q" in hit["gst"] or "BIRLANU LIMITED" in A.nm(hit["name"]):
        return "BirlaNu's own GSTIN (excluded)"
    if not hit["gst"]:
        return "No GST — not checked"
    if "JEEVISH" in A.nm(hit["name"]):
        return "In master — not matched"
    return "Not in master"


def write_party_sheet(wb, title: str, parties: list[dict], with_master: bool) -> None:
    ws = wb.create_sheet(title)
    cols = ["Party (as in Amit's sheet)", "GST in his sheet", "Rows in his list",
            "Turn Over Report status", "Report party name", "GST in report", "Matched on"]
    if with_master:
        cols.append("Distributor master (your check, 02-Nov)")
    ws.append(cols)
    for p in parties:
        row = [p["name"], ", ".join(p["gsts"]) or None, p["rows"], p["status"],
               p["rname"] or None, p["rgst"] or None, p["basis"] or None]
        if with_master:
            row.append(p["master"])
        ws.append(row)
        fill_cell(ws.cell(row=ws.max_row, column=4), p["status"])
    widths = [46, 34, 8, 22, 40, 20, 26] + ([30] if with_master else [])
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for c in ws[1]:
        head(c)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{ws.max_row}"


def write_master_sheet(wb, strict: list[dict], his_parties: list[dict]) -> None:
    ws = wb.create_sheet("Distributor master check")
    ws.append(["#", "Distributor in the report", "GST", "Report lines", "In Amit's sheet?",
               "Manual check (your 02-Nov verification)"])
    his_names = {p["key"] for p in his_parties}
    his_gsts = {g for p in his_parties for g in p["gsts"]}
    for i, s in enumerate(strict, start=1):
        nm = A.nm(s["name"])
        in_sheet = "Yes" if nm in his_names or (s["gst"] and s["gst"] in his_gsts) else ""
        if "AAACH2676Q" in s["gst"] or "BIRLANU LIMITED" in nm:
            verdict = "BirlaNu's own GSTIN — not a distributor to onboard"
        elif not s["gst"]:
            verdict = "No GST on the invoice — not part of the GST check"
        elif "JEEVISH" in nm:
            verdict = "In master — not matched"
        else:
            verdict = "Not in master"
        ws.append([i, s["name"], s["gst"] or None, s["lines"], in_sheet or None, verdict])
        if verdict == "Not in master":
            ws.cell(row=ws.max_row, column=6).fill = PatternFill("solid", start_color="FFC7CE")
        elif verdict.startswith("In master"):
            ws.cell(row=ws.max_row, column=6).fill = PatternFill("solid", start_color="FFEB9C")
        else:
            ws.cell(row=ws.max_row, column=6).fill = PatternFill("solid", start_color="EDEDED")
    for i, w in enumerate([5, 50, 20, 10, 14, 44], start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for c in ws[1]:
        head(c)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:F{ws.max_row}"


def write_summary(wb, dists: list[dict], rets: list[dict], source: str, strict: list[dict]) -> None:
    ws = wb.create_sheet("Status summary", 0)
    ws["A1"] = "Amit's issue list — unique retailers and distributors, with report status"
    ws["A1"].font = Font(bold=True, size=14, color="14402A")
    ws["A2"] = f"Report: {source}"
    ws["A3"] = ("One row per party, grouped by normalised name (his sheet is invoice-level; duplicate "
                "rows for the same shop are collapsed, all GSTs kept in one cell).")
    ws["A4"] = ("Status is the party's mapping state in the deduplicated report. Match inside the "
                "invoice: GST, then GST-near + name, then phone (retailers), then name / fuzzy; "
                "`Matched on` names the key and flags `(GST differs)`.")
    ws["A5"] = ("`Distributor master check` applies the 02-Nov manual verification: every GST-bearing "
                "distributor among the 45 report-unmapped is absent from the master except Jeevish "
                "Sales Corporation; BirlaNu's own GSTINs and no-GST records are marked separately.")
    for r in range(2, 6):
        ws.cell(row=r, column=1).font = Font(italic=True, size=10, color="595959")

    r0 = 7
    ws.cell(row=r0, column=1, value="Unique parties").font = Font(bold=True, size=11)
    heads = ["Side", "Parties", "Not mapped", "Partly mapped", "Fully mapped", "No match found"]
    for j, h in enumerate(heads, start=1):
        head(ws.cell(row=r0 + 1, column=j, value=h))
    row = r0 + 2
    for label, parties in (("Distributor", dists), ("Retailer", rets)):
        counts = Counter(p["status"] for p in parties)
        ws.cell(row=row, column=1, value=label)
        ws.cell(row=row, column=2, value=len(parties))
        for j, h in enumerate(heads[2:], start=3):
            ws.cell(row=row, column=j, value=counts.get(h, 0))
        row += 1

    r1 = row + 2
    ws.cell(row=r1, column=1, value="Distributor master check (45 report-unmapped)").font = Font(bold=True, size=11)
    heads2 = ["Verdict", "Distributors"]
    for j, h in enumerate(heads2, start=1):
        head(ws.cell(row=r1 + 1, column=j, value=h))
    verdicts = []
    for s in strict:
        nm = A.nm(s["name"])
        if "AAACH2676Q" in s["gst"] or "BIRLANU LIMITED" in nm:
            verdicts.append("BirlaNu's own GSTIN")
        elif not s["gst"]:
            verdicts.append("No GST — not checked")
        elif "JEEVISH" in nm:
            verdicts.append("In master — not matched")
        else:
            verdicts.append("Not in master")
    row = r1 + 2
    for k, v in Counter(verdicts).most_common():
        ws.cell(row=row, column=1, value=k)
        ws.cell(row=row, column=2, value=v)
        row += 1
    ws.cell(row=row, column=1, value="Total")
    ws.cell(row=row, column=2, value=len(strict))

    for i, w in enumerate([34, 10, 14, 14, 14, 16], start=1):
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

    wb_src = openpyxl.load_workbook(a.amit, data_only=True)
    d_rows = read_party_rows(wb_src, "Distributor not mapped", A.SIDES["distributors"], "dist", head_by_inv)
    r_rows = read_party_rows(wb_src, "retailer not mapped", A.SIDES["retailers"], "ret", head_by_inv)

    dists = unique_parties(d_rows, "Distributor Name")
    rets = unique_parties(r_rows, "Retailer Name")
    strict = strict_unmapped_distributors(df)
    for p in dists:
        p["master"] = master_check(p, strict)

    print(f"distributors: {len(d_rows)} rows -> {len(dists)} unique | " +
          " · ".join(f"{k} {v}" for k, v in Counter(p["status"] for p in dists).most_common()))
    print(f"retailers:    {len(r_rows)} rows -> {len(rets)} unique | " +
          " · ".join(f"{k} {v}" for k, v in Counter(p["status"] for p in rets).most_common()))
    print("master:", dict(Counter(p["master"] for p in dists)))

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    write_summary(wb, dists, rets, Path(a.report).name, strict)
    write_party_sheet(wb, "Distributors", dists, with_master=True)
    write_party_sheet(wb, "Retailers", rets, with_master=False)
    write_master_sheet(wb, strict, dists)
    wb.active = wb.sheetnames.index("Distributors")
    wb.save(a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
