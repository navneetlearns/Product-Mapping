#!/usr/bin/env python3
"""Retailers present in the customer master but not mapped on the Turn Over Report.

The matching-fix queue behind Amit's retailer tab: parties whose every invoice in his sheet is
unidentified in the report, yet the platform's presence check says the account exists in the
customer master. One row per party, with the identifiers a developer needs to write/adjust the
match rule (his GST/phone, the workspace record the presence check matched, and how it matched).

    python3 code/build_master_match_queue.py \
        --report "<01_source/Turn Over Report.xlsx>" \
        --out "<03_review/Retailers present but unmapped - 51.xlsx>"

Reads the same sources and reuses the same party logic as annotate_amit_status.py, so the 51 here
are exactly the "present / not mapped" cell of that workbook's summary.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amit_overlap as A  # noqa: E402
import annotate_amit_status as AA  # noqa: E402
import build_turnover_truth as B  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
F_DIR = Path("/mnt/f/Product/zotok/Product Mapping/Mapping Inconsistency")
DEFAULT_AMIT = F_DIR / "01_source" / "Mapping issue list shared by amit.xlsx"
DEFAULT_PRESENCE = F_DIR / "01_source" / "Mapping issues - presence check in catalogue.xlsx"
DEFAULT_REPORT = F_DIR / "01_source" / "Turn Over Report.xlsx"
DEFAULT_OUT = F_DIR / "03_review" / "Retailers present but unmapped - 51.xlsx"

HEAD_FILL = PatternFill("solid", start_color="14402A")
THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def head(cell) -> None:
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = HEAD_FILL
    cell.alignment = Alignment(horizontal="left", vertical="center")
    cell.border = BORDER


def presence_index(path: Path) -> dict[str, dict]:
    """Per party (normalised name): presence verdict, match basis and the workspace record."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["retailer not mapped"]
    it = ws.iter_rows(values_only=True)
    hdr = [str(c).strip() if c is not None else "" for c in next(it)]
    k = {h: i for i, h in enumerate(hdr)}
    out: dict[str, dict] = defaultdict(lambda: {"flags": set(), "basis": set(), "name": "",
                                                "gst": "", "phone": "", "his_phone": "", "rows": 0})
    for row in it:
        if row[k["Retailer Name"]] is None:
            continue
        key = A.nm(str(row[k["Retailer Name"]]))
        rec = out[key]
        rec["rows"] += 1
        flag = str(row[k["In Workspace"]] or "").strip()
        rec["flags"].add(flag)
        if not rec["his_phone"] and k.get("Retailer Mobile No") is not None:
            v = row[k["Retailer Mobile No"]]
            if v not in (None, "", "NA"):
                rec["his_phone"] = str(v).strip()
        if flag != "Yes":
            continue
        basis = str(row[k["Matched By"]] or "").strip()
        if basis:
            rec["basis"].add(basis)
        for src, dst in (("Workspace Name", "name"), ("Workspace GST", "gst"), ("Workspace Phone", "phone")):
            if k.get(src) is not None and not rec[dst]:
                v = row[k[src]]
                if v not in (None, ""):
                    rec[dst] = str(v).strip()
    wb.close()
    return dict(out)


def gst_relation(his: list[str], ws: str) -> str:
    """How the workspace GST relates to the GST(s) in his sheet — the matching signal for dev."""
    gsts = [g for g in his if len(g) == 15]
    if not ws:
        return "no GST in workspace"
    if ws in gsts:
        return "same GST"
    if any(sum(a != b for a, b in zip(g, ws)) <= 2 for g in gsts):
        return "near GST (≤2 chars)"
    return "different GST"


def build(report: Path, amit: Path, presence: Path, cache_dir: Path) -> tuple[list[dict], dict]:
    df, cache_name = B.load_report(report, cache_dir, refresh=False)
    df = B.normalise(df)
    head_df = df[~df["dup"]].copy()
    head_df["inv_n"] = head_df["Invoice Number"].map(A.inv)
    head_df["ret_phone"] = head_df["Retailer Mobile No"].map(A.phone)
    head_by_inv = {k: v for k, v in head_df.groupby("inv_n")}

    wb_src = openpyxl.load_workbook(amit, data_only=True)
    recs = AA.read_party_rows(wb_src, "retailer not mapped", A.SIDES["retailers"], "ret", head_by_inv)
    wb_src.close()
    parties = AA.unique_parties(recs, "Retailer Name")

    pres = presence_index(presence)
    queue = []
    for p in parties:
        if p["status"] != "Not mapped":
            continue
        info = pres.get(p["key"], {})
        if "Yes" not in info.get("flags", set()):
            continue
        queue.append(dict(
            party=p["name"], gsts=p["gsts"], rows=p["rows"], his_phone=info.get("his_phone", ""),
            rname=p["rname"], rgst=p["rgst"], basis=p["basis"],
            ws_name=info.get("name", ""), ws_gst=info.get("gst", ""), ws_phone=info.get("phone", ""),
            ws_basis=" / ".join(sorted(info.get("basis", set()))),
            gst_rel=gst_relation(p["gsts"], info.get("gst", "")),
        ))
    queue.sort(key=lambda q: (-q["rows"], q["party"].upper()))
    meta = dict(cache=cache_name, parties=len(parties), queue=len(queue),
                basis=Counter(q["ws_basis"] for q in queue),
                gst_rel=Counter(q["gst_rel"] for q in queue))
    return queue, meta


def write_workbook(queue: list[dict], meta: dict, out: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Read me"
    rows = [
        ("What this is", "Retailers that exist in the customer master but are not mapped on any of their "
                         "invoices in the Turn Over Report — the matching-fix queue, not onboarding."),
        ("Population", f"Amit's retailer tab (29-Oct): 517 rows → {meta['parties']} unique parties. "
                       f"{meta['queue']} of them are 'not mapped' in the report AND 'In Workspace = Yes'."),
        ("Report", "Turn Over Report.xlsx, 4-Nov export (filters 2026-01-01 → 2026-11-04); deduplicated "
                   "population; Retailer Status = Not Identified on every invoice of the party in his sheet."),
        ("Presence", "Mapping issues - presence check in catalogue.xlsx, 'In Workspace' / 'Matched By' — "
                     "the platform's own check of his 517 rows."),
        ("Fix category", "Matching rule (GST typo / name variant / phone fallback), not master onboarding. "
                         "The 143 absent parties are the onboarding queue."),
        ("Caveat — identity", "His sheet carries GST/phone, but the workspace records returned by the presence "
                              "check carry no GST. 'Matched By' says Name or Phone where the match was not "
                              "GST-based; those are candidates, not confirmed identities."),
        ("Caveat — scope", "Party status aggregates only the invoices in his sheet, not the party's whole "
                           "report history."),
        ("Match basis", " · ".join(f"{k or 'blank'}: {v}" for k, v in sorted(meta["basis"].items(),
                                                                           key=lambda x: -x[1]))),
        ("GST relation", " · ".join(f"{k}: {v}" for k, v in sorted(meta["gst_rel"].items(),
                                                                  key=lambda x: -x[1]))),
        ("How to read GST relation", "same GST = workspace and invoice agree → a pure matching bug. "
                                     "near GST = a typo/variant on one side. different GST = the name "
                                     "match may be a different firm — verify before touching the rule."),
        ("Reproduce", "python3 eval-datasets/code/build_master_match_queue.py --out \"<this file>\""),
    ]
    ws.append(["Retailers present in master but unmapped — 51"])
    ws["A1"].font = Font(bold=True, size=14)
    ws.append([])
    for k, v in rows:
        ws.append([k, v])
        ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
        ws.cell(row=ws.max_row, column=2).alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 118
    for r in range(3, ws.max_row + 1):
        ws.row_dimensions[r].height = None

    ws = wb.create_sheet("Retailers (51)")
    cols = ["#", "Retailer (as in Amit's sheet)", "GST in his sheet", "Phone in his sheet",
            "Rows in his list", "Presence matched by", "GST relation", "Workspace name",
            "Workspace GST", "Workspace phone", "Report party name", "GST in report",
            "Report match basis"]
    ws.append(cols)
    for i, q in enumerate(queue, 1):
        ws.append([i, q["party"], ", ".join(q["gsts"]) or None, q["his_phone"] or None, q["rows"],
                   q["ws_basis"] or None, q["gst_rel"], q["ws_name"] or None, q["ws_gst"] or None,
                   q["ws_phone"] or None, q["rname"] or None, q["rgst"] or None, q["basis"] or None])
    widths = [5, 46, 34, 16, 8, 18, 20, 40, 20, 16, 40, 20, 28]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for c in ws[1]:
        head(c)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{ws.max_row}"

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--amit", default=str(DEFAULT_AMIT))
    ap.add_argument("--presence", default=str(DEFAULT_PRESENCE))
    ap.add_argument("--cache-dir", default=str(HERE / "outputs" / "turnover-truth"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()

    queue, meta = build(Path(a.report), Path(a.amit), Path(a.presence), Path(a.cache_dir))
    out = Path(a.out)
    write_workbook(queue, meta, out)
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["retailer", "gst_in_his_sheet", "phone_in_his_sheet", "rows_in_his_list",
                    "presence_matched_by", "gst_relation", "workspace_name", "workspace_gst",
                    "workspace_phone", "report_party_name", "gst_in_report", "report_match_basis"])
        for q in queue:
            w.writerow([q["party"], ", ".join(q["gsts"]), q["his_phone"], q["rows"], q["ws_basis"],
                        q["gst_rel"], q["ws_name"], q["ws_gst"], q["ws_phone"], q["rname"],
                        q["rgst"], q["basis"]])
    print(f"{meta['queue']} of {meta['parties']} parties → {out}")
    print(f"csv → {csv_path}")
    print("presence match basis:", dict(meta["basis"]))
    print("gst relation:", dict(meta["gst_rel"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
