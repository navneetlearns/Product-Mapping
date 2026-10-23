#!/usr/bin/env python3
"""Apply the approved title-only replacements and write the corrected sentinel dataset.

Reads the APPROVED plan from the workbook's `Replacements (proposed)` sheet (so what gets
written is exactly what was signed off), recovers each replacement's line-item detail
(serial_no, hsn, rate, distributor_exclusive, invoice, expected sku/status/equivalents)
from the source exports, and rewrites the 100-row CSV with the same 11-column schema.

Row identity is kept (sentinel_item_id stays the lane slot); everything that describes the
product — line_item, equivalent_skus, sku, status, distributor_exclusive, source_invoice —
comes from the replacement's own source record.

Outputs
  <stem> - corrected.csv            the dataset, drop-in shape
  <stem> - corrected.xlsx           same data + `Changes` sheet (old vs new per row)
  the approved workbook gains an `Applied (final set)` sheet

Usage:
  python3 apply_titleonly_fix.py --title-only <csv> --plan <xlsx> --export <csv> --export <csv>
"""
import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.properties import PageSetupProperties

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_titleonly_fix import load_export, norm, clean, BIRLA, CODE, codey, fam

HEAD_FILL = PatternFill("solid", fgColor="14342A")
HEAD_FONT = Font(color="FFFFFF", bold=True, size=10, name="Aptos")
CHG_FILL = PatternFill("solid", fgColor="E4EFE7")
SAME_FILL = PatternFill("solid", fgColor="FFFFFF")


def style(ws, widths):
    for c in range(1, len(widths) + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill, cell.font = HEAD_FILL, HEAD_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 28
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(widths))}{ws.max_row}"
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title-only", required=True, dest="title_only")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--export", action="append", required=True, dest="exports")
    args = ap.parse_args()

    src = Path(args.title_only)
    rows = list(csv.DictReader(open(src, newline="", encoding="utf-8-sig")))
    order = [r["sentinel_item_id"] for r in rows]
    by_id = {r["sentinel_item_id"]: r for r in rows}

    # approved plan
    wb = openpyxl.load_workbook(args.plan)
    ws = wb["Replacements (proposed)"]
    hdr = [c.value for c in ws[1]]
    idx = {h: i for i, h in enumerate(hdr)}
    approved = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if not r[idx["sentinel id"]]:
            continue
        verdict = str(r[idx["approved? (Y/N)"]] or "").strip().lower()
        if verdict in ("n", "no", "reject", "x", "0", "false"):
            continue
        approved.append((r[idx["sentinel id"]], r[idx["proposed replacement title"]]))
    print(f"approved replacements: {len(approved)}")

    # replacement records from the exports
    ex = [rec for p in args.exports for rec in load_export(p)]
    rec_by_title = {}
    for rec in ex:
        rec_by_title.setdefault(norm(rec["name"]), rec)

    out_rows, changes, missing = [], [], []
    for rid in order:
        old = by_id[rid]
        plan = dict(approved).get(rid)
        if not plan:
            out_rows.append(old)
            continue
        rec = rec_by_title.get(norm(plan))
        if rec is None:
            missing.append((rid, plan))
            out_rows.append(old)
            continue
        li_old = json.loads(old["line_item"])
        new_li = {
            "serial_no": rec["serial"],
            "hsn_sac_code": rec["hsn"],
            "product_name": clean(rec["name"]),
            "unit_rate_excl_gst": rec["rate"],
        }
        new = dict(old)
        new["line_item"] = json.dumps(new_li, ensure_ascii=False)
        new["equivalent_skus"] = json.dumps(rec["eq"] or [], ensure_ascii=False)
        new["sku"] = rec["sku"] or ""
        new["status"] = rec["st"]
        new["distributor_exclusive"] = "true" if rec["de"] else "false"
        new["source_invoice"] = rec["inv"]
        out_rows.append(new)
        changes.append((rid, li_old.get("product_name"), new_li["product_name"],
                        old["status"], old["sku"], new["status"], new["sku"],
                        old["source_invoice"], new["source_invoice"]))

    if missing:
        print("!! replacements not found in exports:", missing)
        raise SystemExit(2)

    out_csv = src.with_name(src.stem + " - corrected.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("wrote", out_csv)

    # ---- verification ----
    chk = list(csv.DictReader(open(out_csv, newline="", encoding="utf-8")))
    assert len(chk) == 100 and [r["sentinel_item_id"] for r in chk] == order
    titles = [json.loads(r["line_item"])["product_name"] for r in chk]
    print("rows:", len(chk), "| distinct titles:", len(set(titles)),
          "| brand-token titles:", sum(1 for t in titles if BIRLA.search(t)),
          "| titles with code:", sum(1 for t in titles if CODE.search(t) or codey(t)))
    print("status mix:", collections.Counter(r["status"] for r in chk))
    print("changed rows:", len(changes),
          "| unchanged rows:", 100 - len(changes))

    # ---- audit workbook + applied sheet ----
    wb2 = openpyxl.Workbook()
    ws2 = wb2.active
    ws2.title = "Corrected set"
    cols = ["#", "sentinel_item_id", "product_title", "status", "sku", "hsn", "rate", "invoice", "changed?"]
    ws2.append(cols)
    chg_ids = {c[0] for c in changes}
    for i, r in enumerate(chk, 1):
        li = json.loads(r["line_item"])
        ws2.append([i, r["sentinel_item_id"], li["product_name"], r["status"], r["sku"],
                    li["hsn_sac_code"], li["unit_rate_excl_gst"], r["source_invoice"],
                    "replaced" if r["sentinel_item_id"] in chg_ids else ""])
        if r["sentinel_item_id"] in chg_ids:
            for c in range(1, len(cols) + 1):
                ws2.cell(row=ws2.max_row, column=c).fill = CHG_FILL
    style(ws2, [5, 16, 58, 20, 12, 12, 11, 38, 10])

    ws3 = wb2.create_sheet("Changes")
    cols3 = ["sentinel id", "old title", "new title", "old status", "old sku",
             "new status", "new sku", "old source_invoice", "new source_invoice"]
    ws3.append(cols3)
    for c in changes:
        ws3.append(list(c))
    style(ws3, [12, 50, 50, 20, 12, 20, 12, 38, 38])

    for sheet in wb2.worksheets:
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
        sheet.print_title_rows = "1:1"
    out_xlsx = src.with_name(src.stem + " - corrected.xlsx")
    wb2.save(out_xlsx)
    print("wrote", out_xlsx)

    # applied sheet in the approved workbook
    wb4 = openpyxl.load_workbook(args.plan)
    if "Applied (final set)" in wb4.sheetnames:
        del wb4["Applied (final set)"]
    ws4 = wb4.create_sheet("Applied (final set)")
    ws4.append(cols)
    for i, r in enumerate(chk, 1):
        li = json.loads(r["line_item"])
        ws4.append([i, r["sentinel_item_id"], li["product_name"], r["status"], r["sku"],
                    li["hsn_sac_code"], li["unit_rate_excl_gst"], r["source_invoice"],
                    "replaced" if r["sentinel_item_id"] in chg_ids else ""])
        if r["sentinel_item_id"] in chg_ids:
            for c in range(1, len(cols) + 1):
                ws4.cell(row=ws4.max_row, column=c).fill = CHG_FILL
    style(ws4, [5, 16, 58, 20, 12, 12, 11, 38, 10])
    ws4.page_setup.orientation = "landscape"
    ws4.page_setup.fitToWidth = 1
    ws4.page_setup.fitToHeight = 0
    ws4.print_title_rows = "1:1"
    wb4.save(args.plan)
    print("updated", args.plan)


if __name__ == "__main__":
    main()
