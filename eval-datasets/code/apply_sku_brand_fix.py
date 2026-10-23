#!/usr/bin/env python3
"""Apply the approved SKU-lane brand-token fixes and write the corrected dataset.

Reads the approved plan straight out of the review workbook's `Proposed fixes` sheet (so what is
written is exactly what was reviewed), and rewrites only `line_item.product_name` for those 19
rows. Everything else in the row - and the JSON key order / spacing of the untouched fields - is
left byte-identical: the edit is a token deletion, not a product swap, so the code stays in the
title and the row keeps its mapped SKU.

Outputs
  <stem> - corrected.csv    the lane, drop-in shape
  <stem> - corrected.xlsx   same data + `Changes` sheet (old -> new per row)
  the plan workbook gains an `Applied (final set)` sheet

Usage:
  python3 apply_sku_brand_fix.py --sku <csv> --plan <xlsx> [--catalogue <csv>]
"""
from __future__ import annotations

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
from build_catalogue_check import clean
from plan_sku_brand_fix import BN, BRAND, CODE

HEAD_FILL = PatternFill("solid", fgColor="1F3864")
HEAD_FONT = Font(color="FFFFFF", bold=True, size=10)
CHG_FILL = PatternFill("solid", fgColor="DDEBF7")


def style(ws, widths):
    for c in range(1, len(widths) + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill, cell.font = HEAD_FILL, HEAD_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 26
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(widths))}{ws.max_row}"
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def swap_name(raw_li: str, new_name: str) -> str:
    """Replace just the product_name value inside the raw JSON string, keeping every other byte."""
    old = json.loads(raw_li)
    old_enc = json.dumps(old["product_name"], ensure_ascii=False)
    new_enc = json.dumps(new_name, ensure_ascii=False)
    if raw_li.count(old_enc) != 1:
        # fall back to a rebuild, same key order
        rebuilt = {k: (new_name if k == "product_name" else v) for k, v in old.items()}
        return json.dumps(rebuilt, ensure_ascii=False)
    return raw_li.replace(old_enc, new_enc, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", required=True, help="sku lane csv (source)")
    ap.add_argument("--plan", required=True, help="review workbook with the approved plan")
    ap.add_argument("--catalogue", help="optional catalogue csv for the SKU check")
    ap.add_argument("--out-dir", dest="out_dir", help="where to write the corrected files (default: beside the source)")
    a = ap.parse_args()

    src = Path(a.sku)
    rows = list(csv.DictReader(open(src, newline="", encoding="utf-8-sig")))
    order = [r["sentinel_item_id"] for r in rows]
    by_id = {r["sentinel_item_id"]: r for r in rows}
    fieldnames = list(rows[0].keys())

    wb = openpyxl.load_workbook(a.plan)
    ws = wb["Proposed fixes"]
    hdr = [c.value for c in ws[1]]
    idx = {h: i for i, h in enumerate(hdr)}
    plan = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        rid = r[idx["sentinel_item_id"]]
        if rid:
            plan[rid] = (r[idx["current product_title"]], r[idx["proposed product_title"]],
                         r[idx["token removed"]])
    print(f"plan rows: {len(plan)}")

    missing = [rid for rid in plan if rid not in by_id]
    if missing:
        raise SystemExit(f"plan ids not in the lane: {missing}")

    out_rows, changes = [], []
    for r in rows:
        rid = r["sentinel_item_id"]
        if rid not in plan:
            out_rows.append(r)
            continue
        old_title, new_title, tok = plan[rid]
        cur = json.loads(r["line_item"])["product_name"]
        if clean(cur) != clean(old_title):
            raise SystemExit(f"{rid}: plan title does not match the lane title\n  lane: {cur}\n  plan: {old_title}")
        new = dict(r)
        new["line_item"] = swap_name(r["line_item"], new_title)
        out_rows.append(new)
        changes.append((rid, cur, new_title, tok, r["sku"]))

    out_base = Path(a.out_dir) if a.out_dir else src.parent
    out_base.mkdir(parents=True, exist_ok=True)
    out_csv = out_base / (src.stem + " - corrected.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    print("wrote", out_csv)

    # ---------------- verification ----------------
    chk = list(csv.DictReader(open(out_csv, newline="", encoding="utf-8")))
    assert len(chk) == len(rows) == 100, "row count changed"
    assert [r["sentinel_item_id"] for r in chk] == order, "id order changed"
    chg = {c[0] for c in changes}
    byte_same = sum(1 for a_, b_ in zip(rows, chk) if a_ == b_)
    touched_only_name = 0
    for old, new in zip(rows, chk):
        if old == new:
            continue
        od, nd = json.loads(old["line_item"]), json.loads(new["line_item"])
        same_other = {k: v for k, v in od.items() if k != "product_name"} == \
                     {k: v for k, v in nd.items() if k != "product_name"}
        same_fields = all(old[k] == new[k] for k in fieldnames if k != "line_item")
        if same_other and same_fields and list(od) == list(nd):
            touched_only_name += 1
    titles = [json.loads(r["line_item"])["product_name"] for r in chk]
    code_ok = sum(1 for r in chk if CODE.findall(json.loads(r["line_item"])["product_name"]) == [r["sku"]])
    brand_left = [t for t in titles if BRAND.search(t) or BN.search(t)]
    print(f"rows {len(chk)} | unchanged rows {byte_same} | changed {len(changes)} | "
          f"only-product_name-changed {touched_only_name}/{len(changes)}")
    print(f"distinct titles {len(set(titles))} | code==sku {code_ok}/100 | brand tokens left {len(brand_left)}")
    print("status mix:", dict(collections.Counter(r["status"] for r in chk)))
    if brand_left:
        print("!! brand tokens remain:", brand_left)
    if a.catalogue:
        cat = set()
        with open(a.catalogue, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                cat.add(clean(r.get("Sku Code") or ""))
        in_cat = sum(1 for r in chk if r["sku"] in cat)
        print(f"skus in catalogue {in_cat}/100")

    # ---------------- workbook ----------------
    wb2 = openpyxl.Workbook()
    ws2 = wb2.active
    assert ws2 is not None
    ws2.title = "Corrected set"
    cols = ["#", "sentinel_item_id", "product_title", "status", "sku", "hsn", "rate", "invoice", "changed?"]
    ws2.append(cols)
    for i, r in enumerate(chk, 1):
        li = json.loads(r["line_item"])
        ws2.append([i, r["sentinel_item_id"], li["product_name"], r["status"], r["sku"],
                    li["hsn_sac_code"], li["unit_rate_excl_gst"], r["source_invoice"],
                    "token removed" if r["sentinel_item_id"] in chg else ""])
        if r["sentinel_item_id"] in chg:
            for c in range(1, len(cols) + 1):
                ws2.cell(row=ws2.max_row, column=c).fill = CHG_FILL
    style(ws2, [5, 16, 58, 11, 11, 11, 10, 38, 14])

    ws3 = wb2.create_sheet("Changes")
    cols3 = ["sentinel id", "old title", "new title", "token removed", "sku"]
    ws3.append(cols3)
    for c in changes:
        ws3.append(list(c))
    style(ws3, [14, 56, 56, 14, 11])

    for sheet in wb2.worksheets:
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.paperSize = sheet.PAPERSIZE_A4
        sheet.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.print_title_rows = "1:1"
    out_xlsx = out_base / (src.stem + " - corrected.xlsx")
    wb2.save(out_xlsx)
    print("wrote", out_xlsx)

    # applied sheet in the reviewed workbook
    wb4 = openpyxl.load_workbook(a.plan)
    if "Applied (final set)" in wb4.sheetnames:
        del wb4["Applied (final set)"]
    ws4 = wb4.create_sheet("Applied (final set)")
    ws4.append(cols)
    for i, r in enumerate(chk, 1):
        li = json.loads(r["line_item"])
        ws4.append([i, r["sentinel_item_id"], li["product_name"], r["status"], r["sku"],
                    li["hsn_sac_code"], li["unit_rate_excl_gst"], r["source_invoice"],
                    "token removed" if r["sentinel_item_id"] in chg else ""])
        if r["sentinel_item_id"] in chg:
            for c in range(1, len(cols) + 1):
                ws4.cell(row=ws4.max_row, column=c).fill = CHG_FILL
    style(ws4, [5, 16, 58, 11, 11, 11, 10, 38, 14])
    ws4.page_setup.orientation = "landscape"
    ws4.page_setup.fitToWidth = 1
    ws4.page_setup.fitToHeight = 0
    ws4.print_title_rows = "1:1"
    wb4.save(a.plan)
    print("updated", a.plan)


if __name__ == "__main__":
    main()
