#!/usr/bin/env python3
"""Write one workbook with ONE column per lane: cleaned product titles only.

Usage:
  python3 titles_only_sheets.py --out <xlsx> "<lane label>=<csv>" ["<lane label>=<csv>" ...]

Each lane becomes a sheet whose only content is a `product_title` header plus the cleaned
titles — no ids, skus, HSN, rates, invoices, flags or extra sheets. The `brand_named` and
`title_only` lanes get their embedded SKU codes stripped (those lanes must not carry one); the
`sku_only` lane keeps them, because the code in the title IS that lane (`--keep-code sku_only`).
"""
import argparse
import csv
import json
import re

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill


def clean_title(raw, keep_code=False):
    t = " ".join(str(raw or "").split())  # keep the source apostrophes: ’ vs ’’ encodes ft vs inch
    if keep_code:
        return t
    t = re.sub(r"\s*\(\s*\d{8,9}\s*\)", "", t)
    t = re.sub(r"^\s*\d{8,9}\s+", "", t)
    return re.sub(r"[\s\-–,;/]+$", "", t).strip()


def titles(csv_path, keep_code=False):
    out = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            li = json.loads(r["line_item"]) if r["line_item"].strip() else {}
            out.append(clean_title(li.get("product_name"), keep_code=keep_code))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-code", action="append", default=[], dest="keep_code",
                    help="lane label whose titles keep their embedded SKU code (e.g. sku_only)")
    ap.add_argument("lanes", nargs="+", help='label=csv')
    a = ap.parse_args()

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for spec in a.lanes:
        label, path = spec.split("=", 1)
        ws = wb.create_sheet(label[:31])
        ws.append(["product_title"])
        for t in titles(path, keep_code=label in a.keep_code):
            ws.append([t])
        c = ws.cell(row=1, column=1)
        c.font = Font(bold=True, color="FFFFFF", size=10, name="Aptos")
        c.fill = PatternFill("solid", fgColor="14342A")
        c.alignment = Alignment(vertical="center")
        ws.column_dimensions["A"].width = 64
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:A{ws.max_row}"
        ws.page_setup.orientation = "portrait"
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        print(f"{label}: {ws.max_row - 1} titles")
    wb.save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
