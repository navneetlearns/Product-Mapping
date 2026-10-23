#!/usr/bin/env python3
"""Compare the two sentinel lane datasets (brand_named vs title_only) and attach runner fails.

Usage:
  python3 compare_sentinel_lanes.py --brand <Tarun Sentinel Dataset.csv> \
      --title-only <"title only sentinel tarun.csv"> \
      [--results <"sentiel run eval dataset for product mapping.csv">] [--out <xlsx>]

Output workbook sheets:
  Lanes at a glance | Overlap | Bucket check | Fail leads
"""
import argparse
import collections
import csv
import json
import re
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

CODE_RE = re.compile(r"\d{8,9}")
BRAND_RE = re.compile(r"birla|\bhil\b", re.I)

HEAD_FILL = PatternFill("solid", fgColor="14342A")
HEAD_FONT = Font(color="FFFFFF", bold=True, size=10, name="Aptos")
BAD_FILL = PatternFill("solid", fgColor="FCE9C8")


def clean(s):
    return " ".join(str(s or "").replace("\u2019", "'").split())


def clean_title(raw):
    t = re.sub(r"\s*\(\s*\d{8,9}\s*\)", "", clean(raw))
    t = re.sub(r"^\s*\d{8,9}\s+", "", t)
    return re.sub(r"[\s\-–,;/]+$", "", t).strip()


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            li = json.loads(r["line_item"]) if r["line_item"].strip() else {}
            rows.append(
                dict(
                    id=r["sentinel_item_id"],
                    case=r["case"],
                    src=r["source_dataset"],
                    status=r["status"],
                    name=li.get("product_name") or "",
                    title=clean_title(li.get("product_name") or ""),
                    hsn=li.get("hsn_sac_code"),
                    rate=li.get("unit_rate_excl_gst"),
                    serial=li.get("serial_no"),
                    sku=(r["sku"] or "").strip(),
                    eq=json.loads(r["equivalent_skus"]) if r["equivalent_skus"].strip() else [],
                    inv=(r["source_invoice"] or "").strip(),
                    exclusive=(r["distributor_exclusive"] or "").strip().lower() == "true",
                )
            )
    return rows


def bucket(title):
    if CODE_RE.search(title):
        return "sku"
    if BRAND_RE.search(title):
        return "brand"
    return "title_only"


def counts(rows, rows_all):
    skus = collections.Counter(r["sku"] for r in rows_all)
    hs = [str(r["hsn"]) if r["hsn"] else "" for r in rows_all]
    return [
        ("rows", len(rows_all)),
        ("sentinel ids", f"{min(r['id'] for r in rows_all)} … {max(r['id'] for r in rows_all)}"),
        ("case", ", ".join(sorted({r['case'] for r in rows_all}))),
        ("source_dataset", ", ".join(sorted({r['src'] for r in rows_all}))),
        ("status values", ", ".join(sorted({r['status'] for r in rows_all}))),
        ("distinct product titles (cleaned)", len({r["title"] for r in rows_all})),
        ("titles carrying a SKU code", sum(1 for r in rows_all if CODE_RE.search(r["name"]))),
        ("titles carrying a Birla/HIL token", sum(1 for r in rows_all if BRAND_RE.search(r["name"]))),
        ("titles with neither code nor brand",
         sum(1 for r in rows_all if not CODE_RE.search(r["name"]) and not BRAND_RE.search(r["name"]))),
        ("distinct mapped SKUs", len({r["sku"] for r in rows_all})),
        ("SKUs reached from >1 title (groups)", sum(1 for v in skus.values() if v > 1)),
        ("  rows inside those groups", sum(v for v in skus.values() if v > 1)),
        ("rows with equivalent_skus", sum(1 for r in rows_all if r["eq"])),
        ("distributor_exclusive = true", sum(1 for r in rows_all if r["exclusive"])),
        ("distinct invoices", len({r["inv"] for r in rows_all})),
        ("HSN 8-digit", sum(1 for h in hs if len(h) == 8)),
        ("HSN 4-digit (family only)", sum(1 for h in hs if len(h) == 4)),
        ("HSN missing", sum(1 for h in hs if not h)),
        ("rows with serial_no missing", sum(1 for r in rows_all if r["serial"] is None)),
        ("unit_rate excl GST: min", min(r["rate"] for r in rows_all)),
        ("unit_rate excl GST: max", max(r["rate"] for r in rows_all)),
        ("unit_rate excl GST: total", sum(r["rate"] for r in rows_all)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", required=True)
    ap.add_argument("--title-only", required=True, dest="title_only")
    ap.add_argument("--results")
    ap.add_argument("--out")
    a = ap.parse_args()

    A = load(a.brand)
    B = load(a.title_only)
    out = Path(a.out) if a.out else Path(a.brand).with_name("Sentinel lanes - comparison.xlsx")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Lanes at a glance"
    ws.append(["metric", "brand_named", "title_only"])
    ca, cb = counts(A, A), counts(B, B)
    assert [k for k, _ in ca] == [k for k, _ in cb]
    for (k, va), (_, vb) in zip(ca, cb):
        ws.append([k, va, vb])
    for c in range(1, 4):
        cell = ws.cell(row=1, column=c)
        cell.fill, cell.font = HEAD_FILL, HEAD_FONT
        cell.alignment = Alignment(vertical="center")
    ws.freeze_panes = "A2"
    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["B"].width = 62
    ws.column_dimensions["C"].width = 62

    # ---- overlap ----
    ws2 = wb.create_sheet("Overlap")
    sa, sb = {r["sku"] for r in A}, {r["sku"] for r in B}
    ta, tb = {r["title"] for r in A}, {r["title"] for r in B}
    ia, ib = {r["id"] for r in A}, {r["id"] for r in B}
    va, vb = {r["inv"] for r in A}, {r["inv"] for r in B}
    ws2.append(["comparison", "value"])
    for k, v in [
        ("shared sentinel ids", len(ia & ib)),
        ("shared raw product titles", len({r["name"] for r in A} & {r["name"] for r in B})),
        ("shared cleaned product titles", len(ta & tb)),
        ("shared mapped SKUs", len(sa & sb)),
        ("shared invoices", len(va & vb)),
        ("SKU overlap as % of brand lane", round(100 * len(sa & sb) / len(sa), 1)),
        ("SKU overlap as % of title-only lane", round(100 * len(sa & sb) / len(sb), 1)),
    ]:
        ws2.append([k, v])
    ws2.append([])
    ws2.append(["shared sku", "brand_named title(s)", "title_only title(s)"])
    hdr = ws2.max_row
    for s in sorted(sa & sb):
        ws2.append([s,
                    " || ".join(r["title"] for r in A if r["sku"] == s),
                    " || ".join(r["title"] for r in B if r["sku"] == s)])
    for c in range(1, 4):
        ws2.cell(row=1, column=c).fill = HEAD_FILL
        ws2.cell(row=1, column=c).font = HEAD_FONT
        ws2.cell(row=hdr, column=c).fill = HEAD_FILL
        ws2.cell(row=hdr, column=c).font = HEAD_FONT
    ws2.freeze_panes = "A2"
    for col, w in zip("ABC", [42, 66, 66]):
        ws2.column_dimensions[col].width = w

    # ---- bucket check ----
    ws3 = wb.create_sheet("Bucket check")
    ws3.append(["shipped lane label", "titles", "of which: 8-digit code in title",
                "of which: Birla/HIL token", "of which: neither",
                "bucket under build_datasets rule"])
    for label, rows in (("brand_named", A), ("title_only", B)):
        n_code = sum(1 for r in rows if CODE_RE.search(r["name"]))
        n_brand = sum(1 for r in rows if BRAND_RE.search(r["name"]) and not CODE_RE.search(r["name"]))
        n_plain = len(rows) - n_code - n_brand
        ws3.append([label, len(rows), n_code, n_brand, n_plain,
                    f"sku {n_code} / brand {n_brand} / title_only {n_plain}"])
    for c in range(1, 7):
        ws3.cell(row=1, column=c).fill = HEAD_FILL
        ws3.cell(row=1, column=c).font = HEAD_FONT
        ws3.cell(row=1, column=c).alignment = Alignment(wrap_text=True, vertical="center")
    ws3.row_dimensions[1].height = 32
    for col, w in zip("ABCDEF", [20, 9, 26, 24, 20, 34]):
        ws3.column_dimensions[col].width = w
    ws3.cell(row=4, column=1).value = (
        "Rule (eval-datasets/code/build_datasets.py): 8-digit code -> sku bucket; "
        "birla|hil token -> brand bucket; else title_only. The shipped labels match it for the "
        "brand lane (all 100 carry a brand token) but not for 54 of the title-only lane.")

    # ---- fail leads ----
    if a.results:
        ws4 = wb.create_sheet("Fail leads")
        ws4.append(["lane", "sentinel id", "result", "product_title", "sku in dataset",
                    "equivalent skus", "hsn", "source_invoice"])
        by_id = {r["id"]: r for r in A + B}
        with open(a.results, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if str(r.get("item_result", "")).strip().lower() == "pass":
                    continue
                x = by_id.get(r["item"])
                lane = "brand_named" if r["item"].startswith("brand") else "title_only"
                ws4.append([lane, r["item"], r.get("item_result"),
                            x["title"] if x else "(id not in either dataset)",
                            x["sku"] if x else "", ", ".join(x["eq"]) if x else "",
                            (x["hsn"] if x else "") or "", x["inv"] if x else ""])
                if x and not x["sku"]:
                    for c in range(1, 9):
                        ws4.cell(row=ws4.max_row, column=c).fill = BAD_FILL
        for c in range(1, 9):
            ws4.cell(row=1, column=c).fill = HEAD_FILL
            ws4.cell(row=1, column=c).font = HEAD_FONT
        ws4.freeze_panes = "A2"
        for col, w in zip("ABCDEFGH", [12, 13, 10, 56, 13, 20, 12, 38]):
            ws4.column_dimensions[col].width = w

    for sheet in wb.worksheets:
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0

    wb.save(out)
    print("wrote", out)
    print("shared skus", len(sa & sb), "| shared invoices", len(va & vb), "| shared titles", len(ta & tb))


if __name__ == "__main__":
    main()
