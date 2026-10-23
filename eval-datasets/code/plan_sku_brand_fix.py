#!/usr/bin/env python3
"""SKU lane (case=with_sku) - brand-token audit and minimal fix plan.

The with-sku lane's cue is the SKU code inside the title. A brand name in the same title gives
the mapper a second, unintended route (brand path), so the lane should not carry one.

This script finds brand-carrying titles in the lane and proposes the *minimal* edit: drop the
brand token, keep the 8-9 digit code and therefore keep the row's product and SKU. Every
proposal is checked: exactly one code left, that code equals the row's mapped SKU, the title is
still unique in the lane, the token is gone, and the SKU exists in the BirlaNu catalogue.

The plan is written to a review workbook; nothing is applied here.

Usage:
    python3 plan_sku_brand_fix.py \
        --sku "/mnt/f/.../01_source/sku only tarun sentinel.csv" \
        --catalogue "/mnt/f/.../products-18-October-2026-10-52-16.csv" \
        --out "/mnt/f/.../03_review/SKU lane - brand-name fixes.xlsx"
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.properties import PageSetupProperties

from build_catalogue_check import clean

CODE = re.compile(r"(?<!\d)\d{8,9}(?!\d)")

# Tokens that identify the manufacturer inside a title. BIRLA / BIRLANU are matched
# case-insensitively; BN only as an uppercase standalone token (it is the BirlaNu abbreviation
# in these exports, and lowercase "bn" would be noise).
BRAND = re.compile(r"\b(?:birla(?:nu)?|hil)\b", re.I)
BN = re.compile(r"(?<![A-Za-z])BN(?![A-Za-z])")
# Any other manufacturer we would want to flag rather than silently fix.
OTHER_BRANDS = [
    "prince", "astral", "vectus", "watertec", "bondtite", "solvobond", "kaju", "dr. fixit",
    "dr fixit", "supreme", "finolex", "ashirvad", "jaquar", "ciba", "newkor", "prayag",
    "sanghvi", "polymax", "kesh", "kisan",
]

HEAD_FILL = PatternFill("solid", fgColor="1F3864")
HEAD_FONT = Font(color="FFFFFF", bold=True)
BAD_FILL = PatternFill("solid", fgColor="FFC7CE")
OK_FILL = PatternFill("solid", fgColor="C6EFCE")


def squash(text: str) -> str:
    """Collapse the whitespace left behind when a word is cut out of a title."""
    t = " ".join(str(text).split())
    # only tidy what the cut itself created: no rewriting of the source's own spacing
    t = re.sub(r"\s*([,;])(?=\s)", r"\1", t)
    t = re.sub(r"\s*-\s*-\s*", " - ", t)
    t = re.sub(r"^[\s\-–,;/]+", "", t)
    t = re.sub(r"[\s\-–,;/]+$", "", t)
    return t.strip()


def drop_brand(title: str):
    """Return (new_title, tokens_removed). Keeps the code and everything else untouched."""
    tokens = []
    t = title
    for rx in (BRAND, BN):
        for m in rx.finditer(t):
            tokens.append(m.group(0))
        t = rx.sub(" ", t)
    return squash(t), tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", required=True, help="sku lane csv")
    ap.add_argument("--catalogue", required=True, help="BirlaNu catalogue csv (Product Name, Sku Code)")
    ap.add_argument("--out", required=True, help="review workbook to write")
    a = ap.parse_args()

    with open(a.catalogue, newline="", encoding="utf-8-sig") as fh:
        cat = {}
        for r in csv.DictReader(fh):
            code = clean(r.get("Sku Code") or "")
            if code:
                cat.setdefault(code, clean(r.get("Product Name") or ""))

    with open(a.sku, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    recs = []
    for r in rows:
        li = json.loads(r["line_item"]) if r["line_item"].strip() else {}
        title = clean(li.get("product_name") or "")
        sku = clean(r.get("sku") or "")
        codes = CODE.findall(title)
        brands = [m.group(0) for m in BRAND.finditer(title)] + [m.group(0) for m in BN.finditer(title)]
        others = [b for b in OTHER_BRANDS if re.search(rf"\b{re.escape(b)}\b", title, re.I)]
        proposed, removed = drop_brand(title) if brands else (title, [])
        recs.append(dict(
            id=r["sentinel_item_id"], sku=sku, title=title, codes=codes, brands=brands,
            others=others, proposed=proposed, removed=removed, status=r["status"],
        ))

    fixes = [r for r in recs if r["brands"]]
    titles_after = {r["proposed"] for r in fixes}
    other_titles = {r["title"] for r in recs if not r["brands"]}

    for r in fixes:
        r["code_ok"] = len(CODE.findall(r["proposed"])) == 1 and CODE.findall(r["proposed"])[0] == r["sku"]
        r["token_gone"] = not (BRAND.search(r["proposed"]) or BN.search(r["proposed"]))
        r["unique"] = r["proposed"] not in other_titles and list(titles_after).count(r["proposed"]) == 1
        r["in_cat"] = r["sku"] in cat
        r["cat_name"] = cat.get(r["sku"], "")
        r["ok"] = all((r["code_ok"], r["token_gone"], r["unique"], r["in_cat"]))

    wb = Workbook()

    ws = wb.active
    assert ws is not None
    ws.title = "Summary"
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 96
    ws["A1"] = "SKU lane (case=with_sku) - brand-token audit and fix plan"
    ws["A1"].font = Font(bold=True, size=13)
    ws["A2"] = "Source"
    ws["B2"] = a.sku
    ws["A3"] = "Catalogue"
    ws["B3"] = a.catalogue
    lines = [
        ("Rows in lane", f"{len(recs)}"),
        ("Titles carrying a brand token", f"{len(fixes)}  (BIRLANU {sum(1 for r in fixes if any('BIRLANU' in x.upper() for x in r['brands']))}, "
                                          f"BN {sum(1 for r in fixes if any(x.upper() == 'BN' for x in r['brands']))}, "
                                          f"BIRLA {sum(1 for r in fixes if any(x.upper() == 'BIRLA' for x in r['brands']))})"),
        ("Titles with another manufacturer's name", f"{sum(1 for r in recs if r['others'])}"),
        ("Proposed fix", "drop the brand token only - the 8-9 digit code stays, so the row keeps its product and its SKU"),
        ("All proposals keep one code == mapped SKU", f"{sum(1 for r in fixes if r['code_ok'])}/{len(fixes)}"),
        ("All proposals brand-free after the edit", f"{sum(1 for r in fixes if r['token_gone'])}/{len(fixes)}"),
        ("All proposed titles still unique in the lane", f"{sum(1 for r in fixes if r['unique'])}/{len(fixes)}"),
        ("All SKUs present in the BirlaNu catalogue", f"{sum(1 for r in fixes if r['in_cat'])}/{len(fixes)}"),
        ("Rows failing any check", f"{sum(1 for r in fixes if not r['ok'])}"),
        ("Applied yet?", "NO - this is a plan; say 'replace' to write the corrected lane"),
    ]
    for i, (k, v) in enumerate(lines, start=5):
        ws.cell(row=i, column=1, value=k).font = Font(bold=True)
        ws.cell(row=i, column=2, value=v)

    ws2 = wb.create_sheet("Proposed fixes")
    cols = ["#", "sentinel_item_id", "sku", "current product_title", "token removed",
            "proposed product_title", "code in proposed", "code == sku", "brand gone",
            "still unique", "in catalogue", "catalogue name", "status", "other brand?", "ok"]
    ws2.append(cols)
    for r in sorted(fixes, key=lambda x: x["id"]):
        ws2.append([
            len(ws2["A"]), r["id"], r["sku"], r["title"], ", ".join(r["removed"]), r["proposed"],
            " ".join(CODE.findall(r["proposed"])), "YES" if r["code_ok"] else "NO",
            "YES" if r["token_gone"] else "NO", "YES" if r["unique"] else "NO",
            "YES" if r["in_cat"] else "NO", r["cat_name"], r["status"],
            ", ".join(r["others"]), "OK" if r["ok"] else "CHECK",
        ])
    for j, c in enumerate(cols, start=1):
        ws2.column_dimensions[get_column_letter(j)].width = {
            1: 5, 2: 15, 3: 10, 4: 46, 5: 12, 6: 46, 7: 10, 8: 10, 9: 10, 10: 10, 11: 11,
            12: 46, 13: 12, 14: 14, 15: 8,
        }[j]
    for row in ws2.iter_rows(min_row=1, max_row=ws2.max_row):
        for c in row:
            c.alignment = Alignment(vertical="top", wrap_text=True)
        if row[0].row == 1:
            for c in row:
                c.fill, c.font = HEAD_FILL, HEAD_FONT
        elif row[14].value != "OK":
            for c in row:
                c.fill = BAD_FILL
    ws2.freeze_panes = "C2"

    ws3 = wb.create_sheet("Lane check (all 100)")
    cols3 = ["sentinel_item_id", "sku", "code in title", "code == sku", "brand token?",
             "token(s) found", "other brand?", "proposed title", "status"]
    ws3.append(cols3)
    for r in recs:
        ws3.append([r["id"], r["sku"], " ".join(r["codes"]), "YES" if r["sku"] in r["codes"] else "NO",
                    "YES" if r["brands"] else "", ", ".join(r["brands"]), ", ".join(r["others"]),
                    r["proposed"] if r["brands"] else "", r["status"]])
    for j, w in enumerate([15, 10, 12, 10, 11, 14, 14, 56, 12], start=1):
        ws3.column_dimensions[get_column_letter(j)].width = w
    for row in ws3.iter_rows(min_row=1, max_row=1):
        for c in row:
            c.fill, c.font = HEAD_FILL, HEAD_FONT
    for row in ws3.iter_rows(min_row=2, max_row=ws3.max_row):
        for c in row:
            c.alignment = Alignment(vertical="top", wrap_text=True)
        if row[4].value == "YES":
            for c in row:
                c.fill = BAD_FILL
    ws3.freeze_panes = "B2"

    # print setup: every sheet on one page wide, landscape, header repeated
    for sheet in wb.worksheets:
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.paperSize = sheet.PAPERSIZE_A4
        sheet.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.print_options.horizontalCentered = False
    for sheet in (ws2, ws3):
        sheet.print_title_rows = "1:1"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(a.out)

    print(f"rows {len(recs)} | brand-token titles {len(fixes)} | "
          f"ok {sum(1 for r in fixes if r['ok'])}/{len(fixes)} | other-brand rows {sum(1 for r in recs if r['others'])}")
    print(f"wrote {a.out}")
    for r in fixes:
        print(f"  {r['id']}  {r['title']}\n        -> {r['proposed']}")
    if any(not r["ok"] for r in fixes):
        raise SystemExit("some proposals failed a check - see the workbook")


if __name__ == "__main__":
    main()
