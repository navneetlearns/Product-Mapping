#!/usr/bin/env python3
"""Remove the two remaining brand abbreviations from the corrected title-only lane.

The lane must carry no brand and no SKU code. Two titles survived the 54-title replacement
because the token rule only knew BIRLA / BIRLANU / HIL:

    plain-066  BN UPVC PIPES SCH 40 3MTR 1"                  -> BN  = BirlaNu
    plain-036  SWR Pipe 110mm 3m Type A Pushfit S/S - B      -> " - B" = Birla

Both are labelled MAPPED_BRAND_NOT_FOUND, i.e. the expected label already says "no brand in the
title", so leaving the abbreviation in the text contradicts the ground truth. Fix is token-only:
delete the abbreviation, keep the product, the SKU and the status.

Guard: the script refuses to run unless the detected candidate set is exactly these two ids.

Usage:
  python3 apply_titleonly_token_fix.py --title-only <corrected.csv> [--xlsx <corrected.xlsx>]
                                       [--plan <replacements workbook>] [--out-dir <dir>]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.properties import PageSetupProperties

EXPECTED = {"plain-036", "plain-066"}

TRAIL = re.compile(r"\s*[-–]\s*B\s*$")          # "... S/S - B"
BN = re.compile(r"(?<![A-Za-z])BN\s+(?=[A-Z0-9])")  # "BN UPVC ..." (abbreviation only, uppercase)
BRAND_LEFT = re.compile(r"\b(?:birla(?:nu)?|hil)\b", re.I)
CODE = re.compile(r"(?<!\d)\d{8,9}(?!\d)")

HEAD_FILL = PatternFill("solid", fgColor="14342A")
HEAD_FONT = Font(color="FFFFFF", bold=True, size=10)
CHG_FILL = PatternFill("solid", fgColor="E4EFE7")


def squash(t: str) -> str:
    t = " ".join(t.split())
    return t.strip(" -–,;/")


def fix_title(title: str):
    t = TRAIL.sub("", title)
    t = BN.sub("", t)
    return squash(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title-only", required=True, dest="title_only")
    ap.add_argument("--xlsx", help="corrected workbook to update in place")
    ap.add_argument("--plan", help="title-only replacements workbook to update in place")
    a = ap.parse_args()

    src = Path(a.title_only)
    rows = list(csv.DictReader(open(src, newline="", encoding="utf-8-sig")))
    fieldnames = list(rows[0].keys())

    cand = []
    for r in rows:
        t = json.loads(r["line_item"])["product_name"]
        new = fix_title(t)
        if new != t:
            cand.append(r["sentinel_item_id"])
    print("candidates:", cand)
    if set(cand) != EXPECTED:
        raise SystemExit(f"expected exactly {sorted(EXPECTED)}, got {sorted(cand)} - refusing to write")

    out_rows, changes = [], []
    for r in rows:
        if r["sentinel_item_id"] not in EXPECTED:
            out_rows.append(r)
            continue
        old = json.loads(r["line_item"])["product_name"]
        new = fix_title(old)
        # checks: nothing but the token changed, no brand/code left, one title
        old_words, new_words = old.split(), new.split()
        dropped = [w for w in old_words if w not in new_words]
        if any(not re.fullmatch(r"(?i)bn|-|b", w) for w in dropped):
            raise SystemExit(f"{r['sentinel_item_id']}: unexpected word change {dropped}")
        if BRAND_LEFT.search(new) or re.search(r"(?<![A-Za-z])BN(?![A-Za-z])", new):
            raise SystemExit(f"{r['sentinel_item_id']}: brand still present in {new!r}")
        if CODE.search(new):
            raise SystemExit(f"{r['sentinel_item_id']}: a code appeared in {new!r}")
        rec = dict(r)
        old_enc = json.dumps(old, ensure_ascii=False)
        if r["line_item"].count(old_enc) != 1:
            raise SystemExit(f"{r['sentinel_item_id']}: could not locate product_name in line_item")
        rec["line_item"] = r["line_item"].replace(old_enc, json.dumps(new, ensure_ascii=False), 1)
        out_rows.append(rec)
        changes.append((r["sentinel_item_id"], old, new, r["status"], r["sku"], r["source_invoice"]))

    with open(src, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    print("rewrote", src)

    chk = list(csv.DictReader(open(src, newline="", encoding="utf-8")))
    titles = [json.loads(r["line_item"])["product_name"] for r in chk]
    print(f"rows {len(chk)} | distinct {len(set(titles))} | brand-free {sum(1 for t in titles if not BRAND_LEFT.search(t))}"
          f" | code-free {sum(1 for t in titles if not CODE.search(t))} | changed {len(changes)}")
    assert len(chk) == 100 and len(set(titles)) == 100
    assert not any(BRAND_LEFT.search(t) or re.search(r"(?<![A-Za-z])BN(?![A-Za-z])", t) for t in titles)
    assert not any(CODE.search(t) for t in titles)

    newt = {c[0]: c[2] for c in changes}

    # --- update the corrected workbook (keep the 54 earlier changes, append these two) ---
    if a.xlsx:
        wb = openpyxl.load_workbook(a.xlsx)
        ws = wb["Corrected set"]
        hdr = [c.value for c in ws[1]]
        ci = {h: i + 1 for i, h in enumerate(hdr)}
        for row in ws.iter_rows(min_row=2):
            rid = row[ci["sentinel_item_id"] - 1].value
            if rid in newt:
                row[ci["product_title"] - 1].value = newt[rid]
                row[ci["changed?"] - 1].value = "token removed"
                for c in row:
                    c.fill = CHG_FILL
        ws3 = wb["Changes"]
        for rid, old, new, st, sku, inv in changes:
            ws3.append([rid, old, new, st, sku, st, sku, inv, inv])
        for row in ws3.iter_rows(min_row=2, max_row=ws3.max_row):
            for c in row:
                c.alignment = Alignment(vertical="top", wrap_text=True)
        wb.save(a.xlsx)
        print("updated", a.xlsx)

    # --- same for the reviewed plan workbook's Applied sheet ---
    if a.plan:
        wb = openpyxl.load_workbook(a.plan)
        if "Applied (final set)" in wb.sheetnames:
            ws = wb["Applied (final set)"]
            hdr = [c.value for c in ws[1]]
            ci = {h: i + 1 for i, h in enumerate(hdr)}
            for row in ws.iter_rows(min_row=2):
                rid = row[ci["sentinel_item_id"] - 1].value
                if rid in newt:
                    row[ci["product_title"] - 1].value = newt[rid]
                    row[ci["changed?"] - 1].value = "token removed"
                    for c in row:
                        c.fill = CHG_FILL
            wb.save(a.plan)
            print("updated", a.plan)

    for c in changes:
        print(f"  {c[0]}  {c[1]}\n        -> {c[2]}")


if __name__ == "__main__":
    main()
