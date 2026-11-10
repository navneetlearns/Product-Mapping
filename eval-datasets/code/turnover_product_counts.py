#!/usr/bin/env python3
"""Unique-product coverage from a Turn Over Report export.

Companion to turnover_entity_counts.py: one read-only pass over the xlsx, no cache, no
workbook. A "product" is the printed invoice title (`Material Description in Secondary
Invoice`) deduped with the classifier's norm_text, because that is the text the mapper sees.
Status is three-way per product from `Parsing Status`, with MBNF counted as mapped:

  fully  every line Mapped / Mapped - Branding Not Found
  partly some mapped, some Unmapped
  never  no line mapped

Both populations are reported from the SAME pass: all rows as delivered, and deduplicated
(drop rows whose Validation Status contains "Duplicate Invoice"). MBNF is never counted as
a failure.

Usage:
  python3 turnover_product_counts.py "<Turn Over Report.xlsx>" [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify_mapping_issues import norm_text  # noqa: E402

MAPPED = {"Mapped", "Mapped - Branding Not Found"}
REQUIRED = ("Invoice Number", "Validation Status", "Parsing Status", "Material No",
            "Material Description in Secondary Invoice")
BLANK_TITLES = {"", "na", "n a", "-", "none", "null"}


def is_duplicate(value) -> bool:
    return "duplicate invoice" in str(value or "").lower()


def summarize(rows, cols):
    idx = {c.strip(): i for i, c in enumerate(cols)}
    missing = [c for c in REQUIRED if c not in idx]
    if missing:
        raise SystemExit(f"missing columns: {missing}")
    d_i = idx["Invoice Number"]
    v_i = idx["Validation Status"]
    p_i = idx["Parsing Status"]
    sku_i = idx["Material No"]
    desc_i = idx["Material Description in Secondary Invoice"]

    per = {"all": {}, "dedup": {}}   # title -> [lines, mapped]
    lines = {"all": 0, "dedup": 0}
    skus: set[str] = set()

    for row in rows:
        if row[d_i] is None:
            continue
        dup = is_duplicate(row[v_i])
        title = norm_text(row[desc_i] if row[desc_i] is not None else "")
        mapping = 1 if str(row[p_i] or "").strip() in MAPPED else 0
        for frame in ("dedup", "all"):
            if frame == "dedup" and dup:
                continue
            entry = per[frame].setdefault(title, [0, 0])
            entry[0] += 1
            entry[1] += mapping
            lines[frame] += 1
        sku = str(row[sku_i] or "").strip()
        if sku and sku.upper() not in {"NA", "-"}:
            skus.add(sku)

    out: dict[str, object] = {"skus": len(skus)}
    for frame, table in per.items():
        full = sum(1 for n, m in table.values() if m == n)
        never = sum(1 for _, m in table.values() if m == 0)
        out[frame] = {
            "lines": lines[frame],
            "products": len(table),
            "full": full,
            "partial": len(table) - full - never,
            "never": never,
            "blank_titles": sum(1 for t in table if t in BLANK_TITLES),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    import openpyxl
    wb = openpyxl.load_workbook(args.report, read_only=True, data_only=True)
    try:
        name = "Secondary Invoice Report" if "Secondary Invoice Report" in wb.sheetnames else wb.sheetnames[0]
        it = wb[name].iter_rows(values_only=True)
        cols = None
        filter_line = ""
        for _ in range(10):
            row = next(it, None)
            if row is None:
                break
            cells = [str(c).strip() if c is not None else "" for c in row]
            if not filter_line:
                filter_line = next((c for c in cells if c.startswith("Filters Applied")), "")
            if cells[0] == "Invoice Number":
                cols = cells
                break
        if cols is None:
            raise SystemExit("header row with 'Invoice Number' not found in the first 10 rows")
        out = {"source": args.report, "filter": filter_line, **summarize(it, cols)}
    finally:
        wb.close()

    if args.json:
        print(json.dumps(out, default=str))
        return
    print(f"REPORT {out['source']}")
    print(f"  {out['filter']}")
    for label in ("all", "dedup"):
        s = out[label]
        print(f"  {label:5s} lines={s['lines']:,} products={s['products']:,} "
              f"full={s['full']:,} partial={s['partial']:,} never={s['never']:,} "
              f"blank_titles={s['blank_titles']}")
    print(f"  distinct mapped SKUs={out['skus']:,}")


if __name__ == "__main__":
    main()
