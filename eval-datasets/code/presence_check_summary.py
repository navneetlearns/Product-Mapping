#!/usr/bin/env python3
"""Bucket the presence-check workbook against the unique-party status view.

Input:  `01_source/Mapping issues - presence check in catalogue.xlsx` — Amit's three tabs with the
        platform's own presence flags (`In Workspace` for parties, `In Catalog` for SKUs).
        `03_review/Mapping issue list shared by amit - unique parties.xlsx` — our report status.

Output: `outputs/turnover-truth/presence-check.json` + a markdown block on stdout.

    python3 code/presence_check_summary.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import openpyxl
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
F_DIR = Path("/mnt/f/Product/zotok/Product Mapping/Mapping Inconsistency")
SRC = F_DIR / "01_source" / "Mapping issues - presence check in catalogue.xlsx"
UNIQUE = F_DIR / "03_review" / "Mapping issue list shared by amit - unique parties.xlsx"
OUT = HERE / "outputs" / "turnover-truth" / "presence-check.json"


def nm(s) -> str:
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", str(s).upper())
    return re.sub(r"\s+", " ", s).strip()


def product_buckets(xl: pd.ExcelFile) -> dict:
    df = xl.parse("SKU not mapped", dtype=str).dropna(how="all")
    df["desc_n"] = df["Material Description in Secondary Invoice"].map(nm)
    out = {
        "rows": len(df),
        "distinct_descriptions": int(df["desc_n"].nunique()),
        "in_catalog": {},
        "not_in_catalog": {},
    }
    for flag in ("Yes", "No"):
        sub = df[df["In Catalog"] == flag]
        detail = (sub.groupby("Detail").agg(rows=("desc_n", "size"), descriptions=("desc_n", "nunique"))
                  .sort_values("rows", ascending=False))
        out["in_catalog" if flag == "Yes" else "not_in_catalog"] = {
            "rows": len(sub),
            "descriptions": int(sub["desc_n"].nunique()),
            "detail": {k: {"rows": int(v["rows"]), "descriptions": int(v["descriptions"])}
                       for k, v in detail.iterrows()},
        }
    return out


def party_buckets(xl: pd.ExcelFile, wb, sheet: str, name_col: str, party_sheet: str) -> dict:
    df = xl.parse(sheet, dtype=str).dropna(how="all")
    df["nm"] = df[name_col].map(nm)
    per = df.groupby("nm").agg(
        presence=("In Workspace", lambda x: "No" if set(x) == {"No"} else ("Mixed" if "No" in set(x) else "Yes")),
        basis=("Matched By", lambda x: ", ".join(sorted({str(v) for v in x if str(v) != "nan"}))),
    )
    ours = [r for r in wb[party_sheet].iter_rows(min_row=2, values_only=True) if any(r)]
    cross: dict[str, Counter] = {}
    for r in ours:
        key, status = nm(r[0]), r[3]
        p = per.loc[key, "presence"] if key in per.index else "(not in sheet)"
        cross.setdefault(status, Counter())[p] += 1
    return {
        "rows": len(df),
        "unique_names": int(df["nm"].nunique()),
        "in_workspace_rows": {k: int(v) for k, v in df["In Workspace"].fillna("(blank)").value_counts().items()},
        "matched_by_rows": {k: int(v) for k, v in df["Matched By"].fillna("(blank)").value_counts().items()},
        "unique_presence": {k: int(v) for k, v in per["presence"].value_counts().items()},
        "status_cross": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in cross.items()},
    }


def main() -> int:
    xl = pd.ExcelFile(SRC)
    wb = openpyxl.load_workbook(UNIQUE, data_only=True)
    out = {
        "source": str(SRC),
        "status_source": str(UNIQUE),
        "products": product_buckets(xl),
        "distributors": party_buckets(xl, wb, "Distributor not mapped", "Distributor Name", "Distributors"),
        "retailers": party_buckets(xl, wb, "retailer not mapped", "Retailer Name", "Retailers"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    p = out["products"]
    print(f"PRODUCTS {p['rows']} rows / {p['distinct_descriptions']} descriptions")
    for label, key in (("In Catalog = Yes", "in_catalog"), ("In Catalog = No", "not_in_catalog")):
        b = p[key]
        print(f"  {label}: {b['rows']} rows / {b['descriptions']} descriptions")
        for k, v in b["detail"].items():
            print(f"    {k}: {v['rows']} rows / {v['descriptions']} descriptions")
    for side in ("distributors", "retailers"):
        b = out[side]
        print(f"\n{side.upper()} {b['rows']} rows / {b['unique_names']} unique")
        print(f"  In Workspace rows: {b['in_workspace_rows']}")
        print(f"  unique presence: {b['unique_presence']}")
        for status, c in b["status_cross"].items():
            print(f"  {status}: {c}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
