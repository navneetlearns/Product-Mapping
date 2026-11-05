#!/usr/bin/env python3
"""Amit's SKU-tab products vs a Turn Over Report snapshot — product-grain status.

Answers: of the UNIQUE products in Amit's "SKU not mapped" tab, how many are now
mapped / partly mapped / never mapped in the platform's line-level successor?

Method
- Unique product = Material Description in Secondary Invoice after cmi.norm_text
  (case, quotes/inch, unit spacing, known typo pairs). 3,216 rows -> 2,359 products.
- Join to the report on the same normaliser (the report's `desc_norm` column).
- Product status is aggregated from `Parsing Status`, which is decided per line and
  is distributor-specific (product_mapping.cfaId), so "partly mapped" is a real state:
    Fully mapped      every line for the description is Mapped / MBNF
    Partially mapped  at least one mapped line, at least one not
    Not mapped        no mapped line at all
- Two populations are reported because the export carries duplicate-invoice rows
  (Validation Status = Duplicate Invoice): the dedup population is the baseline
  convention; all rows is what the platform shows.

Usage
  python3 amit_products_vs_report.py            # newest cache / default paths
  python3 amit_products_vs_report.py --refresh  # re-read the xlsx (slow)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import build_turnover_truth as btt  # noqa: E402
import classify_mapping_issues as cmi  # noqa: E402

DEFAULT_SRC = Path("/mnt/f/Product/zotok/Product Mapping/Mapping Inconsistency/01_source")
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "outputs/turnover-truth"
DESC = "Material Description in Secondary Invoice"


def load_amit_products(xlsx: Path) -> pd.DataFrame:
    wb = load_workbook(xlsx, read_only=True, data_only=True)
    rows = list(wb["SKU not mapped"].iter_rows(min_row=1, values_only=True))
    wb.close()
    hdr = [str(c).strip() for c in rows[0]]
    i = hdr.index(DESC)
    raw = [str(r[i]).strip() for r in rows[1:] if r[i] not in (None, "")]
    df = pd.DataFrame({"desc_raw": raw})
    df["desc_norm"] = df["desc_raw"].map(cmi.norm_text)
    return df.drop_duplicates("desc_norm").reset_index(drop=True)


def status_table(report: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    g = report.groupby("desc_norm").agg(
        lines=("parse", "size"),
        mapped=("parse", lambda s: s.isin(btt.MAPPED_TIERS).sum()),
        unmapped=("parse", lambda s: (s == "Unmapped").sum()),
        mbnf=("parse", lambda s: (s == "Mapped - Branding Not Found").sum()),
        invoices=("inv_key", "nunique"),
        value=("value", "sum"),
    ).reset_index()
    m = products.merge(g, on="desc_norm", how="left")
    for c in ("lines", "mapped", "unmapped", "mbnf", "invoices"):
        m[c] = m[c].fillna(0).astype(int)
    m["value"] = m["value"].fillna(0.0)
    m["status"] = [
        "Fully mapped" if mp == ln else ("Partially mapped" if mp > 0 else "Not mapped")
        for ln, mp in zip(m["lines"], m["mapped"])
    ]
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=DEFAULT_SRC / "Turn Over Report.xlsx")
    ap.add_argument("--amit", type=Path, default=DEFAULT_SRC / "Mapping issue list shared by amit.xlsx")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <cache-dir>/amit-products-vs-report-<snapshot-date>.csv")
    args = ap.parse_args()

    df, cache = btt.load_report(args.report, args.cache_dir, args.refresh)
    df = btt.normalise(df)
    products = load_amit_products(args.amit)
    print(f"report: {args.report.name} (cache {cache}) rows={len(df)}")
    print(f"Amit products: 3216 rows -> {len(products)} unique (cmi.norm_text)")

    dedup = status_table(df[~df["dup"]], products)
    allrows = status_table(df, products)

    print("\nstatus of the unique products (dedup population):")
    print(dedup["status"].value_counts().to_string())
    print("\nstatus of the unique products (all rows, incl duplicate-invoice):")
    print(allrows["status"].value_counts().to_string())

    never = dedup[dedup["status"] == "Not mapped"]
    print(f"\nnever mapped (dedup): {len(never)} products · {never.lines.sum()} lines · "
          f"₹{never.value.sum():,.0f}")
    print(never.sort_values("lines", ascending=False)[["desc_raw", "lines", "invoices"]]
          .head(10).to_string(index=False))

    out = allrows[["desc_raw", "status", "lines", "mapped", "unmapped", "mbnf", "invoices", "value"]]
    stamp = cache.replace("turnover-truth-", "").replace(".csv.gz", "")
    out_path = args.out or (args.cache_dir / f"amit-products-vs-report-{stamp}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
