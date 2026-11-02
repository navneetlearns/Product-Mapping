#!/usr/bin/env python3
"""Check the platform team's "Mapping Status" note against the 1-Nov Turn Over Report.

For each tab of Amit's sheet it reports the cuts needed to test the note's claims:
  - rows vs unique party names (the note quotes rows as if they were parties),
  - presence-check rows vs unique names,
  - report status at row / present-row / unique-party / distinct-invoice level,
  - products: rows vs unique descriptions, and how many are wholly / partly Unmapped.

    python3 code/verify_platform_note.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amit_overlap as A  # noqa: E402
import build_turnover_truth as B  # noqa: E402
import classify_mapping_issues as cmi  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
F_DIR = Path("/mnt/f/Product/zotok/Product Mapping/Mapping Inconsistency")
REPORT = F_DIR / "01_source" / "Turn Over Report.xlsx"
AMIT = F_DIR / "01_source" / "Mapping issue list shared by amit.xlsx"
PRES = F_DIR / "01_source" / "Mapping issues - presence check in catalogue.xlsx"
CACHE = HERE / "outputs" / "turnover-truth"


def main() -> int:
    df, _ = B.load_report(REPORT, CACHE, refresh=False)
    df = B.normalise(df)
    head = df[~df["dup"]].copy()
    head["inv_n"] = head["Invoice Number"].map(A.inv)
    head["ret_phone"] = head["Retailer Mobile No"].map(A.phone)
    by_inv = {k: v for k, v in head.groupby("inv_n")}

    wb = openpyxl.load_workbook(AMIT, read_only=True, data_only=True)
    px = pd.ExcelFile(PRES)

    for side, spec, sheet in (
        ("DISTRIBUTORS", A.SIDES["distributors"], "Distributor not mapped"),
        ("RETAILERS", A.SIDES["retailers"], "retailer not mapped"),
    ):
        tab = A.prep_amit(wb, spec)
        tab["_v"] = [A.row_verdict(r, by_inv, spec) for r in tab.to_dict("records")]
        p = px.parse(sheet, dtype=str).dropna(how="all")
        tab["pres"] = p["In Workspace"].values
        tab["inv"] = tab["Invoice Number"].map(A.inv)

        pv = tab.groupby("_party")["_v"].apply(list).map(A.party_verdict)
        pres = tab[tab["pres"] == "Yes"]
        pres_pv = pres.groupby("_party")["_v"].apply(list).map(A.party_verdict)
        un = tab[tab["_v"] == "unmapped"]
        pres_un = pres[pres["_v"] == "unmapped"]

        print(f"\n{side}")
        print(f"  rows {len(tab)} | unique names {tab['_nm'].nunique()} | unique GST {tab['_gst'].replace('', pd.NA).nunique()}")
        print(f"  presence: rows Yes {int((tab['pres'] == 'Yes').sum())} / No {int((tab['pres'] == 'No').sum())} | "
              f"unique all-Yes {int((p['In Workspace'] == 'Yes').groupby(p[spec['name_hdr']].map(A.nm)).all().sum())}")
        print(f"  row verdicts: {tab['_v'].value_counts().to_dict()}")
        print(f"  present rows: {len(pres)} | verdicts {pres['_v'].value_counts().to_dict()}")
        print(f"  unique-party verdicts: {pv.value_counts().to_dict()}")
        print(f"  present unique-party verdicts: {pres_pv.value_counts().to_dict()}")
        print(f"  unmapped rows {len(un)} | present unmapped rows {len(pres_un)} | "
              f"distinct invoices {un['inv'].nunique()} / present {pres_un['inv'].nunique()}")
        any_unm = sum(1 for i in pres["inv"].unique()
                      if i in by_inv and (by_inv[i][spec["flag"]] == 0).any())
        all_unm = sum(1 for i in pres["inv"].unique()
                      if i in by_inv and (by_inv[i][spec["flag"]] == 0).all())
        print(f"  present invoices: {pres['inv'].nunique()} | any unmapped line {any_unm} | all unmapped {all_unm}")

    sku = pd.ExcelFile(AMIT).parse("SKU not mapped", dtype=str)
    sku["d"] = sku["Material Description in Secondary Invoice"].map(cmi.norm_text)
    sp = px.parse("SKU not mapped", dtype=str).dropna(how="all")
    sp["d"] = sp["Material Description in Secondary Invoice"].map(cmi.norm_text)
    m = head.groupby("desc_norm")["Parsing Status"].agg(lambda x: set(x.dropna()))
    print("\nPRODUCTS")
    print(f"  rows {len(sku)} | unique descriptions {sku['d'].nunique()}")
    for flag in ("Yes", "No"):
        ds = set(sp.loc[sp["In Catalog"] == flag, "d"])
        allunm = sum(1 for d in ds if d in m.index and m[d] == {"Unmapped"})
        anyunm = sum(1 for d in ds if d in m.index and "Unmapped" in m[d])
        print(f"  In Catalog {flag}: rows {int((sp['In Catalog'] == flag).sum())} | descriptions {len(ds)} | "
              f"all-Unmapped {allunm} | any-Unmapped {anyunm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
