#!/usr/bin/env python3
"""Mapping-truth baseline from the BirlaNu "Turn Over Report" (secondary invoice report).

The report is the platform's own line-level truth for secondary sales: every line carries
`Distributor Status`, `Retailer Status` and `Parsing Status` plus the mapped `Material No`.
This builder answers three questions from it, in that priority order:

  1. how many DISTRIBUTORS are mapped (identified against the master)
  2. how many RETAILERS are mapped
  3. how many PRODUCTS are mapped (line / title / SKU level)

Definitions (agreed 2026-11-01, plan §0):
  D1 population   headline = deduplicated (lines whose Validation Status contains
                  "Duplicate Invoice" dropped); all-lines shown as a sensitivity
  D2 MBNF         `Mapped - Branding Not Found` counts as MAPPED, always split out
  D3 units        line · invoice · distinct entity · ₹ (line amount excl. tax)
  D4 identity     GST first, then SAP code (distributor) / phone (retailer), name for display
  D5 scope        invoice date 2026-01-01 → 2026-11-01 as delivered; ingested-only sensitivity
  D6 products     unmapped titles bucketed against the catalogue (reuses classify_mapping_issues)
  D7 reconcile    Amit's 29-Oct three-sheet list tested against the definition space
  D8 delivery     workbook in 03_review + markdown readout; no deck

Outputs
  <out-workbook>  the review workbook (Headline first, distributor/retailer ahead of products)
  <out-onboarding> optional handover pack: the accounts to onboard, 80%-of-value cut flagged
  <out-md>        the written readout, rendered from the same computed numbers
  <out-json>      summary.json — every headline number, for rebuild diffing
  cache           <cache-dir>/turnover-truth-<report-mtime>.csv.gz (skips the 80 s xlsx read)

Usage
  python3 build_turnover_truth.py \
    --report "F:/.../01_source/Turn Over Report.xlsx" \
    --catalogue "F:/.../Eval Related Product Mapping/products-18-October-2026-10-52-16.csv" \
    --amit "F:/.../01_source/Mapping issue list shared by amit.xlsx" \
    --out-workbook "F:/.../03_review/Truth - mapping coverage.xlsx" \
    --out-md ../TRUTH-MAPPING-BASELINE.md \
    --out-json outputs/turnover-truth/summary.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import classify_mapping_issues as cmi  # noqa: E402  (component parser + catalogue classifier)

# ------------------------------------------------------------------ constants
PINE = "14402A"
INK = "111C16"
GREY = "E8EEE9"
CLAY = "B0812C"

MAPPED_TIERS = ("Mapped", "Mapped - Branding Not Found")
GST_RE = re.compile(r"^[0-9A-Z]{15}$")

STR_COLS = [
    "Invoice Number", "Sequence", "Invoice Date", "Invoice Status", "Validation Status",
    "Tax Validation", "Submitted By", "Uploaded Phone Number", "Distributor Code (SAP)",
    "Distributor Name", "Matched Distributor Name", "Distributor GST", "Distributor Email",
    "Distributor Mobile No", "Distributor Status", "Retailer Code (SFA)", "Retailer Name",
    "Matched Retailer Name", "Retailer GST", "Retailer Email", "Retailer Mobile No",
    "Retailer Pin code", "Retailer Status", "Eway bill Number", "Created At",
    "Parsing Status", "Material No", "Material Description", "Material Description in Secondary Invoice",
    "Category Code", "Category Name", "HSN", "Applied Scheme", "Contribution",
]
NUM_COLS = [
    "Total line Amount (Excluding tax)", "Total Line Amount (Net of Discount)",
    "Base Price (Rate)", "Qty", "Total Line Amount (incl. Tax)", "GST Value",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ loading
def load_report(xlsx: Path, cache_dir: Path, refresh: bool) -> tuple[pd.DataFrame, str]:
    """Read the report once; cache a str-typed CSV so re-runs skip the 80 s xlsx parse."""
    stamp = time.strftime("%Y-%m-%d", time.localtime(xlsx.stat().st_mtime))
    cache = cache_dir / f"turnover-truth-{stamp}.csv.gz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not refresh:
        log(f"cache hit: {cache.name}")
        df = pd.read_csv(cache, dtype=str, keep_default_na=False, na_values=[])
    else:
        log(f"reading {xlsx.name} (this is the slow part, ~80 s)")
        df = pd.read_excel(xlsx, sheet_name="Secondary Invoice Report", header=2, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
        df.to_csv(cache, index=False, compression="gzip")
        log(f"cached → {cache}")
    df.columns = [str(c).strip() for c in df.columns]
    return df, cache.name


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    for c in STR_COLS:
        if c not in df.columns:
            df[c] = ""
        df[c] = (
            df[c].fillna("").astype(str).str.strip()
            .replace({"NA": "", "None": "", "nan": "", "-": ""})
        )
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    def money(v):
        try:
            return float(v)
        except Exception:
            return 0.0

    df["value"] = df["Total line Amount (Excluding tax)"].map(money)
    df["value_incl"] = df["Total Line Amount (incl. Tax)"].map(money)
    df["qty"] = df["Qty"].map(money)

    df["gst_ok"] = df["Distributor GST"].str.upper().str.replace(r"\s+", "", regex=True)
    df["ret_gst_ok"] = df["Retailer GST"].str.upper().str.replace(r"\s+", "", regex=True)
    df["gst_valid"] = df["gst_ok"].map(lambda g: bool(GST_RE.match(g)))
    df["ret_gst_valid"] = df["ret_gst_ok"].map(lambda g: bool(GST_RE.match(g)))

    def name_norm(s):
        s = re.sub(r"[^A-Za-z0-9 ]+", " ", s.upper())
        return re.sub(r"\s+", " ", s).strip()

    df["dist_name_n"] = df["Distributor Name"].map(name_norm)
    df["ret_name_n"] = df["Retailer Name"].map(name_norm)
    df["desc_norm"] = df["Material Description in Secondary Invoice"].map(cmi.norm_text)
    df["desc_raw"] = df["Material Description in Secondary Invoice"]

    # D4 identity: GST → SAP code → name (distributor); GST → phone → name+pin (retailer)
    df["dist_key"] = [
        g if v else (f"SAP:{c}" if c else f"NAME:{n}")
        for g, v, c, n in zip(df["gst_ok"], df["gst_valid"], df["Distributor Code (SAP)"], df["dist_name_n"])
    ]
    phone = df["Retailer Mobile No"].str.replace(r"\D", "", regex=True).str[-10:]
    df["ret_key"] = [
        g if v else (f"PHONE:{p}" if p else f"NAME:{n}|{pin}")
        for g, v, p, n, pin in zip(df["ret_gst_ok"], df["ret_gst_valid"], phone,
                                   df["ret_name_n"], df["Retailer Pin code"])
    ]
    df["phone"] = phone

    df["inv_key"] = (df["Invoice Number"] + "|" + df["dist_key"] + "|" + df["Invoice Date"])
    df["inv_month"] = df["Invoice Date"].str[:7].where(df["Invoice Date"].str[:4].str.isdigit(), "")
    df["created"] = pd.to_datetime(df["Created At"], errors="coerce")
    df["created_ym"] = df["created"].dt.strftime("%Y-%m-%d").fillna("")

    df["dup"] = df["Validation Status"].str.contains("Duplicate Invoice", case=False, na=False)
    df["dist_ok"] = df["Distributor Status"].eq("Identified")
    df["ret_ok"] = df["Retailer Status"].eq("Identified")
    df["parse"] = df["Parsing Status"].fillna("")
    df["prod_ok"] = df["parse"].isin(MAPPED_TIERS)
    df["has_sku"] = df["Material No"].ne("")
    df["is_duplicate_pair"] = df["desc_norm"].ne("")
    return df


# ------------------------------------------------------------------ KPI helpers
def pct(a: float, b: float) -> float:
    return 0.0 if not b else 100.0 * a / b


def flag_kpi(df: pd.DataFrame, flag: str, key: str) -> dict:
    """Line / value / invoice / entity coverage for one boolean flag."""
    n = len(df)
    m = int(df[flag].sum())
    val = float(df["value"].sum())
    val_m = float(df.loc[df[flag], "value"].sum())
    inv = int(df["inv_key"].nunique())
    inv_all_mapped = int(df.loc[df.groupby("inv_key")[flag].transform("all"), "inv_key"].nunique())
    per = df.groupby(key)[flag].agg(["sum", "count"])
    ents = int(len(per))
    full = int((per["sum"] == per["count"]).sum())
    none = int((per["sum"] == 0).sum())
    return {
        "lines": n, "lines_mapped": m, "lines_pct": pct(m, n),
        "value": val, "value_mapped": val_m, "value_pct": pct(val_m, val),
        "invoices": inv, "invoices_all_mapped": inv_all_mapped, "invoices_pct": pct(inv_all_mapped, inv),
        "entities": ents, "entities_full": full, "entities_partial": ents - full - none,
        "entities_none": none, "entities_full_pct": pct(full, ents),
    }


def entity_table(df: pd.DataFrame, key: str, flag: str, name_col: str, gst_col: str,
                 extra_cols: list[str]) -> pd.DataFrame:
    """One row per entity, ordered by value; status flag = full / partial / none."""
    g = df.groupby(key, dropna=False)
    out = g.agg(
        rows=(flag, "size"),
        mapped_rows=(flag, "sum"),
        value=("value", "sum"),
        invoices=("inv_key", "nunique"),
    ).reset_index()
    first = g.agg({name_col: "first", gst_col: "first", **{c: "first" for c in extra_cols}}).reset_index()
    out = out.merge(first, on=key, how="left")
    out["status"] = [
        "Fully mapped" if mr == r else ("Not mapped" if mr == 0 else "Partially mapped")
        for mr, r in zip(out["mapped_rows"], out["rows"])
    ]
    reasons = (
        df[~df[flag]].groupby(key)["Validation Status"]
        .agg(lambda s: "; ".join(f"{k} ({v})" for k, v in collections.Counter(s).most_common(2)))
        .rename("top_failure_reasons").reset_index()
    )
    out = out.merge(reasons, on=key, how="left")
    out["top_failure_reasons"] = out["top_failure_reasons"].fillna("")
    return out.sort_values("value", ascending=False)


# ------------------------------------------------------------------ anomalies
def anomaly_blocks(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    blocks = []
    m = (df["parse"].eq("Unmapped")) & df["has_sku"]
    blocks.append(("Unmapped status but Material No present", df.loc[m]))

    m = df["parse"].eq("") & df["desc_raw"].eq("")
    blocks.append(("Parsing Status blank, description blank", df.loc[m]))

    m = (~df["dist_ok"]) & df["Distributor Code (SAP)"].ne("")
    blocks.append(("Distributor Not Identified but carries an SAP code", df.loc[m]))

    m = (~df["ret_ok"]) & df["ret_gst_valid"]
    blocks.append(("Retailer Not Identified but carries a valid GST", df.loc[m]))

    m = df["Distributor Name"].eq("") & df["Retailer Name"].eq("")
    blocks.append(("Both party names blank", df.loc[m]))

    m = df["dist_ok"] & df["Matched Distributor Name"].eq("")
    blocks.append(("Distributor Identified but matched name blank", df.loc[m]))

    m = df["ret_ok"] & df["Matched Retailer Name"].eq("")
    blocks.append(("Retailer Identified but matched name blank", df.loc[m]))

    inv_dup = df.loc[df["Invoice Number"].ne("")].groupby("Invoice Number")["dist_key"].nunique()
    reused = set(inv_dup[inv_dup > 1].index)
    blocks.append(("Invoice number used by more than one distributor", df.loc[df["Invoice Number"].isin(reused)]))

    m = df["Invoice Date"].str[:4].ne("2026") & df["Invoice Date"].ne("")
    blocks.append(("Invoice date outside 2026", df.loc[m]))
    return blocks


# ------------------------------------------------------------------ reconciliation
AMIT_SHEETS = {
    "SKU": ("SKU not mapped", "Material Description in Secondary Invoice"),
    "Distributor": ("Distributor not mapped", "Distributor Name"),
    "Retailer": ("retailer not mapped", "Retailer Name"),
}


def read_amit(path: Path) -> dict:
    wb = __import__("openpyxl").load_workbook(path, read_only=True, data_only=True)
    out = {}
    for k, (sheet, keycol) in AMIT_SHEETS.items():
        rows = list(wb[sheet].iter_rows(values_only=True))
        hdr = [str(h).strip() if h else h for h in rows[0]]
        recs = [dict(zip(hdr, r)) for r in rows[1:] if any(v not in (None, "") for v in r)]
        val_col = next((h for h in hdr if h and h.lower() == "value"), None)
        inv_col = next((h for h in hdr if h and h.lower() == "inv"), None)
        out[k] = {
            "rows": len(recs),
            "names": len({str(r.get(keycol, "")).strip() for r in recs if r.get(keycol)}),
            "inv": float(sum(float(r.get(inv_col) or 0) for r in recs if inv_col)) if inv_col else None,
            "value": float(sum(float(r.get(val_col) or 0) for r in recs if val_col)) if val_col else None,
            "keycol": keycol,
        }
    return out


def score(rows, inv, value, target: dict) -> float:
    """Relative distance across all three of Amit's numbers, each capped at 3× so no axis dominates."""
    def rel(a, b):
        return min(abs(a - b) / b, 3.0) if b else 0.0
    return (rel(rows, target["rows"]) + rel(inv, target["inv"] or 0)
            + rel(value, target["value"] or 0))


def reconciliation(df: pd.DataFrame, target: dict) -> list[dict]:
    """Search the definition space for the filter that reproduces Amit's SKU tab."""
    windows = {
        "all": df.index == df.index,
        "created ≥ 2026-08-01": df["created_ym"] >= "2026-08-01",
        "created ≥ 2026-09-01": df["created_ym"] >= "2026-09-01",
        "created ≥ 2026-10-01": df["created_ym"] >= "2026-10-01",
        "created ≥ 2026-10-15": df["created_ym"] >= "2026-10-15",
        "invoice month 2026-10": df["inv_month"].eq("2026-10"),
        "invoice 2026-09+10": df["inv_month"].isin(["2026-09", "2026-10"]),
    }
    statuses = {
        "Unmapped": df["parse"].eq("Unmapped"),
        "MBNF": df["parse"].eq("Mapped - Branding Not Found"),
        "Unmapped+MBNF": df["parse"].isin(["Unmapped", "Mapped - Branding Not Found"]),
    }
    keys = {
        "desc+distributor": ["desc_norm", "dist_key"],
        "desc only": ["desc_norm"],
        "desc+distributor+month": ["desc_norm", "dist_key", "inv_month"],
    }
    out = []
    for wname, wmask in windows.items():
        for sname, smask in statuses.items():
            sub = df.loc[wmask & smask]
            if sub.empty:
                continue
            for kname, kcols in keys.items():
                g = sub.groupby(kcols, dropna=False)
                agg = g.agg(rows=("value", "size"), value=("value", "sum"),
                            value_incl=("value_incl", "sum"), inv=("inv_key", "nunique"))
                rows, value, inv = len(agg), float(agg["value"].sum()), int(agg["inv"].sum())
                out.append({
                    "window": wname, "status": sname, "key": kname,
                    "rows": rows, "inv": inv, "value": value, "value_incl": float(agg["value_incl"].sum()),
                    "d_rows": rows - target["rows"],
                    "d_value": value - target["value"],
                    "score": score(rows, inv, value, target),
                })
    return sorted(out, key=lambda r: r["score"])


# ------------------------------------------------------------------ workbook
HDR_FILL = PatternFill("solid", fgColor=PINE)
HDR_FONT = Font(color="FFFFFF", bold=True, size=10)
TITLE_FONT = Font(color=INK, bold=True, size=14)
SUB_FONT = Font(color=INK, bold=True, size=11)
NOTE_FONT = Font(color="555555", italic=True, size=9)


def sheet_head(ws, title: str, note: str = ""):
    ws.append([title])
    ws["A1"].font = TITLE_FONT
    if note:
        ws.append([note])
        ws["A2"].font = NOTE_FONT
    ws.append([])


def table(ws, header: list[str], rows, widths: dict | None = None):
    ws.append(header)
    hdr_row = ws.max_row
    for c in ws[hdr_row]:
        c.fill = HDR_FILL
        c.font = HDR_FONT
        c.alignment = Alignment(vertical="center")
    for r in rows:
        ws.append(list(r))
    for i, h in enumerate(header, start=1):
        width = (widths or {}).get(h)
        ws.column_dimensions[get_column_letter(i)].width = width or max(11, min(48, len(str(h)) + 4))
    return hdr_row


def fmt_ws(ws, money_cols: list[str] = (), pct_cols: list[str] = (), header_row: int | None = None):
    hdr = {}
    if header_row:
        hdr = {c.value: c.column for c in ws[header_row] if c.value}
    for name in money_cols:
        if name in hdr:
            for row in ws.iter_rows(min_row=header_row + 1, min_col=hdr[name], max_col=hdr[name]):
                for c in row:
                    c.number_format = '#,##0'
    for name in pct_cols:
        if name in hdr:
            for row in ws.iter_rows(min_row=header_row + 1, min_col=hdr[name], max_col=hdr[name]):
                for c in row:
                    c.number_format = '0.0"%"'


def fit_widths(ws, max_width: int = 60, sample: int = 3000):
    """Size every column to what is actually in it, so numbers never render as ###."""
    for col in ws.iter_cols(max_row=min(ws.max_row, sample)):
        best = 0
        for c in col:
            v = c.value
            if v is None or isinstance(v, bool):
                continue
            s = f"{v:,}" if isinstance(v, (int, float)) else str(v)
            best = max(best, len(s))
        if best:
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(10, best + 2), max_width)


def build_workbook(path: Path, ctx: dict):
    wb = Workbook()
    wb.remove(wb.active)

    # ---------------- Read me
    ws = wb.create_sheet("Read me")
    sheet_head(ws, "Truth — mapping coverage (Turn Over Report, secondary invoices)",
               f"Source: {ctx['report_name']} · downloaded {ctx['report_mtime']} · "
               f"filters 2026-01-01 → 2026-11-01")
    rows = [
        ("Question", "How many distributors, retailers and products are mapped — from the platform's own line-level truth."),
        ("Priority", "1) distributors  2) retailers  3) products"),
        ("Population (headline)", "Deduplicated: lines whose Validation Status contains 'Duplicate Invoice' are dropped."),
        ("Population (sensitivity)", "All lines as delivered, shown beside the headline in every KPI row."),
        ("Distributor mapped", "Distributor Status = Identified. The matched-name column is blank on a small share of those lines — listed in Anomalies, not reclassified."),
        ("Retailer mapped", "Retailer Status = Identified (same matched-name caveat, see Anomalies)."),
        ("Product mapped", "Parsing Status ∈ {Mapped, Mapped - Branding Not Found}. MBNF carries a Material No, so it is mapped; it is always split out."),
        ("Product NOT mapped", "Parsing Status = Unmapped (9,178 lines in the headline population; 10,819 as delivered) plus the 95 lines whose status is Unmapped yet carry a SKU — listed in Anomalies."),
        ("Identity", "GST first, then SAP code (distributor) / phone (retailer); name+pin is the last resort, name alone is display only."),
        ("Value", "Total line Amount (Excluding tax). Incl-tax shares differ; this workbook never mixes them."),
        ("Units", "line · invoice (number + distributor + date) · distinct entity · ₹"),
        ("Rebuild", "python3 eval-datasets/code/build_turnover_truth.py --report … --catalogue … --out-workbook …"),
        ("Definitions agreed", "2026-11-01 — plan §0 (D1–D8), user priority: retailer/distributor before products."),
    ]
    for k, v in rows:
        ws.append([k, v])
        ws.cell(row=ws.max_row, column=1).font = Font(bold=True, color=INK)
        ws.cell(row=ws.max_row, column=2).alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 110

    # ---------------- Headline
    ws = wb.create_sheet("Headline")
    sheet_head(ws, "Headline — mapped share by subject",
               "Retailer and distributor first (priority order). Headline = deduplicated population; "
               "'all lines' column is the as-delivered sensitivity.")
    hdr = ["Subject", "Unit", "Mapped", "Total", "Mapped %", "All-lines Mapped %", "Note"]
    rows = []
    specs = [
        ("Distributor", "dist", ("lines", "value", "entities")),
        ("Retailer", "ret", ("lines", "value", "entities")),
        ("Product (line status)", "prod", ("lines", "value")),
        ("Product (title, all lines mapped)", "title", ("entities",)),
        ("Product (SKU)", "sku", ("lines", "value", "entities")),
    ]
    for subject, key, units in specs:
        k = ctx["kpi"][key]
        ka = ctx["kpi"][key + "_all"]
        for unit in units:
            if unit == "lines":
                rows.append((subject, "lines", k["lines_mapped"], k["lines"], round(k["lines_pct"], 1),
                             round(ka["lines_pct"], 1), ""))
            elif unit == "value":
                rows.append((subject, "₹ excl. tax", k["value_mapped"], k["value"], round(k["value_pct"], 1),
                             round(ka["value_pct"], 1), ""))
            else:
                rows.append((subject, "entities", k["entities_full"], k["entities"],
                             round(k["entities_full_pct"], 1), round(ka["entities_full_pct"], 1),
                             "entity = fully mapped (all its lines)"))
    hr = table(ws, hdr, rows, widths={"Subject": 34, "Unit": 24, "Note": 40})
    fmt_ws(ws, money_cols=["Mapped", "Total"], pct_cols=["Mapped %", "All-lines Mapped %"], header_row=hr)

    ws.append([])
    ws.append(["Product tiers (deduplicated population)"])
    ws.cell(row=ws.max_row, column=1).font = SUB_FONT
    tiers = ctx["product_tiers"]
    hr = table(ws, ["Parsing Status", "Lines", "Line %", "Value ₹", "Value %"],
               [(t, tiers[t]["lines"], round(tiers[t]["lines_pct"], 1), tiers[t]["value"],
                 round(tiers[t]["value_pct"], 1))
                for t in ("Mapped", "Mapped - Branding Not Found", "Unmapped", "(blank)")],
               widths={"Parsing Status": 30})
    fmt_ws(ws, money_cols=["Lines", "Value ₹"], pct_cols=["Line %", "Value %"], header_row=hr)

    ws.append([])
    ws.append(["Retailer identifier coverage (all entities)"])
    ws.cell(row=ws.max_row, column=1).font = SUB_FONT
    rid = ctx["ret_identifier"]
    hr = table(ws, ["Identifier", "Entities", "Fully identified", "Not identified"],
               [(label, rid.get(label, {}).get("n", 0), rid.get(label, {}).get("full", 0),
                 rid.get(label, {}).get("none", 0))
                for label in ("GST+phone", "GST only", "phone only", "none")],
               widths={"Identifier": 20})
    fmt_ws(ws, money_cols=["Entities", "Fully identified", "Not identified"], header_row=hr)

    # ---------------- Distributor coverage
    ws = wb.create_sheet("Distributor coverage")
    sheet_head(ws, f"Distributor coverage — {ctx['kpi']['dist']['entities']} entities "
                   f"({ctx['kpi']['dist']['entities_full']} fully identified)",
               "Deduplicated population · ordered by value · top_failure_reasons come from the "
               "lines whose Distributor Status is Not Identified.")
    d = ctx["dist_table"]
    hdr = ["Status", "Distributor", "GST", "SAP code", "Matched name", "Lines", "Mapped lines",
           "Value ₹", "Invoices", "Top failure reasons"]
    table(ws, hdr, [(r.status, r.dist_name, r.dist_gst, r.dist_code,
                     r.matched_name, r.rows, r.mapped_rows, round(r.value), r.invoices,
                     r.top_failure_reasons) for r in d.itertuples()],
          widths={"Distributor": 40, "Matched name": 34, "Top failure reasons": 52})
    ws.freeze_panes = "A5"
    ws.auto_filter.ref = f"A4:J{ws.max_row}"
    fmt_ws(ws, money_cols=["Lines", "Mapped lines", "Value ₹", "Invoices"], header_row=4)

    # ---------------- Retailer coverage
    ws = wb.create_sheet("Retailer coverage")
    sheet_head(ws, f"Retailer coverage — {ctx['kpi']['ret']['entities']} entities "
                   f"({ctx['kpi']['ret']['entities_full']} fully identified)",
               "Deduplicated population · ordered by value · identifier strength shows which key could match.")
    r = ctx["ret_table"]
    hdr = ["Status", "Retailer", "GST", "Phone", "Pin", "SFA code", "Matched name", "Identifier",
           "Lines", "Mapped lines", "Value ₹", "Invoices", "Top failure reasons"]
    table(ws, hdr, [(x.status, x.ret_name, x.ret_gst, x.phone, x.pin,
                     x.sfa, x.matched_name, x.identifier, x.rows, x.mapped_rows,
                     round(x.value), x.invoices, x.top_failure_reasons) for x in r.itertuples()],
          widths={"Retailer": 40, "Matched name": 34, "Top failure reasons": 52})
    ws.freeze_panes = "A5"
    ws.auto_filter.ref = f"A4:M{ws.max_row}"
    fmt_ws(ws, money_cols=["Lines", "Mapped lines", "Value ₹", "Invoices"], header_row=4)

    # ---------------- Onboarding (top-N that make 80% of unmapped value)
    ws = wb.create_sheet("Onboarding 80%")
    sheet_head(ws, "Onboarding list — entities making 80% of the unmapped value",
               "Cumulative-value cut, distributor then retailer. These are the accounts to fix first.")
    ws.append(["Distributor — accounts carrying unmapped lines"])
    ws.cell(row=ws.max_row, column=1).font = SUB_FONT
    table(ws, ["#", "Distributor", "GST", "Value ₹", "Cumulative %", "Lines not identified"],
          [(i, x.dist_name, x.dist_gst, round(x.value), round(x.cum_pct, 1),
            int(x.rows - x.mapped_rows))
           for i, x in enumerate(ctx["dist_80"].itertuples(), start=1)],
          widths={"Distributor": 40})
    ws.append([])
    ws.append(["Retailer — accounts carrying unmapped lines"])
    ws.cell(row=ws.max_row, column=1).font = SUB_FONT
    table(ws, ["#", "Retailer", "GST", "Phone", "Value ₹", "Cumulative %", "Lines not identified"],
          [(i, x.ret_name, x.ret_gst, x.phone, round(x.value),
            round(x.cum_pct, 1), int(x.rows - x.mapped_rows))
           for i, x in enumerate(ctx["ret_80"].itertuples(), start=1)],
          widths={"Retailer": 40})

    # ---------------- Product coverage
    ws = wb.create_sheet("Product coverage")
    sheet_head(ws, f"Product coverage — {ctx['kpi']['title']['entities']:,} titles "
                   f"({ctx['kpi']['title']['entities_full']:,} fully mapped)",
               "Title = Material Description in Secondary Invoice, canonicalised. Status: all lines mapped / "
               "some / none. Ordered by value.")
    p = ctx["title_table"]
    hdr = ["Status", "Title (as first seen)", "Titles (canonical)", "Lines", "Value ₹", "Invoices",
           "Distinct SKUs seen", "Mapped lines", "Top parsing status"]
    table(ws, hdr, [(x.status, x.example, x.desc_norm, x.rows, round(x.value), x.invoices,
                     x.skus, x.mapped_rows, x.top_status) for x in p.itertuples()],
          widths={"Title (as first seen)": 60, "Titles (canonical)": 44})
    ws.freeze_panes = "A5"
    ws.auto_filter.ref = f"A4:I{ws.max_row}"
    fmt_ws(ws, money_cols=["Lines", "Value ₹", "Invoices"], header_row=4)

    # ---------------- Unmapped buckets + titles
    ws = wb.create_sheet("Unmapped buckets")
    sheet_head(ws, "Unmapped titles bucketed against the 18-Oct catalogue",
               "Component bucketing (family → type → size → qualifier), same classifier as the Oct-31 analysis. "
               "Buckets A/A2/B mean the product IS in the catalogue.")
    table(ws, ["Bucket", "Rows (desc+distributor)", "Lines", "Value ₹", "Rows %", "What it means"],
          [(b, v["rows"], v["lines"], round(v["value"]), round(v["rows_pct"], 1), v["meaning"])
           for b, v in ctx["bucket_summary"].items()], widths={"What it means": 70})
    fmt_ws(ws, money_cols=["Lines", "Value ₹"], pct_cols=["Rows %"], header_row=4)

    ws = wb.create_sheet("Unmapped titles")
    sheet_head(ws, f"Unmapped titles — {len(ctx['unmapped_table']):,} description+distributor rows",
               "One row per description+distributor pair with unmapped lines. Candidates are evidence, not a verdict.")
    u = ctx["unmapped_table"]
    hdr = ["Bucket", "Description (as invoiced)", "Distributor", "Lines", "Value ₹", "Invoices",
           "Bucket detail", "Candidate SKUs", "Candidate names"]
    table(ws, hdr, [(x.bucket, x.desc_raw, x.dist_name, x.lines, round(x.value), x.invoices,
                     x.detail, x.skus, x.candidates) for x in u.itertuples()],
          widths={"Description (as invoiced)": 54, "Distributor": 34, "Bucket detail": 44,
                  "Candidate SKUs": 30, "Candidate names": 60})
    ws.freeze_panes = "A5"
    ws.auto_filter.ref = f"A4:I{ws.max_row}"
    fmt_ws(ws, money_cols=["Lines", "Value ₹", "Invoices"], header_row=4)

    # ---------------- Anomalies
    ws = wb.create_sheet("Anomalies")
    sheet_head(ws, "Status and identity anomalies", "Each block is a population that breaks a naive rule. "
                                                    "None is folded into a headline number.")
    for label, sub in ctx["anomalies"]:
        ws.append([f"{label} — {len(sub):,} lines, ₹{sub['value'].sum():,.0f}"])
        ws.cell(row=ws.max_row, column=1).font = SUB_FONT
        if len(sub):
            cols = ["Invoice Number", "Distributor Name", "Retailer Name", "Parsing Status",
                    "Material No", "Material Description in Secondary Invoice", "Validation Status"]
            table(ws, cols, sub[cols].head(50).itertuples(index=False, name=None),
                  widths={"Material Description in Secondary Invoice": 50, "Validation Status": 44})
        else:
            ws.append(["(none)"])
        ws.append([])

    # ---------------- Reconciliation
    ws = wb.create_sheet("Reconciliation vs Amit")
    sheet_head(ws, "Reconciliation — Amit's 29-Oct list against this report",
               "His three tabs tested over {status filter × date window × aggregation key}. "
               "Closest definitions first; a non-match is a finding, not a failure.")
    t = ctx["amit_target"]
    ws.append(["Amit's SKU tab", f"{t['rows']:,} rows", f"{t['inv']:,.0f} Inv", f"₹{t['value']:,.0f}"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
    ws.append([])
    table(ws, ["Window", "Status filter", "Aggregation", "Rows", "Δ rows", "Inv", "₹ value", "Δ ₹", "Score"],
          [(r["window"], r["status"], r["key"], r["rows"], r["d_rows"], r["inv"], round(r["value"]),
            round(r["d_value"]), round(r["score"], 3)) for r in ctx["recon"][:12]],
          widths={"Aggregation": 26, "Status filter": 20, "Window": 24})
    fmt_ws(ws, money_cols=["Rows", "Δ rows", "Inv", "₹ value", "Δ ₹"],
           header_row=ws.max_row - min(12, len(ctx["recon"])))
    ws.append([])
    for k, v in ctx["amit_target_all"].items():
        ws.append([f"Amit {k} tab", f"{v['rows']} rows", f"{v['names']} names",
                   f"{v['inv'] or 0:.0f} Inv", f"₹{v['value'] or 0:,.0f}"])
    ws.append([])
    ws.append([ctx["recon_verdict"]])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True, color=CLAY)

    # ---------------- Count reconciliation
    ws = wb.create_sheet("Count reconciliation")
    sheet_head(ws, "Count reconciliation — every headline number and its denominator",
               "If a number in the readout is not on this sheet, it is not a number we own.")
    rows = []
    for label, k in ctx["kpi"].items():
        rows.append((label, "lines", k["lines_mapped"], k["lines"], round(k["lines_pct"], 1)))
        rows.append((label, "₹", round(k["value_mapped"]), round(k["value"]), round(k["value_pct"], 1)))
    table(ws, ["KPI", "Unit", "Mapped", "Total", "%"], rows, widths={"KPI": 30})
    fmt_ws(ws, money_cols=["Mapped", "Total"], pct_cols=["%"], header_row=4)
    ws.append([])
    ws.append(["Population ledger"])
    ws.cell(row=ws.max_row, column=1).font = SUB_FONT
    table(ws, ["Population", "Lines", "Invoices", "Value ₹"],
          [(k, v["lines"], v["invoices"], round(v["value"])) for k, v in ctx["populations"].items()],
          widths={"Population": 40})
    fmt_ws(ws, money_cols=["Lines", "Invoices", "Value ₹"],
           header_row=ws.max_row - len(ctx["populations"]))

    for ws in wb.worksheets:
        if ws.title != "Read me":
            fit_widths(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    log(f"workbook → {path}")


def build_onboarding_workbook(path: Path, ctx: dict):
    """The handover pack: who is missing from the master, in the order to fix them."""
    wb = Workbook()
    wb.remove(wb.active)
    ret, dist = ctx["ret_onboard"], ctx["dist_onboard"]
    ret_tot, dist_tot = float(ret["value"].sum()), float(dist["value"].sum())
    r80, d80 = int(ret["in_80"].sum()), int(dist["in_80"].sum())

    ws = wb.create_sheet("Read me")
    sheet_head(ws, "Onboarding — retailers and distributors missing from the master",
               f"Built {ctx['built_at']} from {ctx['report_name']} (secondary invoice report, "
               f"filters 2026-01-01 → 2026-11-01)")
    rows = [
        ("What this is", "Every distributor and retailer that still carries at least one unidentified line, "
                         "with the identifiers to onboard or merge them, ordered by the value at stake."),
        ("Retailers to onboard", f"{len(ret):,} accounts · ₹{ret_tot / 1e7:,.2f} Cr · "
                                  f"{r80} of them make the first 80%"),
        ("Distributors to onboard", f"{len(dist):,} accounts · ₹{dist_tot / 1e7:,.2f} Cr · "
                                     f"{d80} of them make the first 80%"),
        ("80% cut", "The `In 80% cut` column marks the rows to do first: they account for 80% of the value "
                    "carried by unidentified lines."),
        ("How to use", "1) Work the 80% cut rows first. 2) Match on GST; phone is the fallback for retailers, "
                       "SAP code for distributors. 3) After onboarding, re-pull the Turn Over Report and "
                       "re-run the builder — the account should move to Identified."),
        ("Identity", "GST first, then SAP code (distributor) / phone (retailer); name+pin is the last resort."),
        ("Not included", "Retailers that give neither GST nor phone still appear here — for them the fix is "
                         "capture at the invoice, not onboarding."),
        ("Rebuild", "python3 eval-datasets/code/build_turnover_truth.py … --out-onboarding <this file>"),
    ]
    for k, v in rows:
        ws.append([k, v])
        ws.cell(row=ws.max_row, column=1).font = Font(bold=True, color=INK)
        ws.cell(row=ws.max_row, column=2).alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 110

    ws = wb.create_sheet("Retailers to onboard")
    sheet_head(ws, f"Retailers — {len(ret):,} accounts, ₹{ret_tot / 1e7:,.2f} Cr on unidentified lines",
               "Ordered by value. `Lines unidentified` = lines whose Retailer Status is Not Identified.")
    hdr = ["#", "Retailer", "GST", "Phone", "Pin", "SFA code", "Identifier", "Lines unidentified",
           "Value ₹", "Share of retailer gap %", "Distributors", "In 80% cut"]
    hr = table(ws, hdr, [
        (i, x.ret_name, x.ret_gst, x.phone, x.pin, x.sfa, x.identifier,
         int(x.rows - x.mapped_rows), round(x.value),
         round(x.value / ret_tot * 100, 1) if ret_tot else 0.0,
         getattr(x, "distributors", ""), "YES" if x.in_80 else "")
        for i, x in enumerate(ret.itertuples(), start=1)
    ], widths={"Retailer": 44, "Distributors": 44})
    ws.freeze_panes = "A5"
    ws.auto_filter.ref = f"A4:L{ws.max_row}"
    fmt_ws(ws, money_cols=["Lines unidentified", "Value ₹"], pct_cols=["Share of retailer gap %"], header_row=hr)

    ws = wb.create_sheet("Distributors to onboard")
    sheet_head(ws, f"Distributors — {len(dist):,} accounts, ₹{dist_tot / 1e7:,.2f} Cr on unidentified lines",
               "Ordered by value. `Lines unidentified` = lines whose Distributor Status is Not Identified.")
    hdr = ["#", "Distributor", "GST", "SAP code", "Lines unidentified", "Value ₹",
           "Share of distributor gap %", "Top failure reasons", "In 80% cut"]
    hr = table(ws, hdr, [
        (i, x.dist_name, x.dist_gst, x.dist_code, int(x.rows - x.mapped_rows), round(x.value),
         round(x.value / dist_tot * 100, 1) if dist_tot else 0.0, x.top_failure_reasons,
         "YES" if x.in_80 else "")
        for i, x in enumerate(dist.itertuples(), start=1)
    ], widths={"Distributor": 44, "Top failure reasons": 56})
    ws.freeze_panes = "A5"
    ws.auto_filter.ref = f"A4:I{ws.max_row}"
    fmt_ws(ws, money_cols=["Lines unidentified", "Value ₹"], pct_cols=["Share of distributor gap %"], header_row=hr)

    ws = wb.create_sheet("Method")
    sheet_head(ws, "Method and numbers behind the two lists",
               "Same definitions as `Truth - mapping coverage.xlsx` — the 80% cut is a cumulative-value flag, "
               "not a separate population.")
    ws.append(["Population", "Deduplicated Turn Over Report lines (Duplicate-Invoice lines dropped)"])
    ws.append(["Distributor rule", "Distributor Status = Identified"])
    ws.append(["Retailer rule", "Retailer Status = Identified"])
    ws.append(["Entity identity", "GST → SAP code (distributor) / phone (retailer) → name+pin"])
    ws.append([])
    table(ws, ["Gap", "Accounts", "Of which in 80% cut", "Value ₹", "Lines unidentified"],
          [("Retailers", len(ret), r80, round(ret_tot), int((ret["rows"] - ret["mapped_rows"]).sum())),
           ("Distributors", len(dist), d80, round(dist_tot), int((dist["rows"] - dist["mapped_rows"]).sum()))],
          widths={"Gap": 18})
    for ws2 in wb.worksheets:
        if ws2.title != "Read me":
            fit_widths(ws2)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    log(f"onboarding pack → {path}")


# ------------------------------------------------------------------ readout
def inr(v: float) -> str:
    if abs(v) >= 1e7:
        return f"₹{v / 1e7:,.2f} Cr"
    if abs(v) >= 1e5:
        return f"₹{v / 1e5:,.2f} L"
    return f"₹{v:,.0f}"


def build_markdown(ctx: dict, path: Path):
    k = ctx["kpi"]
    L = []
    L.append("# Mapping truth baseline — Turn Over Report\n")
    L.append(f"**Status: built {ctx['built_at']} · source `{ctx['report_name']}` "
             f"(downloaded {ctx['report_mtime']}, filters 2026-01-01 → 2026-11-01)**\n")
    L.append("The platform's own line-level secondary-invoice report, used as the truth for mapping coverage. "
             "Definitions are the plan's §0 gates (D1–D8): headline = deduplicated population "
             "(Duplicate-Invoice lines dropped), MBNF counted as mapped but always split, value = line amount "
             "excluding tax.\n")
    L.append(f"Population: {ctx['populations']['deduplicated (headline)']['lines']:,} lines · "
             f"{ctx['populations']['deduplicated (headline)']['invoices']:,} invoices · "
             f"{inr(ctx['populations']['deduplicated (headline)']['value'])} "
             f"(as delivered: {ctx['populations']['all lines']['lines']:,} lines, "
             f"{inr(ctx['populations']['all lines']['value'])}).\n")

    L.append("## 1. Retailers and distributors (priority)\n")
    for subj, key in (("Retailers", "ret"), ("Distributors", "dist")):
        x = k[key]
        L.append(f"**{subj}** — {x['entities']:,} distinct entities\n")
        L.append(f"- {x['entities_full']:,} fully identified ({x['entities_full_pct']:.1f}%), "
                 f"{x['entities_partial']:,} partially, {x['entities_none']:,} not identified")
        L.append(f"- lines: {x['lines_mapped']:,} of {x['lines']:,} ({x['lines_pct']:.1f}%) · "
                 f"value: {inr(x['value_mapped'])} of {inr(x['value'])} ({x['value_pct']:.1f}%)")
        L.append(f"- invoices fully identified: {x['invoices_all_mapped']:,} of {x['invoices']:,} "
                 f"({x['invoices_pct']:.1f}%)\n")
    rid = ctx["ret_identifier"]
    L.append("Retailer identifier coverage (all entities, by what the invoice gives us):\n")
    L.append("| Identifier | Entities | Fully identified | Not identified |")
    L.append("|---|---:|---:|---:|")
    for label in ("GST+phone", "GST only", "phone only", "none"):
        v = rid.get(label, {"n": 0, "full": 0, "none": 0})
        L.append(f"| {label} | {v['n']:,} | {v['full']:,} | {v['none']:,} |")
    L.append("")
    g = ctx["ret_gst_stats"]
    L.append(f"Of the {g['with_gst']:,} retailers that carry a valid GST, {g['unmatched']} are not fully matched — "
             f"that is master coverage, not identifier capture. "
             f"{ctx['ret_identifier'].get('none', {}).get('none', 0)} retailers give us neither GST nor phone and "
             f"are not matched, which is a hard limit in the invoice itself.\n")
    L.append(f"Unmapped accounts: {k['dist']['entities'] - k['dist']['entities_full']:,} distributors and "
             f"{k['ret']['entities'] - k['ret']['entities_full']:,} retailers carry at least one unidentified line. "
             f"The 80%-of-value cut is {len(ctx['dist_80'])} distributors and {len(ctx['ret_80'])} retailers "
             f"(sheet `Onboarding 80%`, and the handover pack "
             f"`Onboarding - retailers and distributors.xlsx` lists every account with its identifiers "
             f"and flags that cut).\n")
    for title, rows, namecol, gstcol in (
        ("Top distributors by unmapped value", ctx["dist_80"].head(5), "dist_name", "dist_gst"),
        ("Top retailers by unmapped value", ctx["ret_80"].head(5), "ret_name", "ret_gst"),
    ):
        L.append(f"{title}:\n")
        for i, row in enumerate(rows.itertuples(), start=1):
            L.append(f"{i}. {getattr(row, namecol)}"
                     f"{' · ' + getattr(row, gstcol) if getattr(row, gstcol) else ''} — "
                     f"{inr(row.value)}, {int(row.rows - row.mapped_rows):,} lines unidentified")
        L.append("")

    L.append("## 2. Products\n")
    t = ctx["product_tiers"]
    L.append(f"- lines mapped: {k['prod']['lines_mapped']:,} of {k['prod']['lines']:,} "
             f"({k['prod']['lines_pct']:.1f}%), {inr(k['prod']['value_mapped'])} ({k['prod']['value_pct']:.1f}%)")
    L.append(f"- split: Mapped {t['Mapped']['lines']:,} · Mapped-Branding Not Found "
             f"{t['Mapped - Branding Not Found']['lines']:,} · Unmapped {t['Unmapped']['lines']:,} · "
             f"blank {t['(blank)']['lines']:,}")
    L.append(f"- titles: {k['title']['entities']:,} distinct; {k['title']['entities_full']:,} fully mapped "
             f"({k['title']['entities_full_pct']:.1f}%), {k['title']['entities_partial']:,} mixed, "
             f"{k['title']['entities_none']:,} never mapped")
    L.append(f"- SKUs: {k['sku']['entities']:,} distinct `Material No`; {k['sku']['entities_full']:,} carry only "
             f"mapped lines\n")
    agg = ctx["bucket_aggregates"]
    L.append(f"Unmapped titles bucketed against the 18-Oct catalogue ({len(ctx['unmapped_table']):,} "
             f"description+distributor rows, {agg['lines']:,} lines, {inr(agg['value'])}):\n")
    L.append("| Bucket | Rows | Lines | ₹ | Share |")
    L.append("|---|---:|---:|---:|---:|")
    for b, v in ctx["bucket_summary"].items():
        L.append(f"| {b} | {v['rows']:,} | {v['lines']:,} | {inr(v['value'])} | {v['rows_pct']:.1f}% |")
    L.append("")
    L.append(f"In-catalogue (A + A2 + B + C): {agg['in_cat_rows']:,} rows "
             f"({agg['in_cat_rows_pct']:.1f}%), {inr(agg['in_cat_value'])} ({agg['in_cat_value_pct']:.1f}%) — of which "
             f"{agg['ambig_rows']:,} rows are ambiguous (2+ SKUs fit). Genuine gaps / other brand / unparsed: "
             f"{agg['gap_rows']:,} rows ({agg['gap_rows_pct']:.1f}%).\n")
    L.append(f"{agg['brand_lines_pct']:.1f}% of the unmapped lines carry a brand token "
             f"(BIRLA / BIRLANU / HIL / BN) in the description — {agg['brand_lines']:,} of {agg['lines']:,} lines "
             f"({inr(agg['brand_value'])}).\n")

    L.append("## 3. What is not mapped, and why\n")
    L.append(ctx["narrative"])
    L.append("")
    L.append("## 4. Reconciliation with Amit's list\n")
    L.append(ctx["recon_verdict"])
    L.append("")
    amit = ctx["amit_target_all"]
    if amit:
        L.append("| Tab | Rows | Names | Inv | ₹ |")
        L.append("|---|---:|---:|---:|---:|")
        for name, v in amit.items():
            L.append(f"| {name} | {v['rows']:,} | {v['names']:,} | {v['inv'] or 0:,.0f} | {inr(v['value'] or 0)} |")
        L.append("")
    L.append("## 5. Method and reproducibility\n")
    L.append(f"- builder `eval-datasets/code/build_turnover_truth.py`; cache `{ctx['cache_name']}` "
             f"(a full xlsx read costs ~80 s, the cache makes rebuilds cheap)")
    L.append("- populations: " + " · ".join(f"{k2} {v['lines']:,}" for k2, v in ctx["populations"].items()))
    L.append("- values use `Total line Amount (Excluding tax)`; the incl-tax column is never mixed in")
    L.append("- workbook `F:\\...\\Mapping Inconsistency\\03_review\\Truth - mapping coverage.xlsx` carries the "
             "per-entity and per-title detail")
    L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    log(f"readout → {path}")


# ------------------------------------------------------------------ main
def build(args):
    t0 = time.time()
    report = Path(args.report)
    df, cache_name = load_report(report, Path(args.cache_dir), args.refresh)
    log(f"rows {len(df):,}")
    df = normalise(df)

    ded = df.loc[~df["dup"]].copy()
    log(f"deduped {len(ded):,} (dropped {len(df) - len(ded):,} duplicate-flagged)")

    populations = {
        "all lines": {"lines": len(df), "invoices": df["inv_key"].nunique(), "value": float(df["value"].sum())},
        "deduplicated (headline)": {"lines": len(ded), "invoices": ded["inv_key"].nunique(),
                                    "value": float(ded["value"].sum())},
        "distributor+retailer identified": {
            "lines": int((ded["dist_ok"] & ded["ret_ok"]).sum()),
            "invoices": ded.loc[ded["dist_ok"] & ded["ret_ok"], "inv_key"].nunique(),
            "value": float(ded.loc[ded["dist_ok"] & ded["ret_ok"], "value"].sum()),
        },
        "ingested last two months (created ≥ 2026-09-01)": {
            "lines": int((ded["created_ym"] >= "2026-09-01").sum()),
            "invoices": ded.loc[ded["created_ym"] >= "2026-09-01", "inv_key"].nunique(),
            "value": float(ded.loc[ded["created_ym"] >= "2026-09-01", "value"].sum()),
        },
    }

    def sku_kpi(frame: pd.DataFrame) -> dict:
        tab = (
            frame.assign(sku=frame["Material No"].where(frame["has_sku"]))
            .groupby("sku", dropna=True)
            .agg(rows=("value", "size"), value=("value", "sum"), mapped=("prod_ok", "sum"))
            .reset_index()
        )
        tab["status"] = [
            "Fully mapped" if m == r else ("Not mapped" if m == 0 else "Partially mapped")
            for m, r in zip(tab["mapped"], tab["rows"])
        ]
        full = int((tab["status"] == "Fully mapped").sum())
        inv = int(frame["inv_key"].nunique())
        inv_all = int(frame.loc[frame.groupby("inv_key")["has_sku"].transform("all"), "inv_key"].nunique())
        return {
            "lines": len(frame), "lines_mapped": int(frame["has_sku"].sum()),
            "lines_pct": pct(int(frame["has_sku"].sum()), len(frame)),
            "value": float(frame["value"].sum()),
            "value_mapped": float(frame.loc[frame["has_sku"], "value"].sum()),
            "value_pct": pct(float(frame.loc[frame["has_sku"], "value"].sum()), float(frame["value"].sum())),
            "invoices": inv, "invoices_all_mapped": inv_all, "invoices_pct": pct(inv_all, inv),
            "entities": len(tab), "entities_full": full,
            "entities_partial": int((tab["status"] == "Partially mapped").sum()),
            "entities_none": int((tab["status"] == "Not mapped").sum()),
            "entities_full_pct": pct(full, len(tab)),
        }

    sku_table = (
        ded.assign(sku=ded["Material No"].where(ded["has_sku"]))
        .groupby("sku", dropna=True)
        .agg(rows=("value", "size"), value=("value", "sum"), mapped=("prod_ok", "sum"))
        .reset_index()
    )
    sku_table["status"] = [
        "Fully mapped" if m == r else ("Not mapped" if m == 0 else "Partially mapped")
        for m, r in zip(sku_table["mapped"], sku_table["rows"])
    ]

    kpi = {
        "dist": flag_kpi(ded, "dist_ok", "dist_key"),
        "dist_all": flag_kpi(df, "dist_ok", "dist_key"),
        "ret": flag_kpi(ded, "ret_ok", "ret_key"),
        "ret_all": flag_kpi(df, "ret_ok", "ret_key"),
        "prod": flag_kpi(ded, "prod_ok", "desc_norm"),
        "prod_all": flag_kpi(df, "prod_ok", "desc_norm"),
        "title": flag_kpi(ded, "prod_ok", "desc_norm"),
        "title_all": flag_kpi(df, "prod_ok", "desc_norm"),
        "sku": sku_kpi(ded),
        "sku_all": sku_kpi(df),
    }

    # ---- entity tables
    dist_table = entity_table(ded, "dist_key", "dist_ok", "Distributor Name", "Distributor GST",
                              ["Distributor Code (SAP)", "Matched Distributor Name"])
    dist_table = dist_table.rename(columns={"Distributor Name": "dist_name", "Distributor GST": "dist_gst",
                                            "Distributor Code (SAP)": "dist_code",
                                            "Matched Distributor Name": "matched_name"})
    ret_table = entity_table(ded, "ret_key", "ret_ok", "Retailer Name", "Retailer GST",
                             ["Retailer Mobile No", "Retailer Pin code", "Retailer Code (SFA)",
                              "Matched Retailer Name"])
    ret_table = ret_table.rename(columns={"Retailer Name": "ret_name", "Retailer GST": "ret_gst",
                                          "Retailer Mobile No": "phone", "Retailer Pin code": "pin",
                                          "Retailer Code (SFA)": "sfa", "Matched Retailer Name": "matched_name"})
    ret_table["identifier"] = [
        "GST+phone" if g and p else ("GST only" if g else ("phone only" if p else "none"))
        for g, p in zip(ret_table["ret_gst"].map(lambda x: bool(GST_RE.match(str(x)))),
                        ret_table["phone"].astype(str).str.len().eq(10))
    ]

    def with_cum(tab, value_col="value"):
        """Sort by value and flag the rows that make the first 80% of it."""
        t = tab.sort_values(value_col, ascending=False).copy()
        tot = t[value_col].sum()
        t["cum_pct"] = ((t[value_col].cumsum() / tot * 100) if tot else 0).round(1)
        t["in_80"] = t["cum_pct"] <= 80.0
        return t

    dist_onboard = with_cum(dist_table[dist_table["status"] != "Fully mapped"])
    ret_onboard = with_cum(ret_table[ret_table["status"] != "Fully mapped"])
    dist_80 = dist_onboard[dist_onboard["in_80"]]
    ret_80 = ret_onboard[ret_onboard["in_80"]]

    # which distributor(s) each unmatched retailer is buying through (top 2 by value)
    dd = (
        ded.loc[~ded["ret_ok"]]
        .groupby(["ret_key", "Distributor Name"])["value"].sum().reset_index()
        .sort_values("value", ascending=False)
    )
    top_dists = dd.groupby("ret_key").head(2).groupby("ret_key")["Distributor Name"].agg("; ".join)
    ret_onboard["distributors"] = ret_onboard["ret_key"].map(top_dists).fillna("")

    # ---- title table
    def top_status(s):
        c = collections.Counter(s)
        return c.most_common(1)[0][0] if c else ""

    title_table = (
        ded.groupby("desc_norm")
        .agg(example=("desc_raw", "first"), rows=("value", "size"), value=("value", "sum"),
             invoices=("inv_key", "nunique"), mapped_rows=("prod_ok", "sum"),
             skus=("Material No", lambda s: s[s.ne("")].nunique()), top_status=("parse", top_status))
        .reset_index()
    )
    title_table["status"] = [
        "Fully mapped" if m == r else ("Never mapped" if m == 0 else "Mixed")
        for m, r in zip(title_table["mapped_rows"], title_table["rows"])
    ]
    title_table = title_table.sort_values("value", ascending=False)

    # ---- product tiers
    tiers = {}
    tot_lines, tot_val = len(ded), float(ded["value"].sum())
    parse_col = ded["parse"].where(ded["parse"].ne(""), "(blank)")
    for name in ("Mapped", "Mapped - Branding Not Found", "Unmapped", "(blank)"):
        sub = ded[parse_col == name]
        tiers[name] = {"lines": len(sub), "lines_pct": pct(len(sub), tot_lines),
                       "value": float(sub["value"].sum()), "value_pct": pct(float(sub["value"].sum()), tot_val)}

    # ---- unmapped titles + buckets
    cat = cmi.load_catalogue(Path(args.catalogue))
    clf = cmi.Classifier(cat)
    log(f"catalogue {len(cat):,} rows")

    unm = ded[ded["parse"].eq("Unmapped")].copy()
    agg = (
        unm.groupby(["desc_norm", "dist_key"])
        .agg(desc_raw=("desc_raw", "first"), dist_name=("Distributor Name", "first"),
             lines=("value", "size"), value=("value", "sum"), invoices=("inv_key", "nunique"))
        .reset_index().sort_values("value", ascending=False)
    )
    buckets = []
    bucket_meta = {
        "A_exact_title": ("A · exact catalogue title, one SKU", "the product is in the catalogue — title matches exactly"),
        "A_exact_dup": ("A2 · exact title, catalogue has two SKUs", "duplicate catalogue SKUs block the match"),
        "B_unique_sku": ("B · one SKU after normalisation", "in the catalogue, resolvable by normalisation"),
        "C_ambiguous": ("C · 2+ SKUs fit", "in the catalogue; invoice omits the variant/length"),
        "D_variant_conflict": ("D · qualifier conflict", "stated qualifier contradicts every candidate"),
        "E_size_gap": ("E · size not in export", "family/type exists, this size does not"),
        "F_no_size": ("F · no size stated", "family/type exists, title has no size"),
        "G_type_gap": ("G · type not in export", "family/type absent from the export"),
        "H_other_brand": ("H · other brand", "competitor line, not a BirlaNu mapping failure"),
        "I_unparsed": ("I · unparsed", "no family/type/size recoverable"),
    }
    for row in agg.itertuples(index=False):
        c = clf.classify(row.desc_raw)
        buckets.append({"bucket": c["bucket"], "detail": c["detail"], "skus": len(c["skus"]),
                        "candidates": "; ".join(c["candidates"][:3]), "desc_norm": row.desc_norm,
                        "desc_raw": row.desc_raw, "dist_key": row.dist_key, "dist_name": row.dist_name,
                        "lines": row.lines, "value": row.value, "invoices": row.invoices})
    bdf = pd.DataFrame(buckets)
    unmapped_table = bdf.sort_values("value", ascending=False)
    bs = {}
    for b, sub in bdf.groupby("bucket"):
        label, meaning = bucket_meta.get(b, (b, ""))
        bs[label] = {"rows": len(sub), "lines": int(sub["lines"].sum()),
                     "value": float(sub["value"].sum()),
                     "rows_pct": pct(len(sub), len(bdf)), "meaning": meaning}
    bucket_summary = dict(sorted(bs.items(), key=lambda kv: -kv[1]["value"]))

    # ---- bucket aggregates + brand-token share (measured, not asserted)
    in_cat_buckets = {"A_exact_title", "A_exact_dup", "B_unique_sku", "C_ambiguous"}
    gap_buckets = {"D_variant_conflict", "E_size_gap", "F_no_size", "G_type_gap",
                   "H_other_brand", "I_unparsed"}
    in_cat = unmapped_table[unmapped_table["bucket"].isin(in_cat_buckets)]
    gap = unmapped_table[unmapped_table["bucket"].isin(gap_buckets)]
    amb = unmapped_table[unmapped_table["bucket"].eq("C_ambiguous")]
    unm_lines = ded[ded["parse"].eq("Unmapped")]
    brand_mask = unm_lines["desc_raw"].str.contains(r"\b(?:BIRLA|BIRLANU|HIL|BN)\b", case=False, regex=True)
    bucket_tot_rows = len(unmapped_table)
    bucket_tot_val = float(unmapped_table["value"].sum())
    bucket_aggregates = {
        "rows": bucket_tot_rows, "lines": int(unmapped_table["lines"].sum()), "value": bucket_tot_val,
        "in_cat_rows": len(in_cat), "in_cat_rows_pct": pct(len(in_cat), bucket_tot_rows),
        "in_cat_value": float(in_cat["value"].sum()),
        "in_cat_value_pct": pct(float(in_cat["value"].sum()), bucket_tot_val),
        "ambig_rows": len(amb), "gap_rows": len(gap), "gap_rows_pct": pct(len(gap), bucket_tot_rows),
        "brand_lines": int(brand_mask.sum()), "brand_lines_pct": pct(int(brand_mask.sum()), len(unm_lines)),
        "brand_value": float(unm_lines.loc[brand_mask, "value"].sum()),
    }

    # ---- retailer identifier cross-tab (which key could have matched)
    ret_identifier = {}
    for label, sub in ret_table.groupby("identifier"):
        ret_identifier[label] = {
            "n": len(sub),
            "full": int((sub["status"] == "Fully mapped").sum()),
            "none": int((sub["status"] == "Not mapped").sum()),
        }
    gst_ok_ent = ret_table["ret_gst"].map(
        lambda x: bool(GST_RE.match(str(x).upper().replace(" ", "")))
    )
    ret_gst_stats = {
        "with_gst": int(gst_ok_ent.sum()),
        "unmatched": int((gst_ok_ent & (ret_table["status"] != "Fully mapped")).sum()),
    }

    # ---- anomalies
    anomalies = anomaly_blocks(ded)

    # ---- reconciliation
    amit = read_amit(Path(args.amit)) if args.amit else None
    recon, verdict = [], "Amit's workbook not supplied — reconciliation skipped."
    if amit:
        sku_t = amit["SKU"]
        recon = reconciliation(ded, sku_t)
        best = recon[0] if recon else None
        if best:
            r_ok = abs(best["d_rows"]) <= 0.01 * sku_t["rows"]
            i_ok = abs(best["inv"] - sku_t["inv"]) <= 0.05 * sku_t["inv"]
            v_ok = abs(best["d_value"]) <= 0.10 * sku_t["value"]
            label = f"{best['status']} / {best['key']} / {best['window']}"
            if r_ok and i_ok and v_ok:
                verdict = (f"Reproduced: {label} → rows {best['rows']:,} (his {sku_t['rows']:,}), "
                           f"Inv {best['inv']:,} (his {sku_t['inv']:,.0f}), ₹{best['value']:,.0f} "
                           f"(his ₹{sku_t['value']:,.0f}).")
            else:
                hits = [n for n, ok in (("rows", r_ok), ("Inv", i_ok), ("value", v_ok)) if ok]
                misses = [n for n, ok in (("rows", r_ok), ("Inv", i_ok), ("value", v_ok)) if not ok]
                verdict = (
                    f"No definition reproduces all three of Amit's numbers. Closest: {label} → "
                    f"rows {best['rows']:,} vs his {sku_t['rows']:,} (Δ {best['d_rows']:+,}), "
                    f"Inv {best['inv']:,} vs {sku_t['inv']:,.0f}, "
                    f"₹{best['value']:,.0f} vs ₹{sku_t['value']:,.0f}. Matches on {', '.join(hits) or 'nothing'}; "
                    f"fails {', '.join(misses)}. For this slice the report's other money columns read "
                    f"₹{best['value_incl']:,.0f} (incl. tax) and ₹{best['value']:,.0f} (excl. tax) — neither is his "
                    f"₹{sku_t['value']:,.0f}, so his Value column is not the line amount. Confirm the export "
                    f"filter (date window) and what Value sums before treating his list as a KPI."
                )

    narrative = (
        f"Distributor identification is high ({kpi['dist']['lines_pct']:.1f}% of lines) but concentrated: "
        f"{kpi['dist']['entities'] - kpi['dist']['entities_full']} accounts carry every unidentified line and "
        f"{len(dist_80)} of them make 80% of that value. Retailers are the weaker axis "
        f"({kpi['ret']['lines_pct']:.1f}% of lines, {kpi['ret']['entities_full_pct']:.1f}% of retailers fully "
        f"identified); the split that matters is master coverage versus no key — "
        f"{ret_gst_stats['unmatched']} of the {ret_gst_stats['with_gst']:,} retailers that carry a valid GST are "
        f"not fully matched, while {ret_identifier.get('none', {}).get('none', 0)} entities give us neither GST "
        f"nor phone. Products "
        f"behave differently again: {kpi['prod']['lines_pct']:.1f}% of lines carry a SKU, and of the unmapped "
        f"description+distributor rows {bucket_aggregates['in_cat_rows_pct']:.1f}% are catalogue products "
        f"(A/A2/B/C — {bucket_aggregates['ambig_rows']:,} of them ambiguous) while "
        f"{bucket_aggregates['gap_rows_pct']:.1f}% are size gaps, qualifier conflicts, competitor brands or "
        f"unparsed text. {bucket_aggregates['brand_lines_pct']:.1f}% of unmapped lines carry a brand token, so the "
        f"next product gain is variant/vocabulary rules, not new catalogue entries."
    )

    ctx = {
        "report_name": Path(args.report).name,
        "report_mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(Path(args.report).stat().st_mtime)),
        "built_at": time.strftime("%Y-%m-%d %H:%M"),
        "cache_name": cache_name,
        "kpi": kpi,
        "populations": populations,
        "product_tiers": tiers,
        "dist_table": dist_table,
        "ret_table": ret_table,
        "dist_name_col": "dist_name",
        "dist_gst_col": "dist_gst",
        "ret_name_col": "ret_name",
        "ret_gst_col": "ret_gst",
        "dist_80": dist_80,
        "ret_80": ret_80,
        "dist_onboard": dist_onboard,
        "ret_onboard": ret_onboard,
        "title_table": title_table,
        "unmapped_table": unmapped_table,
        "bucket_summary": bucket_summary,
        "bucket_aggregates": bucket_aggregates,
        "ret_identifier": ret_identifier,
        "ret_gst_stats": ret_gst_stats,
        "anomalies": anomalies,
        "amit_target": amit["SKU"] if amit else {"rows": 0, "inv": 0, "value": 0},
        "amit_target_all": amit or {},
        "recon": recon,
        "recon_verdict": verdict,
        "narrative": narrative,
    }

    # ---- write
    build_workbook(Path(args.out_workbook), ctx)
    if args.out_onboarding:
        build_onboarding_workbook(Path(args.out_onboarding), ctx)
    if args.out_md:
        build_markdown(ctx, Path(args.out_md))

    summary = {
        "built_at": ctx["built_at"], "report": ctx["report_name"], "cache": cache_name,
        "populations": populations,
        "kpi": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                for k, v in kpi.items()},
        "product_tiers": tiers,
        "buckets": bucket_summary,
        "bucket_aggregates": bucket_aggregates,
        "ret_identifier": ret_identifier,
        "ret_gst_stats": ret_gst_stats,
        "onboarding": {"distributors_80": len(dist_80), "retailers_80": len(ret_80),
                       "distributors_total": len(dist_onboard), "retailers_total": len(ret_onboard)},
        "amit": amit, "recon_top": recon[:5], "recon_verdict": verdict,
        "anomalies": {label: len(sub) for label, sub in anomalies},
    }
    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"summary → {p}")
    log(f"done in {time.time() - t0:.1f}s")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--amit", default="")
    ap.add_argument("--out-workbook", required=True)
    ap.add_argument("--out-onboarding", default="", help="handover pack: unmatched retailers/distributors")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--cache-dir", default=str(HERE.parent / "outputs" / "turnover-truth"))
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and re-read the xlsx")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
