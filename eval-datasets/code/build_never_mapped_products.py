#!/usr/bin/env python3
"""Workbook: products on Amit's list that never map, with their invoice lines.

Sets: a product is "never mapped" when none of its report lines carry a mapped-tier
`Parsing Status` (Mapped / Mapped - Branding Not Found) anywhere in the export — the
56-product answer at the 4-Nov snapshot. Two sheets:
  Never mapped products  one row per product with line / distributor / invoice counts
  Unmapped lines         one row per report line: distributor, retailer, invoice, value
The client workbook is written to the F: review folder (contains party identifiers);
never commit it to the public repo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import build_turnover_truth as btt  # noqa: E402
import classify_mapping_issues as cmi  # noqa: E402

SRC = Path("/mnt/f/Product/zotok/Product Mapping/Mapping Inconsistency/01_source")
REVIEW = Path("/mnt/f/Product/zotok/Product Mapping/Mapping Inconsistency/03_review")
DESC = "Material Description in Secondary Invoice"
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(bold=True, color="FFFFFF", size=10)
BODY_FONT = Font(size=10)

# Brand read for the 56 (2026-11-04 snapshot). "platform" = the platform's own presence check
# flags it Competitor brand; "description" = brand name / non-BirlaNu category visible in the text.
BRAND_READ: dict[str, tuple[str, str]] = {
    "NEROLAC WALL PUTTY _20K": ("Other brand", "platform: Competitor brand (Nerolac)"),
    'FINOLEX - 4" RCC COUPLER': ("Other brand", "platform: Competitor brand (Finolex)"),
    "25MM CPVC FABT ( SUDHAKAR)": ("Other brand", "platform: Competitor brand (Sudhakar)"),
    "32MM CPVC FABT ( SUDHAKAR)": ("Other brand", "platform: Competitor brand (Sudhakar)"),
    "ROTO TANK 1000 LITRE 3PLY": ("Other brand", "description: Roto tank brand; not BIRLA HIL tank naming"),
    "QUICKFIX PVC SOL TIN 500ML": ("Other brand", "description: Quickfix solvent-cement brand (Wembley)"),
    "HC EWC 31 W S TRAP": ("Other brand", "description: EWC sanitaryware; not a BirlaNu category"),
    "BODY OF CONC. WALL MIXER CAST BODY": ("Other brand", "description: concealed mixer body; not a BirlaNu category"),
    "Diverter Jal": ("Other brand", "description: faucet diverter; not a BirlaNu category"),
    'CPVC Bibcock 1"': ("Other brand", "description: bibcock/tap; no bibcock in the BirlaNu catalogue"),
    "4 LAYER GRAND TANK 1000 LTR ( WHITE )": ("Other brand?",
                                              "description: 4-layer tank, Grand naming; BirlaNu tanks are BIRLA HIL 3Layer"),
}


def load_amit_products(xlsx: Path) -> pd.DataFrame:
    wb = load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb["SKU not mapped"]
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    wb.close()
    hdr = [str(c).strip() for c in rows[0]]
    i = hdr.index(DESC)
    raw = [str(r[i]).strip() for r in rows[1:] if r[i] not in (None, "")]
    df = pd.DataFrame({"amit_desc": raw})
    df["desc_norm"] = df["amit_desc"].map(cmi.norm_text)
    return df.drop_duplicates("desc_norm").reset_index(drop=True)


def sheet(wb: Workbook, title: str, headers: list[str], rows: list[list], widths: list[int],
          money_cols: set[int] | None = None) -> None:
    ws = wb.create_sheet(title)
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
        cell.alignment = Alignment(vertical="center")
        ws.column_dimensions[get_column_letter(c)].width = widths[c - 1]
    for r in rows:
        ws.append(r)
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=len(headers)):
        for cell in row:
            cell.font = BODY_FONT
            if money_cols and cell.column in money_cols:
                cell.number_format = "#,##0.00"
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(ws.max_row, 1)}"
    ws.sheet_view.zoomScale = 90


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=SRC / "Turn Over Report.xlsx")
    ap.add_argument("--amit", type=Path, default=SRC / "Mapping issue list shared by amit.xlsx")
    ap.add_argument("--cache-dir", type=Path,
                    default=CODE.parent / "outputs/turnover-truth")
    ap.add_argument("--out-dir", type=Path, default=REVIEW)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    df, cache = btt.load_report(args.report, args.cache_dir, args.refresh)
    df = btt.normalise(df)
    amit = load_amit_products(args.amit)
    stamp = cache.replace("turnover-truth-", "").replace(".csv.gz", "")

    per = df.groupby("desc_norm").agg(
        lines=("parse", "size"),
        mapped=("parse", lambda s: s.isin(btt.MAPPED_TIERS).sum()),
    ).reset_index()
    key = amit.merge(per, on="desc_norm", how="left").fillna({"lines": 0, "mapped": 0})
    never = key.loc[key["mapped"].eq(0), ["desc_norm", "amit_desc"]]
    lines = df[df["desc_norm"].isin(set(never["desc_norm"]))].copy()

    g = lines.groupby("desc_norm").agg(
        lines=("parse", "size"), dups=("dup", "sum"),
        dists=("dist_key", "nunique"), invs=("inv_key", "nunique"), value=("value", "sum"),
    ).reset_index()
    prod = never.merge(g, on="desc_norm", how="left").sort_values(
        ["lines", "amit_desc"], ascending=[False, True])
    example = (lines.sort_values("Invoice Date")
               .groupby("desc_norm")["Distributor Name"]
               .agg(lambda s: ", ".join(dict.fromkeys(s))[:120]).reset_index(name="dist_examples"))
    retailer_example = (lines.sort_values("Invoice Date")
                        .groupby("desc_norm")["Retailer Name"]
                        .agg(lambda s: ", ".join(dict.fromkeys(s))[:120]).reset_index(name="ret_examples"))
    prod = prod.merge(example, on="desc_norm", how="left").merge(retailer_example, on="desc_norm", how="left")

    prod_rows = [[r.amit_desc, int(r.lines), int(r.dups), int(r.dists), int(r.invs),
                  round(float(r.value), 2), r.dist_examples or "", r.ret_examples or "",
                  BRAND_READ.get(r.amit_desc, ("BirlaNu / generic", ""))[0],
                  BRAND_READ.get(r.amit_desc, ("BirlaNu / generic", ""))[1]]
                 for r in prod.itertuples()]
    total_value = round(float(prod["value"].sum()), 2)

    l = lines.sort_values(["desc_norm", "Invoice Date", "Invoice Number"]).copy()
    amit_name = dict(zip(never["desc_norm"], never["amit_desc"]))
    line_rows = []
    for _, r in l.iterrows():
        line_rows.append([
            amit_name.get(r["desc_norm"], ""),
            r["Material Description in Secondary Invoice"],
            r["Distributor Name"], r["Distributor GST"], r["Distributor Code (SAP)"],
            r["Retailer Name"], r["Retailer GST"], r["Retailer Mobile No"], r["Retailer Pin code"],
            r["Invoice Number"], str(r["Invoice Date"])[:10],
            "Yes" if r["dup"] else "",
            float(r["qty"]), round(float(r["value"]), 2),
        ])

    wb = Workbook()
    wb.remove(wb.active)
    sheet(wb, "Never mapped products",
          ["Product (Amit list)", "Lines", "Duplicate lines", "Distributors", "Invoices",
           "Value (₹, excl tax)", "Distributor(s) on those lines", "Retailer(s) on those lines",
           "Brand read", "Evidence"],
          prod_rows, [52, 8, 10, 12, 9, 16, 40, 40, 16, 52], money_cols={6})
    ws = wb["Never mapped products"]
    ws.cell(row=ws.max_row + 1, column=1, value="TOTAL").font = Font(bold=True, size=10)
    c = ws.cell(row=ws.max_row, column=6, value=total_value)
    c.font, c.number_format = Font(bold=True, size=10), "#,##0.00"

    sheet(wb, "Unmapped lines",
          ["Product (Amit list)", "Invoice description", "Distributor", "Distributor GST",
           "Distributor SAP code", "Retailer", "Retailer GST", "Retailer mobile", "Retailer pin",
           "Invoice Number", "Invoice Date", "Duplicate invoice?", "Qty", "Line value (₹, excl tax)"],
          line_rows, [42, 42, 30, 18, 12, 30, 18, 14, 9, 18, 11, 10, 8, 14], money_cols={14})

    notes = wb.create_sheet("Method")
    other = [r for r in prod_rows if str(r[8]).startswith("Other brand")]
    other_val = sum(float(r[5]) for r in other)
    other_platform = sum(1 for r in other if "platform:" in str(r[9]))
    yr = l["Invoice Date"].str[:4]
    date_note = (f"as exported: 2018 ×{int(yr.eq('2018').sum())}, 2016 ×{int(yr.eq('2016').sum())}, "
                 f"2025 ×{int(yr.eq('2025').sum())}, blank ×{int(l['Invoice Date'].eq('').sum())} — "
                 "date-parsing noise (invoice HRE5AM2018 reads as 2018-05-17); treat dates with care")
    rep = l.groupby(["Invoice Number", "desc_norm", "Qty", "value"]).size()
    reps = rep[rep > 1]
    repeat_note = (f"{int(reps.sum())} of {len(l)} lines sit in {len(reps)} same-invoice / same-product / "
                   "same-qty / same-amount groups; kept as exported (genuine repeats or multi-page re-capture)")
    for row in [
        ["Source", f"Turn Over Report.xlsx — export dated {stamp} (Secondary Invoice Report sheet)"],
        ["Product list", "Mapping issue list shared by amit.xlsx — 'SKU not mapped' tab"],
        ["Uniqueness", "3,216 rows -> 2,359 unique products after description normalisation (cmi.norm_text)"],
        ["Definition", "Never mapped = no line for the description carries Parsing Status Mapped or "
                       "Mapped - Branding Not Found anywhere in the export"],
        ["Counts", f"{len(prod)} products / {len(l)} lines / ₹{total_value:,.2f} (line amount excl. tax)"],
        ["Other brand", f"{len(other)} products / ₹{other_val:,.2f} — {other_platform} flagged by the platform's "
                        "presence check (Competitor brand); the rest read from the brand name / non-BirlaNu "
                        "category. The other 45 are BirlaNu-family or brandless descriptions."],
        ["Duplicates", "Lines flagged 'Duplicate invoice?' are duplicate-invoice records in the report; "
                       "they are included so the list matches the platform view"],
        ["Repeated lines", repeat_note],
        ["Dates", date_note],
        ["Grain", "Product status is per line and distributor-specific: a product mapped for one "
                  "distributor can be Unmapped for another"],
    ]:
        notes.append(row)
    notes.column_dimensions["A"].width = 16
    notes.column_dimensions["B"].width = 110
    for row in notes.iter_rows(min_row=1, max_row=notes.max_row, max_col=2):
        row[0].font = Font(bold=True, size=10)
        row[1].font = BODY_FONT
        row[1].alignment = Alignment(wrap_text=True, vertical="top")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"Products never mapped - {len(prod)} (report {stamp}).xlsx"
    wb.save(out)
    csv_out = out.with_suffix(".csv")
    pd.DataFrame(line_rows, columns=["Product (Amit list)", "Invoice description", "Distributor",
                                     "Distributor GST", "Distributor SAP code", "Retailer",
                                     "Retailer GST", "Retailer mobile", "Retailer pin",
                                     "Invoice Number", "Invoice Date", "Duplicate invoice?",
                                     "Qty", "Line value (Rs, excl tax)"]).to_csv(csv_out, index=False)

    print(f"products {len(prod)} · lines {len(l)} · value ₹{total_value:,.2f}")
    print(f"other-brand read: {len(other)} products / ₹{other_val:,.2f} ({other_platform} platform-flagged)")
    print("wrote", out)
    print("wrote", csv_out)

    # read-back gate: sizes and counts
    chk = load_workbook(out, read_only=True)
    print("readback:", {s: (chk[s].max_row, chk[s].max_column) for s in chk.sheetnames})
    chk.close()


if __name__ == "__main__":
    main()
