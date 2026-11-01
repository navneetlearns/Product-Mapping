#!/usr/bin/env python3
"""Render workbook sheets to HTML/PNG and measure text clipping — the no-Excel visual gate.

Why: the delivered .xlsx is never opened by a human in this environment (no LibreOffice), so a
column too narrow for its numbers (Excel's ###) would ship unnoticed. This renders each sheet the
way Excel lays it out (fixed table layout, Excel column widths), then asks the browser which cells
overflow their box. Any hit is a cell the user would see truncated.

Usage
  python3 qa_render.py --workbook "F:/.../Truth - mapping coverage.xlsx" \
      --sheets "Read me,Headline,Onboarding 80%,Reconciliation vs Amit" --out-dir /tmp/qa_render
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import openpyxl
from openpyxl.utils import get_column_letter

CHARS_PER_PX = 7.0  # Excel column width unit ≈ 7 px at 96 dpi


def cell_text(c) -> str:
    v = c.value
    if v is None:
        return ""
    fmt = c.number_format or ""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "%" in fmt:
            return f"{v:.1f}%"
        if "#,##0" in fmt:
            return f"{v:,.0f}"
        return f"{v}"
    return str(v)


def sheet_html(ws, max_rows: int) -> tuple[str, int, int]:
    n_cols = ws.max_column or 1
    widths = []
    for i in range(1, n_cols + 1):
        w = ws.column_dimensions[get_column_letter(i)].width
        widths.append(max(40, int((w or 8.43) * CHARS_PER_PX)))
    rows = list(ws.iter_rows(min_row=1, max_row=min(ws.max_row, max_rows)))
    parts = ['<table style="table-layout:fixed;border-collapse:collapse;font:11px Calibri,Arial">']
    parts.append("<colgroup>" + "".join(f'<col style="width:{w}px">' for w in widths) + "</colgroup>")
    for r_i, row in enumerate(rows, start=1):
        parts.append("<tr>")
        for c in row:
            txt = html.escape(cell_text(c))
            style = "border:1px solid #d5dcd7;padding:2px 4px;overflow:hidden;white-space:nowrap"
            if r_i <= 3:
                style += ";font-weight:600;background:#e8eee9"
            parts.append(f'<td style="{style}">{txt}</td>')
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts), len(rows), n_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", required=True)
    ap.add_argument("--sheets", default="")
    ap.add_argument("--out-dir", default="/tmp/qa_render")
    ap.add_argument("--max-rows", type=int, default=250)
    args = ap.parse_args()

    wb = openpyxl.load_workbook(args.workbook, data_only=True)
    names = [s.strip() for s in args.sheets.split(",") if s.strip()] or wb.sheetnames
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    report = {"workbook": args.workbook, "sheets": {}}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name in names:
            ws = wb[name]
            table, n_rows, n_cols = sheet_html(ws, args.max_rows)
            page = browser.new_page(viewport={"width": 1800, "height": 1000})
            page.set_content(f"<body style='margin:8px'>{table}</body>")
            clipped = page.evaluate(
                """() => {
                    const bad = [];
                    document.querySelectorAll('td').forEach((td, i) => {
                        if (td.scrollWidth > td.clientWidth + 1) {
                            bad.push({i, text: td.innerText.slice(0, 80),
                                      need: td.scrollWidth, have: td.clientWidth});
                        }
                    });
                    return bad;
                }"""
            )
            size = page.evaluate("() => ({w: document.body.scrollWidth, h: document.body.scrollHeight})")
            shot = out / f"{name.replace(' ', '_').replace('%', 'pct')}.png"
            page.screenshot(path=str(shot), full_page=True)
            page.close()
            report["sheets"][name] = {
                "rendered_rows": n_rows, "cols": n_cols,
                "page_px": size, "clipped_cells": len(clipped),
                "clipped_sample": clipped[:8], "screenshot": str(shot),
            }
        browser.close()

    (out / "qa_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    bad_total = sum(v["clipped_cells"] for v in report["sheets"].values())
    for name, v in report["sheets"].items():
        print(f"{name:28s} rows={v['rendered_rows']:>4} cols={v['cols']:>2} "
              f"clipped={v['clipped_cells']:>3} page={v['page_px']['w']}x{v['page_px']['h']}")
    print("TOTAL clipped cells:", bad_total)
    for name, v in report["sheets"].items():
        for c in v["clipped_sample"]:
            print(f"  [{name}] '{c['text']}' needs {c['need']}px, has {c['have']}px")
    print("report:", out / "qa_report.json")
    raise SystemExit(1 if bad_total else 0)


if __name__ == "__main__":
    main()
