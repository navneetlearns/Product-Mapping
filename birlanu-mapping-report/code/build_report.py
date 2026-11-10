#!/usr/bin/env python3
"""Render the BirlaNu invoice-mapping report to one self-contained HTML file.

Numbers live in code/numbers.json and are asserted to add up before a page is
written, so a stale or mistyped figure fails the build instead of shipping:

  lines_all - duplicates == lines
  entity total == full + partial + none
  products total == full + partial + never ; any == full + partial
  mapped + mbnf + unmapped == lines

Fonts are embedded as base64 so the HTML can be emailed as a single file.

Usage:
  python3 code/build_report.py
  python3 code/build_report.py --numbers code/numbers.json --out outputs/x.html
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "assets" / "fonts"
DEFAULT_OUT = ROOT / "outputs" / "BirlaNu-Mapping-Coverage-2026-11-08.html"

FONT_FILES = {
    "FONT_SANS": "InstrumentSans-Regular.ttf",
    "FONT_SANS_B": "InstrumentSans-Bold.ttf",
    "FONT_SERIF": "InstrumentSerif-Italic.ttf",
    "FONT_MONO": "JetBrainsMono-Regular.ttf",
    "FONT_MONO_B": "JetBrainsMono-Bold.ttf",
}


def fail(msg: str) -> None:
    sys.exit(f"FAIL: {msg}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}" if d else "0.0"


def width(n: int, d: int) -> str:
    return f"{100 * n / d:.3f}" if d else "0.000"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", default=str(ROOT / "code" / "numbers.json"))
    ap.add_argument("--template", default=str(ROOT / "code" / "report.template.html"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    n = json.loads(Path(args.numbers).read_text(encoding="utf-8"))
    dist, ret, prod, split = n["distributors"], n["retailers"], n["products"], n["lines_split"]

    # --- identities: a bad figure must not reach the page -----------------
    check(n["lines_all"] - n["duplicates"] == n["lines"],
          f"lines: {n['lines_all']} - {n['duplicates']} != {n['lines']}")
    for name, e in (("distributors", dist), ("retailers", ret)):
        check(e["total"] == e["full"] + e["partial"] + e["none"],
              f"{name}: total {e['total']} != {e['full']}+{e['partial']}+{e['none']}")
    check(prod["total"] == prod["full"] + prod["partial"] + prod["never"],
          "products: total != full + partial + never")
    check(prod["any"] == prod["full"] + prod["partial"], "products: any != full + partial")
    check(split["mapped"] + split["mbnf"] + split["unmapped"] == n["lines"],
          "line split does not sum to the counted lines")
    check(n["invoices"] <= n["lines_all"], "invoice count exceeds line count")
    check(prod["skus"] <= prod["any"], "distinct SKUs exceed mapped descriptions")

    # --- reasons: every unmapped description must be accounted for once ----
    rea = n["reasons"]
    rprods = rea["products"]
    check(sum(r["desc"] for r in rprods) == prod["never"],
          "product reasons do not cover the never-mapped descriptions")
    check(sum(r["lines"] for r in rprods) == prod["never_lines"],
          "product reasons do not cover the never-mapped lines")
    for axis in ("retailers", "distributors"):
        block = rea[axis]
        check(sum(r["parties"] for r in block["rows"]) == block["none"],
              f"{axis}: reasons do not cover every never-identified party")
        check(sum(r["lines"] for r in block["rows"]) == block["lines"],
              f"{axis}: reason lines do not sum to the block total")
        check(block["none"] == n[axis]["none"], f"{axis}: reason total != entity total")
        check(block["partial"] == n[axis]["partial"], f"{axis}: reason partial != entity partial")

    tokens = {
        "WINDOW": n["window"],
        "DATE": n["prepared"],
        "LINES_ALL": f"{n['lines_all']:,}",
        "DUPS": f"{n['duplicates']:,}",
        "LINES": f"{n['lines']:,}",
        "INVOICES": f"{n['invoices']:,}",
    }
    for key, e in (("DIST", dist), ("RET", ret)):
        tokens[f"{key}_TOTAL"] = f"{e['total']:,}"
        tokens[f"{key}_FULL"] = f"{e['full']:,}"
        tokens[f"{key}_PART"] = f"{e['partial']:,}"
        tokens[f"{key}_NONE"] = f"{e['none']:,}"
        tokens[f"{key}_FULL_PCT"] = pct(e["full"], e["total"])
        tokens[f"{key}_NONE_PCT"] = pct(e["none"], e["total"])
        tokens[f"{key}_W_FULL"] = width(e["full"], e["total"])
        tokens[f"{key}_W_PART"] = width(e["partial"], e["total"])
    tokens.update({
        "PROD_TOTAL": f"{prod['total']:,}",
        "PROD_FULL": f"{prod['full']:,}",
        "PROD_PART": f"{prod['partial']:,}",
        "PROD_NEVER": f"{prod['never']:,}",
        "PROD_NEVER_LINES": f"{prod['never_lines']:,}",
        "PROD_ANY": f"{prod['any']:,}",
        "PROD_ANY_PCT": pct(prod["any"], prod["total"]),
        "SKUS": f"{prod['skus']:,}",
        "LINE_MAP": f"{split['mapped']:,}",
        "LINE_MBNF": f"{split['mbnf']:,}",
        "LINE_UNMAP": f"{split['unmapped']:,}",
        "LINE_MAP_PCT": pct(split["mapped"], n["lines"]),
        "LINE_MBNF_PCT": pct(split["mbnf"], n["lines"]),
        "LINE_UNMAP_PCT": pct(split["unmapped"], n["lines"]),
        "LINE_W_MAP": width(split["mapped"], n["lines"]),
        "LINE_W_MBNF": width(split["mbnf"], n["lines"]),
        "LINE_W_UNMAP": f"{100 - round(100 * split['mapped'] / n['lines'], 3) - round(100 * split['mbnf'] / n['lines'], 3):.3f}",
    })
    for r in rprods:
        k = f"R_{r['key'].upper()}"
        tokens[f"{k}_DESC"] = f"{r['desc']:,}"
        tokens[f"{k}_LINES"] = f"{r['lines']:,}"
        tokens[f"{k}_PCT"] = pct(r["lines"], prod["never_lines"])
        if "brand_desc" in r:
            tokens[f"{k}_BRAND"] = f"{r['brand_desc']:,}"
    for axis, tag in (("retailers", "RET"), ("distributors", "DST")):
        block = rea[axis]
        tokens[f"R_{tag}_NONE"] = f"{block['none']:,}"
        tokens[f"R_{tag}_LINES"] = f"{block['lines']:,}"
        tokens[f"R_{tag}_PARTIAL"] = f"{block['partial']:,}"
        for r in block["rows"]:
            k = f"R_{tag}_{r['key'].upper()}"
            tokens[f"{k}_P"] = f"{r['parties']:,}"
            tokens[f"{k}_L"] = f"{r['lines']:,}"
    tokens["R_RET_NOPHONE_PCT"] = f"{rea['retailers']['missed_no_phone_pct']}"
    tokens["CAT_SKUS"] = f"{n['catalogue_skus']:,}"
    tokens["CAT_DATE"] = n["catalogue_extract"]
    for key, fname in FONT_FILES.items():
        path = FONTS / fname
        if not path.exists():
            fail(f"missing font {path}")
        tokens[key] = base64.b64encode(path.read_bytes()).decode("ascii")

    html = Path(args.template).read_text(encoding="utf-8")
    for key, value in tokens.items():
        html = html.replace(f"@@{key}@@", value)
    leftover = [line.strip()[:80] for line in html.splitlines() if "@@" in line]
    check(not leftover, f"unreplaced tokens: {leftover[:3]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"OK  {n['window']}")
    print(f"    {n['lines']:,} lines / {n['invoices']} invoices / "
          f"{dist['total']} distributors / {ret['total']} retailers / {prod['total']:,} products")
    print(f"    wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
