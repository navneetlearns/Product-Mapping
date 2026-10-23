#!/usr/bin/env python3
"""Replace brand-named titles inside the sentinel "title only" lane with non-brand titles.

Why: the title-only lane is supposed to hold product titles with no brand name, but 54 of its
100 rows carry a Birla/BIRLANU/HIL token. This proposes, per offending row, a replacement drawn
from the same invoice universe (the two BRAND NAME exports), so the lane keeps its size and its
expected-status mix.

Rules
  * replacement must contain NO birla/hil token, no known other-brand token, no promotional
    item, and no 8-digit code;
  * it must not already be used by either sentinel lane;
  * it must carry the same expected status as the row it replaces (all offending rows are MAPPED);
  * preference: same product family -> same HSN family -> same status only;
  * ties broken by sha1(title) so the pick is reproducible; each replacement used once.

Output workbook: Summary | Replacements (proposed) | Brand check (all 100) | Candidate pool
The verdict column ships EMPTY — the user verifies by hand.

Usage:
  python3 build_titleonly_fix.py --title-only <csv> --brand <csv> \
      --export <turnover-*.csv> --export <producttitle only.csv> \
      [--catalogue products-*.csv] --out <xlsx>
"""
import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

import sys

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.properties import PageSetupProperties

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_catalogue_check import clean  # mojibake/whitespace cleaner used by the eval builds

CODE = re.compile(r"\d{8,9}")
BIRLA = re.compile(r"birla|\bhil\b", re.I)
OTHER_BRANDS = ["ASTRAL", "SUPREME", "FINOLEX", "ASHIRVAD", "PRINCE", "VECTUS", "WATERTEC",
                "JAQUAR", "CERA", "HINDWARE", "PARRYWARE", "SINTEX", "KAJU", "KIRLOSKAR",
                "BONDTITE", "SOLVOBOND", "WELDRITE", "EFLO", "ELDO", "TRUBORE", "CAPTAIN",
                "PRAKASH", "NOVEX", "SONEX", "UTKARSH", "PRAYAG", "TAPLON", "PEARL", "HITECH",
                "PRG", "COCO", "EZY", "DR.FIXIT", "AQUAFIT", "NEWKOR", "PLASSON", "FLOWGUARD",
                "EUROFLEX", "ASTON", "PRATIK", "RESHMA", "GOLD", "SILVER"]
BRAND_OTHER = re.compile(r"\b(" + "|".join(OTHER_BRANDS) + r")\b", re.I)
PROMO = re.compile(r"PROMOTIONAL|MERCHANDIZE|KEY\s*RING|BALL\s*PEN|ESTIMATE\s*PAD", re.I)
SUSPECT = re.compile(r"\bBN\b|-\s*B\s*$|\bCK\b", re.I)

FAMILIES = [
    ("SOLVENT", r"SOLVENT|CEMENT|PRIMER|CLEANER"),
    ("TANK", r"\bTANK\b"),
    ("ADHESIVE", r"PUTTY|ADHESIVE|GLUE"),
    ("PIPE", r"\bPIPES?\b"),
    ("END CAP", r"END\s?CAP"),
    ("REDUCER", r"REDUCER|REDUCING|\bBUSH"),
    ("COUPLER", r"COUPLER|SOCKET|\bSOC\b"),
    ("UNION", r"\bUNION\b"),
    ("ELBOW", r"ELBOW|ELBO|\bBEND\b"),
    ("TEE", r"\bTEE\b"),
    ("VALVE", r"VALVE|\bCOCK\b"),
    ("TRAP", r"\bTRAP\b"),
    ("NIPPLE", r"NIPPLE"),
    ("ADAPTOR", r"\bFTA\b|\bMTA\b|ADAPT?OR|ADAPTER"),
    ("Y", r"SINGLE\s*Y|DOUBLE\s*Y"),
    ("CLAMP", r"CLAMP"),
    ("TAPE", r"\bTAPE\b"),
    ("SANITARY", r"CISTERN|FLUSH|BASIN|SEAT\s*COVER"),
    ("FAUCET", r"BIB|SHOWER|MIXER|FAUCET|SINK\s*COCK|ANGLE\s*VALVE"),
    ("SINK", r"\bSINK\b"),
    ("PUMP", r"\bPUMP\b"),
]

HEAD_FILL = PatternFill("solid", fgColor="14342A")
HEAD_FONT = Font(color="FFFFFF", bold=True, size=10, name="Aptos")
FLAG_FILL = PatternFill("solid", fgColor="FCE9C8")
NEW_FILL = PatternFill("solid", fgColor="E4EFE7")


def norm(s):
    return " ".join(str(s or "").split()).casefold()


def fam(title):
    for name, pat in FAMILIES:
        if re.search(pat, title, re.I):
            return name
    return "OTHER"


UNIT_TAIL = re.compile(r"^\s*(?:MM|MMX|CM|MTR|M\b|LTR|L\b|ML|KG|GM|INCH|HP|W\b|V\b|A\b)", re.I)
ALNUM_CODE = re.compile(r"\b[A-Z]{1,3}\.[A-Z0-9]{1,6}[.\-/]\d{2,}[A-Z0-9./-]*")


def codey(title):
    """Catalogue-style code tokens: 4+ digit runs not followed by a unit, plus dotted codes
    (P.CPS-154, P.KS-030). Sizes and specs (160MM, 3MTR, 1000Ltr, SDR11) do not count."""
    hits = []
    for m in re.finditer(r"(?<!\d)(\d{4,})(?!\d)", title):
        if not UNIT_TAIL.match(title[m.end():]):
            hits.append(m.group(1))
    hits += ALNUM_CODE.findall(title)
    return hits


NOISE = {"mm", "cm", "cm2", "m", "mtr", "kg", "ml", "ltr", "gm", "d", "x", "no", "size"}


def toks(s):
    raw = re.findall(r"[a-z0-9]+", str(s).lower())
    return {t for t in raw if t not in NOISE and not t.isdigit() and len(t) > 1}


def jac(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def sha1(s):
    return hashlib.sha1(s.encode()).hexdigest()


def load_sentinel(path):
    out = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            li = json.loads(r["line_item"]) if r["line_item"].strip() else {}
            out.append(dict(id=r["sentinel_item_id"], name=li.get("product_name") or "",
                            hsn=li.get("hsn_sac_code"), rate=li.get("unit_rate_excl_gst"),
                            sku=(r["sku"] or "").strip(), st=r["status"],
                            eq=json.loads(r["equivalent_skus"]) if r["equivalent_skus"].strip() else [],
                            inv=r["source_invoice"]))
    return out


def load_export(path):
    """One record per line item; expected values joined by product_name (not by position)."""
    out = []
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if (r["outputs"] or "").strip() in ("", "{}"):
            continue
        inp = json.loads(r["inputs"])
        ref = json.loads(r["reference_outputs"]) if (r["reference_outputs"] or "").strip() else {}
        exp = ref.get("expected") or []
        byname = collections.defaultdict(list)
        for e in exp:
            byname[norm(e.get("product_name"))].append(e)
        for i, li in enumerate(inp.get("line_items", [])):
            e = (byname.get(norm(li["product_name"])) or (exp[i:i + 1] if i < len(exp) else [{}]))[0]
            out.append(dict(src=Path(path).name, inv=r["id"], name=li["product_name"],
                            hsn=li.get("hsn_sac_code"), rate=li.get("unit_rate_excl_gst"),
                            serial=li.get("serial_no"), de=inp.get("distributor_exclusive"),
                            st=e.get("expected_status"), sku=e.get("expected_sku"),
                            eq=e.get("equivalent_skus") or []))
    return out


def style(ws, ncols, widths, height=30):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill, cell.font = HEAD_FILL, HEAD_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = height
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(ncols)}{ws.max_row}"
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title-only", required=True, dest="title_only")
    ap.add_argument("--brand", required=True)
    ap.add_argument("--export", action="append", required=True, dest="exports")
    ap.add_argument("--catalogue")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    B = load_sentinel(a.title_only)
    A = load_sentinel(a.brand)
    ex = [rec for p in a.exports for rec in load_export(p)]

    cat = {}
    if a.catalogue:
        for r in csv.DictReader(open(a.catalogue, newline="", encoding="utf-8")):
            cat[(r.get("Sku Code") or "").strip()] = clean(r.get("Product Name"))

    brand_rows = [r for r in B if BIRLA.search(r["name"])]
    suspect_rows = [r for r in B if not BIRLA.search(r["name"]) and SUSPECT.search(r["name"])]

    used = {norm(r["name"]) for r in A} | {norm(r["name"]) for r in B}
    pool = {}
    for x in ex:
        if CODE.search(x["name"]) or BIRLA.search(x["name"]):
            continue
        pool.setdefault(norm(x["name"]), x)
    excluded_used = [n for n in pool if n in used]
    screened_out = [x for n, x in pool.items() if n not in used
                    and (BRAND_OTHER.search(x["name"]) or PROMO.search(x["name"]))]
    cand = {n: dict(x, fam=fam(x["name"]), codey=codey(x["name"])) for n, x in pool.items()
            if n not in used and not BRAND_OTHER.search(x["name"]) and not PROMO.search(x["name"])}
    selectable = {n: c for n, c in cand.items() if not c["codey"]}

    taken = set()
    plans = []
    for r in sorted(brand_rows, key=lambda z: z["id"]):
        f, h4 = fam(r["name"]), str(r["hsn"])[:4] if r["hsn"] else ""
        open_c = [c for n, c in selectable.items() if n not in taken]

        def rank(c):
            if c["st"] == r["st"] and c["fam"] == f:
                basis = 0
            elif c["st"] == r["st"] and str(c["hsn"] or "")[:4] == h4:
                basis = 1
            elif c["st"] == r["st"]:
                basis = 2
            else:
                basis = 3
            return (basis, 0 if c["sku"] not in {p["sku"] for p in B} else 1, sha1(norm(c["name"])))

        pick = sorted(open_c, key=rank)[0]
        taken.add(norm(pick["name"]))
        pick = dict(pick, name=clean(pick["name"]))
        plans.append(dict(row=r, fam=f, pick=pick, basis=rank(pick)[0],
                          canned=cat.get(pick["sku"] or "", "")))

    # ---------------- workbook ----------------
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["item", "value"])
    n_b = len(brand_rows)
    rows = [
        ("title-only rows", len(B)),
        ("rows carrying a Birla/BIRLANU/HIL token", n_b),
        ("  of which leading token", sum(1 for r in brand_rows
                                         if re.match(r"^\s*(ck\s+)?(birla|hil)\b", r["name"], re.I))),
        ("  BIRLA", sum(1 for r in brand_rows if re.search(r"birla(?!nu)", r["name"], re.I))),
        ("  BIRLANU", sum(1 for r in brand_rows if re.search(r"birlanu", r["name"], re.I))),
        ("  HIL", sum(1 for r in brand_rows if re.search(r"\bhil\b", r["name"], re.I))),
        ("rows with a SKU/code in the title (8-9 digit or catalogue-style)",
         sum(1 for r in B if CODE.search(r["name"]) or codey(r["name"]))),
        ("expected status of those rows", ", ".join(f"{k} {v}" for k, v in
                                                    collections.Counter(r["st"] for r in brand_rows).items())),
        ("rows with a suspected brand abbreviation (BN / '- B') — not counted above",
         len(suspect_rows)),
        ("", ""),
        ("replacement universe: distinct invoice titles", len({norm(x['name']) for x in ex})),
        ("  no code, no Birla/HIL token", len(pool)),
        ("  minus titles already used by either lane", len(excluded_used)),
        ("  minus other-brand / promotional titles", len(screened_out)),
        ("  usable candidates", len(cand)),
        ("  of which carry a catalogue-style code (never auto-selected)", len(cand) - len(selectable)),
        ("  auto-selectable candidates", len(selectable)),
        ("  usable candidates with expected status MAPPED", sum(1 for c in selectable.values() if c["st"] == "MAPPED")),
        ("replacements proposed", len(plans)),
        ("  matched on same product family", sum(1 for p in plans if p["basis"] == 0)),
        ("  matched on same HSN family only", sum(1 for p in plans if p["basis"] == 1)),
        ("  matched on expected status only", sum(1 for p in plans if p["basis"] == 2)),
        ("  proposed expected SKU present in the BirlaNu catalogue", sum(
            1 for p in plans if p["pick"]["sku"] and p["pick"]["sku"] in cat)),
        ("  proposed expected SKU NOT in the catalogue", sum(
            1 for p in plans if not (p["pick"]["sku"] and p["pick"]["sku"] in cat))),
        ("  proposed SKU already used in the lane (duplicate)", sum(
            1 for p in plans if p["pick"]["sku"] and any(p["pick"]["sku"] == r["sku"] for r in B))),
        ("resulting set after replacement",
         f"{len(B)} titles; 0 carry a brand token; expected-status mix unchanged "
         f"({', '.join(f'{k} {v}' for k, v in collections.Counter(r['st'] for r in B).items())})"),
    ]
    for k, v in rows:
        ws.append([k, v])
    style(ws, 2, [58, 74])
    for row in ws.iter_rows(min_row=2, min_col=2, max_col=2):
        row[0].alignment = Alignment(horizontal="left", wrap_text=True)

    # ---- replacements ----
    ws2 = wb.create_sheet("Replacements (proposed)")
    cols = ["sentinel id", "current title (has brand)", "brand token", "token position",
            "product family", "expected status", "current sku", "proposed replacement title",
            "replacement family", "replacement status", "replacement sku",
            "catalogue name for that sku", "in BirlaNu catalogue?",
            "title vs catalogue similarity", "spec only in catalogue name (title is silent on it)",
            "replacement hsn", "match basis", "source export", "approved? (Y/N)"]
    ws2.append(cols)
    for p in plans:
        r, c = p["row"], p["pick"]
        tok = ("BIRLANU" if re.search(r"birlanu", r["name"], re.I)
               else "HIL" if re.search(r"\bhil\b", r["name"], re.I) else "BIRLA")
        pos = "leading" if re.match(r"^\s*(ck\s+)?(birla|hil)\b", r["name"], re.I) else "inside"
        ws2.append([r["id"], r["name"], tok, pos, p["fam"], r["st"], r["sku"], c["name"],
                    c["fam"], c["st"], c["sku"], p["canned"],
                    "YES" if (c["sku"] and c["sku"] in cat) else "NO",
                    round(jac(toks(c["name"]), toks(p["canned"])), 2) if p["canned"] else "",
                    " ".join(sorted(toks(p["canned"]) - toks(c["name"]))) if p["canned"] else "",
                    c["hsn"],
                    {0: "same family", 1: "same HSN", 2: "same status"}.get(p["basis"], "fallback"),
                    c["src"], ""])
        for i in range(1, len(cols) + 1):
            ws2.cell(row=ws2.max_row, column=i).fill = NEW_FILL if i >= 8 else FLAG_FILL
    style(ws2, len(cols), [12, 46, 10, 11, 12, 14, 12, 46, 12, 14, 12, 42, 14, 12, 30, 11, 13, 30, 14])
    for row in ws2.iter_rows(min_row=2):
        row[len(cols) - 1].fill = PatternFill("solid", fgColor="FFFFFF")

    # ---- lane check: all 100 rows (brand token + SKU code) ----
    ws3 = wb.create_sheet("Lane check (all 100)")
    ws3.append(["sentinel id", "product title", "brand token found", "token", "leading?",
                "suspected abbreviation", "SKU/code in title?", "code found", "expected status",
                "sku", "action"])
    for r in B:
        tok = "YES" if BIRLA.search(r["name"]) else ""
        which = ("BIRLANU" if re.search(r"birlanu", r["name"], re.I)
                 else "HIL" if re.search(r"\bhil\b", r["name"], re.I)
                 else "BIRLA" if re.search(r"birla", r["name"], re.I) else "")
        lead = "yes" if re.match(r"^\s*(ck\s+)?(birla|hil)\b", r["name"], re.I) else ""
        sus = "CHECK" if (not tok and SUSPECT.search(r["name"])) else ""
        codes = CODE.findall(r["name"]) + codey(r["name"])
        has_code = "YES" if codes else ""
        action = "replace" if tok else ("confirm" if sus else "keep")
        ws3.append([r["id"], r["name"], tok, which, lead, sus, has_code,
                    ", ".join(codes), r["st"], r["sku"], action])
        if tok:
            for i in range(1, 12):
                ws3.cell(row=ws3.max_row, column=i).fill = FLAG_FILL
        if codes:
            ws3.cell(row=ws3.max_row, column=7).fill = PatternFill("solid", fgColor="F6C9C9")
    style(ws3, 11, [12, 52, 14, 10, 9, 14, 16, 16, 14, 12, 10])

    # ---- candidate pool ----
    ws4 = wb.create_sheet("Candidate pool")
    ws4.append(["product title", "product family", "expected status", "expected sku",
                "catalogue name for sku", "in BirlaNu catalogue?", "hsn", "unit rate",
                "source export", "used in proposal?", "SKU/code in title?"])
    picked = {norm(p["pick"]["name"]) for p in plans}
    for n, c in sorted(cand.items(), key=lambda kv: (kv[1]["st"] != "MAPPED", kv[1]["fam"], kv[0])):
        incat = bool(c["sku"] and c["sku"] in cat)
        ws4.append([c["name"], c["fam"], c["st"], c["sku"], cat.get(c["sku"] or "", ""),
                    "YES" if incat else "NO", c["hsn"], c["rate"], c["src"],
                    "yes" if n in picked else "", ", ".join(c["codey"]) if c["codey"] else ""])
        if c["codey"]:
            ws4.cell(row=ws4.max_row, column=11).fill = PatternFill("solid", fgColor="F6C9C9")
        if not incat:
            ws4.cell(row=ws4.max_row, column=6).fill = PatternFill("solid", fgColor="EDEDED")
    style(ws4, 11, [50, 12, 14, 12, 42, 14, 11, 10, 30, 12, 16])

    for sheet in wb.worksheets:
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
        sheet.print_title_rows = "1:1"

    wb.save(a.out)
    print(f"brand rows {n_b} | suspects {len(suspect_rows)} | pool {len(cand)} | proposed {len(plans)}")
    print("basis:", collections.Counter(p["basis"] for p in plans))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
