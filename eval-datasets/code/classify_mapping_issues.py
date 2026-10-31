#!/usr/bin/env python3
"""Classify the BirlaNu "Mapping issue list shared by amit" (2026-10-29).

Three sheets come in: distributors not mapped, retailers not mapped, SKU/material
descriptions not mapped.  This script buckets all three against the catalogue export
and writes a review workbook.

SKU bucketing is component-based, not fuzzy-string based: family (CPVC/UPVC/SWR/...),
type (PIPE/ELBOW/TEE/...), diameter sizes (mm, inch and mixed fractions) and variant
qualifiers (SCH/SDR/CLASS/kg, Type A/B, Pushfit/Solfit, S/S-D/S, length, colour,
volume/capacity).  A description is matched only inside its family+type cell.  That
keeps a tee from matching an elbow just because the strings are close.

Buckets
  A_exact_title      description == a catalogue name after case/space/quote/plural/
                     brand-suffix/word-order normalisation, exactly one SKU
  A_exact_dup        same, but the catalogue carries the name twice (two SKUs)
  B_unique_sku       same family/type/sizes and exactly one candidate survives the
                     qualifier test -> product is in catalogue and resolvable
  C_ambiguous        same family/type/sizes but 2+ candidates survive (invoice omits
                     variant/length, or catalogue has duplicate variants)
  D_variant_conflict family/type/sizes exist but no candidate matches the stated
                     qualifier (e.g. CLASS-4 asked, only CLASS-2/3 carry that size;
                     kg-vs-CLASS shorthand also lands here)
  E_size_gap         family/type exists in catalogue but not at this size
  F_no_size          family/type exists but the description states no parsable size
  G_type_gap         type/family not present in the catalogue export
  H_other_brand      competitor / non-BirlaNu brand (Astral, Supreme, JSW, ...)
  I_unparsed         no family, type or size recoverable (paints, hardware, codes)

Caveats baked into the report:
  * The catalogue export is the 18-Oct-2026 2,335-row export; live-only 412xxx SKUs
    are out of scope.  "Not in catalogue" means "not in this export".
  * The workbook has no unit price, so C cannot be resolved by price here; the
    production mapper does have unit_rate and catalogue price (product_match_context).
  * "Inv" is taken as the number of invoices the row aggregates (the workbook does
    not define it).

Usage:
  python3 classify_mapping_issues.py \
    --workbook "/mnt/f/.../Mapping issue list shared by amit.xlsx" \
    --catalogue "/mnt/f/.../products-18-October-2026-10-52-16.csv" \
    --out "/mnt/f/.../03_review/Mapping issue analysis.xlsx"
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
import unicodedata
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# ---------------------------------------------------------------- vocabulary
BRAND_SUFFIXES = {"b", "bn", "bl", "trd"}
COMPETITORS = [
    "astral", "supreme", "jsw", "nerolac", "prince", "paras", "indiano", "kisan",
    "watertec", "water tec", "prakash", "finolex", "ashirvad", "plasto", "kankhal",
    "trimurti", "utkarsh", "flowton", "abro", "viking", "g k supreme", "adcolac",
    "hindware", "cera", "parryware", "essco", "prayag", "sanghvi", "polymax",
]
FAMILIES = [
    ("cpvc", "CPVC"), ("upvc", "UPVC"), ("swr", "SWR"), ("ugd", "UGD"),
    ("silent", "SILENT"), ("newkor", "NEWKOR"), ("agri", "AGRI"),
    ("submersible", "SUBMERSIBLE"), ("column", "SUBMERSIBLE"), ("tank", "TANK"),
    ("solvent", "SOLVENT"), ("pressure", "PRESSURE"), ("pvc", "PVC"),
]
TYPE_RULES = [
    ("multi floor trap", "TRAP"), ("nahani trap", "TRAP"), ("trap", "TRAP"),
    ("ball valve", "VALVE"), ("valve", "VALVE"), ("concealed", "VALVE"),
    ("pipe clamp", "CLAMP"), ("clamp", "CLAMP"),
    ("solvent", "SOLVENT"), ("solution", "SOLVENT"),
    ("water tank", "TANK"), ("tank", "TANK"),
    ("newkor", "SHEET"), ("sheet", "SHEET"),
    ("elbow", "ELBOW"), ("bend", "ELBOW"), ("shoe", "ELBOW"), ("bow", "ELBOW"),
    ("tee", "TEE"), ("socket", "SOCKET"), ("coupler", "COUPLER"), ("union", "UNION"),
    ("bush", "BUSH"), ("end cap", "ENDCAP"), ("cap", "ENDCAP"),
    ("tank nipple", "NIPPLE"), ("nipple", "NIPPLE"),
    ("mta", "MTA"), ("fta", "FTA"), ("plug", "PLUG"), ("pipe", "PIPE"),
    ("cowl", "COWL"), ("stainer", "STRAINER"), ("strainer", "STRAINER"),
    ("tape", "TAPE"), ("filter", "FILTER"), ("flange", "FLANGE"),
    ("saddle", "SADDLE"), ("ferrule", "FERRULE"),
]
INCH2MM_STD = {0.25: 8, 0.375: 10, 0.5: 15, 0.75: 20, 1: 25, 1.25: 32, 1.5: 40,
               2: 50, 2.5: 65, 3: 80, 4: 110, 5: 125, 6: 160, 8: 200}
INCH2MM_LARGE = {0.5: 15, 0.75: 20, 1: 25, 1.25: 32, 1.5: 40, 2: 63,
                 2.5: 75, 3: 90, 4: 110, 5: 140, 6: 160, 8: 200}


def norm_text(value) -> str:
    s = str(value)
    s = unicodedata.normalize("NFKC", s)
    for a, b in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'),
                 ("\u201d", '"'), ("\u2033", '"'), ("\u2032", "'")):
        s = s.replace(a, b)
    s = s.replace("\u00d7", "x").replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\u00bd", " 1/2 ").replace("\u00bc", " 1/4 ").replace("\u00be", " 3/4 ")
    s = s.lower()
    s = s.replace('"', " in ").replace("''", " in ")
    s = re.sub(r"\b(?:rs\.?|mrp|rate)\b", " ", s)
    s = re.sub(r"\d+(?:\.\d+)?\s*/-", " ", s)
    s = re.sub(r"[-\u2013]\s*\d+(?:\.\d+)?\s*$", " ", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*-\s*(mtrs?|mtr|mm|m|ft)\b", r"\1 \2", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*(mm|mtrs?|mtr|ml|mi|ltr|kg|pc|pcs|no)\b", r"\1 \2", s)
    s = s.replace(" mi ", " ml ")
    s = re.sub(r"(\d)m(\d{2})\b", r"\1 m \2", s)
    s = re.sub(r"(\d+)\s*mt\b", r"\1 m", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*(sdr|sch)\b", r"\1 \2", s)
    s = re.sub(r"\b(sdr|sch|class)\s*[-.]?\s*(\d+(?:\.\d+)?)", r"\1 \2", s)
    s = re.sub(r"(?<![\d/])([1-9])([1-4])/([24])(?![\d/])",
               lambda m: f"{m.group(1)} {m.group(2)}/{m.group(3)}", s)
    s = re.sub(r"\b(swr|cpvc|upvc|pvc)(pipe|pipes)\b", r"\1 pipe", s)
    s = re.sub(r"\b([a-z])\.([a-z])\.([a-z])\.([a-z])\.", r"\1\2\3\4", s)
    s = s.replace("cpvp", "cpvc")
    s = re.sub(r"\b(pipes|pipe)\b", "pipe", s)
    s = re.sub(r"\b(elbo|elbow)\b", "elbow", s)
    s = re.sub(r"\b(sofit|softi|selffit|selfit|solfit)\b", "solfit", s)
    s = re.sub(r"\b(pusfit|pufshift|push\s*fit|pushfit)\b", "pushfit", s)
    s = re.sub(r"\b(socket|soc)\b", "socket", s)
    s = s.replace("busing", "bush")
    s = re.sub(r"\b(bushing|bush)\b", "bush", s)
    s = re.sub(r"\b(a\s*type|type\s*-?\s*a)\b", "type a", s)
    s = re.sub(r"\b(b\s*type|type\s*-?\s*b)\b", "type b", s)
    s = re.sub(r"\b(s\s*/\s*s|ss)\b", "s/s", s)
    s = re.sub(r"\b(d\s*/\s*s|ds)\b", "d/s", s)
    s = re.sub(r"\bpipe\s+(40|80)\b", r"pipe sch \1", s)
    s = re.sub(r"\bc\s+(brass\s+)?(mta|fta|elbow|tee|union|socket|coupler)\b",
               r"cpvc \1\2", s)
    s = re.sub(r"\bu\s+(brass\s+)?(mta|fta|elbow|tee|union|socket|coupler)\b",
               r"upvc \1\2", s)
    s = re.sub(r"\br\s*/\s*(tee|elbow|bush|socket|coupler)\b", r"reducing \1", s)
    s = re.sub(r"(?<=\d)\s*[x*]\s*(?=\d)", " x ", s)
    s = re.sub(r"[^a-z0-9./]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def keyset(value) -> tuple[str, ...]:
    toks = norm_text(value).split()
    while toks and toks[-1] in BRAND_SUFFIXES:
        toks.pop()
    return tuple(sorted(toks))


def family(value) -> str:
    t = norm_text(value)
    if any(re.search(rf"\b{re.escape(b)}\b", t) for b in COMPETITORS):
        return "OTHER_BRAND"
    # "tank nipple" is a fitting, not a water tank
    tt = re.sub(r"\btank\s+nipple\b", "nipple", t)
    for key, fam in FAMILIES:
        if re.search(rf"\b{key}\b", tt):
            return fam
    if "brass" in tt:
        return "BRASS"
    return "UNKNOWN"


def ptype(value) -> str:
    t = re.sub(r"\btank\s+nipple\b", "nipple", norm_text(value))
    if "reducing" in t and not any(w in t for w in ("tee", "socket", "bush", "elbow")):
        return "REDUCER"
    for key, typ in TYPE_RULES:
        if re.search(rf"\b{key}\b", t):
            return typ
    return "UNKNOWN"


def size_mm(value: float, fam: str) -> int | None:
    table = INCH2MM_LARGE if fam in ("SWR", "PRESSURE", "PVC", "AGRI", "UGD") else INCH2MM_STD
    for k, mm in table.items():
        if abs(value - k) < 0.01:
            return mm
    return None


def parse_sizes(value) -> tuple[int, ...]:
    raw = re.sub(r"\(\s*300\s*mm\s*\)", " ", str(value), flags=re.I)
    t = norm_text(raw)
    fam, typ = family(raw), ptype(raw)
    t = t.replace("inch", "in")
    t = re.sub(r"(?<=[\d/])\s*x\s*(?=[\d/])", " _X_ ", t)
    t = re.sub(r"\s+x\s*(?=[\d/])", " _X_ ", t)
    t = re.sub(r"\b(?:sch|sdr|class)\s*\d+(?:\.\d+)?(?![\d/])", " ", t)
    t = re.sub(r"\btype\s*[ab]\b", " ", t)
    t = re.sub(r"\b\d+(?:\.\d+)?\s*(?:kg|pc|pcs|no|mtr|mtrs|ft)\b", " ", t)
    t = re.sub(r"\b(?:19|20)\d{2}(?:[-/]\d{2,4})?\b", " ", t)
    t = re.sub(r"\b\d{8,9}\b", " ", t)
    t = re.sub(r"\b(\d{3})m\b", r"\1 mm", t)
    t = re.sub(r"\b\d+(?:\.\d+)?\s*m\b", " ", t)
    t = re.sub(r"\b(\d+)\s*(ltr|litre|liter|l)\b", r"\1 l", t)
    toks = t.split()
    sizes: list[int] = []

    def add_in(v: float) -> None:
        mm = size_mm(v, fam)
        if mm and mm not in sizes:
            sizes.append(mm)

    i = 0
    while i < len(toks):
        tok = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        prev = toks[i - 1] if i > 0 else ""
        if re.fullmatch(r"\d+", tok) and re.fullmatch(r"\d/[24]", nxt):
            add_in(int(tok) + int(nxt.split("/")[0]) / int(nxt.split("/")[1]))
            i += 2
            continue
        m = re.fullmatch(r"(\d+)/([24])", tok)
        if m:
            add_in(int(m.group(1)) / int(m.group(2)))
            i += 1
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
            if nxt in ("in", "mm"):
                v = float(tok)
                if nxt == "in":
                    add_in(v)
                elif int(v) not in sizes:
                    sizes.append(int(v))
                i += 2
                continue
            if typ == "SOLVENT" and nxt == "ml":
                v = int(float(tok))
                if v not in sizes:
                    sizes.append(v)
                i += 2
                continue
            if typ == "TANK" and nxt == "l":
                v = int(float(tok))
                if v not in sizes:
                    sizes.append(v)
                i += 2
                continue
            if prev == "_X_" or nxt == "_X_" or re.fullmatch(r"\d/[24]", prev) \
                    or re.fullmatch(r"\d/[24]", nxt):
                v = float(tok)
                if 15 <= v <= 400 and v == int(v) and nxt == "_X_" and prev != "_X_":
                    if int(v) not in sizes:
                        sizes.append(int(v))
                else:
                    add_in(v)
        i += 1
    return tuple(sorted(sizes))


def parse_variants(value) -> frozenset[str]:
    t = norm_text(value)
    v: set[str] = set()
    for pat, tag in (
        (r"\bsch\s*(\d+)", "SCH"), (r"\bsdr\s*(\d+(?:\.\d+)?)(?![\d/])", "SDR"),
        (r"\bclass\s*-?\s*(\d)", "CLASS"), (r"\b(\d+(?:\.\d+)?)\s*kg\b", "KG"),
        (r"\btype\s*([ab])\b", "TYPE"),
    ):
        m = re.search(pat, t)
        if m:
            v.add(tag + m.group(1).upper())
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*sdr\b", t)
    if m:
        v.add("SDR" + m.group(1))
    for token, tag in (("pushfit", "PUSHFIT"), ("solfit", "SOLFIT"),
                       ("s/s", "SS"), ("d/s", "DS"), ("fab", "FAB"),
                       ("plain", "PLAIN"), ("lw", "LW"), ("isi", "ISI"),
                       ("mfg", "MFG"), ("door", "DOOR"), ("trd", "TRD")):
        if re.search(rf"\b{re.escape(token)}\b", t):
            v.add(tag)
    for c in ("white", "green", "blue", "yellow", "red", "black", "ivory",
              "ash", "grey", "gray", "pink"):
        if re.search(rf"\b{c}\b", t):
            v.add("COL:" + c.upper())
    for layer in ("3layer", "4layer", "foam"):
        if layer in t.replace(" ", ""):
            v.add("LAYER:" + layer.upper())
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:mtr|mtrs|m)\b", t)
    if m and not (len(m.group(1)) == 3 and family(value) in ("SWR", "PRESSURE", "PVC", "AGRI", "UGD")):
        v.add("LEN:" + m.group(1) + "M")
    m = re.search(r"\b(\d+)\s*(?:ft|feet)\b", t)
    if m:
        v.add("LEN:" + m.group(1) + "FT")
    if re.search(r"\(\s*300\s*mm\s*\)", t):
        v.add("LEN:300MM")
    m = re.search(r"\b(\d+)\s*ml\b", t)
    if m:
        v.add("VOL:" + m.group(1) + "ML")
    if family(value) == "TANK":
        m = re.search(r"\b(\d+)\s*l(?:tr|itre|iter)?\b", t)
        if m:
            v.add("CAP:" + m.group(1) + "L")
    if family(value) == "NEWKOR":
        m = re.search(r"\b(\d+)\s*mm\b", t)
        if m:
            v.add("THK:" + m.group(1) + "MM")
    return frozenset(v)


# ---------------------------------------------------------------- loading
def load_workbook(path: Path) -> dict[str, list[dict]]:
    wb = openpyxl.load_workbook(path, data_only=True)
    out: dict[str, list[dict]] = {}
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        hdr = [str(h).strip() if h is not None else h for h in rows[0]]
        out[ws.title] = [dict(zip(hdr, r)) for r in rows[1:]
                         if any(v not in (None, "") for v in r)]
    return out


def load_catalogue(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as fh:
        return [{"name": r["Product Name"], "sku": str(r["Sku Code"]).strip()}
                for r in csv.DictReader(fh)]


# ---------------------------------------------------------------- classify
class Classifier:
    def __init__(self, catalogue: list[dict]):
        self.cat = catalogue
        self.by_keyset: dict[tuple[str, ...], list[dict]] = collections.defaultdict(list)
        for c in catalogue:
            c["family"] = family(c["name"])
            c["type"] = ptype(c["name"])
            c["sizes"] = parse_sizes(c["name"])
            c["variants"] = parse_variants(c["name"])
            self.by_keyset[keyset(c["name"])].append(c)
        self.by_cell: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
        for c in catalogue:
            self.by_cell[(c["family"], c["type"])].append(c)

    def classify(self, desc: str) -> dict:
        if any(re.search(rf"\b{re.escape(b)}\b", str(desc).lower()) for b in COMPETITORS):
            return {"bucket": "H_other_brand", "skus": [], "candidates": [],
                    "detail": "competitor / non-BirlaNu brand"}
        key = keyset(desc)
        if key in self.by_keyset:
            hits = self.by_keyset[key]
            skus = sorted({c["sku"] for c in hits})
            bucket = "A_exact_dup" if len(skus) > 1 else "A_exact_title"
            return {"bucket": bucket, "skus": skus,
                    "candidates": [c["name"] for c in hits],
                    "detail": "exact catalogue title (normalised)" if len(skus) == 1
                              else f"exact title, catalogue carries {len(skus)} SKUs"}
        fam, typ = family(desc), ptype(desc)
        sizes, variants = parse_sizes(desc), parse_variants(desc)
        if fam == "UNKNOWN" and typ == "UNKNOWN":
            return {"bucket": "I_unparsed", "skus": [], "candidates": [],
                    "detail": "no family/type/size recovered"}
        cands = list(self.by_cell.get((fam, typ), []))
        if not cands and typ in ("COUPLER", "SOCKET") and fam in ("CPVC", "UPVC"):
            alt = "SOCKET" if typ == "COUPLER" else "COUPLER"
            cands = list(self.by_cell.get((fam, alt), []))
            if cands:
                typ = alt + "(syn)"
        if not cands:
            cands = [c for (cf, ct), rows in self.by_cell.items()
                     if ct.split("(")[0] == typ.split("(")[0] for c in rows]
        if not cands:
            return {"bucket": "G_type_gap", "skus": [], "candidates": [],
                    "detail": f"{fam}/{typ} not in catalogue export"}
        same = [c for c in cands if set(c["sizes"]) == set(sizes)]
        if same:
            compatible = [c for c in same if variants <= c["variants"]]
            if len(compatible) == 1:
                return {"bucket": "B_unique_sku", "skus": [compatible[0]["sku"]],
                        "candidates": [compatible[0]["name"]],
                        "detail": f"unique {fam}/{typ} {sizes}"}
            if len(compatible) > 1:
                return {"bucket": "C_ambiguous",
                        "skus": sorted({c["sku"] for c in compatible}),
                        "candidates": [c["name"] for c in compatible],
                        "detail": f"{len(compatible)} candidates fit {fam}/{typ} {sizes}"}
            return {"bucket": "D_variant_conflict",
                    "skus": sorted({c["sku"] for c in same}),
                    "candidates": [c["name"] for c in same],
                    "detail": f"qualifier conflict {sorted(variants)} vs "
                              f"{[sorted(c['variants']) for c in same][:3]}"}
        if not sizes:
            return {"bucket": "F_no_size", "skus": sorted({c["sku"] for c in cands}),
                    "candidates": [c["name"] for c in cands],
                    "detail": f"{fam}/{typ} but no size in description"}
        return {"bucket": "E_size_gap", "skus": sorted({c["sku"] for c in cands}),
                "candidates": [c["name"] for c in cands],
                "detail": f"{fam}/{typ} exists but not at {sizes}"}


# ---------------------------------------------------------------- workbook
HEADER_FILL = PatternFill("solid", fgColor="14402A")
HEADER_FONT = Font(color="FFFFFF", bold=True)
SUB_FILL = PatternFill("solid", fgColor="E8EEE9")


def autosize(ws, max_width=70):
    for col in ws.columns:
        width = max((len(str(c.value)) if c.value is not None else 0) for c in col)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(10, width + 2), max_width)


def write_table(ws, header, rows):
    ws.append(header)
    for c in ws[1]:
        c.fill = HEADER_FILL
        c.font = HEADER_FONT
        c.alignment = Alignment(vertical="center", wrap_text=True)
    for row in rows:
        ws.append(list(row))
    ws.freeze_panes = "A2"
    autosize(ws)


def build(args):
    data = load_workbook(Path(args.workbook))
    cat = load_catalogue(Path(args.catalogue))
    clf = Classifier(cat)
    D = data["Distributor not mapped"]
    R = data["retailer not mapped"]
    S = data["SKU not mapped"]

    # --- SKU classification (unique titles then rows)
    unique = list(dict.fromkeys(str(r["Material Description in Secondary Invoice"]) for r in S))
    cres = {d: clf.classify(d) for d in unique}
    sku_rows = []
    for r in S:
        d = str(r["Material Description in Secondary Invoice"])
        c = cres[d]
        sku_rows.append({
            "description": d,
            "distributor": r["Distributor Name"],
            "sap_code": r["Distributor Code (SAP)"],
            "invoices": r["Inv"],
            "value": float(r["Value"] or 0),
            "family": family(d),
            "type": ptype(d),
            "sizes": ", ".join(str(x) for x in parse_sizes(d)),
            **c,
        })

    def agg(bucket):
        rows = [r for r in sku_rows if r["bucket"] == bucket]
        return len(rows), sum(r["value"] for r in rows)

    total_value = sum(r["value"] for r in sku_rows)
    total_rows = len(sku_rows)
    bucket_order = ["A_exact_title", "A_exact_dup", "B_unique_sku", "C_ambiguous",
                    "D_variant_conflict", "E_size_gap", "F_no_size", "G_type_gap",
                    "H_other_brand", "I_unparsed"]
    summary_rows = []
    for b in bucket_order:
        n, v = agg(b)
        summary_rows.append([b, n, n / total_rows, v, v / total_value])
    summary_rows.append(["TOTAL", total_rows, 1.0, total_value, 1.0])

    # headline aggregates
    def bucket_sum(names):
        n = sum(agg(b)[0] for b in names)
        v = sum(agg(b)[1] for b in names)
        return n, v

    exact_n, exact_v = bucket_sum(["A_exact_title", "A_exact_dup"])
    unique_n, unique_v = bucket_sum(["A_exact_title"])
    dup_n, dup_v = bucket_sum(["A_exact_dup"])
    resolv_n, resolv_v = bucket_sum(["A_exact_title", "B_unique_sku"])
    amb_n, amb_v = bucket_sum(["C_ambiguous"])
    gap_n, gap_v = bucket_sum(["D_variant_conflict", "E_size_gap", "F_no_size",
                               "G_type_gap", "I_unparsed"])
    other_n, other_v = bucket_sum(["H_other_brand"])

    # --- duplicate catalogue names
    dup_groups = {k: v for k, v in clf.by_keyset.items() if len({c["sku"] for c in v}) > 1}
    dup_rows = []
    for key, hits in sorted(dup_groups.items(), key=lambda kv: -len(kv[1])):
        skus = sorted({c["sku"] for c in hits})
        affected = [r for r in sku_rows if r["bucket"] == "A_exact_dup"
                    and set(r["skus"]) == set(skus)]
        if not affected:
            continue
        dup_rows.append([hits[0]["name"], " | ".join(skus), len(affected),
                         round(sum(r["value"] for r in affected), 2),
                         "Retire or hide one SKU; return a single canonical SKU per name"])
    dup_rows.sort(key=lambda x: -x[3])

    # --- distributor sheet
    def gst_ok(g):
        g = str(g).strip().upper()
        return bool(re.fullmatch(r"\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]", g))

    def nname(s):
        s = str(s).upper()
        s = re.sub(r"\(.*?\)|\[.*?\]", "", s)
        s = re.sub(r"\b(?:19|20)\d{2}\s*[-/]?\s*\d{0,4}\b", "", s)
        s = re.sub(r"\bFROM\b.*$", "", s)
        s = re.sub(r"[^A-Z0-9]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    import difflib
    def variant_map(names, threshold=0.88):
        names = sorted(set(names)); out = {}; used = set()
        for i, a in enumerate(names):
            if a in used:
                continue
            grp = [a]; used.add(a)
            for b in names[i + 1:]:
                if b in used:
                    continue
                if difflib.SequenceMatcher(None, nname(a), nname(b)).ratio() >= threshold:
                    grp.append(b); used.add(b)
            if len(grp) > 1:
                for n in grp:
                    out[n] = grp
        return out

    d_variants = variant_map([r["Distributor Name"] for r in D])
    d_gst_names = collections.defaultdict(set)
    for r in D:
        g = str(r["Distributor GST"]).strip().upper()
        if g and g != "NA":
            d_gst_names[g].add(str(r["Distributor Name"]).strip())
    d_name_gsts = collections.defaultdict(set)
    for r in D:
        d_name_gsts[str(r["Distributor Name"]).strip()].add(str(r["Distributor GST"]).strip().upper())
    sku_dists = {str(r["Distributor Name"]).strip().upper() for r in S}
    ret_dists = {str(r["Distributor Name"]).strip().upper() for r in R}
    dist_rows = []
    for r in D:
        name = str(r["Distributor Name"]).strip()
        gst = str(r["Distributor GST"]).strip().upper()
        flags = []
        if gst in ("", "NA", "-"):
            flags.append("GST missing")
        elif not gst_ok(gst):
            flags.append("GST format invalid")
        if name in d_variants:
            flags.append("name variant")
        if len(d_gst_names.get(gst, set())) > 1:
            flags.append("same GST under >1 name")
        if len(d_name_gsts.get(name, set())) > 1:
            flags.append("same name under >1 GST")
        if name.upper() not in sku_dists:
            flags.append("not a seller in SKU sheet")
        if name.upper() in ret_dists:
            flags.append("seller elsewhere in retailer sheet")
        if not flags:
            flags.append("row data clean")
        # root cause + action
        if "GST missing" in flags or "GST format invalid" in flags:
            cause = "Identifier missing/invalid — cannot key the distributor"
            action = "Capture/repair GST at invoice intake; re-run entity match"
        elif "name variant" in flags or "same GST under >1 name" in flags \
                or "same name under >1 GST" in flags:
            cause = "Entity-resolution gap — name/GST variants and duplicate master records"
            action = "Canonicalise name (strip year/punctuation), keep GST as key, merge duplicate CFA records"
        elif "seller elsewhere in retailer sheet" in flags:
            cause = "Invoice-to-master link gap — the distributor trades elsewhere but this invoice is unlinked"
            action = "Re-link invoice seller to the existing CFA record; add a seller/GST consistency check"
        else:
            cause = "Distributor not present in the platform master — onboarding/coverage gap"
            action = "Onboard the CFA (GST + phone) and re-process its secondary invoices"
        dist_rows.append([name, gst, r["Invoice Number"], r["Inv"], float(r["Value"] or 0),
                          "; ".join(flags), cause, action])

    # --- retailer sheet
    r_variants = variant_map([r["Retailer Name"] for r in R], 0.92)
    r_gst_names = collections.defaultdict(set)
    for r in R:
        g = str(r["Retailer GST"]).strip().upper()
        if g and g != "NA":
            r_gst_names[g].add(str(r["Retailer Name"]).strip())
    ret_rows = []
    for r in R:
        name = str(r["Retailer Name"]).strip()
        gst = str(r["Retailer GST"]).strip().upper()
        phone = str(r["Retailer Mobile No"]).strip()
        sap = str(r["Distributor Code (SAP)"]).strip()
        flags = []
        if gst in ("", "NA", "-"):
            flags.append("no GST")
        elif not gst_ok(gst):
            flags.append("GST format invalid")
        if phone in ("", "NA", "-"):
            flags.append("no phone")
        if sap in ("", "NA", "-"):
            flags.append("no distributor SAP code")
        if len(r_gst_names.get(gst, set())) > 1:
            flags.append("same GST under >1 name")
        if name in r_variants:
            flags.append("name variant")
        strength = ("no identifier" if gst in ("", "NA", "-") and phone in ("", "NA", "-")
                    else "GST only" if phone in ("", "NA", "-")
                    else "phone only" if gst in ("", "NA", "-")
                    else "GST + phone")
        if strength == "no identifier":
            cause = "No GST and no phone — the retailer cannot be keyed"
            action = "Capture a GST or mobile at invoice intake; otherwise exclude from the mapping KPI"
        elif "name variant" in flags or "same GST under >1 name" in flags:
            cause = "Entity-resolution gap — name variants on the same GST"
            action = "Canonicalise retailer name; match on GST first, then phone"
        elif strength == "GST only":
            cause = "Retailer not in the customer master (or match is phone-only); GST present"
            action = "Enable GST-based customer match (AP-14) and bulk-onboard the GST list"
        elif strength == "phone only":
            cause = "Retailer not in the customer master; only a phone is available"
            action = "Match on phone and onboard the retailer; add GST capture"
        else:
            cause = "Retailer has both identifiers but is not in the master — coverage gap"
            action = "Bulk-onboard the retailer (GST + phone); check duplicate customer records"
        ret_rows.append([name, gst, phone, sap, r["Distributor Name"],
                         r["Invoice Number"], r["Inv"], float(r["Value"] or 0),
                         strength, "; ".join(flags) if flags else "row data clean",
                         cause, action])

    # --- write workbook
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()

    ws = wb.active
    ws.title = "Read me"
    readme = [
        ["Mapping issue list shared by amit — analysis", ""],
        ["Source workbook", str(args.workbook)],
        ["Catalogue export", str(args.catalogue)],
        ["Created", "2026-10-31"],
        ["", ""],
        ["Purpose", "Bucket the three issue sheets, verify against the catalogue, state root cause and action."],
        ["Scope", "Catalogue = 2,335-row export (18-Oct-2026). Live-only 412xxx SKUs are out of scope."],
        ["", ""],
        ["Sheets", ""],
        ["Summary", "SKU bucket counts and values; sheet-level profile."],
        ["SKU buckets", "All 3,216 SKU rows with bucket, catalogue candidates and reason."],
        ["SKU titles", "2,913 unique material descriptions, same classification."],
        ["Duplicate SKU pairs", "Catalogue names carried by two SKUs (old/new code) and the rows they block."],
        ["Distributor issues", "69 rows with data-defect flags and root cause."],
        ["Retailer issues", "517 rows with identifier strength and data-defect flags."],
        ["", ""],
        ["Bucket key", ""],
        ["A_exact_title", "Exact catalogue title (case/space/quote/plural/brand-suffix/word-order normalised), one SKU."],
        ["A_exact_dup", "Exact title but the catalogue has two SKUs for the same name."],
        ["B_unique_sku", "Same family/type/size and one candidate survives qualifiers: in catalogue, resolvable."],
        ["C_ambiguous", "Same family/type/size but 2+ SKUs fit; invoice omits variant/length or catalogue duplicates."],
        ["D_variant_conflict", "Family/type/size exists, stated qualifier (CLASS/SDR/colour/kg) matches no candidate."],
        ["E_size_gap", "Family/type exists but not at this size."],
        ["F_no_size", "Family/type exists but the description states no parsable size."],
        ["G_type_gap", "Type/family not present in the catalogue export."],
        ["H_other_brand", "Competitor / non-BirlaNu brand; should not map to the BirlaNu catalogue."],
        ["I_unparsed", "No family/type/size recoverable (paints, hardware, coded items)."],
        ["", ""],
        ["Caveats", "The workbook has no unit price; C cannot be resolved by price in this analysis."],
        ["", "Inv is read as the number of invoices a row aggregates (not defined by the workbook)."],
        ["", "Classification is deterministic but parser-limited; bucket counts are a review aid, not a verdict."],
    ]
    for row in readme:
        ws.append(row)
    ws["A1"].font = Font(bold=True, size=14)
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 120
    for row in ws.iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")

    ws = wb.create_sheet("Summary")
    write_table(ws, ["Bucket", "Rows", "Row share", "Value", "Value share"], summary_rows)
    for r in range(2, ws.max_row + 1):
        ws.cell(r, 3).number_format = "0.0%"
        ws.cell(r, 5).number_format = "0.0%"
        ws.cell(r, 4).number_format = "#,##0"
    # headline block
    start = ws.max_row + 3
    ws.cell(start, 1, "Headline").font = Font(bold=True, size=12)
    headline = [
        ["In catalogue — exact title (incl. duplicate-SKU titles)", exact_n, exact_n / total_rows,
         exact_v, exact_v / total_value],
        ["  exact title, one SKU", unique_n, unique_n / total_rows,
         unique_v, unique_v / total_value],
        ["  exact title, catalogue carries two SKUs", dup_n, dup_n / total_rows,
         dup_v, dup_v / total_value],
        ["In catalogue — one SKU after normalisation (B)", bucket_sum(["B_unique_sku"])[0],
         bucket_sum(["B_unique_sku"])[0] / total_rows, bucket_sum(["B_unique_sku"])[1],
         bucket_sum(["B_unique_sku"])[1] / total_value],
        ["Uniquely resolvable today (exact one SKU + B)", resolv_n, resolv_n / total_rows,
         resolv_v, resolv_v / total_value],
        ["In catalogue — 2+ SKUs fit, variant/length not stated", amb_n, amb_n / total_rows,
         amb_v, amb_v / total_value],
        ["Catalogue gap / qualifier conflict / unparsed", gap_n, gap_n / total_rows,
         gap_v, gap_v / total_value],
        ["Competitor / non-BirlaNu brand", other_n, other_n / total_rows,
         other_v, other_v / total_value],
    ]
    for i, row in enumerate(headline, start + 1):
        for j, val in enumerate(row, 1):
            ws.cell(i, j, val)
        ws.cell(i, 3).number_format = "0.0%"
        ws.cell(i, 5).number_format = "0.0%"
        ws.cell(i, 4).number_format = "#,##0"
    # sheet-level profile
    start = ws.max_row + 3
    ws.cell(start, 1, "Issue sheet profile").font = Font(bold=True, size=12)
    profile = [
        ["Sheet", "Rows", "Unique entity", "Invoices (Inv sum)", "Value"],
        ["Distributor not mapped", len(D), len({r["Distributor Name"] for r in D}),
         sum(float(r["Inv"] or 0) for r in D), sum(float(r["Value"] or 0) for r in D)],
        ["Retailer not mapped", len(R), len({r["Retailer Name"] for r in R}),
         sum(float(r["Inv"] or 0) for r in R), sum(float(r["Value"] or 0) for r in R)],
        ["SKU not mapped", len(S), len(unique),
         sum(float(r["Inv"] or 0) for r in S), total_value],
    ]
    for i, row in enumerate(profile, start + 1):
        for j, val in enumerate(row, 1):
            ws.cell(i, j, val)
            if i == start + 1:
                ws.cell(i, j).fill = HEADER_FILL
                ws.cell(i, j).font = HEADER_FONT
        ws.cell(i, 4).number_format = "#,##0"
        ws.cell(i, 5).number_format = "#,##0"

    ws = wb.create_sheet("SKU buckets")
    write_table(ws, ["Material description", "Distributor", "Distributor SAP", "Inv", "Value",
                     "Family", "Type", "Sizes (mm)", "Bucket", "Catalogue SKU(s)",
                     "Candidate count", "Reason"],
                [[r["description"], r["distributor"], r["sap_code"], r["invoices"], r["value"],
                  r["family"], r["type"], r["sizes"], r["bucket"], ", ".join(r["skus"]),
                  len(r["skus"]), r["detail"]] for r in sku_rows])
    for r in range(2, ws.max_row + 1):
        ws.cell(r, 5).number_format = "#,##0"

    ws = wb.create_sheet("SKU titles")
    title_rows = []
    for d in unique:
        c = cres[d]
        rows = [r for r in sku_rows if r["description"] == d]
        title_rows.append([d, len(rows), sum(r["value"] for r in rows),
                           family(d), ptype(d), ", ".join(str(x) for x in parse_sizes(d)),
                           c["bucket"], ", ".join(c["skus"]), len(c["skus"]), c["detail"]])
    title_rows.sort(key=lambda x: (x[6], -x[2]))
    write_table(ws, ["Material description", "Rows", "Value", "Family", "Type", "Sizes (mm)",
                     "Bucket", "Catalogue SKU(s)", "Candidate count", "Reason"], title_rows)
    for r in range(2, ws.max_row + 1):
        ws.cell(r, 3).number_format = "#,##0"

    ws = wb.create_sheet("Duplicate SKU pairs")
    write_table(ws, ["Catalogue name", "SKUs", "Issue rows", "Value", "Action"], dup_rows)
    for r in range(2, ws.max_row + 1):
        ws.cell(r, 4).number_format = "#,##0"

    ws = wb.create_sheet("Distributor issues")
    write_table(ws, ["Distributor name", "GST", "Invoice", "Inv", "Value", "Flags",
                     "Root cause", "Action"], dist_rows)
    for r in range(2, ws.max_row + 1):
        ws.cell(r, 5).number_format = "#,##0"

    ws = wb.create_sheet("Retailer issues")
    write_table(ws, ["Retailer name", "GST", "Mobile", "Distributor SAP", "Distributor",
                     "Invoice", "Inv", "Value", "Identifier strength", "Flags",
                     "Root cause", "Action"], ret_rows)
    for r in range(2, ws.max_row + 1):
        ws.cell(r, 8).number_format = "#,##0"

    ws = wb.create_sheet("Evidence")
    ev = [
        ["Evidence", "Detail"],
        ["Trace run", "trace-01a15db1-… (Vaibhav Enterprises, invoice GST-492, 21-Oct-2026)"],
        ["Trace: distributor catalog", "fetch_distributor_catalog returned [] for all 18 line items; fetch_catalog_text fallback returned candidates and 17/18 mapped."],
        ["Trace: master record", "fetch_cfa returned two seller records for Vaibhav (ids 366a06fa… code 0001121653 and ff09993d… code 1121653) — duplicate CFA master record."],
        ["Trace: workbook overlap", "Of the 18 trace lines, only the two MAPPED_BRAND_NOT_FOUND descriptions appear in the workbook; the 15 MAPPED lines do not."],
        ["Historical corpus", "554 of 3,216 issue rows (21.7% of value) appear verbatim in the human mapping corpus (3,746 rows); every one carries a SKU still present in the current catalogue."],
        ["Exact-title share", f"{agg('A_exact_title')[0] + agg('A_exact_dup')[0]} rows ({(agg('A_exact_title')[0]+agg('A_exact_dup')[0])/total_rows:.1%}) and {agg('A_exact_title')[1]+agg('A_exact_dup')[1]:,.0f} value are exact catalogue titles; {agg('A_exact_dup')[0]} of those rows are blocked only by duplicate SKUs."],
        ["Catalogue duplicates", f"{len(dup_groups)} names carry two SKUs in the export."],
        ["Caveat", "No unit price in the workbook; variant ambiguity cannot be settled by price here."],
    ]
    for row in ev:
        ws.append(row)
    for c in ws[1]:
        c.fill = HEADER_FILL; c.font = HEADER_FONT
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 130
    for row in ws.iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")

    wb.save(out)
    print(f"wrote {out}")

    # console summary
    print("\nSKU buckets")
    for b, n, share, v, vshare in summary_rows:
        print(f"  {b:20s} {n:5d} {share:6.1%} {v:14,.0f} {vshare:6.1%}")
    print("\nSheet profile")
    for row in profile[1:]:
        print(" ", row)
    return clf, sku_rows, cres, summary_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
