#!/usr/bin/env python3
"""Why did the unmapped ones stay unmapped? Evidence pass over the Turn Over Report.

Reuses the counting rules of turnover_entity_counts.py / turnover_product_counts.py so the
populations reconcile with the report: dedup = drop rows whose Validation Status contains
"Duplicate Invoice"; a product is the printed title under the classifier's norm_text; a party
is keyed GST -> SAP/phone -> name.

Reasons are assigned, not guessed:

  products   matched against the catalogue extract on the invoice line's own terms —
             a shared product word plus the title's size numbers. A catalogue name that
             carries the same words and the same sizes means the product IS in the
             catalogue and the title is written differently; a shared word with no size
             in common means the family exists but not that pack; no shared word at all
             means the catalogue does not carry the product.
  parties    read off the identifier the invoice carried: a valid 15-character GSTIN that
             did not resolve, a PAN or other short string in the GST field, a phone with
             no GSTIN, or nothing usable. The company's own GSTIN is its own category.

Usage:
  python3 unmapped_reasons.py "<Turn Over Report.xlsx>" "<catalogue csv>" [--json] [--samples N]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.home() / "AgentWork/product-mapping/eval-datasets/code"))
from classify_mapping_issues import norm_text  # noqa: E402

MAPPED = {"Mapped", "Mapped - Branding Not Found"}
GST_RE = re.compile(r"^[0-9A-Z]{15}$")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
OWN_NAMES = ("birlanu", "birla nu", "hil limited", "hindustan insecticides")

# words that never identify a product on their own
STOP = set("""pcs pc nos no nr na n a mm cm mtr mtrs mtr. m ft in inch kg kgs gm gms gr ltr lt ml
ltrs type class plain fab solfit d s heavy light medium isi new old white black grey gray blue
green red yellow brown silver golden gold steel brass iron pvc cpvc upvc swr sdr sch series
branded unbranded with without and or of for the birla birlanu hil birla nu""".split())

# generic words that appear in catalogue names but never pin a product family
GENERIC = set("""kit set combo part parts item pack piece pieces metre meter length assorted
misc other misc. suitable use application""".split())

CHARGE = ("freight", "discount", "round off", "coolie", "labour", "hamali", "transport",
          "packing", "forwarding", "octroi", "cartage", "loading", "unloading", "scheme",
          "damage", "shortage", "less", "rebate", "total", "sub total", "subtotal")
CHARGE_RE = re.compile(r"\b(" + "|".join(re.escape(c) for c in CHARGE) + r")\b")


def is_duplicate(value) -> bool:
    return "duplicate invoice" in str(value or "").lower()


def scalar(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    s = str(value).strip()
    return s[:-2] if s.endswith(".0") else s


def digits(value) -> str:
    return re.sub(r"\D", "", scalar(value))


def name_norm(value) -> str:
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", str(value or "").upper())
    return re.sub(r"\s+", " ", s).strip()


def numbers(text: str) -> set[str]:
    out = set()
    text = re.sub(r"(\d+)\.(\d+)/(\d+)", r"\2/\3", text)   # "1.1/2 in" is a 1 1/2 in
    for raw in NUM_RE.findall(text):
        try:
            out.add(str(float(raw)).rstrip("0").rstrip(".") if "." in raw else str(int(raw)))
        except ValueError:
            pass
    return out


def words(text: str) -> set[str]:
    return {t for t in text.split() if t.isalpha() and t not in STOP and len(t) > 1}


def load_catalogue(path: Path):
    """Name -> sku, plus token / size dictionaries for reason matching."""
    names: list[tuple[str, str]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            nm = norm_text(row.get("Product Name") or "")
            sku = str(row.get("Sku Code") or "").strip()
            if nm and sku:
                names.append((nm, sku))
    index: dict[str, list[tuple[set[str], set[str], str]]] = defaultdict(list)
    for nm, sku in names:
        for tok in words(nm) - GENERIC:
            index[tok].append((words(nm), numbers(nm), sku))
    return names, index


BRANDS = {
    "WATERTEC", "ASTRAL", "PRINCE", "FINOLEX", "SUPREME", "ASHIRVAD", "TATA", "HERO",
    "SPARKLE", "CALISTA", "COZI", "PRAKASH", "BHARTI", "WELDRITE", "FAVICOL", "FEVICOL",
    "SGE", "SUPERGUARD", "HITECH", "VICTOR", "JAQUAR", "HINDWARE", "PARRYWARE", "CERA",
    "SOMANY", "KAJARIA", "BERGER", "NEROLAC", "PIDILITE", "ULTRATECH", "AMBUJA", "POLYCAB",
    "HAVELLS", "CROMPTON", "BAJAJ", "VGUARD", "BOSCH", "MAKITA", "STANLEY", "GODREJ",
    "DORSET", "EBCO", "HETTICH", "HAFELE", "ANCHOR", "LEGRAND", "SCHNEIDER",
}


def classify_product(title: str, index, cat_names: set[str]):
    """Return (reason, evidence) for one never-mapped title.

    Every label is literally what the check established against the catalogue extract:
    a shared product word and the title's own sizes, not a judgement about the world.
    """
    toks = title.split()
    if not toks or title in {"na", "n a", "-", "none", "null"} or CHARGE_RE.search(title):
        return "not a product line", ""
    if title in cat_names:
        return "worded exactly as the catalogue", ""
    upper = {t.upper() for t in toks}
    brand = next((b for b in BRANDS if b in upper), "")
    if brand:
        return "another manufacturer's brand", brand
    tw = words(title) - GENERIC
    tn = numbers(title)
    best = None
    for tok in tw:
        for ew, en, sku in index.get(tok, ()):
            score = (len(tw & ew), len(tn & en))
            if best is None or score > best[0]:
                best = (score, ew, en, sku)
    if best:
        _score, _ew, en, sku = best
        if tn and tn <= en:
            return "catalogue product, title written differently", sku
        if not tn:
            return "catalogue family, title carries no size", sku
        return "catalogue family, this size or grade not listed", sku
    return "not in the catalogue", ""


def party_reason(name: str, gst: str, phone: str, sap: str) -> str:
    nm = name.lower()
    if any(o in nm for o in OWN_NAMES):
        return "company's own GSTIN, not a trade account"
    gst = (gst or "").upper().replace(" ", "")
    if GST_RE.match(gst):
        return "GSTIN carried, no matching account"
    if len(gst) == 10:
        return "PAN in the GST field, not a GSTIN"
    if phone:
        return "no GSTIN, phone only"
    if sap:
        return "SAP code only, no GSTIN"
    return "no usable identifier on the invoice"


def analyze(report: Path, catalogue: Path):
    cat_names, index = load_catalogue(catalogue)
    cat_set = {n for n, _ in cat_names}

    import openpyxl
    wb = openpyxl.load_workbook(report, read_only=True, data_only=True)
    try:
        ws = wb["Secondary Invoice Report"] if "Secondary Invoice Report" in wb.sheetnames else wb[wb.sheetnames[0]]
        it = ws.iter_rows(values_only=True)
        cols = None
        for _ in range(10):
            row = next(it, None)
            if row is None:
                break
            cells = [str(c).strip() if c is not None else "" for c in row]
            if cells[0] == "Invoice Number":
                cols = cells
                break
        if cols is None:
            raise SystemExit("header not found")
        idx = {c: i for i, c in enumerate(cols)}

        prods: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        parties: dict[str, dict[str, dict]] = {"dist": {}, "ret": {}}

        for row in it:
            if row[idx["Invoice Number"]] is None:
                continue
            if is_duplicate(row[idx["Validation Status"]]):
                continue
            title = norm_text(row[idx["Material Description in Secondary Invoice"]] or "")
            e = prods[title]
            e[0] += 1
            e[1] += int(str(row[idx["Parsing Status"]] or "").strip() in MAPPED)

            for axis, gstc, namec, phonec, sapc, statc in (
                ("dist", "Distributor GST", "Distributor Name", "Distributor Mobile No", "Distributor Code (SAP)", "Distributor Status"),
                ("ret", "Retailer GST", "Retailer Name", "Retailer Mobile No", "Retailer Code (SFA)", "Retailer Status"),
            ):
                gst = str(row[idx[gstc]] or "").upper().replace(" ", "")
                phone = digits(row[idx[phonec]])[-10:]
                nm = name_norm(row[idx[namec]])
                sap = scalar(row[idx[sapc]])
                key = gst if GST_RE.match(gst) else (f"SAP:{sap}" if sap not in ("", "NA") else (f"PHONE:{phone}" if phone else f"NAME:{nm}"))
                entry = parties[axis].setdefault(
                    key, {"lines": 0, "identified": 0, "missed": 0, "missed_no_phone": 0,
                          "name": nm, "gst": gst, "phone": phone}
                )
                entry["lines"] += 1
                if str(row[idx[statc]] or "").strip() == "Identified":
                    entry["identified"] += 1
                else:
                    entry["missed"] += 1
                    entry["missed_no_phone"] += int(not phone)
    finally:
        wb.close()

    never = {t: n for t, (n, m) in prods.items() if m == 0}
    prod_reason: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    examples: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for title, lines in sorted(never.items(), key=lambda kv: -kv[1]):
        reason, ev = classify_product(title, index, cat_set)
        prod_reason[reason][0] += 1
        prod_reason[reason][1] += lines
        if len(examples[reason]) < 8:
            examples[reason].append((title, ev))

    party_out = {}
    for axis in ("ret", "dist"):
        table = parties[axis]
        zero = [(k, v) for k, v in table.items() if v["identified"] == 0]
        reasons = Counter()
        lines = Counter()
        samples = defaultdict(list)
        for _key, v in zero:
            r = party_reason(v["name"], v["gst"], v["phone"], "")
            reasons[r] += 1
            lines[r] += v["lines"]
            if len(samples[r]) < 5:
                samples[r].append(v["name"])
        partial = [(k, v) for k, v in table.items() if 0 < v["identified"] < v["lines"]]
        missed = sum(v["missed"] for _k, v in partial)
        missed_no_phone = sum(v["missed_no_phone"] for _k, v in partial)
        missed_pct = round(100 * missed_no_phone / missed) if missed else 0
        party_out[axis] = {
            "total": len(table),
            "none": len(zero),
            "partial": len(partial),
            "missed_no_phone_pct": missed_pct,
            "reasons": [{"reason": r, "parties": reasons[r], "lines": lines[r], "examples": samples[r]}
                        for r in sorted(reasons, key=lambda r: -reasons[r])],
        }

    return {
        "products": {
            "never": len(never),
            "lines": sum(never.values()),
            "reasons": [
                {"reason": r, "products": prod_reason[r][0], "lines": prod_reason[r][1], "examples": examples[r]}
                for r in sorted(prod_reason, key=lambda r: -prod_reason[r][0])
            ],
        },
        "parties": party_out,
    }


PROD_KEYS = {
    "catalogue product, title written differently": "wording",
    "not in the catalogue": "outside",
    "another manufacturer's brand": "outside",
    "catalogue family, this size or grade not listed": "size",
    "catalogue family, title carries no size": "nosize",
    "worded exactly as the catalogue": "exact",
    "not a product line": "outside",
}
PARTY_KEYS = {
    "GSTIN carried, no matching account": "gst",
    "no GSTIN, phone only": "phone",
    "no usable identifier on the invoice": "no_id",
    "company's own GSTIN, not a trade account": "own",
}


def to_numbers(out: dict) -> dict:
    """Fold the analysis into the reasons block of numbers.json (display order)."""
    prod_rows: dict[str, dict] = {}
    for r in out["products"]["reasons"]:
        key = PROD_KEYS[r["reason"]]
        row = prod_rows.setdefault(key, {"key": key, "desc": 0, "lines": 0})
        row["desc"] += r["products"]
        row["lines"] += r["lines"]
        if key == "outside" and r["reason"] == "another manufacturer's brand":
            row["brand_desc"] = r["products"]
    order = ("wording", "outside", "size", "nosize", "exact")
    products = [prod_rows[k] for k in order if k in prod_rows]

    blocks = {}
    for axis, tag in (("ret", "retailers"), ("dist", "distributors")):
        src = out["parties"][axis]
        rows = [{"key": PARTY_KEYS[r["reason"]], "parties": r["parties"], "lines": r["lines"]}
                for r in src["reasons"] if r["reason"] in PARTY_KEYS]
        by_key = {r["key"]: r for r in rows}
        blocks[tag] = {
            "none": src["none"],
            "lines": sum(r["lines"] for r in rows),
            "partial": src["partial"],
            "rows": [by_key[k] for k in ("gst", "phone", "no_id", "own") if k in by_key],
        }
    blocks["retailers"]["missed_no_phone_pct"] = out["parties"]["ret"]["missed_no_phone_pct"]
    return {
        "never_lines": out["products"]["lines"],
        "products": products,
        "retailers": blocks["retailers"],
        "distributors": blocks["distributors"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("catalogue")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--update-numbers", metavar="NUMBERS_JSON",
                    help="write the reasons block into this numbers.json")
    args = ap.parse_args(argv)
    out = analyze(Path(args.report), Path(args.catalogue))

    if args.update_numbers:
        path = Path(args.update_numbers)
        numbers = json.loads(path.read_text(encoding="utf-8"))
        block = to_numbers(out)
        numbers["products"]["never_lines"] = block["never_lines"]
        numbers["reasons"] = {"products": block["products"], "retailers": block["retailers"],
                              "distributors": block["distributors"]}
        path.write_text(json.dumps(numbers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"updated {path}")
        for r in block["products"]:
            print(f"  {r['key']:8s} {r['desc']:4d} desc {r['lines']:5d} lines")
        for tag in ("retailers", "distributors"):
            print(f"  {tag:12s} none={block[tag]['none']} partial={block[tag]['partial']} lines={block[tag]['lines']}")
        return
    if args.json:
        print(json.dumps(out, indent=2))
        return
    p = out["products"]
    print(f"PRODUCTS never mapped: {p['never']:,} descriptions / {p['lines']:,} lines")
    for r in p["reasons"]:
        print(f"  {r['products']:4d} desc {r['lines']:5d} lines  {r['reason']}")
        for t, ev in r["examples"][:3]:
            print(f"        e.g. {t[:60]}" + (f"  -> {ev}" if ev else ""))
    for axis, label in (("ret", "RETAILERS"), ("dist", "DISTRIBUTORS")):
        s = out["parties"][axis]
        print(f"\n{label}: {s['total']} unique, {s['none']} never identified, {s['partial']} partly")
        for r in s["reasons"]:
            print(f"  {r['parties']:3d} parties {r['lines']:4d} lines  {r['reason']}")


if __name__ == "__main__":
    main()
