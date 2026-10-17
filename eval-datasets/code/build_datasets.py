#!/usr/bin/env python3
"""Build three 100-case title-level LangSmith eval datasets from the BRAND NAME exports.

Source exports (read-only):
  A = /mnt/f/Langsmith/BRAND NAME/turnover-product-mapping-136d0d1e.csv  (71 inv / 899 lines)
  B = /mnt/f/Langsmith/BRAND NAME/producttitle only.csv                  (79 inv / 943 lines)

Categories (mutually exclusive, decided by the title text):
  sku       — title carries an embedded 8-digit SKU code            (A only; B has none)
  brand     — no code, title carries the birla/hil literal token
  titleonly — no code, no brand token (pure fuzzy name-matching)

Selection (deterministic, no RNG):
  * one example per distinct normalized title; variant = the one with the largest
    candidate set (tie broken by hash of its candidate SKUs)
  * stratified by ground-truth expected_status, quotas proportional to the pool mix
    (largest-remainder rounding to exactly 100)
  * within a stratum, order by sha1(normalized title) and take the quota

Output: ../outputs/{sku_cases_100, brand_in_title_cases_100, titleonly_cases_100}.jsonl
        ../outputs/selection_summary.csv
Each JSONL line: {"inputs": {...pipeline-shaped 1-line invoice...},
                  "outputs": {"expected": [...]}, "metadata": {...}}
"""
import collections
import csv
import hashlib
import json
import os
import re

SRC = {
    "A": "/mnt/f/Langsmith/BRAND NAME/turnover-product-mapping-136d0d1e.csv",
    "B": "/mnt/f/Langsmith/BRAND NAME/producttitle only.csv",
}
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
DIG8 = re.compile(r"\d{8}")
BRAND = re.compile(r"birla|\bhil\b")


def norm(s):
    return " ".join(s.split()).strip().casefold()


def h(s):
    return hashlib.sha1(s.encode()).hexdigest()


def load_occurrences(src_id, path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8", errors="replace")))
    occ = []
    for rr in rows:
        if (rr["outputs"] or "").strip() in ("", "{}"):
            continue  # errored attempt row (no scored output)
        inp = json.loads(rr["inputs"])
        cr = {}          # norm key -> (raw key, candidates)
        for k, v in inp.get("catalog_results", {}).items():
            cr[norm(k)] = (k, v)
        ref = json.loads(rr["reference_outputs"])["expected"] if (rr["reference_outputs"] or "").strip() else []
        refn = collections.defaultdict(list)
        for e in ref:
            refn[norm(e.get("product_name", ""))].append(e)
        for x in inp.get("line_items", []):
            raw = x.get("product_name", "")
            t = norm(raw)
            raw_key, cand = cr.get(t, (raw, []))
            gt = (refn.get(t) or [{}])[0]
            occ.append(dict(
                src=src_id, inv=rr["id"], raw=raw, title=t, key=raw_key,
                hsn=x.get("hsn_sac_code"), rate=x.get("unit_rate_excl_gst"), serial=x.get("serial_no"),
                cand=cand,
                gt_name=gt.get("product_name", raw), gt_sku=gt.get("expected_sku"),
                gt_st=gt.get("expected_status"), gt_eq=sorted(gt.get("equivalent_skus", []) or []),
                de=inp.get("distributor_exclusive"), ws=inp.get("workspace_id"),
                guidance=inp.get("product_mapping_guidance", ""),
            ))
    return occ


def bucket_of(title):
    if DIG8.search(title):
        return "sku"
    if BRAND.search(title):
        return "brand"
    return "titleonly"


def representative(variants):
    """Deterministic pick: most candidates first, then hash of its candidate SKUs."""
    def sort_key(e):
        sig = "|".join(sorted(str(c.get("sku_code") or c.get("product_name", "")) for c in e["cand"]))
        return (-len(e["cand"]), h(sig))
    return sorted(variants, key=sort_key)[0]


def quotas(mix, target=100):
    total = sum(mix.values())
    raw = {k: target * n / total for k, n in mix.items()}
    base = {k: int(v) for k, v in raw.items()}
    rem = target - sum(base.values())
    order = sorted(raw, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:rem]:
        base[k] += 1
    return base


def main():
    occ = load_occurrences("A", SRC["A"]) + load_occurrences("B", SRC["B"])

    # Guidance standardization: the source exports carry TWO prompt versions
    # (12494 chars = current, 10668 = older draft on 4 of 151 rows). Use the
    # modal (current) text for every example so the dataset runs one prompt.
    gcount = collections.Counter(hashlib.md5(e["guidance"].encode()).hexdigest() for e in occ)
    mode_md5 = gcount.most_common(1)[0][0]
    standard_g = next(e["guidance"] for e in occ
                      if hashlib.md5(e["guidance"].encode()).hexdigest() == mode_md5)
    nonstd = sum(v for k, v in gcount.items() if k != mode_md5)
    print(f"guidance: standard {mode_md5} ({len(standard_g)} chars) | nonstandard occurrences: {nonstd}")

    buckets = collections.defaultdict(list)
    for e in occ:
        buckets[bucket_of(e["title"])].append(e)

    print("=== availability (occurrences / titles) ===")
    title_occ = collections.Counter(e["title"] for e in occ)
    for name in ("sku", "brand", "titleonly"):
        es = buckets[name]
        print(f"  {name}: {len(es)} occ / {len({e['title'] for e in es})} titles")

    os.makedirs(OUTDIR, exist_ok=True)
    all_titles = set()
    chosen_all = {}

    for name in ("sku", "brand", "titleonly"):
        es = buckets[name]
        # one variant per title
        by_title = collections.defaultdict(list)
        for e in es:
            by_title[e["title"]].append(e)
        reps = [representative(v) for v in by_title.values()]

        # stratified by expected_status, proportional
        strata = collections.defaultdict(list)
        for e in reps:
            strata[e["gt_st"]].append(e)
        q = quotas({k: len(v) for k, v in strata.items()}, 100)
        chosen = []
        for st in sorted(strata):
            pool = sorted(strata[st], key=lambda e: h(e["title"]))
            if q[st] > len(pool):
                raise SystemExit(f"{name}/{st}: quota {q[st]} > pool {len(pool)}")
            chosen += pool[: q[st]]
        assert len(chosen) == 100, len(chosen)

        fname = {"sku": "sku_cases_100", "brand": "brand_in_title_cases_100",
                 "titleonly": "titleonly_cases_100"}[name]
        path = os.path.join(OUTDIR, fname + ".jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for e in sorted(chosen, key=lambda e: e["title"]):
                example = {
                    "inputs": {
                        "catalog_results": {e["key"]: e["cand"]},
                        "distributor_exclusive": e["de"],
                        "line_items": [{
                            "hsn_sac_code": e["hsn"],
                            "product_name": e["raw"],
                            "serial_no": 1,
                            "unit_rate_excl_gst": e["rate"],
                        }],
                        "product_mapping_guidance": standard_g,
                        "workspace_id": e["ws"],
                    },
                    "outputs": {"expected": [{
                        "equivalent_skus": e["gt_eq"],
                        "expected_sku": e["gt_sku"],
                        "expected_status": e["gt_st"],
                        "product_name": e["gt_name"],
                    }]},
                    "metadata": {
                        "category": name,
                        "source_experiment": "turnover-product-mapping-" + ("136d0d1e" if e["src"] == "A" else "676119ee"),
                        "source_invoice_id": e["inv"],
                        "source_serial_no": e["serial"],
                        "normalized_title": e["title"],
                        "title_occurrences_in_pool": title_occ[e["title"]],
                        "n_candidates": len(e["cand"]),
                    },
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        all_titles.update(e["title"] for e in chosen)
        chosen_all[name] = chosen

        mix = collections.Counter(e["gt_st"] for e in chosen)
        zc = sum(1 for e in chosen if not e["cand"])
        print(f"\n{name}: wrote {path}")
        print(f"   mix: {dict(mix)} | zero-candidate: {zc} | unique titles: {len({e['title'] for e in chosen})}")

    print(f"\nall-300 unique titles: {len(all_titles)} (expected 300 = disjoint buckets)")

    # summary csv
    spath = os.path.join(OUTDIR, "selection_summary.csv")
    with open(spath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "normalized_title", "expected_status", "expected_sku",
                    "source_experiment", "source_invoice_id", "source_serial_no", "n_candidates"])
        for name in ("sku", "brand", "titleonly"):
            for e in sorted(chosen_all[name], key=lambda e: e["title"]):
                w.writerow([name, e["title"], e["gt_st"], e["gt_sku"],
                            "turnover-product-mapping-" + ("136d0d1e" if e["src"] == "A" else "676119ee"),
                            e["inv"], e["serial"], len(e["cand"])])
    print("summary written:", spath)


if __name__ == "__main__":
    main()
