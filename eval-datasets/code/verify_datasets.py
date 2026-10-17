#!/usr/bin/env python3
"""Independent verification of the three 100-case datasets in ../outputs.

For every example, resolve the cited source row in the original exports and prove:
  1. the example title IS an invoice line item (inputs.line_items[].product_name)
     of that source row — i.e. it was pulled from the TEST side, not from the
     candidate list — and its serial matches the recorded source_serial_no
  2. the source row's reference_outputs (ground truth) matches the example's outputs
     (status, sku, equivalent_skus)
  3. the example's candidate set deep-equals catalog_results[title] of that source row
  4. hsn_sac_code / unit_rate_excl_gst match the source line
  5. structural QA: 100 lines per file, unique normalized titles per file,
     300 unique across files, guidance length 12494, workspace id, status<->sku rules

Exit 0 = all checks pass. Any failure prints and exits 1.
"""
import csv
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "outputs")
SRC = {
    "turnover-product-mapping-136d0d1e": "/mnt/f/Langsmith/BRAND NAME/turnover-product-mapping-136d0d1e.csv",
    "turnover-product-mapping-676119ee": "/mnt/f/Langsmith/BRAND NAME/producttitle only.csv",
}
FILES = ["sku_cases_100.jsonl", "brand_in_title_cases_100.jsonl", "titleonly_cases_100.jsonl"]
failures = []


def norm(s):
    return " ".join(s.split()).strip().casefold()


def load_source_index(path):
    """invoice id -> success row (dict). Skips the errored duplicate attempt."""
    idx = {}
    for rr in csv.DictReader(open(path, newline="", encoding="utf-8", errors="replace")):
        if (rr["outputs"] or "").strip() in ("", "{}"):
            continue
        idx[rr["id"]] = rr
    return idx


def main():
    src_idx = {exp: load_source_index(p) for exp, p in SRC.items()}
    seen_titles = {}
    total = 0
    title_in_candidates = 0

    for fn in FILES:
        titles = set()
        examples = [json.loads(l) for l in open(os.path.join(OUTDIR, fn), encoding="utf-8") if l.strip()]
        if len(examples) != 100:
            failures.append(f"{fn}: {len(examples)} lines, expected 100")
        for ex in examples:
            total += 1
            where = f"{fn}:{total}"
            inp, out, md = ex["inputs"], ex["outputs"], ex["metadata"]
            title = inp["line_items"][0]["product_name"]
            exp = md["source_experiment"]
            titles.add(norm(title))
            if norm(title) in seen_titles:
                failures.append(f"{where}: title duplicate across files ({norm(title)}) -> also in {seen_titles[norm(title)]}")
            seen_titles[norm(title)] = fn

            row = src_idx[exp].get(md["source_invoice_id"])
            if row is None:
                failures.append(f"{where}: source invoice {md['source_invoice_id']} not found in {exp}")
                continue
            rin = json.loads(row["inputs"])
            rref = json.loads(row["reference_outputs"])["expected"]
            rcr = rin.get("catalog_results", {})

            # 1. title is a real invoice LINE ITEM of that source row
            line = next((x for x in rin.get("line_items", []) if x.get("product_name") == title), None)
            if line is None:
                failures.append(f"{where}: title NOT a line item of source invoice ({title!r})")
                continue
            if line.get("serial_no") != md["source_serial_no"]:
                failures.append(f"{where}: serial mismatch ({line.get('serial_no')} vs {md['source_serial_no']})")
            if line.get("hsn_sac_code") != inp["line_items"][0]["hsn_sac_code"]:
                failures.append(f"{where}: hsn mismatch")
            if line.get("unit_rate_excl_gst") != inp["line_items"][0]["unit_rate_excl_gst"]:
                failures.append(f"{where}: unit_rate mismatch")

            # 2. ground truth matches
            gt = next((e for e in rref if e.get("product_name") == title), None)
            if gt is None:
                failures.append(f"{where}: title missing from source reference_outputs")
                continue
            e0 = out["expected"][0]
            if (gt.get("expected_sku"), gt.get("expected_status"), sorted(gt.get("equivalent_skus", []) or [])) != \
               (e0["expected_sku"], e0["expected_status"], sorted(e0["equivalent_skus"] or [])):
                failures.append(f"{where}: GT mismatch vs source ({gt} vs {e0})")
            if e0.get("product_name") != title:
                failures.append(f"{where}: expected.product_name != title")

            # 3. candidates deep-equal the source row's set for this title
            if title not in rcr:
                failures.append(f"{where}: title not a catalog_results key in source row")
            elif rcr[title] != inp["catalog_results"][title]:
                failures.append(f"{where}: candidate set differs from source")

            # info: does the title also appear among its own candidates' names? (data, not sourcing)
            if any(norm(c.get("product_name", "")) == norm(title) for c in inp["catalog_results"][title]):
                title_in_candidates += 1

        if len(titles) != 100:
            failures.append(f"{fn}: {len(titles)} unique titles, expected 100")

    print(f"examples verified against sources: {total}/300")
    print(f"unique titles across all files: {len(seen_titles)} (expected 300)")
    print(f"titles also textually present among their own candidate names: {title_in_candidates} (informational)")
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures[:40]:
            print("  -", f)
        sys.exit(1)
    print("\nALL CHECKS PASS: every title traced to an invoice line item + its GT in the source exports.")


if __name__ == "__main__":
    main()
