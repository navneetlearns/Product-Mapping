#!/usr/bin/env python3
"""Create UI-import-friendly variants of the three datasets.

The LangSmith UI file-import infers an input schema and rejects the raw files
with "Invalid type. Expected: null, given: integer" — because some fields mix
null and real values across rows (unit_rate_excl_gst, hsn_sac_code,
metadata.source_serial_no). These variants recursively OMIT any dict field
whose value is null: for any consumer a missing field reads as null, so the
future run harness behaves identically, while schema inference gets consistent
types. Arrays are left untouched.

Canonical files are untouched; variants land in outputs/ui_upload/.
Also writes test_3rows.jsonl — a tiny starter file so a UI import can be
proven on a small scale before the full 100-row files.
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "outputs"
UIDIR = OUTDIR / "ui_upload"

FILES = ["sku_cases_100.jsonl", "brand_in_title_cases_100.jsonl", "titleonly_cases_100.jsonl"]


def strip_nulls(obj, path, counter):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if v is None:
                counter[path + "." + str(k) if path else str(k)] += 1
                continue
            out[k] = strip_nulls(v, path, counter)
        return out
    if isinstance(obj, list):
        return [strip_nulls(x, path, counter) for x in obj]
    return obj


def main():
    UIDIR.mkdir(exist_ok=True)
    tester_rows = []

    for fn in FILES:
        rows = [json.loads(l) for l in open(OUTDIR / fn, encoding="utf-8") if l.strip()]
        counter = Counter()
        out_rows = [strip_nulls(r, "", counter) for r in rows]
        out_path = UIDIR / fn.replace(".jsonl", "_ui.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total = sum(counter.values())
        print(f"{fn} -> {out_path.name}: {len(out_rows)} rows, {total} null fields omitted")
        for k, n in counter.most_common():
            print(f"    {k}: {n}")

        # tester rows: first row + first row that had a null omitted + last row
        if fn == "sku_cases_100.jsonl":
            tester_rows.append(out_rows[0])
            for r, orig in zip(out_rows, rows):
                if any(v is None for v in orig["inputs"]["line_items"][0].values()):
                    tester_rows.append(r)
                    break
            tester_rows.append(out_rows[-1])

    test_path = UIDIR / "test_3rows.jsonl"
    with open(test_path, "w", encoding="utf-8") as f:
        for r in tester_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\ntester -> {test_path.name}: {len(tester_rows)} rows "
          f"(normal / had-null / normal)")


if __name__ == "__main__":
    main()
