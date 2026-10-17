#!/usr/bin/env python3
"""Upload the title-level eval datasets to LangSmith via the SDK.

WHY THIS EXISTS
    The LangSmith UI file-import runs JSON-schema inference on the example
    inputs. Our inputs contain `catalog_results` — an object whose keys are
    per-example product titles (dynamic keys, special characters like quotes
    and slashes) — which makes that step fail with
    "Couldn't modify dataset. failed to modify input schema."
    The SDK stores examples as arbitrary JSON with NO schema step, and is how
    the existing invoice-wise mapping dataset is maintained.

USAGE
    export LANGSMITH_API_KEY=lsv2_...        # key for the account/workspace you upload to
    # optional: export LANGSMITH_ENDPOINT=https://api.smith.langchain.com

    # dry run — validates files, touches nothing on the network:
    python3 upload_to_langsmith.py --all --dry-run

    # upload all three sets (names below), split "base":
    python3 upload_to_langsmith.py --all

    # or one file:
    python3 upload_to_langsmith.py --file ../outputs/sku_cases_100.jsonl \
        --dataset turnover-product-mapping-titlelevel-sku

NOTES
    - If the target dataset already has examples, the script refuses to add
      more unless --append is passed (prevents accidental duplicates on retry).
    - Chunked upload (default 50/call) with progress output; safe to re-run
      only after deleting the dataset if a run was interrupted mid-way.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "outputs"

ALL_SETS = [
    ("sku_cases_100.jsonl", "turnover-product-mapping-titlelevel-sku"),
    ("brand_in_title_cases_100.jsonl", "turnover-product-mapping-titlelevel-brand"),
    ("titleonly_cases_100.jsonl", "turnover-product-mapping-titlelevel-titleonly"),
]

DESCRIPTION = (
    "Title-level BirlaNu turnover product-mapping cases (100 per category: "
    "sku-in-title / brand-in-title / title-only). Built by eval-datasets/code/"
    "build_datasets.py from the invoice-wise experiment exports; each example "
    "is a one-line invoice, candidate set, and reference mapping. "
    "Uploaded by upload_to_langsmith.py (SDK path — no UI schema inference)."
)


def load_rows(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i}: invalid JSON: {e}")
            if not isinstance(obj, dict) or "inputs" not in obj or "outputs" not in obj:
                raise SystemExit(f"{path}:{i}: missing inputs/outputs")
            rows.append(obj)
    return rows


def to_examples(rows, split):
    examples = []
    for r in rows:
        ex = {"inputs": r["inputs"], "outputs": r["outputs"]}
        if r.get("metadata"):
            ex["metadata"] = r["metadata"]
        if split:
            ex["split"] = split
        examples.append(ex)
    return examples


def upload_one(file_name: Optional[str], dataset_name: str, args):
    if args.file:
        path = Path(args.file)
    else:
        assert file_name is not None
        path = OUTDIR / file_name
    rows = load_rows(path)
    examples = to_examples(rows, args.split)
    print(f"\n== {dataset_name}")
    print(f"   file: {path} ({len(rows)} examples)")

    if args.dry_run:
        r0 = rows[0]
        print(f"   first title: {r0['inputs']['line_items'][0]['product_name']!r}")
        print(f"   inputs keys: {sorted(r0['inputs'].keys())}")
        print(f"   outputs: {json.dumps(r0['outputs'])[:100]}...")
        print(f"   [dry-run] would create/verify dataset and upload {len(examples)} examples "
              f"(split={args.split!r}, chunk={args.chunk})")
        return

    from langsmith import Client  # imported late so --dry-run needs no SDK
    client = Client()
    try:
        ds = client.read_dataset(dataset_name=dataset_name)
        n = ds.example_count or 0
        print(f"   dataset exists: {ds.id} ({n} examples)")
        if n > 0 and not args.append:
            print("   SKIP: dataset already has examples; pass --append to add more "
                  "(or delete it in the UI first).")
            return
    except Exception:
        ds = client.create_dataset(dataset_name=dataset_name, description=DESCRIPTION)
        print(f"   created dataset: {ds.id}")

    total = 0
    for i in range(0, len(examples), args.chunk):
        batch = examples[i:i + args.chunk]
        client.create_examples(dataset_id=ds.id, examples=batch)
        total += len(batch)
        print(f"   uploaded {total}/{len(examples)}")
    print(f"   done. dataset: {dataset_name} — "
          f"https://smith.langchain.com/datasets/{ds.id}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("USAGE")[0].strip())
    ap.add_argument("--file", help="single JSONL file to upload (omit with --all)")
    ap.add_argument("--dataset", help="dataset name (required with --file)")
    ap.add_argument("--all", action="store_true", help="upload the three standard sets")
    ap.add_argument("--split", default="base", help="example split (house style: base)")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--append", action="store_true",
                    help="allow adding to an existing non-empty dataset")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.all and args.file:
        raise SystemExit("use either --all or --file, not both")
    if not args.all and not (args.file and args.dataset):
        raise SystemExit("either --all, or --file with --dataset")

    if not args.dry_run and not os.environ.get("LANGSMITH_API_KEY"):
        raise SystemExit("LANGSMITH_API_KEY is not set. "
                         "Create a key in LangSmith → Settings → API Keys.")

    if args.all:
        for file_name, dataset_name in ALL_SETS:
            upload_one(file_name, dataset_name, args)
    else:
        upload_one(None, args.dataset, args)

    print("\ndone." if not args.dry_run else "\ndry-run complete (nothing uploaded).")


if __name__ == "__main__":
    main()
