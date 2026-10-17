#!/usr/bin/env python3
"""Fire a prepared upload-experiment payload at the LangSmith API.

The payload files are built by build_linewise_payload.py
(outputs/linewise/*_upload_payload.json) and follow the schema of
POST /api/v1/datasets/upload-experiment (docs: "How to upload experiments run
outside of LangSmith with the REST API").

This endpoint is the ONLY way to make externally-produced results appear as an
EXPERIMENT in LangSmith (the UI can only create datasets). It creates (or
appends to) an externally-managed dataset that groups the uploaded experiments.

Usage:
  # offline check - no key needed, sends nothing
  python3 upload_experiment.py ../outputs/linewise/producttitle_only_upload_payload.json --dry-run

  # fire it (key is read from the environment, never stored or printed)
  LANGSMITH_API_KEY=... python3 upload_experiment.py ../outputs/linewise/producttitle_only_upload_payload.json
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payload", help="path to *_upload_payload.json")
    ap.add_argument("--dry-run", action="store_true", help="validate and print, send nothing")
    ap.add_argument("--endpoint", default=os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
                    help="API base (change for self-hosted / regional SaaS)")
    args = ap.parse_args()

    path = Path(args.payload)
    body = json.loads(path.read_text(encoding="utf-8"))
    results = body.get("results", [])

    # --- offline validation ---
    problems = []
    for k in ("experiment_name", "experiment_start_time", "experiment_end_time"):
        if not body.get(k):
            problems.append(f"top-level missing: {k}")
    if not (body.get("dataset_name") or body.get("dataset_id")):
        problems.append("need dataset_name or dataset_id")
    start, end = body.get("experiment_start_time"), body.get("experiment_end_time")
    for i, r in enumerate(results):
        for k in ("row_id", "inputs", "start_time", "end_time"):
            if r.get(k) is None:
                problems.append(f"row {i}: missing {k}")
        if r.get("start_time") and start and r["start_time"] < start:
            problems.append(f"row {i}: start_time before experiment start")
        if r.get("end_time") and end and r["end_time"] > end:
            problems.append(f"row {i}: end_time after experiment end")

    print(f"payload      : {path}")
    print(f"experiment   : {body.get('experiment_name')}")
    print(f"dataset      : {body.get('dataset_name')}")
    print(f"rows         : {len(results)}")
    print(f"size         : {path.stat().st_size / 1e6:.1f} MB")
    print(f"validation   : {'OK' if not problems else str(len(problems)) + ' PROBLEM(S)'}")
    for p in problems[:10]:
        print("   -", p)
    if problems:
        sys.exit(1)

    if args.dry_run:
        print("dry run - nothing sent")
        return

    key = os.environ.get("LANGSMITH_API_KEY")
    if not key:
        sys.exit("LANGSMITH_API_KEY is not set - refusing to send")

    url = f"{args.endpoint.rstrip('/')}/api/v1/datasets/upload-experiment"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": key},
        method="POST",
    )
    print(f"POST {url} ...")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} {e.reason}")
        print(e.read().decode("utf-8", "replace")[:3000])
        sys.exit(1)

    exp = out.get("experiment") or {}
    ds = out.get("dataset") or {}
    print("OK")
    print(f"  experiment : {exp.get('name')}  id={exp.get('id')}")
    print(f"  dataset    : {ds.get('name')}  id={ds.get('id')}  externally_managed={ds.get('externally_managed')}")
    print("  (stats may take a few seconds to populate in the UI)")


if __name__ == "__main__":
    main()
