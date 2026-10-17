#!/usr/bin/env python3
"""Build a product-line-wise, experiment-upload-ready payload from a downloaded
LangSmith experiment CSV (the invoice-wise exports).

For every invoice row, every product line becomes ONE experiment row carrying:
  inputs           = the full pipeline input, narrowed to that single line
                     (line_items:[line] + that title's candidates + guidance +
                      workspace) — the same shape the mapping service consumes.
  expected_outputs = that line's reference (expected_status / expected_sku /
                     equivalent_skus).
  actual_outputs   = that line's RECORDED run result (status / sku_code /
                     mapped_product_name / confidence / matched_brand).
  evaluation_scores= the five house metrics, recomputed per line with the EXACT
                     original formulas (reverse-engineered and verified against
                     the source CSV's own per-row scores on all 150 rows):
                       false_map       = 1 if expected==UNMAPPED and actual!=UNMAPPED
                       missed          = 1 if expected in {MAPPED, MBNF} and actual==UNMAPPED
                       wrong_sku       = 1 if expected_sku and actual sku both
                                         present, differ, not in equivalent_skus
                       status_accuracy = 1 if actual status == expected status
                       sku_accuracy    = 1 if status_ok and not wrong_sku
                                         (per-line share of the original
                                          (status_ok - wrong_sku)/n formula)

Outputs (into outputs/linewise/):
  <slug>_linewise_rows.jsonl     lean per-line records for review
  <slug>_linewise_review.csv     flat table (open in Excel)
  <slug>_upload_payload.json     exact POST body for /datasets/upload-experiment

Caveat carried into the payload metadata: results are the recorded outputs of
the ORIGINAL invoice-context run, re-keyed per line — not a fresh run.
"""
import argparse
import csv
import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTROOT = HERE.parent / "outputs" / "linewise"
NS = uuid.UUID("6f6a1e2c-2f43-4c9a-9f5e-2b8a11c0d7e3")  # deterministic row ids

MBNF = "MAPPED_BRAND_NOT_FOUND"


def jget(s):
    try:
        return json.loads(s) if s else None
    except Exception:
        return None


def norm(s):
    return " ".join((s or "").split()).casefold()


def score_of(cell):
    v = jget(cell)
    if isinstance(v, list) and v and isinstance(v[0], dict):
        return v[0].get("score")
    return None


def line_scores(exp, act):
    """Per-line version of the five house metrics. Returns (dict, verdict)."""
    es = exp.get("expected_status")
    as_ = (act or {}).get("status")
    ex_sku = exp.get("expected_sku")
    a_sku = (act or {}).get("sku_code")
    equiv = set(exp.get("equivalent_skus") or [])

    status_ok = as_ is not None and as_ == es
    wrong_sku = (ex_sku is not None and a_sku is not None
                 and a_sku != ex_sku and a_sku not in equiv)
    false_map = 1 if (es == "UNMAPPED" and as_ is not None and as_ != "UNMAPPED") else 0
    missed = 1 if (es in ("MAPPED", MBNF) and as_ == "UNMAPPED") else 0

    if act is None:
        verdict = "errored"
    elif false_map:
        verdict = "false_map"
    elif missed:
        verdict = "missed"
    elif wrong_sku:
        verdict = "wrong_sku"
    elif as_ == es:
        verdict = "ok"
    else:
        verdict = "tier_diff"

    scores = {
        "false_map": false_map,
        "missed": missed,
        "wrong_sku": 1 if wrong_sku else 0,
        "status_accuracy": 1 if status_ok else 0,
        "sku_accuracy": 1 if (status_ok and not wrong_sku) else 0,
    }
    return scores, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="experiment CSV downloaded from LangSmith UI")
    ap.add_argument("--slug", required=True, help="short name used in output file names")
    ap.add_argument("--dataset-name", default="turnover-product-mapping-linewise",
                    help="external dataset name used to group the uploaded experiments")
    ap.add_argument("--experiment-name", default=None,
                    help="experiment name shown in the UI (default: <slug>-linewise)")
    ap.add_argument("--base-time", default="2026-10-16T06:00:00+00:00",
                    help="synthetic start of the experiment window (source CSV has no wall-clock times)")
    ap.add_argument("--no-guidance", action="store_true",
                    help="drop product_mapping_guidance from inputs (smaller payload)")
    args = ap.parse_args()

    with open(args.file, newline="", encoding="utf-8") as f:
        src_rows = list(csv.DictReader(f))

    # Some examples ran more than once (repetitions). Keep the BEST run per example:
    # prefer a successful run with results; fall back to the first run otherwise.
    def usable(r):
        o = jget(r.get("outputs"))
        return r.get("status") == "success" and bool((o or {}).get("line_items"))

    groups = {}
    for r in src_rows:
        groups.setdefault(r["id"], []).append(r)
    chosen = []
    skipped = 0
    for rid, grp in groups.items():
        g = sorted(grp, key=lambda x: int(x.get("repetition") or 1))
        good = [r for r in g if usable(r)]
        chosen.append(good[-1] if good else g[0])
        skipped += max(0, len(g) - 1)

    base = datetime.fromisoformat(args.base_time)
    rows_out = []
    review_rows = []
    stats = {
        "src_rows": len(src_rows), "skipped_reps": skipped, "line_rows": 0,
        "align_fallback": 0, "name_mismatch": 0, "cand_missing": 0,
        "repeats_of_first": 0,
    }
    src_totals = {"false_map": 0.0, "missed": 0.0, "wrong_sku": 0.0,
                  "sum_status_ok": 0.0, "sum_sku_like": 0.0, "n": 0}
    built_totals = {"false_map": 0.0, "missed": 0.0, "wrong_sku": 0.0,
                    "sum_status_ok": 0.0, "sum_sku_like": 0.0, "n": 0}

    for r in chosen:
        inputs = jget(r.get("inputs")) or {}
        ref = (jget(r.get("reference_outputs")) or {}).get("expected") or []
        out = jget(r.get("outputs"))
        act_list = (out or {}).get("line_items") or []
        in_lines = inputs.get("line_items") or []

        # source side aggregates (for validation against our per-line totals)
        if act_list and len(act_list) == len(ref):
            st_ok = sum(1 for a, b in zip(act_list, ref) if a.get("status") == b.get("expected_status"))
            ws = 0
            for a, b in zip(act_list, ref):
                ex, ax = b.get("expected_sku"), a.get("sku_code")
                if ex is not None and ax is not None and ax != ex and ax not in set(b.get("equivalent_skus") or []):
                    ws += 1
            src_totals["false_map"] += score_of(r["false_map"]) or 0
            src_totals["missed"] += score_of(r["missed"]) or 0
            src_totals["wrong_sku"] += score_of(r["wrong_sku"]) or 0
            src_totals["sum_status_ok"] += st_ok
            src_totals["sum_sku_like"] += (st_ok - ws)
            src_totals["n"] += len(ref)

        latency_sec = float(r.get("latency") or 0) or 0
        per_line_lat = max(0.05, min(latency_sec / max(len(ref), 1), 2.0)) if latency_sec else 0.05

        for i, exp in enumerate(ref):
            act = act_list[i] if i < len(act_list) else None
            title = exp.get("product_name") or ""

            # input line item: index alignment, name-checked, with fallback search
            in_li = None
            if i < len(in_lines) and norm(in_lines[i].get("product_name")) == norm(title):
                in_li = in_lines[i]
            else:
                cands = [x for x in in_lines if norm(x.get("product_name")) == norm(title)]
                if cands:
                    in_li = cands[0]
                    stats["align_fallback"] += 1
                    stats["name_mismatch"] += 1
                elif i < len(in_lines):
                    in_li = in_lines[i]
                    stats["name_mismatch"] += 1
                else:
                    in_li = {"product_name": title}
                    stats["name_mismatch"] += 1

            # candidates for this title (exact key, then normalized, else none)
            cat = inputs.get("catalog_results") or {}
            ckey = None
            if title in cat:
                ckey = title
            else:
                for k in cat:
                    if norm(k) == norm(title):
                        ckey = k
                        break
            if ckey is None:
                stats["cand_missing"] += 1

            narrow_inputs = dict(inputs)
            narrow_inputs["line_items"] = [in_li]
            narrow_inputs["catalog_results"] = {ckey: cat[ckey]} if ckey else {}
            if args.no_guidance:
                narrow_inputs.pop("product_mapping_guidance", None)

            scores, verdict = line_scores(exp, act)
            built_totals["false_map"] += scores["false_map"]
            built_totals["missed"] += scores["missed"]
            built_totals["wrong_sku"] += scores["wrong_sku"]
            built_totals["sum_status_ok"] += scores["status_accuracy"]
            built_totals["sum_sku_like"] += scores["sku_accuracy"]
            built_totals["n"] += 1

            start = base + timedelta(seconds=len(rows_out))
            end = start + timedelta(seconds=per_line_lat)
            row_id = str(uuid.uuid5(NS, f"{r['session_name']}|{r['id']}|{i}"))

            meta = {
                "source_experiment": r["session_name"],
                "source_invoice_id": r["id"],
                "line_index": i,
                "serial_no": in_li.get("serial_no"),
                "verdict": verdict,
                "hint": "recorded result from the original invoice-context run",
            }
            evaluation = []
            for key in ["false_map", "missed", "wrong_sku", "status_accuracy", "sku_accuracy"]:
                entry = {"key": key, "score": scores[key],
                         "feedback_source": {"type": "api"}}
                if key in ("false_map", "missed", "wrong_sku") and scores[key]:
                    entry["comment"] = {"false_map": "expected UNMAPPED, but a match was returned",
                                        "missed": "expected a match, but result came back UNMAPPED",
                                        "wrong_sku": "mapped to a different SKU than expected"}[key]
                evaluation.append(entry)

            result_row = {
                "row_id": row_id,
                "inputs": narrow_inputs,
                "expected_outputs": {"expected": [exp]},
                "actual_outputs": {"errored": bool(out and out.get("errored")), "line_items": [act] if act else []},
                "evaluation_scores": evaluation,
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "run_name": "turnover-mapping (recorded)",
                "run_metadata": meta,
            }
            if act is None:
                result_row["error"] = "source run errored for this invoice; no recorded result"
            rows_out.append(result_row)

            review_rows.append({
                "source_invoice_id": r["id"], "line_index": i,
                "serial_no": in_li.get("serial_no"), "product_name": title,
                "n_candidates": len(cat.get(ckey) or []) if ckey else 0,
                "expected_status": exp.get("expected_status"), "expected_sku": exp.get("expected_sku"),
                "actual_status": (act or {}).get("status"), "actual_sku": (act or {}).get("sku_code"),
                "mapped_product_name": (act or {}).get("mapped_product_name"),
                "confidence": (act or {}).get("mapping_confidence_score"),
                "false_map": scores["false_map"], "missed": scores["missed"], "wrong_sku": scores["wrong_sku"],
                "status_accuracy": scores["status_accuracy"], "sku_accuracy": scores["sku_accuracy"],
                "verdict": verdict,
            })

    stats["line_rows"] = len(rows_out)

    total_sec = max(60, len(rows_out) * 2)
    payload = {
        "experiment_name": args.experiment_name or f"{args.slug}-linewise",
        "experiment_description": (
            "Product-line-wise re-key of the recorded experiment "
            f"'{src_rows[0]['session_name']}'. Rows are product lines; results are the "
            "RECORDED outputs of the original invoice-context run (not a fresh run). "
            "Scores recomputed per line with the same formulas as the source evaluators."
        ),
        "experiment_start_time": base.isoformat(),
        "experiment_end_time": (base + timedelta(seconds=total_sec)).isoformat(),
        "dataset_name": args.dataset_name,
        "dataset_description": "Product-line-wise views of turnover mapping experiments (recorded results).",
        "experiment_metadata": {
            "source_file": Path(args.file).name,
            "source_experiment": src_rows[0]["session_name"],
            "built_by": "build_linewise_payload.py",
            "caveat": "recorded results re-keyed per line; not a fresh run",
            "timestamps": "synthesized (source export carries no wall-clock run times)",
        },
        "summary_experiment_scores": [
            {"key": "false_map", "score": built_totals["false_map"],
             "comment": f"total across {built_totals['n']} product lines"},
            {"key": "missed", "score": built_totals["missed"],
             "comment": f"total across {built_totals['n']} product lines"},
            {"key": "wrong_sku", "score": built_totals["wrong_sku"],
             "comment": f"total across {built_totals['n']} product lines"},
            {"key": "status_accuracy", "score": round(built_totals["sum_status_ok"] / built_totals["n"], 4)},
            {"key": "sku_accuracy", "score": round(built_totals["sum_sku_like"] / built_totals["n"], 4)},
        ],
        "results": rows_out,
    }

    OUTROOT.mkdir(parents=True, exist_ok=True)
    rows_path = OUTROOT / f"{args.slug}_linewise_rows.jsonl"
    csv_path = OUTROOT / f"{args.slug}_linewise_review.csv"
    payload_path = OUTROOT / f"{args.slug}_upload_payload.json"

    with open(rows_path, "w", encoding="utf-8") as f:
        for row in review_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()))
        w.writeheader()
        w.writerows(review_rows)
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"source rows: {stats['src_rows']}  |  skipped non-primary reps: {stats['skipped_reps']}")
    print(f"line rows built: {stats['line_rows']}")
    print(f"alignment fallbacks: {stats['align_fallback']}  |  name mismatches: {stats['name_mismatch']}  |  missing candidates: {stats['cand_missing']}")
    print("validation (built vs source):")
    for k in ["false_map", "missed", "wrong_sku"]:
        print(f"  {k}: built_total={built_totals[k]:g} vs source_total={src_totals[k]:g}  -> {'MATCH' if built_totals[k] == src_totals[k] else 'DIFF'}")
    print(f"  status_acc: built={built_totals['sum_status_ok']/built_totals['n']:.4f} vs source={src_totals['sum_status_ok']/src_totals['n']:.4f}")
    print(f"  sku_acc:    built={built_totals['sum_sku_like']/built_totals['n']:.4f} vs source={src_totals['sum_sku_like']/src_totals['n']:.4f}")
    print(f"files -> {rows_path}\n         {csv_path}\n         {payload_path}  ({payload_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
