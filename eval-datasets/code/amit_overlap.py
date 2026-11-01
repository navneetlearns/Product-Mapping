#!/usr/bin/env python3
"""Amit's 29-Oct issue list vs the Turn Over Report: account-level overlap.

Answers two questions and nothing else:
  1. how many of the accounts the report still shows unmapped are already named in his list,
  2. of the parties his list names, how many hold up against the report (wholly unmapped),
     how many are already fully mapped, how many cannot be found in the report at all.

His sheet is invoice-level rows; this collapses them to parties (GST, else phone, else name) and
matches the report by GST first, then phone, then normalised name. Counts only — no value columns,
by request.

    python3 code/amit_overlap.py --amit "<Mapping issue list shared by amit.xlsx>" \
        --report "<Turn Over Report.xlsx>" --out-json outputs/turnover-truth/amit-overlap.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_turnover_truth as B  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = Path("/mnt/f/Product/zotok/Product Mapping/Mapping Inconsistency/01_source/Turn Over Report.xlsx")
DEFAULT_AMIT = Path("/mnt/f/Product/zotok/Product Mapping/Mapping Inconsistency/01_source/Mapping issue list shared by amit.xlsx")

SIDES = {
    "retailers": dict(key="ret_key", flag="ret_ok", gcol="ret_gst_ok", gvalid="ret_gst_valid",
                      ncol="ret_name_n", phone_col="ret_phone",
                      sheet="retailer not mapped", gst_hdr="Retailer GST", name_hdr="Retailer Name",
                      phone_hdr="Retailer Mobile No"),
    "distributors": dict(key="dist_key", flag="dist_ok", gcol="gst_ok", gvalid="gst_valid",
                         ncol="dist_name_n", phone_col=None,
                         sheet="Distributor not mapped", gst_hdr="Distributor GST", name_hdr="Distributor Name",
                         phone_hdr=None),
}


def gst(v) -> str:
    s = "" if v is None else str(v).upper().replace(" ", "").strip()
    return s if B.GST_RE.match(s) else ""


def nm(v) -> str:
    s = "" if v is None else str(v)
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", s.upper())
    return re.sub(r"\s+", " ", s).strip()


def inv(v) -> str:
    return "" if v is None else re.sub(r"\.0+$", "", str(v).strip().upper())


def phone(v) -> str:
    s = "" if v is None else re.sub(r"\D", "", str(v))
    return s[-10:] if len(s) >= 10 else ""


def read_sheet(wb, sheet: str) -> pd.DataFrame:
    it = wb[sheet].iter_rows(values_only=True)
    hdr = [str(h).strip() if h is not None else "" for h in next(it)]
    return pd.DataFrame([dict(zip(hdr, r)) for r in it if not all(v is None for v in r)])


def prep_amit(wb, spec: dict) -> pd.DataFrame:
    tab = read_sheet(wb, spec["sheet"])
    tab["_gst"] = tab[spec["gst_hdr"]].map(gst)
    tab["_nm"] = tab[spec["name_hdr"]].map(nm)
    tab["_phone"] = tab[spec["phone_hdr"]].map(phone) if spec["phone_hdr"] else ""
    # his sheet has no party id: GST where he gives one, else the normalised name (phone is too often
    # blank/"NA" to carry identity, and two shops can share a name but not a GST)
    tab["_party"] = tab["_gst"].where(tab["_gst"] != "", "NAME:" + tab["_nm"])
    return tab


def row_verdict(row, head_by_inv: dict, spec: dict) -> str:
    cand = head_by_inv.get(inv(row["Invoice Number"]))
    if cand is None:
        return "no_invoice"
    if row["_gst"]:
        m = cand[cand[spec["gcol"]] == row["_gst"]]
    elif row["_phone"]:
        m = cand[cand[spec["phone_col"]] == row["_phone"]]
    else:
        m = cand[cand[spec["ncol"]] == row["_nm"]]
    if len(m) == 0:
        return "no_party"
    got, tot = int(m[spec["flag"]].sum()), len(m)
    return "resolved" if got == tot else ("unmapped" if got == 0 else "partial")


def party_verdict(vs: list[str]) -> str:
    matched = [v for v in vs if v in ("resolved", "unmapped", "partial")]
    if not matched:
        return "not in report"
    if all(v == "unmapped" for v in matched):
        return "wholly unmapped"
    if all(v == "resolved" for v in matched):
        return "fully mapped"
    return "partly mapped"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--amit", default=str(DEFAULT_AMIT))
    ap.add_argument("--cache-dir", default=str(HERE / "outputs" / "turnover-truth"))
    ap.add_argument("--out-json", default=str(HERE / "outputs" / "turnover-truth" / "amit-overlap.json"))
    a = ap.parse_args()

    df, _ = B.load_report(Path(a.report), Path(a.cache_dir), refresh=False)
    df = B.normalise(df)
    head = df[~df["dup"]].copy()
    head["inv_n"] = head["Invoice Number"].map(inv)
    head["ret_phone"] = head["Retailer Mobile No"].map(phone)
    head_by_inv = {k: v for k, v in head.groupby("inv_n")}

    import openpyxl
    wb = openpyxl.load_workbook(a.amit, read_only=True, data_only=True)

    out: dict = {"source": {"report": str(a.report), "amit": str(a.amit)}}
    for name, spec in SIDES.items():
        per = head.groupby(spec["key"])[spec["flag"]].agg(["sum", "count"])
        gap = set(per[per["sum"] < per["count"]].index)
        full = set(per[per["sum"] == per["count"]].index)
        gmap = head[head[spec["gvalid"]]].groupby(spec["key"])[spec["gcol"]].agg(lambda x: sorted(set(x))[0]).to_dict()
        nmap = head.groupby(spec["key"])[spec["ncol"]].first().to_dict()

        tab = prep_amit(wb, spec)
        tab["_v"] = [row_verdict(r, head_by_inv, spec) for r in tab.to_dict("records")]

        his_by_gst = {g: p for g, p in zip(tab["_gst"], tab["_party"]) if g}
        his_by_name: dict[str, set[str]] = {}
        for n, p in zip(tab["_nm"], tab["_party"]):
            his_by_name.setdefault(n, set()).add(p)

        named, parties_hit = [], set()

        def match(gapset):
            """gap entities that his sheet names, by GST or normalised name."""
            hits, party = [], set()
            for k in gapset:
                ps = set()
                if gmap.get(k, "") in his_by_gst:
                    ps.add(his_by_gst[gmap[k]])
                ps |= his_by_name.get(nmap.get(k, ""), set())
                if ps:
                    hits.append(k)
                    party |= ps
            return hits, party

        named, parties_hit = match(gap)
        # convention variant: any identified invoice = mapped, so only zero-identified accounts are unmapped
        gap_strict = set(per[per["sum"] == 0].index)
        named_strict, _ = match(gap_strict)
        partial_n = int(((per["sum"] > 0) & (per["sum"] < per["count"])).sum())

        pv = tab.groupby("_party")["_v"].apply(list).map(party_verdict)
        buckets = pv.value_counts().to_dict()
        rows = tab["_v"].map({"unmapped": "still unmapped", "resolved": "already resolved",
                              "partial": "partly identified", "no_invoice": "invoice not in report",
                              "no_party": "invoice exists, party not on it"}).value_counts().to_dict()

        out[name] = {
            "report_unmapped": len(gap),
            "named_in_amit": len(named),
            "absent_from_amit": len(gap) - len(named),
            "report_unmapped_strict": len(gap_strict),
            "named_in_amit_strict": len(named_strict),
            "absent_from_amit_strict": len(gap_strict) - len(named_strict),
            "partial_counted_as_mapped": partial_n,
            "amit_parties": int(tab["_party"].nunique()),
            "amit_parties_named_by_gap": len(parties_hit),
            "amit_party_verdict": {k: int(v) for k, v in buckets.items()},
            "amit_parties_with_any_unmapped_row": int(sum(1 for v in pv if v in ("wholly unmapped", "partly mapped"))),
            "amit_rows": len(tab),
            "amit_row_verdict": {k: int(v) for k, v in rows.items()},
        }
        print(f"\n{name.upper()}: report unmapped {len(gap)} | named in his list {len(named)} | "
              f"absent {len(gap)-len(named)}")
        print(f"  strict (any identified invoice = mapped): unmapped {len(gap_strict)} | "
              f"named {len(named_strict)} | absent {len(gap_strict)-len(named_strict)} | "
              f"partial folded in {partial_n}")
        print(f"  his parties {int(tab['_party'].nunique())}: " +
              ", ".join(f"{k} {v}" for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])))
        print(f"  his rows {len(tab)}: " + ", ".join(f"{k} {v}" for k, v in rows.items()))

    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
