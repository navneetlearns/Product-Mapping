#!/usr/bin/env python3
"""Value-level QA for the BirlaNu mapping report (Playwright + PIL + pdftotext).

Proves the page against code/numbers.json rather than trusting the build log:
two A4 pages, no clipped or overflowing content, fonts actually loaded, every
headline figure on the page, zero internal/tooling terms, and a light paper
ground. Also writes the PDF twin and per-page PNGs to outputs/qa/.

Run:  python3 code/qa_report.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "outputs" / "BirlaNu-Mapping-Coverage-2026-11-08.html"
QA = ROOT / "outputs" / "qa"
PDF = ROOT / "outputs" / "BirlaNu-Mapping-Coverage-2026-11-08.pdf"
NUM = json.loads((ROOT / "code" / "numbers.json").read_text(encoding="utf-8"))

FORBIDDEN = [
    r"\blangsmith\b", r"\bturnover-product-mapping\b", r"\bfalse_map\b", r"\btotal_cost\b",
    r"\beval[- ]dashboard\b", r"\bv8\.4\.19\b", r"\bexcel\b", r"\bvlookup\b", r"\bpower query\b",
    r"\bworkbook\b", r"\bxlsx\b", r"\bopenpyxl\b", r"\bpython\b", r"\bgithub\b", r"\brepo\b",
    r"\bPR\s*#?\d+", r"₹", r"\bRs\.?\b", r"\bINR\b", r"\beval\b", r"\d{10,}",
]

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILS.append(name)


def band_profile(path: Path) -> tuple[int, list[tuple[int, int]]]:
    """Ink percentage per 40px band; returns (blank_runs, samples)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    px = img.load()
    bands = []
    step = 40
    for y0 in range(0, h, step):
        ink = total = 0
        for y in range(y0, min(y0 + step, h), 3):
            for x in range(0, w, 3):
                r, g, b = px[x, y]
                total += 1
                if r < 235 or g < 235 or b < 235:
                    ink += 1
        bands.append(round(100 * ink / total, 2))
    # blank run = consecutive bands under 0.4% ink, deeper than 260px
    run = best = 0
    for v in bands:
        run = run + 1 if v < 0.4 else 0
        best = max(best, run)
    return best * step, bands


def main() -> None:
    QA.mkdir(parents=True, exist_ok=True)
    dist, ret, prod, split = NUM["distributors"], NUM["retailers"], NUM["products"], NUM["lines_split"]

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 900, "height": 1300})
        errors: list[str] = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(HTML.as_uri(), wait_until="load")
        page.wait_for_timeout(400)
        page.evaluate("document.fonts.ready")

        # fonts must be the embedded ones, not a fallback
        fonts_ok = page.evaluate(
            """() => ['12px Inst','700 12px Inst','italic 12px Serif','12px Mono']
                       .map(f => document.fonts.check(f))"""
        )
        check("embedded fonts loaded", all(fonts_ok), str(fonts_ok))
        check("no console/page errors", not errors, "; ".join(errors[:3]))

        # page count + geometry (210x297mm at 96dpi = 794x1123px)
        dims = page.evaluate(
            """() => Array.from(document.querySelectorAll('.page')).map(p => {
                 const r = p.getBoundingClientRect();
                 return [Math.round(r.width), Math.round(r.height)]; })"""
        )
        check("three pages", len(dims) == 3, str(dims))
        check("pages are A4", all(abs(w - 794) <= 2 and abs(h - 1123) <= 2 for w, h in dims), str(dims))

        body_overflow = page.evaluate(
            """() => Array.from(document.querySelectorAll('.page')).map(p => {
                 const b = p.querySelector('.body');
                 return b.scrollHeight - b.clientHeight; })"""
        )
        check("no page content overflow", all(v <= 1 for v in body_overflow), str(body_overflow))

        # footer must not sit on top of body content
        collide = page.evaluate(
            """() => Array.from(document.querySelectorAll('.page')).map(p => {
                 const foot = p.querySelector('.foot').getBoundingClientRect();
                 const secs = Array.from(p.querySelectorAll('.body .sec, .body .glance, .body .rblock, .body .rmethod'));
                 if (!secs.length) return -999;
                 const last = secs[secs.length-1].getBoundingClientRect();
                 return Math.round(last.bottom - foot.top); })"""
        )
        check("footer clears content", all(v < 0 for v in collide), str(collide))

        clipped = page.evaluate(
            """() => Array.from(document.querySelectorAll('h1,h2,p,td,th,.lfoot span,.leg,.note .txt,.gcap,.gpct,.rname,.rnote,.rnote-line,.rmethod'))
                 .filter(el => el.scrollWidth > el.clientWidth + 2 || el.scrollHeight > el.clientHeight + 2).length"""
        )
        check("no clipped text nodes", clipped == 0, f"{clipped} clipped")

        overflow_x = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        check("no horizontal overflow", overflow_x <= 1, f"{overflow_x}px")

        text = page.inner_text("body")
        # every headline figure must be on the page
        needles = [
            f"{NUM['lines_all']:,}", f"{NUM['duplicates']:,}", f"{NUM['lines']:,}", f"{NUM['invoices']}",
            f"{dist['total']}", f"{dist['full']}", f"{ret['total']}", f"{ret['full']}",
            f"{prod['total']:,}", f"{prod['any']:,}", f"{prod['never']:,}", f"{prod['skus']}",
            f"{split['mapped']:,}", f"{split['mbnf']:,}", f"{split['unmapped']:,}",
            f"{NUM['reasons']['products'][0]['desc']}", f"{NUM['reasons']['products'][0]['lines']:,}",
            f"{NUM['reasons']['products'][1]['brand_desc']}",
            f"{NUM['reasons']['retailers']['rows'][0]['parties']}",
            f"{NUM['reasons']['distributors']['rows'][0]['lines']}",
        ]
        # every reason row must be labelled on the page
        labels = ["Catalogue product, title written differently", "Not in the catalogue extract",
                  "this size or grade not listed", "no size in the title",
                  "Exactly as the catalogue", "GSTIN carried, no matching account",
                  "No GSTIN, phone only", "Company"]
        missing_labels = [s for s in labels if s not in text]
        check("reason rows present", not missing_labels, f"missing {missing_labels}")
        missing = [s for s in needles if s not in text]
        check("headline figures present", not missing, f"missing {missing}")

        bad = [pat for pat in FORBIDDEN if re.search(pat, text, re.I)]
        check("no internal/tooling terms", not bad, f"found {bad}")

        # per-page PNGs
        for i, el in enumerate(page.query_selector_all(".page"), start=1):
            el.screenshot(path=str(QA / f"page-{i}.png"))
        page.emulate_media(media="print")
        page.pdf(path=str(PDF), format="A4", print_background=True, prefer_css_page_size=True)
        page.emulate_media(media="screen")
        browser.close()

    page_files = sorted(QA.glob("page-*.png"))
    for f in page_files:
        i = f.stem.split("-")[1]
        run_px, bands = band_profile(f)
        check(f"page {i}: no empty band", run_px <= 260, f"blank run {run_px}px")
        check(f"page {i}: ink present", max(bands) > 2.0, f"max band {max(bands)}%")
        img = Image.open(QA / f"page-{i}.png").convert("RGB")
        w, h = img.size
        corner = img.getpixel((w - 8, h - 8))
        check(f"page {i}: paper ground", min(corner) > 200, str(corner))

    info = subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True).stdout
    pages_line = next((ln for ln in info.splitlines() if ln.startswith("Pages:")), "")
    size_line = next((ln for ln in info.splitlines() if ln.startswith("Page size:")), "")
    check("PDF is three pages", "Pages:" in pages_line and pages_line.split(":")[1].strip() == "3", pages_line)
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", size_line)
    check("PDF is A4", bool(m) and abs(float(m.group(1)) - 594.96) < 1.5 and abs(float(m.group(2)) - 841.92) < 1.5,
          size_line)

    pdf_text = subprocess.run(["pdftotext", "-layout", str(PDF), "-"], capture_output=True, text=True).stdout
    bad_pdf = [pat for pat in FORBIDDEN if re.search(pat, pdf_text, re.I)]
    check("PDF text has no internal terms", not bad_pdf, f"found {bad_pdf}")
    for needle in (NUM["window"], f"{prod['any']:,}", "branding not found"):
        check(f"PDF carries {needle!r}", needle.lower() in pdf_text.lower())

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        sys.exit(1)
    print(f"all checks passed · {PDF.name} ({PDF.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
