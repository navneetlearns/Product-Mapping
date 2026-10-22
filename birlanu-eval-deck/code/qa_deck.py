#!/usr/bin/env python3
"""No-vision QA for the BirlaNu eval deck.

Checks, in order:
  1. PDF integrity  — page count and page size.
  2. Text round-trip — every slot label (or its supplied number) is present, and no
     forbidden internal tokens leaked into the PDF.
  3. Pixel bands     — light slides carry no ink in the outer 3px frame or the right
     strip; the cover is allowed to bleed.
  4. Density         — each slide's ink fraction sits in a plausible range.

Usage: python3 qa_deck.py
"""
import os, re, subprocess, sys, json
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


PDF = _arg("--pdf", os.path.join(BASE, "outputs", "BirlaNu-Mapping-Eval-Deck.pdf"))
QA = _arg("--qa", os.path.join(BASE, "outputs", "qa"))
NUM = json.load(open(_arg("--numbers", os.path.join(BASE, "code", "numbers.json")), encoding="utf-8"))
PAPER = (244, 246, 245)
FAIL = []


def fail(msg):
    FAIL.append(msg)
    print("  FAIL " + msg)


def ok(msg):
    print("  ok   " + msg)


def pdf_meta():
    info = subprocess.run(["pdfinfo", PDF], capture_output=True, text=True).stdout
    pages = re.search(r"Pages:\s+(\d+)", info)
    size = re.search(r"Page size:\s+([\d.]+ x [\d.]+) pts", info)
    if not pages or int(pages.group(1)) != 4:
        fail(f"page count: {pages.group(1) if pages else 'missing'}, expected 4")
    else:
        ok("4 pages")
    if not size or size.group(1) != "1440 x 810":
        fail(f"page size: {size.group(1) if size else 'missing'}, expected 1440 x 810 pts")
    else:
        ok("page size 1440 x 810 pts")


def pdf_text():
    text = subprocess.run(["pdftotext", "-layout", PDF, "-"], capture_output=True, text=True).stdout
    low = re.sub(r"\s+", " ", text.lower())
    for slot, value in NUM.items():
        token = slot if value in (None, "") else str(value)
        words = [w for w in re.split(r"\s+", token.lower()) if len(w) >= 3]
        missing = [w for w in words if w not in low]
        if missing:
            fail(f"text missing {slot!r} (expected {token!r}; absent words: {missing})")
    if all(v not in (None, "") for v in NUM.values()):
        ok("all supplied numbers present in the PDF text")
    else:
        ok("all empty slots render as visible placeholders")
    forbidden = ["excel", "vlookup", "power query", "workbook", "langsmith",
                 "catalogue check", "csv", "total_cost"]
    hits = [f for f in forbidden if f in low]
    if hits:
        fail("forbidden tokens in PDF: " + ", ".join(hits))
    else:
        ok("no internal/tooling tokens")
    return text


def ink_stats(path, bg=PAPER, tol=10):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    px = im.load()
    assert px is not None
    total = 0
    for y in range(0, h, 3):
        for x in range(0, w, 3):
            r, g, b = px[x, y]
            if abs(r - bg[0]) > tol or abs(g - bg[1]) > tol or abs(b - bg[2]) > tol:
                total += 1
    return total / ((h // 3 + 1) * (w // 3 + 1)), im


def band_ink(im, box, bg=PAPER, tol=10):
    x0, y0, x1, y1 = box
    px = im.load()
    assert px is not None
    n = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            r, g, b = px[x, y]
            if abs(r - bg[0]) > tol or abs(g - bg[1]) > tol or abs(b - bg[2]) > tol:
                n += 1
    return n


def pixels():
    for i in range(1, 5):
        p = os.path.join(QA, f"slide_{i:02d}.png")
        if not os.path.exists(p):
            fail(f"missing capture {p}")
            continue
        dens, im = ink_stats(p)
        w, h = im.size
        if i == 1:  # cover bleeds
            ok(f"slide 1 density {dens:.2f} (cover, full bleed)")
            continue
        edges = {
            "top": (0, 0, w, 3), "bottom": (0, h - 3, w, h),
            "left": (0, 0, 3, h), "right": (w - 3, 0, w, h),
        }
        bad = {k: band_ink(im, b) for k, b in edges.items() if band_ink(im, b)}
        if bad:
            fail(f"slide {i} ink in edge band {bad}")
        else:
            ok(f"slide {i} clean edge bands")
        strip = band_ink(im, (w - 40, 0, w, h))
        if strip:
            fail(f"slide {i} right strip ink {strip}")
        # horizontal-band occupancy: flag long empty runs inside the canvas
        from PIL import Image as _I
        g = im.convert("L").resize((w, h // 8))
        rows = []
        for yy in range(g.size[1]):
            band = g.crop((96, yy, w - 96, yy + 1))
            lo, hi = band.getextrema()
            rows.append(lo < 200)
        runs, cur = [], 0
        for r in rows:
            cur = cur + 1 if not r else 0
            runs.append(cur)
        deepest = max(runs) * 8 if runs else 0
        if deepest > 220:
            fail(f"slide {i} has a {deepest}px empty band")
        else:
            ok(f"slide {i} max empty band {deepest}px")

        if not 0.02 <= dens <= 0.40:
            fail(f"slide {i} density {dens:.2f} outside 0.02-0.40")
        else:
            ok(f"slide {i} density {dens:.2f}")


if __name__ == "__main__":
    if not os.path.exists(PDF):
        sys.exit("no PDF yet — run build_deck.py first")
    print("pdf:");     pdf_meta()
    print("text:");    text = pdf_text()
    print("pixels:");  pixels()
    print()
    if FAIL:
        print(f"QA FAILED — {len(FAIL)} problem(s)")
        sys.exit(1)
    print("QA PASSED")
