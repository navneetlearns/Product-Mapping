from PIL import Image
import glob, os

QA = "/home/sumit/AgentWork/product-mapping/birlanu-cadence-deck/outputs/qa"
GREEN = (31, 92, 61); CLAY = (176, 129, 44); EMPTY = (211, 221, 215); TRACK = (231, 236, 233)

def near(p, c, tol=18):
    return all(abs(p[i] - c[i]) <= tol for i in range(3))

def load(n):
    im = Image.open(f"{QA}/{n}").convert("RGB")
    return im, im.load(), im.size

def count(px, w, h, c, box=None, tol=18, step=1):
    x0, y0, x1, y1 = box or (0, 0, w, h)
    n = 0
    for y in range(y0, y1, step):
        for x in range(x0, x1, step):
            if near(px[x, y], c, tol): n += 1
    return n

slides = sorted(glob.glob(f"{QA}/slide_*.png"))
print("slides:", [os.path.basename(s) for s in slides])

# ---------- slide 2: counts ----------
im, px, (w, h) = load("slide_02.png")

# August matrix: DOM box was [96, 489, 1292, 200] (tracks 1170 wide, 10px cells, 6px dots)
box = (96, 489, 96 + 1292, 489 + 200)
g = count(px, w, h, GREEN, box, tol=30)
e = count(px, w, h, EMPTY, box, tol=10)
print(f"matrix: green_px={g} -> dots~{g/36:.0f} (expect 957) | empty_px={e} -> dots~{e/36:.0f} (expect 1366)")

# boxes from code/inspect_bar.py (DOM rects relative to the slide)
BAR = (305, 905, 600, 14)      # track
MINI = (1144, 877, 330, 70)    # 36-mark matrix

# 36-issue marks (clay, 12x12) inside its own box
mx, my, mw, mh = MINI
c = count(px, w, h, CLAY, (mx, my, mx + mw, my + mh), tol=30)
print(f"clay marks: px={c} -> dots~{c/144:.1f} (expect 36)")

# approval bar: measure the green fill inside the track box
x0, y0, bw, bh = BAR
ymid = y0 + bh // 2
fill = 0
for x in range(x0, x0 + bw):
    if near(px[x, ymid], GREEN, 25): fill += 1
print(f"approval bar: fill={fill}px of {bw}px = {100*fill/bw:.2f}% (expect 7.8%)")

# ---------- slide 3: enhancements present ----------
im3, px3, (w3, h3) = load("slide_03.png")
ink = 0
for y in range(0, h3, 3):
    for x in range(0, w3, 3):
        if sum(px3[x, y]) / 3 < 232: ink += 1
print(f"slide_03 ink = {100*ink/((h3//3)*(w3//3)):.1f}%")
