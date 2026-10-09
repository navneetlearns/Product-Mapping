from PIL import Image
import sys

QA = "/home/sumit/AgentWork/product-mapping/birlanu-cadence-deck/outputs/qa"
GREEN = (31, 92, 61); TEAL = (94, 149, 117); CLAY = (176, 129, 44)
EMPTY = (211, 221, 215); TRACK = (231, 236, 233)

def near(p, c, tol=18):
    return abs(p[0]-c[0]) <= tol and abs(p[1]-c[1]) <= tol and abs(p[2]-c[2]) <= tol

def load(n):
    im = Image.open(f"{QA}/{n}").convert("RGB"); return im, im.load(), im.size

def count_color(px, w, h, c, box=None, tol=18):
    x0, y0, x1, y1 = box or (0, 0, w, h)
    n = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            if near(px[x, y], c, tol): n += 1
    return n

# ---------- slide 2: bars ----------
im, px, (w, h) = load("slide_02.png")
runs = []
for y in range(h):
    xs = [x for x in range(w) if near(px[x, y], GREEN, 18) or near(px[x, y], TEAL, 18)]
    if len(xs) > 150:  # long solid run => a bar row
        runs.append((y, min(xs), max(xs)-min(xs)+1))
# group contiguous rows
bars = []
for y, x0, ln in runs:
    if bars and y - bars[-1][2] <= 2:
        b = bars[-1]; bars[-1] = (b[0], min(b[1], x0), max(b[2], ln), y)
    else:
        bars.append((y, x0, ln, y))
print("slide02 bars (y0,x0,len,y1):", bars)
# track width: scan the row just under bar start for TRACK color
for (y0, x0, ln, y1) in bars:
    ty = y1 + 4
    txs = [x for x in range(w) if near(px[x, ty], TRACK, 6)]
    if txs:
        track = max(txs) - min(txs) + 1
        print(f"   bar y={y0}-{y1} fill={ln}px track={track}px ratio={100*ln/track:.1f}%")

# ---------- slide 3: matrix ----------
im, px, (w, h) = load("slide_03.png")
# matrix bbox from DOM: [96,320,1216,624]
box = (96, 320, 96+1216, 320+624)
g = count_color(px, w, h, GREEN, box, tol=30)
e = count_color(px, w, h, EMPTY, box, tol=10)
print(f"slide03 matrix: green_px={g} -> dots~{g/64:.0f} (expect 957), empty_px={e} -> dots~{e/49:.0f} (expect 1366)")

# ---------- slide 4: mini matrices ----------
im, px, (w, h) = load("slide_04.png")
g = count_color(px, w, h, GREEN, None, tol=30)
c = count_color(px, w, h, CLAY, None, tol=30)
print(f"slide04: green dots~{g/121:.0f} (expect 181), clay dots~{c/121:.0f} (expect 36)")

# ---------- cover: ruler + band ----------
im, px, (w, h) = load("slide_01.png")
# light pixels in the bottom third
lt = 0
for y in range(700, 1080):
    for x in range(0, w, 2):
        if sum(px[x, y]) / 3 > 90: lt += 1
print(f"cover bottom (y700-1080) light px sampled={lt}")
