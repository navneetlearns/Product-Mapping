from PIL import Image
import glob, os, sys

QA = sys.argv[1] if len(sys.argv) > 1 else "/home/sumit/AgentWork/product-mapping/birlanu-cadence-deck/outputs/qa"

def ink_profile(f):
    im = Image.open(f).convert("RGB")
    w, h = im.size
    px = im.load()
    name = os.path.basename(f)
    dark = name != "slide_01.png"
    B = 30
    bands = []
    step = max(1, h // 50)
    # sample every `step` pixel for speed
    for i in range(B):
        y0, y1 = i * h // B, (i + 1) * h // B
        cnt = tot = 0
        for y in range(y0, y1, 3):
            for x in range(0, w, 3):
                r, g, b = px[x, y]
                lum = (r + g + b) / 3
                on = lum < 232 if dark else lum > 83
                cnt += on
                tot += 1
        bands.append(round(100 * cnt / max(tot, 1), 2))
    # edge bleed (paper slides only): ink in outer 24px
    edge = None
    if dark:
        cnt = tot = 0
        m = 24
        for y in range(0, h, 3):
            for x in range(0, w, 3):
                if x < m or x >= w - m or y < m or y >= h - m:
                    r, g, b = px[x, y]
                    if (r + g + b) / 3 < 232:
                        cnt += 1
                    tot += 1
        edge = round(100 * cnt / max(tot, 1), 3)
    return name, bands, edge

for f in sorted(glob.glob(f"{QA}/slide_*.png")):
    name, bands, edge = ink_profile(f)
    empty = [i for i, v in enumerate(bands) if v < 0.05]
    print(f"{name}: edge_bleed={edge}% empty_bands={empty}")
    print("   ", bands)
