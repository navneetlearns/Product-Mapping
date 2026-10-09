from PIL import Image, ImageChops
import glob, os

BASE = "/home/sumit/AgentWork/product-mapping/birlanu-cadence-deck/outputs"

def profile(path, dark=True):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    px = im.load()
    B = 24
    bands = []
    for i in range(B):
        y0, y1 = i * h // B, (i + 1) * h // B
        cnt = tot = 0
        for y in range(y0, y1, 2):
            for x in range(0, w, 2):
                lum = sum(px[x, y]) / 3
                on = (lum < 232) if dark else (lum > 83)
                cnt += on; tot += 1
        bands.append(round(100 * cnt / max(tot, 1), 2))
    return im.size, bands

print("== PDF pages (rendered at 60dpi -> 1600x900 expected) ==")
for f in sorted(glob.glob(f"{BASE}/qa/pdf/pg-*.png")):
    size, bands = profile(f, dark=(not f.endswith("-1.png")))
    empty = [i for i, v in enumerate(bands) if v < 0.05]
    print(os.path.basename(f), size, "empty:", empty)
    print("   ", bands)

print("== screen slides vs pdf pages pixel diff ==")
for i in range(1, 6):
    a = Image.open(f"{BASE}/qa/slide_{i:02d}.png").convert("RGB")
    b = Image.open(f"{BASE}/qa/pdf/pg-{i}.png").convert("RGB").resize(a.size)
    diff = ImageChops.difference(a, b).convert("L")
    hist = diff.histogram()
    changed = sum(hist[12:]) / (a.size[0] * a.size[1]) * 100
    print(f"  page {i}: pixels differing >12/255 = {changed:.2f}%")
