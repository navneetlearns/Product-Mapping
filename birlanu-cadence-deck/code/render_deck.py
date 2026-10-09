import asyncio, os, json
from playwright.async_api import async_playwright

BASE = "/home/sumit/AgentWork/product-mapping/birlanu-cadence-deck"
SRC = f"{BASE}/code/birlanu-cadence-deck.html"
OUT_PDF = f"{BASE}/outputs/BirlaNu-Cadence-Deck.pdf"
QA = f"{BASE}/outputs/qa"

async def main():
    os.makedirs(QA, exist_ok=True)
    os.makedirs(f"{BASE}/outputs", exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width":1920,"height":1080}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        await page.goto(f"file://{SRC}")
        await page.evaluate("document.fonts.ready")
        await page.wait_for_timeout(900)

        # structural checks: overflow, element bounds inside slide
        report = await page.evaluate("""() => {
          const out = {slides: document.querySelectorAll('.slide').length, overflow: [], outside: [], marks: {}};
          out.marks.matrix = document.querySelectorAll('#matrix i').length;
          out.marks.matrixOn = document.querySelectorAll('#matrix i.on').length;
          out.marks.mmA = document.querySelectorAll('#mmA i').length;
          out.marks.mmAoff = document.querySelectorAll('#mmA i.off').length;
          out.marks.mmB = document.querySelectorAll('#mmB i').length;
          document.querySelectorAll('.slide').forEach((s, si) => {
            const sr = s.getBoundingClientRect();
            if (s.scrollWidth > s.clientWidth + 1 || s.scrollHeight > s.clientHeight + 1)
              out.overflow.push({slide: si+1, sw: s.scrollWidth, cw: s.clientWidth, sh: s.scrollHeight, ch: s.clientHeight});
            s.querySelectorAll('*').forEach(el => {
              if (!el.children.length && (el.textContent||'').trim().length) {
                const r = el.getBoundingClientRect();
                if (r.right > sr.right + 1 || r.bottom > sr.bottom + 1 || r.left < sr.left - 1 || r.top < sr.top - 1) {
                  out.outside.push({slide: si+1, tag: el.tagName, cls: el.className,
                    txt: (el.textContent||'').trim().slice(0,40),
                    r: [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)],
                    s: [Math.round(sr.left), Math.round(sr.top), Math.round(sr.right), Math.round(sr.bottom)]});
                }
              }
            });
          });
          return out;
        }""")

        await page.pdf(path=OUT_PDF, print_background=True, width="1920px", height="1080px")

        slides = await page.query_selector_all(".slide")
        boxes = []
        for i, s in enumerate(slides, 1):
            await s.screenshot(path=f"{QA}/slide_{i:02d}.png")
            boxes.append(await s.evaluate("""el => {
              const pick = sel => { const n = el.querySelector(sel); if(!n) return null;
                const r = n.getBoundingClientRect(); const s = el.getBoundingClientRect();
                return [Math.round(r.left-s.left), Math.round(r.top-s.top), Math.round(r.width), Math.round(r.height)]; };
              return {h2: pick('h2'), content: pick('.content'), foot: pick('.footbar'),
                      lead: pick('.lead'), matrix: pick('.matrix'), duo: pick('.duo'), ledger: pick('.ledger')};
            }"""))
        print(json.dumps({"errors": errors, "report": report, "boxes": boxes}, indent=1))
        await browser.close()
    print("rendered", OUT_PDF)

asyncio.run(main())
