import asyncio, json
from playwright.async_api import async_playwright
SRC = "/home/sumit/AgentWork/product-mapping/birlanu-cadence-deck/code/birlanu-cadence-deck.html"

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width":1920,"height":1080})
        await pg.goto(f"file://{SRC}")
        await pg.evaluate("document.fonts.ready")
        await pg.wait_for_timeout(700)
        out = await pg.evaluate("""() => {
          const s = document.querySelectorAll('.slide')[1];
          const sr = s.getBoundingClientRect();
          const rel = r => [Math.round(r.left - sr.left), Math.round(r.top - sr.top),
                            Math.round(r.width), Math.round(r.height)];
          const q = sel => { const n = s.querySelector(sel); return n ? rel(n.getBoundingClientRect()) : null; };
          return {bar: q('.p3 .pbar'), fill: q('.p3 .pbar span'), mini36: q('.mini36'),
                  matrix: q('.matrix'), counts: q('.rowcounts')};
        }""")
        print(json.dumps(out))
        await b.close()

asyncio.run(main())
