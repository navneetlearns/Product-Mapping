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
          const s = document.querySelectorAll('.slide')[4];
          const sr = s.getBoundingClientRect();
          const rel = r => [Math.round(r.top - sr.top), Math.round(r.bottom - sr.top)];
          const rows = [...s.querySelectorAll('.irow')].map(r => ({
            row: rel(r.getBoundingClientRect()),
            title: rel(r.querySelector('.ititle').getBoundingClientRect()),
            desc: rel(r.querySelector('.idesc').getBoundingClientRect()),
            descH: Math.round(r.querySelector('.idesc').getBoundingClientRect().height)}));
          return {impl: rel(s.querySelector('.impl').getBoundingClientRect()), rows};
        }""")
        print(json.dumps(out, indent=1))
        await b.close()

asyncio.run(main())
