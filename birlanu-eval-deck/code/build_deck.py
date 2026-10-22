#!/usr/bin/env python3
"""Build the BirlaNu product-mapping eval deck (4 slides, 1920x1080 -> PDF).

Scope: the mapping-logic eval — four logics (SKU-based, Brand-based, Normal
product title, MRP-based), each run on its own set of product titles.

All figures come from code/numbers.json. A null slot renders as a visible
placeholder chip, so the deck can be reviewed and approved before the numbers
land. Fill numbers.json, re-run this script, and the same slides carry them.

Usage:
    python3 build_deck.py                        # build + render PDF and QA captures
    python3 build_deck.py --check                # validate slots only
    python3 build_deck.py --numbers f.json --out /tmp/x.pdf --qa /tmp/qa
"""
import asyncio, json, os, sys, html

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(BASE, "code")
SRC = os.path.join(CODE, "deck.html")


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


NUMBERS = _arg("--numbers", os.path.join(CODE, "numbers.json"))
PDF = _arg("--out", os.path.join(BASE, "outputs", "BirlaNu-Mapping-Eval-Deck.pdf"))
OUT = _arg("--qa", os.path.join(BASE, "outputs", "qa"))

with open(NUMBERS, encoding="utf-8") as f:
    NUM = json.load(f)

LANES = [("SKU-based", "ACC_SKU"), ("Brand-based", "ACC_BRAND"),
         ("Normal title", "ACC_TITLE"), ("MRP-based", "ACC_MRP")]


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def fmt(v):
    return f"{v:,.1f}" if isinstance(v, float) else f"{v:,}"


def V(slot, dark=False, cls=""):
    """Render a slot: the value when supplied, otherwise a labelled chip."""
    v = NUM.get(slot)
    if v is None or v == "":
        c = "slot" + (" on-dark" if dark else "") + (f" {cls}" if cls else "")
        return f'<span class="{c}">{slot}</span>'
    if is_num(v):
        shown = f"{v:.1f}%" if slot.startswith("ACC_") else fmt(v)
        return f"<span class=\"num{' ' + cls if cls else ''}\">{shown}</span>"
    return html.escape(str(v))


def validate():
    problems = []
    for _, slot in LANES:
        v = NUM.get(slot)
        if is_num(v) and not 0 <= v <= 100:
            problems.append(f"{slot} = {v}, expected a percentage 0-100")
    filled = [k for k, v in NUM.items() if v not in (None, "")]
    print(f"  slots filled {len(filled)}/{len(NUM)}: {', '.join(filled) or 'none'}")
    if problems:
        print("\nSLOT CHECK FAILED")
        for p in problems:
            print("  - " + p)
        sys.exit(1)
    print("  slot check OK")


def lane_bar(slot):
    v = NUM.get(slot)
    if is_num(v):
        fill = f'<span style="width:{v:.1f}%"></span>'
        val = f"{v:.1f}%"
        cls = "track"
    else:
        fill, val, cls = '<span style="width:0%"></span>', V(slot), "track pending"
    return f'<div class="{cls}">{fill}</div><div class="lv">{val}</div>'


CSS = r"""
  @font-face { font-family:'Inst'; src:url('assets/fonts/InstrumentSans-Regular.ttf'); font-weight:400; }
  @font-face { font-family:'Inst'; src:url('assets/fonts/InstrumentSans-Bold.ttf'); font-weight:700; }
  @font-face { font-family:'Serif'; src:url('assets/fonts/InstrumentSerif-Italic.ttf'); font-weight:400; font-style:italic; }
  @font-face { font-family:'Mono'; src:url('assets/fonts/JetBrainsMono-Regular.ttf'); font-weight:400; }
  @font-face { font-family:'Mono'; src:url('assets/fonts/JetBrainsMono-Bold.ttf'); font-weight:700; }

  @page { size: 1920px 1080px; margin: 0; }
  * { margin:0; padding:0; box-sizing:border-box; }
  :root{
    --paper:#f4f6f5; --ink:#111c16; --muted:#5c6f66; --hair:#dfe6e2;
    --green:#1f5c3d; --teal:#5e9575; --pine:#14402a; --clay:#b0812c; --line:#cbd6d0;
  }
  html,body{ background:var(--paper); }
  body{ font-family:'Inst',sans-serif; color:var(--ink); -webkit-font-smoothing:antialiased; }
  .num{ font-variant-numeric:tabular-nums; }

  .slide{ width:1920px; height:1080px; position:relative; overflow:hidden; background:var(--paper);
    page-break-after:always; display:flex; flex-direction:column; }
  .slide:last-child{ page-break-after:auto; }

  .content{ flex:1; position:relative; padding:84px 96px 132px; display:flex; flex-direction:column; }
  .secnum{ display:flex; align-items:center; gap:18px; font-family:'Mono'; font-size:19px; color:var(--muted); }
  .secnum .num2{ font-family:'Mono'; font-size:23px; font-weight:700; color:var(--green); letter-spacing:.14em; }
  .secnum .num2::after{ content:' /'; font-weight:400; color:var(--muted); }
  .secnum .lbl{ letter-spacing:.18em; text-transform:uppercase; }
  .secnum .rule{ flex:1; height:1px; background:var(--hair); }
  h2{ font-weight:400; font-size:74px; line-height:1.07; letter-spacing:-.015em; margin-top:22px; max-width:1640px; }
  h2 .serif{ font-family:'Serif'; font-style:italic; font-weight:400; color:var(--green); }
  .lead{ font-size:25px; color:var(--muted); margin-top:22px; line-height:1.5; max-width:1440px; }

  .footbar{ position:absolute; left:96px; right:96px; bottom:44px; display:flex; align-items:center; gap:18px;
    font-family:'Mono'; font-size:16px; color:var(--muted); letter-spacing:.08em; }
  .footbar .rule{ flex:1; height:1px; background:var(--hair); }
  .footbar .pg b{ color:var(--green); font-weight:700; }

  /* placeholder chips */
  .slot{ display:inline-block; font-family:'Mono'; font-size:.52em; letter-spacing:.08em; line-height:1.4;
    border:1px solid var(--clay); color:var(--clay); border-radius:.45em; padding:.04em .34em;
    vertical-align:.14em; white-space:nowrap; text-transform:uppercase; }
  .lead .slot{ font-size:.8em; }
  .cover .kicker .slot{ font-size:.8em; }
  .cover .slot{ border-color:#d8b06a; color:#e7c88d; }

  /* cover */
  .cover{ background:var(--pine); color:#fff; }
  .cover .grid-bg{ position:absolute; inset:0; opacity:.07;
    background-image:radial-gradient(#fff 1px, transparent 1px); background-size:36px 36px; }
  .cover .inner{ position:relative; z-index:2; flex:1; display:flex; flex-direction:column; padding:96px 120px 64px; }
  .cover .kicker{ display:flex; align-items:center; gap:20px; font-family:'Mono'; font-size:20px;
    letter-spacing:.2em; text-transform:uppercase; color:#c9e3d4; }
  .cover .kicker .chip{ border:1px solid rgba(201,227,212,.45); border-radius:999px; padding:7px 18px; }
  .cover h1{ font-weight:400; font-size:118px; line-height:1.03; letter-spacing:-.02em; margin-top:70px; }
  .cover h1 .serif{ font-family:'Serif'; font-style:italic; color:#a9d3bd; }
  .cover .sub{ font-size:30px; line-height:1.5; color:#d6e6dd; margin-top:32px; max-width:1240px; }
  .cover .ruler{ margin-top:auto; height:58px; border-bottom:1px solid rgba(255,255,255,.28);
    background-image:
      repeating-linear-gradient(90deg, rgba(255,255,255,.55) 0 1px, transparent 1px 200px),
      repeating-linear-gradient(90deg, rgba(255,255,255,.3) 0 1px, transparent 1px 40px);
    background-size:100% 58px, 100% 24px;
    background-position:left bottom, left bottom;
    background-repeat:no-repeat, no-repeat; }
  .cover .band{ margin-top:50px; display:grid; grid-template-columns:repeat(4,1fr); }
  .cover .band .cell{ padding-right:38px; border-right:1px solid rgba(255,255,255,.12); }
  .cover .band .cell:last-child{ border-right:none; }
  .cover .band .cell:not(:first-child){ padding-left:38px; }
  .cover .band .v{ font-size:58px; line-height:1.15; font-variant-numeric:tabular-nums; }
  .cover .band .k{ font-family:'Mono'; font-size:15px; letter-spacing:.16em; text-transform:uppercase;
    color:#9db8aa; margin-top:14px; }

  /* steps */
  .steps{ margin-top:60px; border-top:1px solid var(--line); flex:1; display:flex; flex-direction:column;
    justify-content:space-between; padding-bottom:24px; }
  .step{ display:grid; grid-template-columns:90px 300px 1fr; gap:40px; align-items:baseline;
    padding:40px 0; border-bottom:1px solid var(--hair); }
  .step .idx{ font-family:'Mono'; font-size:22px; font-weight:700; color:var(--green); letter-spacing:.12em; }
  .step .t{ font-size:34px; }
  .step .b{ font-size:23px; line-height:1.5; color:var(--muted); }

  /* logic table */
  .lanes{ margin-top:54px; border-top:1px solid var(--line); flex:1; display:flex; flex-direction:column;
    justify-content:space-between; padding-bottom:24px; }
  .lane{ display:grid; grid-template-columns:420px 1fr; gap:56px; align-items:baseline;
    padding:36px 0; border-bottom:1px solid var(--hair); }
  .lane .t{ font-size:40px; line-height:1.15; }
  .lane .b{ font-size:23px; line-height:1.5; color:var(--muted); }

  /* results */
  .results{ margin-top:58px; border-top:1px solid var(--line); flex:1; display:flex; flex-direction:column;
    justify-content:space-between; padding-bottom:26px; }
  .res{ display:grid; grid-template-columns:420px 1fr 260px; gap:56px; align-items:center;
    padding:34px 0; border-bottom:1px solid var(--hair); }
  .res .t{ font-size:40px; line-height:1.15; }
  .res .track{ height:18px; background:#e7ece9; position:relative; }
  .res .track span{ position:absolute; left:0; top:0; bottom:0; background:var(--green); }
  .res .track.pending{ background:repeating-linear-gradient(90deg,#e7ece9 0 14px,#eef2f0 14px 28px); }
  .res .lv{ text-align:right; font-size:44px; font-variant-numeric:tabular-nums; }
  .closing{ font-family:'Mono'; font-size:16px; color:var(--muted); letter-spacing:.06em; padding-top:26px; }
"""


def slide_cover():
    cells = "".join(
        f'<div class="cell"><div class="v">{V(slot, dark=True)}</div><div class="k">{name}</div></div>'
        for name, slot in LANES
    )
    return f"""
<section class="slide cover">
  <div class="grid-bg"></div>
  <div class="inner">
    <div class="kicker"><span class="chip">BirlaNu · Product Mapping Eval</span><span>{V('RUN_DATE', dark=True)}</span></div>
    <h1>Mapping, <span class="serif">measured.</span></h1>
    <div class="sub">Four mapping logics — SKU-based, brand-based, normal product title, MRP-based — each run on its own set of product titles.</div>
    <div class="ruler"></div>
    <div class="band">{cells}</div>
  </div>
</section>"""


def slide_tested():
    return f"""
<section class="slide">
  <div class="content">
    <div class="secnum"><span class="num2">01</span><span class="lbl">What was tested</span><span class="rule"></span></div>
    <h2>One dataset per mapping logic, each built to <span class="serif">exercise that logic</span>.</h2>
    <div class="lead">Each logic was run on its own set of {V('TITLES')} product titles, so the four results are comparable lane by lane.</div>
    <div class="lanes">
      <div class="lane"><div class="t">SKU-based</div>
        <div class="b">Titles that carry an embedded product code — the code path, tested on its own.</div></div>
      <div class="lane"><div class="t">Brand-based</div>
        <div class="b">Titles carrying the brand token and no code — the brand path, tested on its own.</div></div>
      <div class="lane"><div class="t">Normal product title</div>
        <div class="b">Plain titles: no code and no brand token — matching on the words alone.</div></div>
      <div class="lane"><div class="t">MRP-based</div>
        <div class="b">Titles matched with price in the picture — the price-led path, tested on its own.</div></div>
    </div>
  </div>
  <div class="footbar"><span>BirlaNu · Product Mapping Eval</span><span class="rule"></span><span class="pg"><b>02</b> / 04</span></div>
</section>"""


def slide_method():
    return f"""
<section class="slide">
  <div class="content">
    <div class="secnum"><span class="num2">02</span><span class="lbl">How it was run and scored</span><span class="rule"></span></div>
    <h2>One run per logic, every title scored against its <span class="serif">recorded result</span>.</h2>
    <div class="steps">
      <div class="step">
        <div class="idx">01</div><div class="t">The sets</div>
        <div class="b">Four datasets, one per mapping logic, each holding its own product titles as the agent receives them — no codes or hints added.</div>
      </div>
      <div class="step">
        <div class="idx">02</div><div class="t">The runs</div>
        <div class="b">One run per logic, same revision and the same conditions throughout; nothing adjusted afterwards.</div>
      </div>
      <div class="step">
        <div class="idx">03</div><div class="t">The scoring</div>
        <div class="b">Each title carries its expected result, fixed in the set before the run; every title is scored against it the same way.</div>
      </div>
    </div>
  </div>
  <div class="footbar"><span>BirlaNu · Product Mapping Eval</span><span class="rule"></span><span class="pg"><b>03</b> / 04</span></div>
</section>"""


def slide_found():
    rows = "".join(
        f'<div class="res"><div class="t">{name}</div>{lane_bar(slot)}</div>'
        for name, slot in LANES
    )
    return f"""
<section class="slide">
  <div class="content">
    <div class="secnum"><span class="num2">03</span><span class="lbl">What we found</span><span class="rule"></span></div>
    <h2>How each mapping logic <span class="serif">performed</span>.</h2>
    <div class="results">{rows}</div>
    <div class="closing">Each logic scored against the expected result recorded in its own set; results as reported by the run.</div>
  </div>
  <div class="footbar"><span>BirlaNu · Product Mapping Eval</span><span class="rule"></span><span class="pg"><b>04</b> / 04</span></div>
</section>"""


def build_html():
    body = slide_cover() + slide_tested() + slide_method() + slide_found()
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>BirlaNu · Product Mapping Eval</title>
<style>{CSS}</style></head>
<body>{body}
</body></html>
"""
    with open(SRC, "w", encoding="utf-8") as f:
        f.write(doc)
    return SRC


async def render():
    from playwright.async_api import async_playwright
    os.makedirs(os.path.dirname(PDF), exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        await page.goto(f"file://{SRC}")
        await page.evaluate("document.fonts.ready")
        await page.wait_for_timeout(700)

        report = await page.evaluate("""() => {
          const out = {slides: document.querySelectorAll('.slide').length, overflow: [], outside: [], tight: []};
          document.querySelectorAll('.slide').forEach((s, si) => {
            if (s.scrollWidth > s.clientWidth + 1 || s.scrollHeight > s.clientHeight + 1)
              out.overflow.push({slide: si+1, sw: s.scrollWidth, cw: s.clientWidth, sh: s.scrollHeight, ch: s.clientHeight});
            const sr = s.getBoundingClientRect();
            s.querySelectorAll('*').forEach(el => {
              if (!el.children.length && (el.textContent||'').trim().length) {
                const r = el.getBoundingClientRect();
                if (r.right > sr.right + 1 || r.bottom > sr.bottom + 1 || r.left < sr.left - 1 || r.top < sr.top - 1)
                  out.outside.push({slide: si+1, tag: el.tagName, cls: el.className,
                    txt: (el.textContent||'').trim().slice(0,40),
                    r: [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)]});
                if (el.scrollWidth > el.clientWidth + 2)
                  out.tight.push({slide: si+1, cls: el.className,
                    txt: (el.textContent||'').trim().slice(0,40), sw: el.scrollWidth, cw: el.clientWidth});
              }
            });
          });
          return out;
        }""")

        await page.pdf(path=PDF, print_background=True, width="1920px", height="1080px")
        slides = await page.query_selector_all(".slide")
        for i, s in enumerate(slides, 1):
            await s.screenshot(path=os.path.join(OUT, f"slide_{i:02d}.png"))
        print(json.dumps({"errors": errors, "report": report}, indent=1))
        await browser.close()
    print("rendered", PDF)


if __name__ == "__main__":
    print("slot check:")
    validate()
    if "--check" in sys.argv:
        sys.exit(0)
    build_html()
    asyncio.run(render())
