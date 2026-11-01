from playwright.sync_api import sync_playwright

SRC = "file:///home/wuling/pdfbuild/handout.html"
OUT = "/home/wuling/pdfbuild/PyTorch与神经网络讲义.pdf"

HEADER = """
<div style="width:100%;font-size:8pt;color:#8a8a8a;
            font-family:'Noto Serif CJK SC',serif;
            padding:0 18mm;display:flex;justify-content:space-between;">
  <span>PyTorch 与神经网络：从张量到可训练模型</span>
  <span>系统讲义</span>
</div>
"""
FOOTER = """
<div style="width:100%;font-size:8.5pt;color:#666;
            font-family:'Noto Serif CJK SC',serif;text-align:center;">
  <span class="pageNumber"></span> / <span class="totalPages"></span>
</div>
"""

with sync_playwright() as p:
    b = p.chromium.launch(args=["--no-sandbox", "--font-render-hinting=none"])
    pg = b.new_page()
    pg.goto(SRC, wait_until="load", timeout=180000)
    pg.emulate_media(media="print")
    pg.pdf(path=OUT, format="A4", print_background=True,
           display_header_footer=True,
           header_template=HEADER, footer_template=FOOTER,
           margin={"top": "20mm", "bottom": "18mm", "left": "18mm", "right": "18mm"})
    b.close()
print("PDF 生成完成:", OUT)
