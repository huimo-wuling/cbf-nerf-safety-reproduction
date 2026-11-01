# -*- coding: utf-8 -*-
"""为自动生成的目录注入真实页码。

Chromium 不支持 CSS 的 target-counter，因此采用两遍编译：
第一遍得到 PDF，用 pdftotext 定位每章起始页，把页码写进目录，再编译第二遍。
"""
import re
import subprocess
import sys
from pathlib import Path

PDF, HTM = "PyTorch与神经网络讲义.pdf", "handout.html"
HEADER_TITLE = "PyTorch与神经网络：从张量到可训练模型"
HEADER_RIGHT = "系统讲义"


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def page_texts(pdf: str) -> dict:
    info = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
    n = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    out = {}
    for p in range(1, n + 1):
        out[p] = subprocess.run(
            ["pdftotext", "-f", str(p), "-l", str(p), pdf, "-"],
            capture_output=True, text=True).stdout
    return out


def body_of(page_txt: str) -> str:
    """剔除页眉文字与页脚页码，返回归一化正文。"""
    n = norm(page_txt)
    n = n.replace(HEADER_TITLE, "").replace(HEADER_RIGHT, "")
    n = re.sub(r"\d+/\d+$", "", n)
    return n


def main() -> int:
    texts = page_texts(PDF)
    pages = sorted(texts)
    body = {p: body_of(t) for p, t in texts.items()}

    src = Path(HTM).read_text(encoding="utf-8")
    nav = re.search(r'<nav id="TOC".*?</nav>', src, re.S).group(0)
    nav = re.sub(r'<span class="toc-page">\d+</span>', "", nav)   # 清掉旧页码

    anchors = re.findall(r'<a href="#([^"]+)"[^>]*>(.*?)</a>', nav, re.S)
    pagemap, detail = {}, []
    for href, inner in anchors:
        label = re.sub(r"<[^>]+>", "", inner)
        label = re.sub(r"\s+", "", label)
        if len(label) < 3:
            continue
        got = None
        m = re.match(r"(第\d+章|附录[A-C])", label)
        if m:
            head = m.group(1)
            for p in pages:
                if p > 2 and body[p].startswith(head):
                    got = p
                    break
        if got is None:
            for p in pages:
                if p > 2 and label in body[p]:
                    got = p
                    break
        if got:
            pagemap[href] = got
            detail.append((got, re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", inner)).strip()))

    def repl(mm):
        href, inner = mm.group(1), mm.group(2)
        p = pagemap.get(href)
        if p is None:
            return mm.group(0)
        return f'<a href="#{href}">{inner}</a><span class="toc-page">{p}</span>'

    new_nav = re.sub(r'<a href="#([^"]+)"[^>]*>(.*?)</a>', repl, nav, flags=re.S)
    Path(HTM).write_text(src.replace(nav, new_nav), encoding="utf-8")

    print(f"目录条目 {len(anchors)}，注入页码 {len(pagemap)}")
    for p, l in sorted(detail):
        print(f"   p{p:<3d} {l[:50]}")
    missing = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", i)).strip()
               for h, i in anchors if h not in pagemap]
    if missing:
        print("未定位:", missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
