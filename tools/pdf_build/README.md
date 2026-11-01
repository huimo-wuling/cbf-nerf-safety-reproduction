# 讲义 PDF 生成流程

本目录保存 `docs/06_PyTorch与神经网络讲义.md` → `docs/06_PyTorch与神经网络讲义.pdf` 的可复现构建链。**只有需要重新排版时才用得到**，阅读讲义直接看 PDF 或 Markdown 即可。

## 依赖

| 组件 | 用途 | 说明 |
| --- | --- | --- |
| pandoc ≥ 3.x | Markdown → HTML（含 MathML 公式） | 可用 `pip install pypandoc_binary` 获得独立二进制 |
| Chromium（Playwright 版） | HTML → PDF | `pip install playwright && python -m playwright install chromium`，无需系统级安装 |
| poppler-utils | `pdfinfo` / `pdftotext`，用于第二遍编译时定位章节页码 | Ubuntu: `sudo apt install poppler-utils` |
| 中文字体 | 正文与标题 | 需要 Noto Serif CJK SC 与 Noto Sans CJK SC |

本机实测：pandoc 3.9、Chromium 156（headless shell）、poppler 24.x、Ubuntu 26.04。

## 三步构建

```bash
# 0) 约定：在仓库根目录执行；产物写到 docs/
SRC=docs/06_PyTorch与神经网络讲义.md
OUT=docs

# 1) Markdown → HTML
pandoc "$SRC" \
  -f markdown-yaml_metadata_block-simple_tables \
  -o "$OUT/handout.html" --standalone --mathml --toc --toc-depth=2 \
  --syntax-highlighting=tango --css=handout.css \
  --metadata title="PyTorch 与神经网络：从张量到可训练模型" \
  --metadata subtitle="系统讲义 · 16 章 + 3 附录" \
  --metadata date="2026 年 11 月" \
  --metadata abstract="（摘要正文）"

# 2) 第一遍编译（此时目录尚无页码）
python tools/pdf_build/mkpdf.py

# 3) 定位章节页码并写回目录 → 第二遍编译
python tools/pdf_build/toc_pages.py
python tools/pdf_build/mkpdf.py
```

## 两个技术要点

**为什么分两遍。** 目录页码需要"标题 → 实际页号"的映射，而 Chromium 不支持 CSS 的 `target-counter`。第二遍编译前，`toc_pages.py` 用 `pdftotext` 逐页取文本，按"页面正文是否以 `第 N 章` / `附录 X` 开头"定位章节起始页，再把 `N` 写进对应目录条目。页面布局不变，因此第二遍得到的页码与最终 PDF 一致。

**为什么禁用两个 Pandoc 扩展。** `yaml_metadata_block`：正文里的 `---` 分隔线会被误判为 YAML 元数据块；`simple_tables`：紧贴标题的 `---` 会被解析成简单表格，把后续标题吞进表格单元格。

## 自定义排版

`handout.css` 控制全部版式：A4 页面、页边距、字体、代码块、表格、目录引导点与页码对齐、封面与摘要样式、分页规则（`h1` 强制分页、代码块允许跨页）。修改后只需重跑第 2、3 步。

`mkpdf.py` 中的 `HEADER` / `FOOTER` 常量控制页眉页脚；`mkpdf.py` 的 `margin` 参数与 `handout.css` 的 `@page` 需保持一致。
