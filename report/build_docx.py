"""Render report/report.md into a Word document with embedded figures.

Run with:  uv run --with python-docx python report/build_docx.py
(kept out of pyproject.toml so `uv sync` stays clean for the marker)

Markdown subset supported: #/##/### headings, paragraphs, bullet lists,
pipe tables, **bold**, `code`, *italic* caption lines, and
[[FIG:filename.png|caption]] image directives.
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "report" / "report.md"
OUT = ROOT / "report" / "AgriGuard_RL_Summative_Enock_Mugisha.docx"
FIGS = ROOT / "figures"

INLINE = re.compile(r"(\*\*.+?\*\*|`.+?`|\*.+?\*)")


def add_runs(par, text: str):
    """Write text into a paragraph, honouring **bold**, `code` and *italic*."""
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(9)
        elif part.startswith("*") and part.endswith("*"):
            par.add_run(part[1:-1]).italic = True
        else:
            par.add_run(part)


def add_table(doc, rows: list[list[str]]):
    header, body = rows[0], rows[1:]
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, cell in enumerate(header):
        p = t.rows[0].cells[i].paragraphs[0]
        add_runs(p, cell)
        for r in p.runs:
            r.bold = True
    for row in body:
        cells = t.add_row().cells
        for i, cell in enumerate(row[:len(header)]):
            add_runs(cells[i].paragraphs[0], cell)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(8)
    doc.add_paragraph()


def add_figure(doc, fname: str, caption: str):
    path = FIGS / fname
    if not path.exists():
        doc.add_paragraph(f"[missing figure: {fname}]")
        return
    doc.add_picture(str(path), width=Inches(6.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.italic = True; r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)


def main():
    md = SRC.read_text().splitlines()
    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.9)
        s.top_margin = s.bottom_margin = Inches(0.8)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"; normal.font.size = Pt(10.5)

    i, table_buf = 0, []
    while i < len(md):
        line = md[i].rstrip()

        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not re.fullmatch(r"[\s:\-|]+", line.strip("|")):
                table_buf.append(cells)
            i += 1
            continue
        if table_buf:
            add_table(doc, table_buf); table_buf = []

        fig = re.fullmatch(r"\[\[FIG:([^|\]]+)\|(.+)\]\]", line.strip())
        if fig:
            add_figure(doc, fig.group(1).strip(), fig.group(2).strip()); i += 1; continue

        if line.startswith("### "):
            doc.add_heading(line[4:], level=3)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=2)
        elif line.startswith("# "):
            h = doc.add_heading(line[2:], level=0)
            h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif line.strip() in {"---", "***"}:
            doc.add_page_break()
        elif line.startswith(("* ", "- ")):
            add_runs(doc.add_paragraph(style="List Bullet"), line[2:])
        elif line.strip():
            add_runs(doc.add_paragraph(), line.strip())
        i += 1

    if table_buf:
        add_table(doc, table_buf)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
