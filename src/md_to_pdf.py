"""Small, dependency-light markdown -> PDF renderer used by Module 8.

Supports exactly what the executive summary needs: #/## headers, **bold**,
`code`, numbered lists, bulleted lists, and an italic footer line. Not a
general-purpose markdown engine by design — the executive summary is the
only PDF this pipeline produces.
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, ListFlowable, ListItem, Paragraph, SimpleDocTemplate

NAVY = colors.HexColor("#1a2a4a")
SLATE = colors.HexColor("#334155")

_styles = getSampleStyleSheet()
_styles.add(ParagraphStyle(name="OITitle", fontName="Helvetica-Bold", fontSize=20,
                            textColor=NAVY, spaceAfter=6, leading=24))
_styles.add(ParagraphStyle(name="OISubtitle", fontName="Helvetica", fontSize=10.5,
                            textColor=SLATE, spaceAfter=18, leading=14))
_styles.add(ParagraphStyle(name="OIH2", fontName="Helvetica-Bold", fontSize=14,
                            textColor=NAVY, spaceBefore=16, spaceAfter=8, leading=17))
_styles.add(ParagraphStyle(name="OIBody", fontName="Helvetica", fontSize=10.5,
                            textColor=colors.HexColor("#1f2937"), spaceAfter=8, leading=15.5))
_styles.add(ParagraphStyle(name="OIListItem", fontName="Helvetica", fontSize=10.5,
                            textColor=colors.HexColor("#1f2937"), spaceAfter=6, leading=15.5))
_styles.add(ParagraphStyle(name="OIFooter", fontName="Helvetica-Oblique", fontSize=8.5,
                            textColor=colors.HexColor("#6b7280"), spaceBefore=10, leading=12))


def _inline(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    return text


def _is_break(line: str) -> bool:
    stripped = line.strip()
    return (not stripped or line.startswith("#") or re.match(r"^\d+\.\s+", line)
            or stripped.startswith("- ") or stripped == "---")


def markdown_to_pdf(md_path: Path, pdf_path: Path) -> None:
    lines = md_path.read_text(encoding="utf-8").split("\n")
    story = []
    numbered: list[str] = []
    bulleted: list[str] = []

    def flush_numbered():
        if numbered:
            items = [ListItem(Paragraph(_inline(t), _styles["OIListItem"]), leftIndent=6) for t in numbered]
            story.append(ListFlowable(items, bulletType="1", start="1", leftIndent=18,
                                       bulletFontSize=10.5, spaceBefore=4, spaceAfter=10))
            numbered.clear()

    def flush_bulleted():
        if bulleted:
            items = [ListItem(Paragraph(_inline(t), _styles["OIListItem"]), leftIndent=6) for t in bulleted]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=18,
                                       bulletFontSize=10.5, spaceBefore=4, spaceAfter=10))
            bulleted.clear()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue

        if line.startswith("# "):
            flush_numbered(); flush_bulleted()
            story.append(Paragraph(_inline(line[2:]), _styles["OITitle"]))
            i += 1
            continue

        if line.startswith("## "):
            flush_numbered(); flush_bulleted()
            story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#d1d5db"),
                                     spaceBefore=2, spaceAfter=2))
            story.append(Paragraph(_inline(line[3:]), _styles["OIH2"]))
            i += 1
            continue

        if line.strip() == "---":
            flush_numbered(); flush_bulleted()
            i += 1
            continue

        if re.match(r"^\d+\.\s+", line):
            flush_bulleted()
            text = re.match(r"^\d+\.\s+(.*)", line).group(1)
            j = i + 1
            while j < len(lines) and lines[j].strip() and not _is_break(lines[j]):
                text += " " + lines[j].strip()
                j += 1
            numbered.append(text)
            i = j
            continue

        if line.strip().startswith("- "):
            flush_numbered()
            text = line.strip()[2:]
            j = i + 1
            while j < len(lines) and lines[j].strip() and not _is_break(lines[j]):
                text += " " + lines[j].strip()
                j += 1
            bulleted.append(text)
            i = j
            continue

        if line.strip().startswith("*") and line.strip().endswith("*") and not line.strip().startswith("**"):
            flush_numbered(); flush_bulleted()
            story.append(Paragraph(_inline(line.strip()[1:-1]), _styles["OIFooter"]))
            i += 1
            continue

        flush_numbered(); flush_bulleted()
        text = line.strip()
        j = i + 1
        while j < len(lines) and lines[j].strip() and not _is_break(lines[j]):
            text += " " + lines[j].strip()
            j += 1
        style = _styles["OISubtitle"] if text.startswith("**Cohort:**") else _styles["OIBody"]
        story.append(Paragraph(_inline(text), style))
        i = j

    flush_numbered(); flush_bulleted()

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=letter,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
        title="OncoInsights Executive Summary",
    )
    doc.build(story)
