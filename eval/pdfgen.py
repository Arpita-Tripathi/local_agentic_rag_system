"""Tiny PDF writers for the sample set, so every test file is reproducible and free to share.

Not a general PDF library. Text PDFs use the standard Helvetica font. Hindi PDFs use a non-embedded Type0
font with a ToUnicode map: the text layer extracts as correct Unicode, but a viewer without that font
shows placeholder glyphs. That is fine for fixtures, which are read by software, not by people.
"""

import textwrap
from typing import List, Sequence, Union

Line = Union[str, Sequence[str]]  # a string, or table cells drawn left to right

PAGE_W, PAGE_H = 612, 792
LEFT, TOP, LEADING, COL_W = 50, 740, 18, 150


def wrap(text: str, width: int) -> List[str]:
    return textwrap.wrap(text, width=width) or [""]


def _latin(s: str) -> str:
    s.encode("latin-1")  # fail loudly on characters the standard font cannot show
    return "(" + s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"


def _hindi(s: str) -> str:
    return "<" + "".join(f"{ord(ch):04X}" for ch in s) + ">"


_TO_UNICODE = (
    "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
    "/CMapName /Adobe-Identity-UCS def /CMapType 2 def\n"
    "1 begincodespacerange <0000> <FFFF> endcodespacerange\n"
    "1 beginbfrange <0000> <FFFF> <0000> endbfrange\n"
    "endcmap CMapName currentdict /CMap defineresource pop end end"
)


def _content(lines: List[Line], encode) -> str:
    ops = ["BT", "/F1 12 Tf"]
    y = TOP
    for line in lines:
        cells = [line] if isinstance(line, str) else list(line)
        for i, cell in enumerate(cells):
            if cell:
                ops.append(f"1 0 0 1 {LEFT + i * COL_W} {y} Tm {encode(cell)} Tj")
        y -= LEADING
    ops.append("ET")
    return "\n".join(ops)


def _assemble(objects: List[str]) -> bytes:
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def build_pdf(pages: List[List[Line]], script: str = "latin") -> bytes:
    """pages: a list of pages, each a list of lines. An empty page is an empty list."""
    encode = _hindi if script == "devanagari" else _latin
    n = len(pages)
    page_ids = [3 + 2 * i for i in range(n)]
    font_id = 3 + 2 * n
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {n} >>",
    ]
    for i, lines in enumerate(pages):
        stream = _content(lines, encode) if lines else ""
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] /Contents {page_ids[i] + 1} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    if script == "devanagari":
        objects.append(
            f"<< /Type /Font /Subtype /Type0 /BaseFont /NotoSansDevanagari /Encoding /Identity-H "
            f"/DescendantFonts [{font_id + 1} 0 R] /ToUnicode {font_id + 3} 0 R >>"
        )
        objects.append(
            "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /NotoSansDevanagari "
            "/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
            f"/FontDescriptor {font_id + 2} 0 R /DW 600 /CIDToGIDMap /Identity >>"
        )
        objects.append(
            "<< /Type /FontDescriptor /FontName /NotoSansDevanagari /Flags 4 /FontBBox [0 -300 1000 1000] "
            "/ItalicAngle 0 /Ascent 900 /Descent -300 /CapHeight 700 /StemV 80 >>"
        )
        objects.append(f"<< /Length {len(_TO_UNICODE)} >>\nstream\n{_TO_UNICODE}\nendstream")
    else:
        objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    return _assemble(objects)


def build_scanned_pdf(pages: List[List[str]], path: str, dpi: int = 150) -> None:
    """Image-only PDF: every page is a rendered picture of its lines, with no text layer."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=34)
    images = []
    for lines in pages:
        img = Image.new("L", (1240, 1754), 255)
        draw = ImageDraw.Draw(img)
        y = 140
        for line in lines:
            draw.text((110, y), line, fill=0, font=font)
            y += 56
        images.append(img)
    images[0].save(path, "PDF", resolution=float(dpi), save_all=True, append_images=images[1:])


def corrupt(data: bytes, keep: float = 0.55) -> bytes:
    """A real PDF cut off partway: it still starts with %PDF- but has no valid page tree or trailer."""
    return data[: int(len(data) * keep)]
