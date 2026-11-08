#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Build the small embedded bold CJK font used for the UI headings.

The board image only ships WenQuanYi Zen Hei (regular only) for Chinese, and
Slint's FemtoVG renderer does not synthesise bold, so `font-weight: 700`
Chinese text renders regular. Instead of installing a full bold CJK font
package (tens of MB), this subsets Noto Sans CJK SC Bold (SIL OFL 1.1) to the
characters that actually occur in the UI sources (~150 glyphs, a few hundred
KB), converts it to TrueType outlines, renames it to "DR Heading SC" (the
OFL forbids reusing the original name for a modified font) and writes it next
to the .slint file, which embeds it via `import "fonts/...";`.

Characters missing from the subset (e.g. new text added later without
re-running this script) fall back to the system font, i.e. render regular.

Usage (needs `pip install fonttools`):
    python3 tools/make_heading_font.py [--source NotoSansCJK-Bold.ttc]
"""

import argparse
import pathlib
import re
import sys

from fontTools import subset
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTCollection, newTable

APP = pathlib.Path(__file__).resolve().parent.parent
OUT = APP / "ui" / "fonts" / "DRHeadingSC-Bold.ttf"
FAMILY = "DR Heading SC"
DEFAULT_SOURCE = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
SOURCE_FAMILY = "Noto Sans CJK SC"


def ui_characters():
    """Printable ASCII plus every non-ASCII character in string literals of
    the UI sources (Slint markup and the Rust status messages)."""
    chars = {chr(c) for c in range(0x20, 0x7F)}
    literal = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
    files = sorted((APP / "ui").glob("*.slint")) + sorted((APP / "src").glob("*.rs"))
    for path in files:
        if path.name.startswith("."):
            continue
        for match in literal.finditer(path.read_text(encoding="utf-8")):
            chars.update(ch for ch in match.group(1) if ord(ch) > 0x7F)
    return "".join(sorted(chars))


def collection_index(source):
    for i, font in enumerate(TTCollection(source).fonts):
        name = font["name"]
        family = name.getDebugName(16) or name.getDebugName(1)
        if family == SOURCE_FAMILY:
            return i
    sys.exit(f"{source}: no '{SOURCE_FAMILY}' face found")


def to_truetype(font):
    """CFF (cubic) -> glyf (quadratic) outlines, after fontTools' otf2ttf
    snippet: TrueType is what every rasteriser on the board handles."""
    order = font.getGlyphOrder()
    glyph_set = font.getGlyphSet()
    glyf = newTable("glyf")
    glyf.glyphOrder = order
    glyf.glyphs = {}
    for name in order:
        pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyf.glyphs[name] = pen.glyph()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf
    for tag in ("CFF ", "VORG"):
        if tag in font:
            del font[tag]
    for glyph in glyf.glyphs.values():
        glyph.recalcBounds(glyf)
    glyf.compile(font)
    hmtx = font["hmtx"]
    for name, glyph in glyf.glyphs.items():
        if hasattr(glyph, "xMin"):
            hmtx[name] = (hmtx[name][0], glyph.xMin)
    maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    maxp.maxZones = 1
    for field in (
        "maxTwilightPoints",
        "maxStorage",
        "maxFunctionDefs",
        "maxInstructionDefs",
        "maxStackElements",
        "maxSizeOfInstructions",
        "maxComponentElements",
    ):
        setattr(maxp, field, 0)
    font["maxp"] = maxp
    maxp.numGlyphs = len(order)
    maxp.recalc(font)
    post = font["post"]
    post.formatType = 3.0
    font.sfntVersion = "\x00\x01\x00\x00"


def rename(font):
    name = font["name"]
    source_version = (name.getDebugName(5) or "unknown version").split(";")[0]
    keep = {0, 7, 8, 9, 11, 12, 13, 14}  # copyright, trademark, vendor, license...
    name.names = [rec for rec in name.names if rec.nameID in keep]
    for name_id, value in (
        (1, FAMILY),
        (2, "Bold"),
        (3, "DRHeadingSC-Bold;subset of Noto Sans CJK SC Bold"),
        (4, f"{FAMILY} Bold"),
        (5, f"Version 1.0; subset of Noto Sans CJK SC Bold {source_version}"),
        (6, "DRHeadingSC-Bold"),
    ):
        name.setName(value, name_id, 3, 1, 0x409)
    font["OS/2"].usWeightClass = 700


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    text = ui_characters()
    options = subset.Options()
    options.font_number = collection_index(args.source)
    options.hinting = False
    options.desubroutinize = True
    options.notdef_outline = True
    options.name_IDs = ["*"]
    options.name_languages = ["*"]
    options.drop_tables += ["VORG", "vhea", "vmtx", "DSIG"]
    font = subset.load_font(args.source, options)
    cmap_before = font.getBestCmap()
    subsetter = subset.Subsetter(options)
    subsetter.populate(text=text)
    subsetter.subset(font)
    to_truetype(font)
    rename(font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    font.save(str(args.output))

    cmap = font.getBestCmap()
    missing = [ch for ch in text if ord(ch) not in cmap]
    cjk = [ch for ch in text if ord(ch) > 0x2E80]
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes, {len(cmap)} code points, {len(cjk)} CJK)")
    if missing:
        absent = "".join(ch for ch in missing if ord(ch) not in cmap_before)
        print(f"not in source font (system fallback, regular weight): {absent!r}")
        if len(absent) != len(missing):
            sys.exit("subset lost characters that the source font has")


if __name__ == "__main__":
    main()
