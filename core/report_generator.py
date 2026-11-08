"""
ATHER - Two-Page IEEE-Style Performance Report
------------------------------------------------
PDF generation using ReportLab only (no LaTeX).

Format (matches the earlier ATHER report):
- Full-width title, author lines and abstract, then two-column body
- Sections I-XI, subsections A-D, formula lines, TABLE I with caption
- Photos stacked one below the other, graphs stacked one below the other
- ONE text size for the whole report (identical on both pages)
- Content flows continuously column -> column -> page, so no blank gaps
- Exactly two A4 pages: only the image scale is adjusted to fit

main.py signature preserved:
  generate_ieee_report(results, output_path, image_name, gamma_value,
                       original_image_path, enhanced_image_paths,
                       algorithm_parameters, system_info)
"""

from pathlib import Path
from io import BytesIO
import platform
import sys

import numpy as np

from PIL import Image as PILImage

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Table, TableStyle,
    Image, KeepTogether, HRFlowable, NextPageTemplate,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


# =========================================================
# CONFIGURATION
# =========================================================

PAGE_WIDTH, PAGE_HEIGHT = A4

LEFT = 0.6 * inch
RIGHT = 0.6 * inch
TOP = 0.5 * inch
BOTTOM = 0.55 * inch            # page number lives below this line

CONTENT_WIDTH = PAGE_WIDTH - LEFT - RIGHT
COL_GAP = 0.25 * inch
COL_W = (CONTENT_WIDTH - COL_GAP) / 2
COL_X = [LEFT, LEFT + COL_W + COL_GAP]

PHOTO_MAX_H = 2.0 * inch        # tallest photo at scale 1.0
CHART_W_PX_H_RATIO = 1.5 / 3.5  # chart height / width

# (body font size, leading). The first size that lets everything fit on
# two pages with reasonably large images is used. Both pages always share
# the same size.
TEXT_SIZES = [(9.0, 10.8), (8.7, 10.4), (8.4, 10.0), (8.1, 9.6)]
GOOD_IMAGE_SCALE = 0.62         # prefer a smaller font over tiny images
MIN_IMAGE_SCALE = 0.30

ALGORITHM_NAMES = {
    "gamma": "Gamma Correction",
    "contrast_stretching": "Contrast Stretching",
    "histogram_equalization": "Histogram Equalization",
    "clahe": "CLAHE",
}

SHORT_NAMES = {
    "gamma": "Gamma",
    "contrast_stretching": "Contrast Str.",
    "histogram_equalization": "Hist. Eq.",
    "clahe": "CLAHE",
}

ALGORITHM_ORDER = [
    "gamma",
    "contrast_stretching",
    "histogram_equalization",
    "clahe",
]

REFERENCES = [
    (
        "S. Khormaeipour and F. Shakeri, "
        "\"Enhancing low-light color image visibility with "
        "hybrid contrast and saturation modification using "
        "a saturation-aware map,\" Journal of Visual "
        "Communication and Image Representation, vol. 107, "
        "Art. no. 104392, Mar. 2025, "
        "doi: 10.1016/j.jvcir.2025.104392."
    ),
    (
        "S.-H. Jin, D.-M. Son, S.-H. Lee, Y.-H. Go, and "
        "S.-H. Lee, \"Enhancing local contrast in low-light "
        "images: A multiscale model with adaptive "
        "redistribution of histogram excess,\" Mathematics, "
        "vol. 13, no. 20, Art. no. 3282, Oct. 2025, "
        "doi: 10.3390/math13203282."
    ),
    (
        "W. Deng and G. Xie, \"Image contrast enhancement "
        "and brightness preservation based on an adaptive "
        "histogram correction framework,\" Applied Optics, "
        "vol. 64, no. 13, pp. 3502-3515, Apr. 2025, "
        "doi: 10.1364/AO.557280."
    ),
    (
        "M. M. Tatana, M. S. Tsoeu, and R. C. Maswanganyi, "
        "\"Low-light image and video enhancement for more "
        "robust computer vision tasks: A review,\" Journal "
        "of Imaging, vol. 11, no. 4, Art. no. 125, Apr. 2025, "
        "doi: 10.3390/jimaging11040125."
    ),
    (
        "D. Mu, Z. Wang, Z. Li, et al., \"A new multi-scale "
        "adaptive enhancement algorithm for low-illumination "
        "coal mine images based on improved CLAHE and ACE,\" "
        "Discover Artificial Intelligence, vol. 5, "
        "Art. no. 406, Dec. 2025, "
        "doi: 10.1007/s44163-025-00663-5."
    ),
]


# =========================================================
# FONTS  (Times New Roman if available, so γ and − render)
# =========================================================

def _register_fonts():
    """
    Returns (regular, bold, italic, bold_italic, has_unicode).
    Tries Times New Roman (Windows/macOS) then Liberation Serif (Linux).
    Falls back to built-in Times, where γ and − are not available.
    """

    candidates = [
        (
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/timesbd.ttf",
            "C:/Windows/Fonts/timesi.ttf",
            "C:/Windows/Fonts/timesbi.ttf",
        ),
        (
            "/Library/Fonts/Times New Roman.ttf",
            "/Library/Fonts/Times New Roman Bold.ttf",
            "/Library/Fonts/Times New Roman Italic.ttf",
            "/Library/Fonts/Times New Roman Bold Italic.ttf",
        ),
        (
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
        ),
    ]

    for files in candidates:

        if not all(Path(f).is_file() for f in files):
            continue

        try:

            names = (
                "ATHERSerif",
                "ATHERSerif-Bold",
                "ATHERSerif-Italic",
                "ATHERSerif-BoldItalic",
            )

            for name, path in zip(names, files):
                pdfmetrics.registerFont(TTFont(name, path))

            addMapping("ATHERSerif", 0, 0, names[0])
            addMapping("ATHERSerif", 1, 0, names[1])
            addMapping("ATHERSerif", 0, 1, names[2])
            addMapping("ATHERSerif", 1, 1, names[3])

            return names + (True,)

        except Exception:
            continue

    return (
        "Times-Roman",
        "Times-Bold",
        "Times-Italic",
        "Times-BoldItalic",
        False,
    )


F_REG, F_BOLD, F_ITAL, F_BI, HAS_UNICODE = _register_fonts()

GAMMA = "γ" if HAS_UNICODE else "<i>gamma</i>"
MINUS = "−" if HAS_UNICODE else "-"


# =========================================================
# STYLES
# =========================================================

def make_styles(size, lead):

    return {

        "title": ParagraphStyle(
            "title", fontName=F_BOLD, fontSize=22, leading=25,
            alignment=TA_CENTER, spaceAfter=3,
        ),

        "author": ParagraphStyle(
            "author", fontName=F_REG, fontSize=10, leading=12,
            alignment=TA_CENTER, spaceAfter=3,
        ),

        "abstract": ParagraphStyle(
            "abstract", fontName=F_REG, fontSize=size, leading=lead,
            alignment=TA_JUSTIFY, spaceAfter=0,
        ),

        "heading": ParagraphStyle(
            "heading", fontName=F_BOLD, fontSize=size + 1.5,
            leading=size + 3.5, alignment=TA_LEFT,
            spaceBefore=5, spaceAfter=2, keepWithNext=1,
        ),

        "subhead": ParagraphStyle(
            "subhead", fontName=F_BI, fontSize=size + 0.3, leading=lead,
            alignment=TA_LEFT, spaceBefore=2, spaceAfter=1,
            keepWithNext=1,
        ),

        "body": ParagraphStyle(
            "body", fontName=F_REG, fontSize=size, leading=lead,
            alignment=TA_JUSTIFY, spaceAfter=2,
        ),

        "formula": ParagraphStyle(
            "formula", fontName=F_BOLD, fontSize=size, leading=lead,
            alignment=TA_LEFT, spaceBefore=1, spaceAfter=1,
            keepWithNext=1,
        ),

        "small": ParagraphStyle(
            "small", fontName=F_REG, fontSize=size - 0.5, leading=lead,
            alignment=TA_JUSTIFY, spaceAfter=1,
        ),

        "caption": ParagraphStyle(
            "caption", fontName=F_REG, fontSize=size - 1.2,
            leading=lead - 1.4, alignment=TA_CENTER,
            spaceBefore=1, spaceAfter=3,
        ),

        "tablecap": ParagraphStyle(
            "tablecap", fontName=F_REG, fontSize=size - 1.4,
            leading=lead - 1.6, alignment=TA_CENTER,
            spaceBefore=1, spaceAfter=2, keepWithNext=1,
        ),

        "refs": ParagraphStyle(
            "refs", fontName=F_REG, fontSize=size - 1.0,
            leading=lead - 1.0, alignment=TA_JUSTIFY, spaceAfter=1,
            leftIndent=0,
        ),

        "tablehead": ParagraphStyle(
            "tablehead", fontName=F_BOLD, fontSize=7, leading=8,
            alignment=TA_CENTER,
        ),
    }


# =========================================================
# GENERAL HELPERS
# =========================================================

def safe_text(value):

    if value is None:
        return "N/A"

    text = str(value)

    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def number(value, decimals=2):

    try:
        return f"{float(value):.{decimals}f}"

    except (TypeError, ValueError):
        return "N/A"


def algorithm_name(key):

    return ALGORITHM_NAMES.get(
        key,
        str(key).replace("_", " ").title()
    )


def draw_paragraph(pdf, text, x, y, width, style):

    paragraph = Paragraph(text, style)

    _, height = paragraph.wrap(width, PAGE_HEIGHT)

    paragraph.drawOn(pdf, x, y - height)

    return y - height


def draw_page_number(pdf, page):

    pdf.saveState()

    pdf.setFont(F_REG, 9)

    pdf.setFillColor(colors.black)

    pdf.drawCentredString(PAGE_WIDTH / 2, 0.28 * inch, str(page))

    pdf.restoreState()


# =========================================================
# BENCHMARK VALIDATION
# =========================================================

def validate_results(results):

    if not results:

        raise ValueError(
            "No benchmark results found. "
            "Run all four benchmarks first."
        )

    lookup = {r.get("algorithm"): r for r in results}

    missing = set(ALGORITHM_ORDER) - set(lookup)

    if missing:

        raise ValueError(
            "Missing benchmark algorithms: "
            + ", ".join(sorted(missing))
        )

    ordered = [lookup[key] for key in ALGORITHM_ORDER]

    for result in ordered:

        if "processing_time_ms" not in result:

            raise ValueError(
                "Missing processing_time_ms for "
                + str(result.get("algorithm"))
            )

        metrics = result.get("enhanced_metrics")

        if not isinstance(metrics, dict):

            raise ValueError(
                "Missing enhanced_metrics for "
                + str(result.get("algorithm"))
            )

        for key in (
            "brightness",
            "contrast",
            "entropy",
            "dynamic_range",
        ):

            if key not in metrics:

                raise ValueError(
                    f"Missing {key} for "
                    + str(result.get("algorithm"))
                )

    return ordered


# =========================================================
# CHARTS (column-sized figures)
# =========================================================

def _mpl_font():

    return "DejaVu Sans"


def create_chart(results, title, metric_key=None, processing_time=False):

    labels = [
        SHORT_NAMES.get(i["algorithm"], algorithm_name(i["algorithm"]))
        for i in results
    ]

    if processing_time:

        values = [float(i["processing_time_ms"]) for i in results]
        ylabel = "Milliseconds"

    else:

        values = [
            float(i["enhanced_metrics"][metric_key]) for i in results
        ]

        ylabel = {
            "brightness": "Intensity",
            "contrast": "Std. deviation",
            "entropy": "Bits",
        }.get(metric_key, metric_key.replace("_", " ").title())

    figure = Figure(figsize=(3.5, 1.5), dpi=220)

    chart_canvas = FigureCanvasAgg(figure)

    axis = figure.add_subplot(111)

    bars = axis.bar(range(len(labels)), values)

    axis.set_title(title, fontsize=7.5, pad=3)

    axis.set_ylabel(ylabel, fontsize=6.5)

    axis.set_xticks(range(len(labels)))

    axis.set_xticklabels(labels, fontsize=6)

    axis.tick_params(axis="y", labelsize=6)

    axis.grid(axis="y", alpha=0.2, linewidth=0.4)

    axis.set_axisbelow(True)

    top = max(values) if values else 1

    axis.set_ylim(0, top * 1.2 if top > 0 else 1)

    for bar, value in zip(bars, values):

        axis.annotate(
            number(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            fontsize=5.5,
        )

    figure.tight_layout(pad=0.5)

    buffer = BytesIO()

    chart_canvas.print_png(buffer)

    buffer.seek(0)

    figure.clear()

    return buffer


def create_histogram(original_path, enhanced_paths):

    if not original_path or not Path(original_path).is_file():
        return None

    figure = Figure(figsize=(3.5, 1.5), dpi=220)

    chart_canvas = FigureCanvasAgg(figure)

    axis = figure.add_subplot(111)

    paths = [("Original", original_path)]

    for algorithm in ALGORITHM_ORDER:

        path = enhanced_paths.get(algorithm)

        if path and Path(path).is_file():

            paths.append((algorithm_name(algorithm), path))

    for label, path in paths:

        try:

            image = PILImage.open(path).convert("L")

            array = np.asarray(image)

            axis.hist(
                array.ravel(),
                bins=64,
                range=(0, 255),
                density=True,
                histtype="step",
                linewidth=0.8,
                label=label,
            )

            image.close()

        except Exception:
            continue

    axis.set_title("Grayscale Intensity Histogram", fontsize=7.5, pad=3)

    axis.set_xlabel("Intensity", fontsize=6.5)

    axis.set_ylabel("Density", fontsize=6.5)

    axis.tick_params(labelsize=6)

    axis.legend(fontsize=5, ncol=2, loc="upper right")

    axis.grid(alpha=0.2, linewidth=0.4)

    figure.tight_layout(pad=0.5)

    buffer = BytesIO()

    chart_canvas.print_png(buffer)

    buffer.seek(0)

    figure.clear()

    return buffer


# =========================================================
# IMAGE HELPERS
# =========================================================

def photo_asset(path):
    """Load a photo once, downsample, keep as JPEG bytes."""

    with PILImage.open(path) as im:

        im = im.convert("RGB")

        im.thumbnail((1100, 1100))

        buf = BytesIO()

        im.save(buf, "JPEG", quality=88)

        return buf.getvalue(), im.width, im.height


def chart_asset(buffer):

    data = buffer.getvalue()

    with PILImage.open(BytesIO(data)) as im:

        return data, im.width, im.height


def picture(asset, k, max_h):
    """Column-width image (times k) that keeps its aspect ratio."""

    data, w_px, h_px = asset

    width = COL_W
    height = width * h_px / w_px

    if height > max_h:

        height = max_h
        width = height * w_px / h_px

    img = Image(BytesIO(data), width=width * k, height=height * k)

    img.hAlign = "CENTER"

    return img


def results_table(results, S):

    head = lambda t: Paragraph(t, S["tablehead"])

    data = [[
        head("Algorithm"),
        head("Time<br/>(ms)"),
        head("Bright."),
        head("Contrast"),
        head("Entropy"),
        head("Range"),
    ]]

    for result in results:

        m = result["enhanced_metrics"]

        data.append([
            algorithm_name(result["algorithm"]),
            number(result["processing_time_ms"]),
            number(m["brightness"]),
            number(m["contrast"]),
            number(m["entropy"], 3),
            safe_text(m["dynamic_range"]),
        ])

    widths = [COL_W * f for f in (0.35, 0.11, 0.13, 0.14, 0.14, 0.13)]

    table = Table(data, colWidths=widths)

    table.setStyle(TableStyle([
        ("FONTNAME", (0, 1), (-1, -1), F_REG),
        ("FONTSIZE", (0, 1), (-1, -1), 7),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))

    return table


# =========================================================
# DOCUMENT (two pages, two columns, continuous flow)
# =========================================================

ABSTRACT = (
    "<b>Abstract—</b> This report presents a measured performance "
    "evaluation of four deterministic, non-AI low-light image "
    "enhancement methods implemented in ATHER: Gamma Correction, "
    "Contrast Stretching, Histogram Equalization, and Contrast "
    "Limited Adaptive Histogram Equalization (CLAHE). The algorithms "
    "are compared using measured execution time and image statistics, "
    "including mean brightness, contrast, entropy, and dynamic range. "
    "Visual and histogram comparisons are included when the "
    "corresponding image files are provided. Results describe the "
    "tested image and computing environment and are not presented as "
    "universal rankings."
)

TITLE = (
    "Performance Evaluation of Classical Low-Light "
    "Image Enhancement Algorithms"
)


def draw_header(pdf, S):
    """Full-width title block. Returns the y where the columns start."""

    y = PAGE_HEIGHT - TOP

    y = draw_paragraph(pdf, TITLE, LEFT, y, CONTENT_WIDTH, S["title"])

    y = draw_paragraph(
        pdf,
        "ATHER Research System<br/>"
        "Classical Image Processing and Analysis",
        LEFT, y - 1, CONTENT_WIDTH, S["author"],
    )

    y = draw_paragraph(
        pdf, ABSTRACT, LEFT, y - 2, CONTENT_WIDTH, S["abstract"]
    )

    return y - 5


def make_doc(target, S, header_bottom):

    doc = BaseDocTemplate(
        target,
        pagesize=A4,
        leftMargin=LEFT,
        rightMargin=RIGHT,
        topMargin=TOP,
        bottomMargin=BOTTOM,
        title=TITLE,
        author="ATHER Research System",
        subject=(
            "Measured evaluation of classical low-light "
            "image enhancement algorithms"
        ),
    )

    kw = dict(
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0
    )

    h1 = header_bottom - BOTTOM
    h2 = PAGE_HEIGHT - TOP - BOTTOM

    first = [
        Frame(COL_X[0], BOTTOM, COL_W, h1, id="p1c1", **kw),
        Frame(COL_X[1], BOTTOM, COL_W, h1, id="p1c2", **kw),
    ]

    later = [
        Frame(COL_X[0], BOTTOM, COL_W, h2, id="p2c1", **kw),
        Frame(COL_X[1], BOTTOM, COL_W, h2, id="p2c2", **kw),
    ]

    def on_first(pdf, d):
        draw_header(pdf, S)
        draw_page_number(pdf, d.page)

    def on_later(pdf, d):
        draw_page_number(pdf, d.page)

    doc.addPageTemplates([
        PageTemplate(id="first", frames=first, onPage=on_first),
        PageTemplate(id="later", frames=later, onPage=on_later),
    ])

    return doc


def build_story(k, S, ctx):

    P = lambda t, s="body": Paragraph(t, S[s])
    H = lambda n, t: Paragraph(f"{n}. {t.upper()}", S["heading"])
    SUB = lambda t: Paragraph(t, S["subhead"])
    FORM = lambda t: Paragraph(t, S["formula"])

    story = [

        NextPageTemplate("later"),

        P(
            "<b>Index Terms—</b> Low-light image enhancement, gamma "
            "correction, contrast stretching, histogram equalization, "
            "CLAHE, image quality assessment, classical image "
            "processing.",
            "small",
        ),

        HRFlowable(
            width="100%", thickness=0.5, color=colors.black,
            spaceBefore=2, spaceAfter=1,
        ),

        # ---------------- I ----------------
        H("I", "Introduction"),

        P(
            "Low-light images can exhibit low mean intensity, weak "
            "local contrast, and reduced visibility of scene details. "
            "Classical image processing methods modify pixel "
            "intensities or local histogram distributions to improve "
            "visibility without training data or neural networks. "
            "Recent studies continue to investigate histogram-based, "
            "gamma-based, and local-contrast enhancement strategies "
            "[1]–[5]. ATHER evaluates four classical methods using "
            "the same input image and records the resulting image "
            "statistics and processing time."
        ),

        # ---------------- II ----------------
        H("II", "Mathematical Foundations"),

        SUB("A. Gamma Correction"),
        P("Gamma correction applies a nonlinear intensity mapping:"),
        FORM(f"s = 255 (r / 255)<super>{GAMMA}</super>"),
        P(
            "where r is the input intensity, s is the output "
            f"intensity, and {GAMMA} controls the nonlinear "
            "brightness mapping."
        ),

        SUB("B. Contrast Stretching"),
        P("A min–max intensity mapping is expressed as:"),
        FORM(
            f"s = 255 × (r {MINUS} r<sub>min</sub>) / "
            f"(r<sub>max</sub> {MINUS} r<sub>min</sub>)"
        ),
        P(
            "where r<sub>min</sub> and r<sub>max</sub> represent the "
            "selected intensity limits."
        ),

        SUB("C. Histogram Equalization"),
        P(
            "Histogram equalization maps intensity levels using the "
            "cumulative distribution function:"
        ),
        FORM(
            f"s<sub>k</sub> = round[(L {MINUS} 1) "
            "CDF(r<sub>k</sub>)]"
        ),
        P("where L is the number of intensity levels."),

        SUB("D. Contrast Limited Adaptive Histogram Equalization"),
        P(
            "CLAHE applies local histogram equalization to contextual "
            "regions and clips histogram excess to limit "
            "over-amplification. Interpolation between neighboring "
            "regions helps reduce visible transitions. Recent work "
            "examines adaptive histogram redistribution and "
            "local-contrast enhancement [2], [3], [5]."
        ),

        # ---------------- III ----------------
        H("III", "Experimental Methodology"),

        P(ctx["methodology"]),
    ]

    if ctx["parameter_text"]:
        story.append(P(ctx["parameter_text"], "small"))

    story += [

        # ---------------- IV ----------------
        H("IV", "Performance Metrics"),

        P(
            "<b>Mean brightness:</b> the average grayscale intensity "
            "of the image. <b>Contrast:</b> represented by the "
            "standard deviation of grayscale intensity. "
            "<b>Entropy:</b> describes the distribution of intensity "
            "information. <b>Dynamic range:</b> the difference "
            "between maximum and minimum intensity. "
            "<b>Processing time:</b> the measured execution duration "
            "in milliseconds. Any noise estimate supplied by the "
            "benchmark is approximate and is not a ground-truth "
            "noise measurement."
        ),

        P(
            "PSNR and SSIM are not reported unless a valid, aligned "
            "ground-truth reference image is available. This avoids "
            "presenting unsupported reference-based quality "
            "measurements."
        ),

        # ---------------- V ----------------
        KeepTogether([
            H("V", "Experimental Results"),
            P("TABLE I. MEASURED ENHANCEMENT RESULTS", "tablecap"),
            results_table(ctx["results"], S),
        ]),

        Paragraph(
            "The table reports the measured values returned by the "
            "ATHER benchmark for the selected input. The values are "
            "specific to this experiment and the computing "
            "environment used during execution.",
            ParagraphStyle(
                "afterTable", parent=S["body"], spaceBefore=3
            ),
        ),
    ]

    # ---------------- VI (photos stacked) ----------------
    photos = ctx["photos"]

    for index, (label, asset) in enumerate(photos):

        block = []

        if index == 0:
            block.append(H("VI", "Visual Comparison"))

        block.append(picture(asset, k, PHOTO_MAX_H))
        block.append(P(label, "caption"))

        if index == len(photos) - 1:
            block.append(P(
                "Fig. 1. Original image and enhanced outputs "
                "generated by the evaluated algorithms.",
                "caption",
            ))

        story.append(KeepTogether(block))

    if ctx["histogram"]:

        story.append(KeepTogether([
            SUB("Histogram Comparison"),
            picture(ctx["histogram"], k, 99 * inch),
            P(
                "Fig. 2. Grayscale intensity distributions of the "
                "original and enhanced images.",
                "caption",
            ),
        ]))

    # ---------------- VII (graphs stacked) ----------------
    charts = ctx["charts"]

    for index, asset in enumerate(charts):

        block = []

        if index == 0:
            block.append(H("VII", "Graphical Performance Comparison"))

        block.append(picture(asset, k, 99 * inch))

        if index == len(charts) - 1:
            block.append(P(
                "Fig. 3. Measured brightness, contrast, entropy, and "
                "processing-time comparisons.",
                "caption",
            ))

        story.append(KeepTogether(block))

    story += [

        H("VIII", "Discussion"),

        P(
            "The measured results characterize the response of the "
            "four algorithms on the selected input image. "
            "Brightness, contrast, entropy, and dynamic range "
            "describe different aspects of the output and should be "
            "interpreted together with visual inspection. Processing "
            "time indicates computational cost on the tested system. "
            "A larger value for an individual metric does not by "
            "itself establish better perceptual quality. The results "
            "therefore support a per-image comparison rather than a "
            "universal ranking of methods."
        ),

        H("IX", "Limitations"),

        P(
            "The reported values depend on the input image, image "
            "dimensions, implementation settings, and computing "
            "environment. Execution time can vary with processor "
            "load and repeated runs. Noise estimation, when "
            "available, is approximate. Reference-based metrics such "
            "as PSNR and SSIM are intentionally omitted when no "
            "valid aligned ground-truth image is provided. The "
            "experiment does not establish that one method is best "
            "for every scene."
        ),

        P("<b>Recorded environment:</b>", "small"),
    ]

    for line in ctx["environment_lines"]:
        story.append(P(line, "small"))

    story += [

        H("X", "Conclusion"),

        P(
            "This report evaluated Gamma Correction, Contrast "
            "Stretching, Histogram Equalization, and CLAHE using "
            "measured image statistics and processing time. The "
            "report presents the benchmark table, visual comparison, "
            "histogram, and comparative graphs in a two-column "
            "technical format. Broader conclusions require testing "
            "across a larger set of low-light images and, where "
            "available, aligned reference images for reference-based "
            "quality assessment."
        ),

        H("XI", "References"),
    ]

    for index, reference in enumerate(REFERENCES, start=1):

        story.append(P(f"[{index}] {safe_text(reference)}", "refs"))

    return story


def pages_needed(size, lead, k, ctx):
    """Build to memory and report how many pages the story needs."""

    S = make_styles(size, lead)

    dummy = canvas.Canvas(BytesIO(), pagesize=A4)

    header_bottom = draw_header(dummy, S)

    doc = make_doc(BytesIO(), S, header_bottom)

    doc.build(build_story(k, S, ctx))

    return doc.page


def choose_layout(ctx):
    """
    Pick the text size and image scale. Text size is identical on both
    pages; only the images shrink until everything fits on two pages.
    """

    fallback = None

    for size, lead in TEXT_SIZES:

        # coarse scan, then refine upward in 0.01 steps
        k = 1.0
        found = None

        while k >= MIN_IMAGE_SCALE:

            if pages_needed(size, lead, k, ctx) <= 2:

                found = k
                break

            k = round(k - 0.04, 3)

        if found is None:
            continue

        for extra in (0.03, 0.02, 0.01):

            trial = round(found + extra, 3)

            if trial <= 1.0 and pages_needed(size, lead, trial, ctx) <= 2:

                found = trial
                break

        if fallback is None:
            fallback = (size, lead, found)

        if found >= GOOD_IMAGE_SCALE:
            return size, lead, found

    if fallback:
        return fallback

    raise RuntimeError(
        "The report cannot fit on two pages even with the smallest "
        "text and images. Shorten the supplied parameter/environment "
        "text or reduce the number of images."
    )


# =========================================================
# MAIN PUBLIC FUNCTION
# =========================================================

def generate_ieee_report(
    results,
    output_path,
    image_name="Input Image",
    gamma_value=0.6,
    original_image_path=None,
    enhanced_image_paths=None,
    algorithm_parameters=None,
    system_info=None,
):
    """
    Generate a two-page A4 IEEE-style PDF.

    Compatible with the existing ATHER main.py function call.
    """

    results = validate_results(results)

    enhanced_image_paths = enhanced_image_paths or {}
    algorithm_parameters = algorithm_parameters or {}
    system_info = system_info or {}

    output_path = Path(output_path)

    if output_path.suffix.lower() != ".pdf":
        output_path = output_path.with_suffix(".pdf")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- text used in the story ----------

    dimension_text = "not available"

    if original_image_path and Path(original_image_path).is_file():

        try:

            with PILImage.open(original_image_path) as im:

                dimension_text = (
                    f"{im.width} × {im.height} pixels and "
                    f"{im.width * im.height} total pixels"
                )

        except Exception:
            pass

    methodology = (
        f"The evaluated input was <b>{safe_text(image_name)}</b>, "
        f"with dimensions {dimension_text}. The gamma parameter "
        f"supplied to the report was {number(gamma_value, 3)}. Each "
        "algorithm was evaluated against the same original image. "
        "Execution time was recorded by the ATHER benchmark process. "
        "Image statistics were calculated for the original and "
        "enhanced outputs."
    )

    parameter_text = ""

    if algorithm_parameters:

        parts = []

        for key, value in algorithm_parameters.items():

            if isinstance(value, dict):

                value = ", ".join(f"{a}: {b}" for a, b in value.items())

            parts.append(f"{algorithm_name(key)}: {safe_text(value)}")

        parameter_text = "<b>Parameters:</b> " + "; ".join(parts) + "."

    environment = dict(system_info) or {
        "Operating System": platform.platform(),
        "Python Version": platform.python_version(),
        "Processor": platform.processor() or "N/A",
    }

    environment_lines = [
        f"{safe_text(a)}: {safe_text(b)}" for a, b in environment.items()
    ]

    # ---------- visual assets (loaded once) ----------

    labels = {
        "gamma": "(b) Gamma Correction",
        "contrast_stretching": "(c) Contrast Stretching",
        "histogram_equalization": "(d) Histogram Equalization",
        "clahe": "(e) CLAHE",
    }

    photos = []

    if original_image_path and Path(original_image_path).is_file():

        photos.append(("(a) Original", photo_asset(original_image_path)))

    for algorithm in ALGORITHM_ORDER:

        path = enhanced_image_paths.get(algorithm)

        if path and Path(path).is_file():

            photos.append((labels[algorithm], photo_asset(path)))

    histogram = create_histogram(original_image_path, enhanced_image_paths)

    charts = [
        chart_asset(
            create_chart(results, "Mean Brightness", "brightness")
        ),
        chart_asset(
            create_chart(results, "Contrast", "contrast")
        ),
        chart_asset(
            create_chart(results, "Entropy", "entropy")
        ),
        chart_asset(
            create_chart(results, "Processing Time", None, True)
        ),
    ]

    ctx = {
        "results": results,
        "methodology": methodology,
        "parameter_text": parameter_text,
        "environment_lines": environment_lines,
        "photos": photos,
        "histogram": chart_asset(histogram) if histogram else None,
        "charts": charts,
    }

    # ---------- fit, then write the real file ----------

    size, lead, k = choose_layout(ctx)

    S = make_styles(size, lead)

    dummy = canvas.Canvas(BytesIO(), pagesize=A4)

    header_bottom = draw_header(dummy, S)

    doc = make_doc(str(output_path), S, header_bottom)

    doc.build(build_story(k, S, ctx))

    if not output_path.is_file() or output_path.stat().st_size == 0:

        raise RuntimeError("PDF generation failed: output file is empty.")

    # exact two-page check (pip install pypdf)
    try:

        from pypdf import PdfReader

        count = len(PdfReader(str(output_path)).pages)

        if count != 2:

            raise RuntimeError(
                f"Expected exactly 2 pages, but generated {count}."
            )

    except ImportError:
        pass

    return str(output_path)


if __name__ == "__main__":

    print("ATHER Two-Page IEEE Report Generator")
    print("Generate reports through the ATHER application.")