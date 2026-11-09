"""
ATHER - Continuous IEEE-Style Performance Report Generator
-----------------------------------------------------------
Classical, non-AI image enhancement research report.

Features:
- Supports one or more of the nine registered algorithms.
- Continuous IEEE-style two-column layout.
- Automatically creates as many pages as required.
- Full-width title and abstract on the first page.
- Consistent typography and column alignment.
- Keeps headings with following content.
- Keeps images with their captions.
- Preserves image aspect ratios.
- Repeating table headers when tables split.
- No forced two-page compression.
- No fabricated benchmark measurements.
"""

from pathlib import Path
from io import BytesIO
import html
import os
import platform
import tempfile

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
    BaseDocTemplate,
    PageTemplate,
    Frame,
    Paragraph,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
    NextPageTemplate,
    Spacer,
)

from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


# =========================================================
# PAGE CONFIGURATION
# =========================================================

PAGE_WIDTH, PAGE_HEIGHT = A4

LEFT = 0.58 * inch
RIGHT = 0.58 * inch
TOP = 0.48 * inch
BOTTOM = 0.48 * inch

CONTENT_WIDTH = PAGE_WIDTH - LEFT - RIGHT

COLUMN_GAP = 0.24 * inch
COLUMN_WIDTH = (CONTENT_WIDTH - COLUMN_GAP) / 2

COLUMN_X = [
    LEFT,
    LEFT + COLUMN_WIDTH + COLUMN_GAP,
]

FIRST_PAGE_TOP_GAP = 0.12 * inch

IMAGE_MAX_HEIGHT = 1.65 * inch
CHART_MAX_HEIGHT = 1.45 * inch

IMAGE_QUALITY = 90
IMAGE_MAX_DIMENSION = 1800

PAGE_NUMBER_Y = 0.25 * inch

# Keep a little safety space above the footer.
FOOTER_SAFE_GAP = 0.10 * inch


# =========================================================
# ALGORITHM INFORMATION
# =========================================================

ALGORITHM_NAMES = {
    "gamma": "Gamma Correction",
    "contrast_stretching": "Contrast Stretching",
    "histogram_equalization": "Histogram Equalization",
    "clahe": "Contrast Limited Adaptive Histogram Equalization",
    "adaptive_gamma": "Adaptive Gamma Correction",
    "ssr": "Single-Scale Retinex",
    "msr": "Multi-Scale Retinex",
    "lime": "LIME-Style Illumination Enhancement",
    "homomorphic": "Homomorphic Filtering",
}

SHORT_NAMES = {
    "gamma": "Gamma",
    "contrast_stretching": "Contrast",
    "histogram_equalization": "Hist. Eq.",
    "clahe": "CLAHE",
    "adaptive_gamma": "Adaptive Gamma",
    "ssr": "SSR",
    "msr": "MSR",
    "lime": "LIME",
    "homomorphic": "Homomorphic",
}

ALGORITHM_ORDER = list(ALGORITHM_NAMES.keys())


ALGORITHM_EXPLANATIONS = {
    "gamma": (
        "Applies a nonlinear power-law intensity transformation. "
        "Values below one brighten darker intensity levels, while "
        "the gamma parameter controls transformation strength."
    ),

    "contrast_stretching": (
        "Expands a selected input intensity interval across the "
        "available output range. This increases global intensity "
        "spread and can improve visibility when the original image "
        "occupies a narrow intensity range."
    ),

    "histogram_equalization": (
        "Uses the cumulative intensity histogram to redistribute "
        "grayscale levels. It improves global contrast but may "
        "increase noise or alter local appearance."
    ),

    "clahe": (
        "Performs adaptive histogram equalization on local tiles. "
        "Histogram clipping limits excessive amplification, while "
        "interpolation reduces transitions between neighboring tiles."
    ),

    "adaptive_gamma": (
        "Adjusts the gamma transformation according to image "
        "brightness characteristics. The mapping adapts its "
        "correction strength to the input illumination."
    ),

    "ssr": (
        "Uses a single-scale Retinex model based on the logarithmic "
        "difference between image intensity and a Gaussian estimate "
        "of illumination."
    ),

    "msr": (
        "Combines Retinex responses at multiple Gaussian scales. "
        "The multiscale representation captures illumination "
        "variation at different spatial extents."
    ),

    "lime": (
        "Uses a simplified illumination-map estimation and "
        "refinement approach to brighten low-light regions. "
        "This implementation is LIME-style and does not claim "
        "to reproduce every operation of the original LIME method."
    ),

    "homomorphic": (
        "Applies logarithmic transformation, frequency-domain "
        "filtering and inverse transformation to adjust slowly "
        "varying illumination and fine image details."
    ),
}


ALGORITHM_FORMULAS = {
    "gamma": (
        "s = 255 × (r / 255)<super>γ</super>"
    ),

    "contrast_stretching": (
        "s = 255 × (r − r<sub>min</sub>) / "
        "(r<sub>max</sub> − r<sub>min</sub>)"
    ),

    "histogram_equalization": (
        "s<sub>k</sub> = round[(L − 1) × CDF(r<sub>k</sub>)]"
    ),

    "clahe": (
        "s = T<sub>local, clipped</sub>(r)"
    ),

    "adaptive_gamma": (
        "s = 255 × (r / 255)<super>γ(I)</super>"
    ),

    "ssr": (
        "R(x,y) = log I(x,y) − log[F(x,y) * I(x,y)]"
    ),

    "msr": (
        "R(x,y) = Σ w<sub>n</sub> "
        "[log I(x,y) − log(F<sub>n</sub> * I)]"
    ),

    "lime": (
        "J(x,y) ≈ I(x,y) / max(T(x,y), ε)"
    ),

    "homomorphic": (
        "S(u,v) = H(u,v) × log[I(x,y)]"
    ),
}


# =========================================================
# REFERENCES
# =========================================================

REFERENCES = [
    (
        "S. Khormaeipour and F. Shakeri, "
        "\"Enhancing low-light color image visibility with hybrid "
        "contrast and saturation modification using a saturation-aware "
        "map,\" Journal of Visual Communication and Image Representation, "
        "vol. 107, Art. no. 104392, Mar. 2025, "
        "doi: 10.1016/j.jvcir.2025.104392."
    ),

    (
        "S.-H. Jin, D.-M. Son, S.-H. Lee, Y.-H. Go, and S.-H. Lee, "
        "\"Enhancing local contrast in low-light images: A multiscale "
        "model with adaptive redistribution of histogram excess,\" "
        "Mathematics, vol. 13, no. 20, Art. no. 3282, Oct. 2025, "
        "doi: 10.3390/math13203282."
    ),

    (
        "W. Deng and G. Xie, \"Image contrast enhancement and brightness "
        "preservation based on an adaptive histogram correction "
        "framework,\" Applied Optics, vol. 64, no. 13, pp. 3502–3515, "
        "Apr. 2025, doi: 10.1364/AO.557280."
    ),

    (
        "M. M. Tatana, M. S. Tsoeu, and R. C. Maswanganyi, "
        "\"Low-light image and video enhancement for more robust "
        "computer vision tasks: A review,\" Journal of Imaging, "
        "vol. 11, no. 4, Art. no. 125, Apr. 2025, "
        "doi: 10.3390/jimaging11040125."
    ),

    (
        "D. Mu, Z. Wang, Z. Li, et al., \"A new multi-scale adaptive "
        "enhancement algorithm for low-illumination coal mine images "
        "based on improved CLAHE and ACE,\" Discover Artificial "
        "Intelligence, vol. 5, Art. no. 406, Dec. 2025, "
        "doi: 10.1007/s44163-025-00663-5."
    ),
]


TITLE = (
    "Performance Evaluation of Classical Low-Light "
    "Image Enhancement Algorithms"
)


# =========================================================
# FONT REGISTRATION
# =========================================================

def _register_fonts():

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
            "/usr/share/fonts/truetype/liberation/"
            "LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/"
            "LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/"
            "LiberationSerif-Italic.ttf",
            "/usr/share/fonts/truetype/liberation/"
            "LiberationSerif-BoldItalic.ttf",
        ),
    ]

    names = (
        "ATHERSerif",
        "ATHERSerif-Bold",
        "ATHERSerif-Italic",
        "ATHERSerif-BoldItalic",
    )

    for files in candidates:

        if not all(Path(path).is_file() for path in files):
            continue

        try:

            for name, path in zip(names, files):

                if name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(
                        TTFont(name, path)
                    )

            addMapping("ATHERSerif", 0, 0, names[0])
            addMapping("ATHERSerif", 1, 0, names[1])
            addMapping("ATHERSerif", 0, 1, names[2])
            addMapping("ATHERSerif", 1, 1, names[3])

            return names

        except Exception:
            continue

    return (
        "Times-Roman",
        "Times-Bold",
        "Times-Italic",
        "Times-BoldItalic",
    )


F_REG, F_BOLD, F_ITAL, F_BI = _register_fonts()


# =========================================================
# TEXT AND NUMBER HELPERS
# =========================================================

def safe_text(value):

    if value is None:
        return "N/A"

    return html.escape(
        str(value)
    ).replace("\n", " ")


def number(value, decimals=2):

    try:

        value = float(value)

        if not np.isfinite(value):
            return "N/A"

        return f"{value:.{decimals}f}"

    except (TypeError, ValueError, OverflowError):
        return "N/A"


def numeric_value(value, field_name):

    try:
        result = float(value)

    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"Invalid numeric value for {field_name}: {value}"
        )

    if not np.isfinite(result):
        raise ValueError(
            f"Non-finite numeric value for {field_name}."
        )

    return result


def algorithm_name(key):

    return ALGORITHM_NAMES.get(
        key,
        str(key).replace("_", " ").title()
    )


def valid_image_path(path):

    if not path:
        return False

    try:
        return Path(path).is_file()

    except (TypeError, ValueError, OSError):
        return False


# =========================================================
# STYLES
# =========================================================

def make_styles():

    return {

        "title": ParagraphStyle(
            "ATHER_Title",
            fontName=F_BOLD,
            fontSize=18,
            leading=21,
            alignment=TA_CENTER,
            spaceAfter=5,
        ),

        "author": ParagraphStyle(
            "ATHER_Author",
            fontName=F_REG,
            fontSize=9.5,
            leading=11.5,
            alignment=TA_CENTER,
            spaceAfter=3,
        ),

        "abstract": ParagraphStyle(
            "ATHER_Abstract",
            fontName=F_REG,
            fontSize=9,
            leading=11,
            alignment=TA_JUSTIFY,
            spaceAfter=0,
        ),

        "heading": ParagraphStyle(
            "ATHER_Heading",
            fontName=F_BOLD,
            fontSize=11,
            leading=13,
            alignment=TA_LEFT,
            spaceBefore=5,
            spaceAfter=3,
            keepWithNext=True,
            keepTogether=True,
            allowWidows=0,
            allowOrphans=0,
        ),

        "subhead": ParagraphStyle(
            "ATHER_Subhead",
            fontName=F_BI,
            fontSize=10,
            leading=12,
            alignment=TA_LEFT,
            spaceBefore=3,
            spaceAfter=2,
            keepWithNext=True,
            allowWidows=0,
            allowOrphans=0,
        ),

        "body": ParagraphStyle(
            "ATHER_Body",
            fontName=F_REG,
            fontSize=9.5,
            leading=11.5,
            alignment=TA_JUSTIFY,
            spaceAfter=4,
            allowWidows=0,
            allowOrphans=0,
        ),

        "formula": ParagraphStyle(
            "ATHER_Formula",
            fontName=F_BOLD,
            fontSize=9,
            leading=11,
            alignment=TA_LEFT,
            spaceBefore=2,
            spaceAfter=3,
            allowWidows=0,
            allowOrphans=0,
        ),

        "small": ParagraphStyle(
            "ATHER_Small",
            fontName=F_REG,
            fontSize=8.7,
            leading=10.5,
            alignment=TA_JUSTIFY,
            spaceAfter=3,
            allowWidows=0,
            allowOrphans=0,
        ),

        "caption": ParagraphStyle(
            "ATHER_Caption",
            fontName=F_ITAL,
            fontSize=8.7,
            leading=10.5,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=4,
        ),

        "tablecap": ParagraphStyle(
            "ATHER_TableCaption",
            fontName=F_REG,
            fontSize=8.5,
            leading=10,
            alignment=TA_CENTER,
            spaceBefore=2,
            spaceAfter=4,
            keepWithNext=True,
        ),

        "refs": ParagraphStyle(
            "ATHER_References",
            fontName=F_REG,
            fontSize=8.5,
            leading=10.2,
            alignment=TA_JUSTIFY,
            spaceAfter=4,
            allowWidows=0,
            allowOrphans=0,
        ),

        "tablehead": ParagraphStyle(
            "ATHER_TableHead",
            fontName=F_BOLD,
            fontSize=8,
            leading=9.5,
            alignment=TA_CENTER,
        ),

        "tablecell": ParagraphStyle(
            "ATHER_TableCell",
            fontName=F_REG,
            fontSize=8,
            leading=9.5,
            alignment=TA_CENTER,
        ),
    }


# =========================================================
# RESULT VALIDATION
# =========================================================

def validate_results(results):

    if not results:
        raise ValueError(
            "No benchmark results were supplied. "
            "Run benchmarks before generating a report."
        )

    lookup = {}

    for result in results:

        if not isinstance(result, dict):
            raise ValueError(
                "A benchmark result is not a dictionary."
            )

        key = result.get("algorithm")

        if not key:
            raise ValueError(
                "A benchmark result has no algorithm key."
            )

        if key not in ALGORITHM_NAMES:
            raise ValueError(
                f"Unsupported algorithm: {key}"
            )

        if key in lookup:
            raise ValueError(
                f"Duplicate result for algorithm: {key}"
            )

        if "processing_time_ms" not in result:
            raise ValueError(
                f"Missing processing time for {key}."
            )

        numeric_value(
            result["processing_time_ms"],
            f"{key}.processing_time_ms"
        )

        metrics = result.get("enhanced_metrics")

        if not isinstance(metrics, dict):
            raise ValueError(
                f"Missing enhanced_metrics for {key}."
            )

        for metric in (
            "brightness",
            "contrast",
            "entropy",
            "dynamic_range",
        ):

            if metric not in metrics:
                raise ValueError(
                    f"Missing {metric} for {key}."
                )

            numeric_value(
                metrics[metric],
                f"{key}.{metric}"
            )

        lookup[key] = result

    return [
        lookup[key]
        for key in ALGORITHM_ORDER
        if key in lookup
    ]


# =========================================================
# CHART GENERATION
# =========================================================

def create_chart(
    results,
    title,
    metric_key=None,
    processing_time=False
):

    labels = [
        SHORT_NAMES[result["algorithm"]]
        for result in results
    ]

    if processing_time:

        values = [
            numeric_value(
                result["processing_time_ms"],
                "processing_time_ms"
            )
            for result in results
        ]

        ylabel = "Milliseconds"

    else:

        values = [
            numeric_value(
                result["enhanced_metrics"][metric_key],
                metric_key
            )
            for result in results
        ]

        ylabel = {
            "brightness": "Mean intensity",
            "contrast": "Standard deviation",
            "entropy": "Bits",
        }.get(metric_key, metric_key)

    figure = Figure(
        figsize=(5.0, 2.0),
        dpi=180
    )

    chart_canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)

    bars = axis.bar(
        range(len(labels)),
        values,
        width=0.62
    )

    axis.set_title(
        title,
        fontsize=9,
        pad=5
    )

    axis.set_ylabel(
        ylabel,
        fontsize=8
    )

    axis.set_xticks(range(len(labels)))

    axis.set_xticklabels(
        labels,
        fontsize=7
    )

    axis.tick_params(
        axis="y",
        labelsize=7
    )

    axis.grid(
        axis="y",
        alpha=0.2,
        linewidth=0.4
    )

    axis.set_axisbelow(True)

    minimum = min(values) if values else 0
    maximum = max(values) if values else 1

    # Safe axis limits for positive, negative and zero values.
    if minimum < 0:

        lower = minimum * 1.2

        upper = (
            maximum * 1.2
            if maximum > 0
            else maximum + max(abs(maximum) * 0.2, 1)
        )

    else:

        lower = 0
        upper = maximum * 1.2 if maximum > 0 else 1

    if upper <= lower:
        upper = lower + 1

    axis.set_ylim(lower, upper)

    for bar, value in zip(bars, values):

        if value >= 0:
            offset = 3
            vertical_alignment = "bottom"
        else:
            offset = -3
            vertical_alignment = "top"

        axis.annotate(
            number(value),
            (
                bar.get_x() + bar.get_width() / 2,
                bar.get_height()
            ),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=vertical_alignment,
            fontsize=6.5
        )

    figure.tight_layout(pad=0.8)

    buffer = BytesIO()

    chart_canvas.print_png(buffer)

    buffer.seek(0)
    figure.clear()

    return buffer


def create_histogram(
    original_path,
    enhanced_paths,
    results
):

    if not valid_image_path(original_path):
        return None

    figure = Figure(
        figsize=(5.0, 2.0),
        dpi=180
    )

    chart_canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)

    paths = [
        ("Original", original_path)
    ]

    for result in results:

        key = result["algorithm"]
        path = enhanced_paths.get(key)

        if valid_image_path(path):

            paths.append((
                SHORT_NAMES.get(key, key),
                path
            ))

    plotted = 0

    for label, path in paths:

        try:

            with PILImage.open(path) as image:

                array = np.asarray(
                    image.convert("L")
                )

            axis.hist(
                array.ravel(),
                bins=64,
                range=(0, 255),
                density=True,
                histtype="step",
                linewidth=0.9,
                label=label
            )

            plotted += 1

        except Exception:
            continue

    if plotted == 0:
        figure.clear()
        return None

    axis.set_title(
        "Grayscale Intensity Distribution",
        fontsize=9,
        pad=5
    )

    axis.set_xlabel(
        "Intensity",
        fontsize=8
    )

    axis.set_ylabel(
        "Density",
        fontsize=8
    )

    axis.tick_params(labelsize=7)

    axis.legend(
        fontsize=6,
        ncol=3,
        loc="upper right"
    )

    axis.grid(
        alpha=0.2,
        linewidth=0.4
    )

    figure.tight_layout(pad=0.8)

    buffer = BytesIO()

    chart_canvas.print_png(buffer)

    buffer.seek(0)
    figure.clear()

    return buffer


# =========================================================
# IMAGE ASSETS
# =========================================================

def photo_asset(path):

    if not valid_image_path(path):
        raise FileNotFoundError(
            f"Image file does not exist: {path}"
        )

    with PILImage.open(path) as source:

        image = source.convert("RGB")

        # Downsampling is only for PDF size.
        # Aspect ratio is always preserved.
        image.thumbnail(
            (
                IMAGE_MAX_DIMENSION,
                IMAGE_MAX_DIMENSION
            ),
            PILImage.Resampling.LANCZOS
        )

        buffer = BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=IMAGE_QUALITY,
            optimize=True
        )

        return (
            buffer.getvalue(),
            image.width,
            image.height
        )


def chart_asset(buffer):

    data = buffer.getvalue()

    with PILImage.open(BytesIO(data)) as image:

        return (
            data,
            image.width,
            image.height
        )


def picture(asset, max_height):

    data, width_px, height_px = asset

    if width_px <= 0 or height_px <= 0:
        raise ValueError(
            "Invalid image dimensions in report."
        )

    # The same scale factor is used for width and height.
    # This prevents image stretching and distortion.
    scale = min(
        COLUMN_WIDTH / width_px,
        max_height / height_px
    )

    display_width = width_px * scale
    display_height = height_px * scale

    image = Image(
        BytesIO(data),
        width=display_width,
        height=display_height,
        kind="proportional"
    )

    image.hAlign = "CENTER"

    return image


# =========================================================
# IMAGE AND CAPTION GROUP
# =========================================================

def image_block(
    asset,
    caption,
    max_height,
    styles
):

    image = picture(
        asset,
        max_height
    )

    caption_paragraph = Paragraph(
        safe_text(caption),
        styles["caption"]
    )

    # Image and caption move together to the next frame
    # when they cannot fit in the remaining column space.
    return KeepTogether([

        image,

        caption_paragraph,

        Spacer(1, 3),

    ])


# =========================================================
# RESULTS TABLE
# =========================================================

def results_table(results, styles):

    def cell(value):

        return Paragraph(
            safe_text(value),
            styles["tablecell"]
        )

    def table_heading(value):

        return Paragraph(
            value,
            styles["tablehead"]
        )

    data = [[

        table_heading("Algorithm"),

        table_heading("Time<br/>(ms)"),

        table_heading("Brightness"),

        table_heading("Contrast"),

        table_heading("Entropy"),

        table_heading("Range"),

    ]]

    for result in results:

        metrics = result["enhanced_metrics"]

        data.append([

            cell(SHORT_NAMES[result["algorithm"]]),

            cell(number(result["processing_time_ms"])),

            cell(number(metrics["brightness"])),

            cell(number(metrics["contrast"])),

            cell(number(metrics["entropy"], 3)),

            cell(number(metrics["dynamic_range"], 2)),

        ])

    widths = [

        COLUMN_WIDTH * 0.27,

        COLUMN_WIDTH * 0.14,

        COLUMN_WIDTH * 0.16,

        COLUMN_WIDTH * 0.16,

        COLUMN_WIDTH * 0.14,

        COLUMN_WIDTH * 0.13,

    ]

    table = Table(
        data,
        colWidths=widths,
        repeatRows=1,
        hAlign="CENTER",
        splitByRow=1,
        spaceBefore=0,
        spaceAfter=0
    )

    table.setStyle(TableStyle([

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("ALIGN", (0, 0), (-1, -1), "CENTER"),

        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),

        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),

        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),

        ("LEFTPADDING", (0, 0), (-1, -1), 3),

        ("RIGHTPADDING", (0, 0), (-1, -1), 3),

        ("TOPPADDING", (0, 0), (-1, -1), 4),

        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),

    ]))

    return table


# =========================================================
# ABSTRACT
# =========================================================

def create_abstract(results):

    names = ", ".join(
        algorithm_name(result["algorithm"])
        for result in results
    )

    return (
        "<b>Abstract—</b> This report presents a measured "
        "evaluation of selected deterministic classical image "
        "enhancement algorithms implemented in ATHER. The selected "
        f"methods are {safe_text(names)}. Each method is evaluated "
        "using the same input image. Processing time, mean brightness, "
        "contrast, entropy and dynamic range are reported using actual "
        "benchmark outputs. Visual and histogram comparisons are "
        "included when image assets are supplied. Results describe "
        "the tested input and computing environment and do not "
        "represent universal algorithm rankings."
    )


# =========================================================
# PAGE HEADER
# =========================================================

def draw_paragraph(
    pdf,
    text,
    x,
    y,
    width,
    style
):

    paragraph = Paragraph(
        text,
        style
    )

    _, height = paragraph.wrap(
        width,
        PAGE_HEIGHT
    )

    paragraph.drawOn(
        pdf,
        x,
        y - height
    )

    return y - height


def draw_header(pdf, styles, abstract):

    y = PAGE_HEIGHT - TOP

    y = draw_paragraph(
        pdf,
        TITLE,
        LEFT,
        y,
        CONTENT_WIDTH,
        styles["title"]
    )

    y = draw_paragraph(
        pdf,
        "ATHER Research System<br/>"
        "Classical Image Processing and Analysis",
        LEFT,
        y - 2,
        CONTENT_WIDTH,
        styles["author"]
    )

    y = draw_paragraph(
        pdf,
        abstract,
        LEFT,
        y - 3,
        CONTENT_WIDTH,
        styles["abstract"]
    )

    y -= 5

    pdf.saveState()

    pdf.setStrokeColor(colors.black)
    pdf.setLineWidth(0.5)

    pdf.line(
        LEFT,
        y,
        PAGE_WIDTH - RIGHT,
        y
    )

    pdf.restoreState()

    return y - FIRST_PAGE_TOP_GAP


# =========================================================
# PAGE NUMBER
# =========================================================

def draw_page_number(pdf, document):

    pdf.saveState()

    pdf.setFont(F_REG, 8)
    pdf.setFillColor(colors.black)

    pdf.drawCentredString(
        PAGE_WIDTH / 2,
        PAGE_NUMBER_Y,
        str(document.page)
    )

    pdf.restoreState()


# =========================================================
# DOCUMENT AND COLUMN FLOW
# =========================================================

def make_doc(
    target,
    styles,
    first_page_header_bottom
):

    document = BaseDocTemplate(
        target,
        pagesize=A4,
        leftMargin=LEFT,
        rightMargin=RIGHT,
        topMargin=TOP,
        bottomMargin=BOTTOM,
        title=TITLE,
        author="ATHER Research System",
        subject=(
            "Measured evaluation of classical "
            "image enhancement algorithms"
        ),
        pageCompression=1
    )

    frame_options = {

        "leftPadding": 0,

        "rightPadding": 0,

        "topPadding": 0,

        "bottomPadding": 0,

        "showBoundary": 0,

    }

    first_height = (
        first_page_header_bottom
        - BOTTOM
        - FOOTER_SAFE_GAP
    )

    later_height = (
        PAGE_HEIGHT
        - TOP
        - BOTTOM
        - FOOTER_SAFE_GAP
    )

    if first_height <= 0:

        raise RuntimeError(
            "The first-page header leaves no room for report content."
        )

    if later_height <= 0:

        raise RuntimeError(
            "Invalid page frame height."
        )

    first_frames = [

        Frame(
            COLUMN_X[0],
            BOTTOM + FOOTER_SAFE_GAP,
            COLUMN_WIDTH,
            first_height,
            id="first_left",
            **frame_options
        ),

        Frame(
            COLUMN_X[1],
            BOTTOM + FOOTER_SAFE_GAP,
            COLUMN_WIDTH,
            first_height,
            id="first_right",
            **frame_options
        ),

    ]

    later_frames = [

        Frame(
            COLUMN_X[0],
            BOTTOM + FOOTER_SAFE_GAP,
            COLUMN_WIDTH,
            later_height,
            id="later_left",
            **frame_options
        ),

        Frame(
            COLUMN_X[1],
            BOTTOM + FOOTER_SAFE_GAP,
            COLUMN_WIDTH,
            later_height,
            id="later_right",
            **frame_options
        ),

    ]

    def first_page(pdf, doc):

        # Full-width first-page title and abstract.
        draw_header(
            pdf,
            styles,
            doc.abstract_text
        )

        draw_page_number(
            pdf,
            doc
        )

    def later_page(pdf, doc):

        draw_page_number(
            pdf,
            doc
        )

    document.addPageTemplates([

        PageTemplate(
            id="first",
            frames=first_frames,
            onPage=first_page
        ),

        PageTemplate(
            id="later",
            frames=later_frames,
            onPage=later_page
        ),

    ])

    return document


# =========================================================
# SECTION HELPERS
# =========================================================

def heading(number_text, title, styles):

    return Paragraph(
        f"{number_text}. {safe_text(title).upper()}",
        styles["heading"]
    )


def paragraph(
    text,
    styles,
    style_name="body"
):

    return Paragraph(
        text,
        styles[style_name]
    )


def algorithm_block(index, result, styles):

    key = result["algorithm"]

    name = algorithm_name(key)

    explanation = ALGORITHM_EXPLANATIONS.get(
        key,
        "A registered classical image enhancement method."
    )

    formula = ALGORITHM_FORMULAS.get(
        key,
        "Refer to the implementation source for the applied transformation."
    )

    return KeepTogether([

        Paragraph(
            f"{index}. {safe_text(name)}",
            styles["subhead"]
        ),

        Paragraph(
            safe_text(explanation),
            styles["body"]
        ),

        Paragraph(
            formula,
            styles["formula"]
        ),

        Spacer(1, 2),

    ])


# =========================================================
# STORY GENERATION
# =========================================================

def build_story(styles, context):

    story = [

        # The first page keeps its full-width title and abstract.
        # The next page uses the regular two-column template.
        NextPageTemplate("later"),

        paragraph(
            "<b>Index Terms—</b> Low-light enhancement, "
            "classical image processing, histogram equalization, "
            "Retinex, illumination correction and image quality assessment.",
            styles,
            "small"
        ),

        HRFlowable(
            width="100%",
            thickness=0.45,
            color=colors.black,
            spaceBefore=2,
            spaceAfter=3
        ),

        heading("I", "Introduction", styles),

        paragraph(
            "Low-light images can exhibit low mean intensity, weak "
            "contrast and reduced visibility of scene details. "
            "Classical image-processing methods modify pixel "
            "intensities, local histograms or illumination estimates "
            "without neural networks or training data. This report "
            "compares the selected ATHER implementations using "
            "identical input data and measured output statistics.",
            styles
        ),

        heading(
            "II",
            "Selected Algorithms and Mathematical Methods",
            styles
        ),

    ]

    for index, result in enumerate(
        context["results"],
        start=1
    ):

        story.append(
            algorithm_block(
                index,
                result,
                styles
            )
        )

    story += [

        heading(
            "III",
            "Experimental Methodology",
            styles
        ),

        paragraph(
            context["methodology"],
            styles
        ),

    ]

    if context["parameter_text"]:

        story.append(
            paragraph(
                context["parameter_text"],
                styles,
                "small"
            )
        )

    story += [

        heading(
            "IV",
            "Performance Metrics",
            styles
        ),

        paragraph(
            "<b>Mean brightness:</b> average grayscale intensity. "
            "<b>Contrast:</b> standard deviation of grayscale "
            "intensity. <b>Entropy:</b> intensity-distribution "
            "information in bits. <b>Dynamic range:</b> maximum "
            "minus minimum intensity. <b>Processing time:</b> "
            "measured algorithm execution time in milliseconds. "
            "Any noise estimate is approximate and is not a "
            "ground-truth noise measurement.",
            styles
        ),

        paragraph(
            "PSNR and SSIM are omitted unless a valid aligned "
            "ground-truth reference image is available. No "
            "reference-based metric is fabricated.",
            styles,
            "small"
        ),

        heading(
            "V",
            "Experimental Results",
            styles
        ),

        KeepTogether([

            Paragraph(
                "TABLE I. MEASURED RESULTS FOR SELECTED ALGORITHMS",
                styles["tablecap"]
            ),

            results_table(
                context["results"],
                styles
            ),

            Spacer(1, 4),

        ]),

        paragraph(
            "Values are taken from the ATHER benchmark output "
            "for the selected image and computing environment.",
            styles,
            "small"
        ),

    ]

    # -----------------------------------------------------
    # VISUAL COMPARISON
    # -----------------------------------------------------

    if context["photos"]:

        story.append(
            heading(
                "VI",
                "Visual Comparison",
                styles
            )
        )

        for label, asset in context["photos"]:

            story.append(
                image_block(
                    asset,
                    label,
                    IMAGE_MAX_HEIGHT,
                    styles
                )
            )

    # -----------------------------------------------------
    # HISTOGRAM
    # -----------------------------------------------------

    if context["histogram"]:

        story.append(
            Paragraph(
                "Histogram Comparison",
                styles["subhead"]
            )
        )

        story.append(
            image_block(
                context["histogram"],
                "Grayscale intensity distributions for the original "
                "and selected enhanced images.",
                CHART_MAX_HEIGHT,
                styles
            )
        )

    # -----------------------------------------------------
    # PERFORMANCE CHARTS
    # -----------------------------------------------------

    story.append(
        heading(
            "VII",
            "Graphical Performance Comparison",
            styles
        )
    )

    for index, (label, asset) in enumerate(
        context["charts"],
        start=1
    ):

        story.append(
            image_block(
                asset,
                f"Fig. {index}. {label}.",
                CHART_MAX_HEIGHT,
                styles
            )
        )

    # -----------------------------------------------------
    # DISCUSSION
    # -----------------------------------------------------

    story += [

        heading(
            "VIII",
            "Discussion",
            styles
        ),

        paragraph(
            "The results characterize the selected algorithms on "
            "the tested input. Brightness, contrast, entropy and "
            "dynamic range describe different properties and must "
            "be interpreted together with visual inspection. "
            "Processing time indicates computational cost on the "
            "tested system. A higher value for an individual metric "
            "does not automatically imply better perceptual quality.",
            styles
        ),

        heading(
            "IX",
            "Limitations",
            styles
        ),

        paragraph(
            "Results depend on image content, image dimensions, "
            "implementation settings and system load. Processing "
            "times can vary between runs. The Retinex and LIME-style "
            "methods are evaluated as implemented in ATHER; their "
            "results should not be interpreted as proof of exact "
            "reproduction of every published variant. Reference-based "
            "metrics are not reported without aligned ground truth.",
            styles
        ),

        paragraph(
            "<b>Recorded environment:</b>",
            styles,
            "small"
        ),

    ]

    for line in context["environment_lines"]:

        story.append(
            paragraph(
                line,
                styles,
                "small"
            )
        )

    # -----------------------------------------------------
    # CONCLUSION
    # -----------------------------------------------------

    story += [

        heading(
            "X",
            "Conclusion",
            styles
        ),

        paragraph(
            "This report evaluated the selected classical "
            "enhancement methods using measured image statistics, "
            "execution time, visual outputs and histogram "
            "comparisons when available. The results provide a "
            "per-image experimental comparison rather than a "
            "universal ranking. Further evaluation across multiple "
            "images is required for broader conclusions.",
            styles
        ),

        heading(
            "XI",
            "References",
            styles
        ),

    ]

    for index, reference in enumerate(
        REFERENCES,
        start=1
    ):

        story.append(
            paragraph(
                f"[{index}] {safe_text(reference)}",
                styles,
                "refs"
            )
        )

    return story


# =========================================================
# INPUT IMAGE INFORMATION
# =========================================================

def get_image_dimensions(original_image_path):

    if not valid_image_path(original_image_path):
        return "not available"

    try:

        with PILImage.open(original_image_path) as image:

            return (
                f"{image.width} × {image.height} pixels "
                f"({image.width * image.height:,} pixels total)"
            )

    except Exception:
        return "not available"


# =========================================================
# PDF HEADER MEASUREMENT
# =========================================================

def calculate_header_bottom(styles, abstract):

    temporary_buffer = BytesIO()

    temporary_canvas = canvas.Canvas(
        temporary_buffer,
        pagesize=A4
    )

    header_bottom = draw_header(
        temporary_canvas,
        styles,
        abstract
    )

    temporary_canvas.save()

    return header_bottom


# =========================================================
# PDF OUTPUT VERIFICATION
# =========================================================

def verify_pdf(path):

    if (
        not path.is_file()
        or path.stat().st_size == 0
    ):

        raise RuntimeError(
            "PDF generation failed: output file is empty."
        )

    try:

        from pypdf import PdfReader

        reader = PdfReader(str(path))

        page_count = len(reader.pages)

        if page_count < 1:

            raise RuntimeError(
                "The generated PDF contains no pages."
            )

        return page_count

    except ImportError:

        return None


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
    Generate a continuous, multi-page, two-column A4 report.

    Supports any selection of one or more registered algorithms.

    The report automatically creates additional pages when needed.

    Existing main.py function signature is preserved.
    """

    # -----------------------------------------------------
    # VALIDATE INPUTS
    # -----------------------------------------------------

    results = validate_results(results)

    enhanced_image_paths = enhanced_image_paths or {}
    algorithm_parameters = algorithm_parameters or {}
    system_info = system_info or {}

    output_path = Path(output_path)

    if output_path.suffix.lower() != ".pdf":
        output_path = output_path.with_suffix(".pdf")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    styles = make_styles()

    # -----------------------------------------------------
    # IMAGE INFORMATION
    # -----------------------------------------------------

    dimension_text = get_image_dimensions(
        original_image_path
    )

    # -----------------------------------------------------
    # METHODOLOGY
    # -----------------------------------------------------

    selected_names = [
        algorithm_name(result["algorithm"])
        for result in results
    ]

    methodology = (
        f"The evaluated input was <b>{safe_text(image_name)}</b>, "
        f"with dimensions {safe_text(dimension_text)}. "
        f"The experiment selected {len(results)} algorithm(s): "
        f"{safe_text(', '.join(selected_names))}. "
        f"The gamma parameter supplied to the report was "
        f"{number(gamma_value, 3)}. Each selected algorithm was "
        "evaluated against the same original input image. "
        "Execution time was recorded by the ATHER benchmark "
        "process, and output statistics were obtained from "
        "the corresponding benchmark results."
    )

    # -----------------------------------------------------
    # ALGORITHM PARAMETERS
    # -----------------------------------------------------

    parameter_parts = []

    for key, value in algorithm_parameters.items():

        if isinstance(value, dict):

            value_text = ", ".join(
                f"{safe_text(name)}: {safe_text(item)}"
                for name, item in value.items()
            )

        else:

            value_text = safe_text(value)

        parameter_parts.append(
            f"{safe_text(algorithm_name(key))}: {value_text}"
        )

    parameter_text = ""

    if parameter_parts:

        parameter_text = (
            "<b>Algorithm parameters:</b> "
            + "; ".join(parameter_parts)
            + "."
        )

    # -----------------------------------------------------
    # SYSTEM ENVIRONMENT
    # -----------------------------------------------------

    environment = dict(system_info) or {

        "Operating System": platform.platform(),

        "Python Version": platform.python_version(),

        "Processor": platform.processor() or "N/A",

    }

    environment_lines = [

        f"{safe_text(key)}: {safe_text(value)}"

        for key, value in environment.items()

    ]

    # -----------------------------------------------------
    # ORIGINAL AND ENHANCED IMAGES
    # -----------------------------------------------------

    photos = []

    if valid_image_path(original_image_path):

        photos.append((
            "(a) Original image",
            photo_asset(original_image_path)
        ))

    for index, result in enumerate(
        results,
        start=1
    ):

        key = result["algorithm"]
        path = enhanced_image_paths.get(key)

        if valid_image_path(path):

            photos.append((
                f"({index}) {algorithm_name(key)}",
                photo_asset(path)
            ))

    # -----------------------------------------------------
    # HISTOGRAM
    # -----------------------------------------------------

    histogram_buffer = create_histogram(
        original_image_path,
        enhanced_image_paths,
        results
    )

    histogram = (
        chart_asset(histogram_buffer)
        if histogram_buffer is not None
        else None
    )

    # -----------------------------------------------------
    # PERFORMANCE CHARTS
    # -----------------------------------------------------

    chart_definitions = [

        (
            "Mean brightness comparison",
            "brightness",
            False
        ),

        (
            "Contrast comparison",
            "contrast",
            False
        ),

        (
            "Entropy comparison",
            "entropy",
            False
        ),

        (
            "Processing time comparison",
            None,
            True
        ),

    ]

    charts = []

    for label, metric, is_time in chart_definitions:

        chart_buffer = create_chart(
            results,
            label,
            metric,
            processing_time=is_time
        )

        charts.append((
            label,
            chart_asset(chart_buffer)
        ))

    # -----------------------------------------------------
    # REPORT CONTEXT
    # -----------------------------------------------------

    context = {

        "results": results,

        "methodology": methodology,

        "parameter_text": parameter_text,

        "environment_lines": environment_lines,

        "photos": photos,

        "histogram": histogram,

        "charts": charts,

    }

    # -----------------------------------------------------
    # FIRST-PAGE HEADER
    # -----------------------------------------------------

    abstract = create_abstract(results)

    header_bottom = calculate_header_bottom(
        styles,
        abstract
    )

    if header_bottom <= BOTTOM + 0.5 * inch:

        raise RuntimeError(
            "The first-page report header is too large "
            "to leave enough space for the report body."
        )

    # -----------------------------------------------------
    # BUILD TO A TEMPORARY PDF FIRST
    # -----------------------------------------------------

    temporary_path = None

    try:

        with tempfile.NamedTemporaryFile(
            prefix="ather_report_",
            suffix=".pdf",
            dir=str(output_path.parent),
            delete=False
        ) as temporary_file:

            temporary_path = Path(
                temporary_file.name
            )

        document = make_doc(
            str(temporary_path),
            styles,
            header_bottom
        )

        # The first-page callback needs the abstract.
        document.abstract_text = abstract

        story = build_story(
            styles,
            context
        )

        document.build(story)

        # -------------------------------------------------
        # VERIFY THE GENERATED PDF
        # -------------------------------------------------

        page_count = verify_pdf(
            temporary_path
        )

        if page_count is None:

            print(
                "PDF generated. Install pypdf to enable "
                "automatic page-count verification."
            )

        else:

            print(
                f"ATHER report generated successfully: "
                f"{page_count} page(s)."
            )

        # -------------------------------------------------
        # REPLACE THE TARGET ONLY AFTER SUCCESS
        # -------------------------------------------------

        os.replace(
            str(temporary_path),
            str(output_path)
        )

        temporary_path = None

    except Exception:

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):

            temporary_path.unlink(
                missing_ok=True
            )

        raise

    return str(output_path)


# =========================================================
# MODULE TEST ENTRY
# =========================================================

if __name__ == "__main__":

    print(
        "ATHER Continuous IEEE Report Generator"
    )

    print(
        "Supported algorithms:",
        len(ALGORITHM_NAMES)
    )

    print(
        "Layout: continuous two-column, "
        "automatic multi-page"
    )

    print("\nSupported algorithms:")

    for key in ALGORITHM_ORDER:

        print(
            f"- {key}: {algorithm_name(key)}"
        )

    print(
        "\nGenerate reports through the ATHER application "
        "after running benchmarks."
    )