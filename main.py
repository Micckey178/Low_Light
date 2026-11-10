"""
ATHER - Advanced Classical Image Enhancement Laboratory

Professional desktop application for classical, non-AI low-light
image enhancement, analysis, benchmarking and research reporting.

Dashboard:
- Integrated Image Workspace and Enhancement Lab
- Upload images and capture webcam frames
- Select and apply enhancement algorithms
- Inspect original and enhanced images
- Save enhanced output

Separate sections:
- Image Workspace
- Enhancement Lab
- Image Analysis
- Algorithm Lab
- Benchmark Lab
- Reports
- Settings

No AI, ML, neural networks or pretrained models.
"""

import sys
import os
import platform
import tempfile
import inspect

from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QFileDialog, QGroupBox, QMessageBox, QComboBox,
    QDoubleSpinBox, QDialog, QPlainTextEdit, QSizePolicy,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QStackedWidget, QCheckBox
)

from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer

from core.enhancement import ALGORITHMS, apply_enhancement
from core.performance import benchmark_algorithm
from core.report_generator import generate_ieee_report
from core.adaptive_engine import build_processing_plan, ENGINE_VERSION as ACP_ENGINE_VERSION
from core.processing_evaluation import (
    evaluate_components,
    ENGINE_VERSION as EVALUATION_ENGINE_VERSION,
    AVAILABLE_COMPONENTS,
)
from core.acp_validation import validate_real_image_plan


# =========================================================
# THEME
# =========================================================

APP_STYLE = """
/* ATHER Light Lavender — consistent application-wide theme */
QMainWindow, QWidget {
    background: #F7F4FF;
    color: #261542;
    font-family: "Segoe UI";
    font-size: 13px;
}
QLabel { background: transparent; color: #302047; }
QWidget#rootWidget, QWidget#mainColumn, QWidget#contentWidget,
QWidget#pageStack { background: #F7F4FF; }
QFrame#sidebar {
    background: #F1EBFC;
    border-right: 1px solid #DED3F1;
}
QFrame#brandPanel {
    background: #FFFFFF;
    border: 1px solid #E2D8F1;
    border-radius: 16px;
}
QLabel#brandMark {
    background: #7C3AED; color: #FFFFFF;
    border-radius: 12px; font-size: 23px; font-weight: 900;
}
QLabel#brand { color: #32145F; font-size: 22px; font-weight: 900; letter-spacing: 2px; }
QLabel#brandSubtitle { color: #75658D; font-size: 9px; font-weight: 800; letter-spacing: 1px; }
QLabel#sidebarEyebrow { color: #8A79A3; font-size: 10px; font-weight: 850; letter-spacing: 1.5px; padding: 7px 9px; }
QFrame#systemStatusPanel { background: #E9DFFF; border: 1px solid #D3C0F2; border-radius: 12px; }
QLabel#statusEyebrow { color: #7656A0; font-size: 9px; font-weight: 850; letter-spacing: 1.1px; }
QLabel#systemStatusText { color: #4C1D95; font-size: 12px; font-weight: 800; }
QFrame#topbar { background: #FFFFFF; border: none; border-bottom: 1px solid #E4DCF1; }
QLabel#topbarEyebrow { color: #8B5CF6; font-size: 9px; font-weight: 850; letter-spacing: 1.3px; }
QLabel#topbarTitle { color: #281342; font-size: 18px; font-weight: 800; }
QLabel#topbarStatus { color: #4C1D95; background: #F0E8FF; border: 1px solid #D8C6F0; border-radius: 10px; padding: 8px 12px; font-size: 10px; font-weight: 850; letter-spacing: .5px; }
QFrame#pageIntro { background: #FFFFFF; border: 1px solid #E2D8F1; border-left: 4px solid #8B5CF6; border-radius: 14px; }
QLabel#pageEyebrow { color: #8B5CF6; font-size: 9px; font-weight: 850; letter-spacing: 1.5px; }
QLabel#pageTitle { color: #281342; font-size: 25px; font-weight: 850; }
QLabel#pageSubtitle { color: #75658D; font-size: 12px; }
QLabel#sectionTitle, QLabel#sectionLabel { color: #6D28D9; font-size: 11px; font-weight: 850; letter-spacing: 1px; }
QLabel#muted { color: #75658D; }
QLabel#statValue { color: #6D28D9; font-size: 28px; font-weight: 850; }
QPushButton {
    background: #6D28D9; color: #FFFFFF; border: 1px solid #7C3AED;
    border-radius: 10px; padding: 10px 16px; min-height: 20px;
    font-size: 12px; font-weight: 750;
}
QPushButton:hover { background: #7C3AED; border-color: #A78BFA; }
QPushButton:pressed { background: #5420B5; }
QPushButton:disabled { background: #E7E0EF; color: #A69AB5; border-color: #DED5E9; }
QPushButton#secondaryButton { background: #F3EEFC; color: #5B318F; border: 1px solid #D8C9EB; }
QPushButton#secondaryButton:hover { background: #E9DFFF; color: #4C1D95; border-color: #A78BFA; }
QPushButton#navButton {
    background: transparent; color: #59466F; border: 1px solid transparent;
    border-radius: 10px; text-align: left; padding: 13px 15px;
    min-height: 24px; font-size: 14px; font-weight: 650;
}
QPushButton#navButton:hover { background: #E9DFFF; color: #4C1D95; border-color: #D6C5F0; }
QPushButton#navButton[active="true"] {
    background: #DCCBFF; color: #4C1D95; border: 1px solid #B99BEF;
    border-left: 4px solid #7C3AED; font-weight: 800;
}
QGroupBox {
    background: #FFFFFF; border: 1px solid #E2D8F1; border-radius: 14px;
    margin-top: 18px; padding: 17px 14px 13px 14px;
    color: #4C1D95; font-size: 12px; font-weight: 800;
}
QGroupBox::title { subcontrol-origin: margin; left: 13px; padding: 0 8px; color: #6D28D9; }
QComboBox, QDoubleSpinBox {
    background: #FFFFFF; color: #302047; border: 1px solid #D8CBEA;
    border-radius: 8px; padding: 8px 10px; min-height: 25px;
    selection-background-color: #7C3AED;
}
QComboBox:hover, QDoubleSpinBox:hover { border-color: #A78BFA; }
QComboBox:focus, QDoubleSpinBox:focus { border: 1px solid #7C3AED; }
QComboBox QAbstractItemView { background: #FFFFFF; color: #302047; selection-background-color: #E9DFFF; selection-color: #4C1D95; border: 1px solid #CBB8E8; padding: 5px; }
QPlainTextEdit { background: #FBF9FF; color: #35244F; border: 1px solid #DED3F1; border-radius: 10px; padding: 11px; font-family: Consolas; font-size: 11px; selection-background-color: #DCCBFF; }
QTableWidget { background: #FFFFFF; alternate-background-color: #F7F3FF; color: #302047; gridline-color: #E8E0F2; border: 1px solid #DED3F1; border-radius: 10px; selection-background-color: #E5D6FF; selection-color: #4C1D95; outline: 0; }
QTableWidget::item { padding: 8px 7px; }
QTableWidget::item:selected { background: #E5D6FF; }
QHeaderView::section { background: #F0E8FF; color: #4C1D95; padding: 10px 8px; font-weight: 800; border: none; border-right: 1px solid #E2D8F1; border-bottom: 1px solid #E2D8F1; }
QCheckBox { color: #35244F; spacing: 9px; padding: 5px; }
QCheckBox::indicator { width: 17px; height: 17px; }
QCheckBox::indicator:unchecked { background: #FFFFFF; border: 1px solid #B9A9CE; border-radius: 4px; }
QCheckBox::indicator:checked { background: #7C3AED; border: 1px solid #6D28D9; border-radius: 4px; }
QScrollArea { background: #F7F4FF; border: none; }
QScrollArea > QWidget > QWidget { background: #F7F4FF; }
QScrollBar:vertical { background: #EEE8F7; width: 11px; margin: 2px; border-radius: 5px; }
QScrollBar::handle:vertical { background: #B9A0E8; min-height: 40px; border-radius: 5px; }
QScrollBar::handle:vertical:hover { background: #8B5CF6; }
QScrollBar:horizontal { background: #EEE8F7; height: 11px; margin: 2px; border-radius: 5px; }
QScrollBar::handle:horizontal { background: #B9A0E8; min-width: 40px; border-radius: 5px; }
QScrollBar::handle:horizontal:hover { background: #8B5CF6; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QDialog { background: #F7F4FF; }
QToolTip { background: #35204F; color: #FFFFFF; border: 1px solid #A78BFA; padding: 7px 9px; border-radius: 6px; }
"""




# =========================================================
# UI HELPERS
# =========================================================

def make_label(text, object_name=None):
    label = QLabel(text)
    if object_name:
        label.setObjectName(object_name)
    return label


def add_table_item(table, row, column, value):
    """Insert a consistently aligned, read-only table cell."""
    item = QTableWidgetItem(str(value))
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    table.setItem(row, column, item)
    return item


def make_section_label(text):
    """Create a consistent section heading for dashboard and research pages."""
    label = QLabel(str(text).upper())
    label.setObjectName("sectionLabel")
    label.setWordWrap(True)
    label.setStyleSheet("""
        QLabel#sectionLabel {
            color: #4C1D95;
            background: transparent;
            border: none;
            border-left: 3px solid #A78BFA;
            padding: 3px 0 3px 10px;
            font-size: 11px;
            font-weight: 850;
            letter-spacing: 1.2px;
        }
    """)
    return label


def make_card(title, value, caption):
    card = QFrame()
    card.setObjectName("dashboardCard")
    card.setMinimumHeight(116)
    card.setStyleSheet("""
        QFrame#dashboardCard {
            background: #FFFFFF;
            border: 1px solid #E2D8F1;
            border-radius: 12px;
        }
        QFrame#dashboardCard:hover {
            background: #F3EEFC;
            border: 1px solid #A78BFA;
        }
    """)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 14, 16, 13)
    layout.setSpacing(6)

    heading = QLabel(title.upper())
    heading.setStyleSheet(
        "color: #8A79A3; font-size: 9px; font-weight: 850; "
        "letter-spacing: 1px; border: none;"
    )
    number = QLabel(value)
    number.setObjectName("statValue")
    number.setStyleSheet(
        "color: #6D28D9; font-size: 27px; font-weight: 850; border: none;"
    )
    description = QLabel(caption)
    description.setStyleSheet(
        "color: #75658D; font-size: 9px; border: none;"
    )
    description.setWordWrap(True)
    layout.addWidget(heading)
    layout.addWidget(number)
    layout.addWidget(description)
    return card, number

def make_preview_label(text, minimum_height=230):
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setMinimumHeight(minimum_height)
    label.setSizePolicy(
        QSizePolicy.Policy.Ignored,
        QSizePolicy.Policy.Expanding
    )
    label.setStyleSheet("""
        background: #FBF9FF;
        color: #75658D;
        border: 1px dashed #B9A0E8;
        border-radius: 10px;
        padding: 12px;
    """)
    return label

# =========================================================
# HISTOGRAM CANVAS
# =========================================================

class HistogramCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 3.2), facecolor="#FFFFFF")
        self.ax = self.figure.add_subplot(111)

        super().__init__(self.figure)
        self.setParent(parent)
        self.setMinimumHeight(270)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.style_axes()

    def style_axes(self):
        self.figure.patch.set_facecolor("#FFFFFF")
        self.ax.set_facecolor("#FFFFFF")
        self.ax.tick_params(colors="#75658D")
        self.ax.xaxis.label.set_color("#75658D")
        self.ax.yaxis.label.set_color("#75658D")
        self.ax.title.set_color("#4C1D95")

        for spine in self.ax.spines.values():
            spine.set_color("#E2D8F1")

    def plot_histograms(self, image):
        self.ax.clear()

        if image is None:
            self.ax.set_title("No image available")
            self.ax.text(
                0.5, 0.5,
                "Load or enhance an image to view its histogram",
                ha="center", va="center",
                color="#75658D",
                transform=self.ax.transAxes,
                fontsize=10
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.style_axes()
            self.figure.tight_layout()
            self.draw()
            return

        if image.ndim == 2:
            self.ax.hist(
                image.ravel(),
                bins=256,
                range=(0, 256),
                color="#8B5CF6",
                alpha=0.85
            )
            self.ax.set_title("Grayscale Intensity Distribution")
        else:
            for channel, color, label in zip(
                cv2.split(image),
                ("#8B5CF6", "#A78BFA", "#4C1D95"),
                ("Blue", "Green", "Red")
            ):
                self.ax.hist(
                    channel.ravel(),
                    bins=256,
                    range=(0, 256),
                    color=color,
                    alpha=0.35,
                    label=label
                )

            self.ax.set_title("RGB Intensity Distribution")
            legend = self.ax.legend(
                facecolor="#F3EEFC",
                edgecolor="#E2D8F1"
            )
            for item in legend.get_texts():
                item.set_color("#261542")

        self.ax.set_xlabel("Pixel Intensity")
        self.ax.set_ylabel("Number of Pixels")
        self.ax.set_xlim(0, 255)
        self.ax.grid(alpha=0.15, color="#D8CBEA")

        self.style_axes()
        self.figure.tight_layout()
        self.draw()


# =========================================================
# BENCHMARK CANVAS
# =========================================================

class BenchmarkCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(9, 3.5), facecolor="#FFFFFF")
        self.ax = self.figure.add_subplot(111)

        super().__init__(self.figure)
        self.setParent(parent)
        self.setMinimumHeight(280)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.style_axes()

    def style_axes(self):
        self.figure.patch.set_facecolor("#FFFFFF")
        self.ax.set_facecolor("#FFFFFF")
        self.ax.tick_params(colors="#75658D")
        self.ax.xaxis.label.set_color("#75658D")
        self.ax.yaxis.label.set_color("#75658D")
        self.ax.title.set_color("#4C1D95")

        for spine in self.ax.spines.values():
            spine.set_color("#E2D8F1")

    def plot_results(self, results, metric, title, ylabel):
        self.ax.clear()

        if not results:
            self.ax.set_title(title)
            self.style_axes()
            self.figure.tight_layout()
            self.draw()
            return

        names = [result["algorithm_name"] for result in results]

        values = [
            result["processing_time_ms"]
            if metric == "processing_time_ms"
            else result["enhanced_metrics"][metric]
            for result in results
        ]

        bars = self.ax.bar(
            names,
            values,
            color="#6D28D9",
            edgecolor="#A78BFA",
            linewidth=0.7
        )

        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)
        self.ax.tick_params(axis="x", rotation=25, labelsize=8)
        self.ax.grid(axis="y", alpha=0.15)

        for bar, value in zip(bars, values):
            self.ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#261542"
            )

        self.style_axes()
        self.figure.tight_layout()
        self.draw()


# =========================================================
# MAIN APPLICATION
# =========================================================

class ATHERApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "ATHER | Advanced Classical Image Enhancement Laboratory"
        )
        self.resize(1500, 960)
        self.setMinimumSize(1120, 720)

        self.original_image = None
        self.enhanced_image = None
        self.image_path = None

        self.original_metrics = None
        self.enhanced_metrics = None

        self.camera = None
        self.benchmark_results = []
        self.last_report_path = None
        self.default_gamma = 0.6
        self.acp_plan = None
        self.processing_evaluation_result = None
        self.processing_evaluation_image = None

        self.navigation_buttons = {}
        self.page_indices = {}
        self.report_algorithm_checks = {}

        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_camera_frame)

        self.setStyleSheet(APP_STYLE)

        self.setup_ui()
        self.populate_algorithms()
        self.populate_algorithm_catalogue()
        self.populate_report_algorithm_selection()

    # =====================================================
    # PAGE CONSTRUCTION
    # =====================================================

    def create_page(self, title, subtitle):
        """Create a clean page canvas; the global top bar owns page identity."""
        scroll = QScrollArea()
        scroll.setObjectName("pageScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        content = QWidget()
        content.setObjectName("contentWidget")
        content.setMinimumWidth(700)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 25, 30, 38)
        layout.setSpacing(18)

        # A compact page intro, deliberately separate from the persistent topbar
        intro = QFrame()
        intro.setObjectName("pageIntro")
        intro_layout = QVBoxLayout(intro)
        intro_layout.setContentsMargins(20, 17, 20, 17)
        intro_layout.setSpacing(5)

        eyebrow = QLabel("RESEARCH MODULE")
        eyebrow.setObjectName("pageEyebrow")
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        heading.setWordWrap(True)
        description = QLabel(subtitle)
        description.setObjectName("pageSubtitle")
        description.setWordWrap(True)

        intro_layout.addWidget(eyebrow)
        intro_layout.addWidget(heading)
        intro_layout.addWidget(description)
        layout.addWidget(intro)

        scroll.setWidget(content)
        return scroll, layout


    def add_page(self, key, page):
        index = self.page_stack.addWidget(page)
        self.page_indices[key] = index

    # =====================================================
    # MAIN UI AND NAVIGATION
    # =====================================================

    def setup_ui(self):
        """Build the persistent ATHER workstation shell."""
        root = QWidget()
        root.setObjectName("rootWidget")
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        # Compact, persistent navigation rail
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(264)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(16, 20, 16, 16)
        sidebar_layout.setSpacing(6)

        brand_panel = QFrame()
        brand_panel.setObjectName("brandPanel")
        brand_layout = QVBoxLayout(brand_panel)
        brand_layout.setContentsMargins(15, 16, 15, 15)
        brand_layout.setSpacing(4)

        brand_mark = QLabel("A")
        brand_mark.setObjectName("brandMark")
        brand_mark.setFixedSize(42, 42)
        brand_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand = QLabel("ATHER")
        brand.setObjectName("brand")
        brand_subtitle = QLabel("CLASSICAL IMAGE LAB")
        brand_subtitle.setObjectName("brandSubtitle")

        brand_row = QHBoxLayout()
        brand_row.setSpacing(10)
        brand_row.addWidget(brand_mark)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(1)
        brand_text.addWidget(brand)
        brand_text.addWidget(brand_subtitle)
        brand_row.addLayout(brand_text, 1)
        brand_layout.addLayout(brand_row)
        sidebar_layout.addWidget(brand_panel)
        sidebar_layout.addSpacing(20)

        menu_label = QLabel("YOUR WORKSPACE")
        menu_label.setObjectName("sidebarEyebrow")
        sidebar_layout.addWidget(menu_label)

        navigation_items = [
            ("Dashboard", "dashboard"),
            ("Image Workspace", "workspace"),
            ("Enhancement Lab", "enhancement"),
            ("Image Analysis & Evaluation", "analysis"),
            ("Adaptive Processing (ACP)", "acp"),
            ("Algorithm Lab", "algorithms"),
            ("Benchmark Lab", "benchmark"),
            ("Research Reports", "reports"),
        ]
        self.navigation_buttons = {}
        for title, key in navigation_items:
            button = QPushButton(title)
            button.setObjectName("navButton")
            button.setProperty("active", False)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(52)
            button.setCheckable(False)
            button.clicked.connect(
                lambda checked=False, destination=key: self.navigate(destination)
            )
            sidebar_layout.addWidget(button)
            self.navigation_buttons[key] = button

        sidebar_layout.addSpacing(14)
        preference_label = QLabel("PREFERENCES")
        preference_label.setObjectName("sidebarEyebrow")
        sidebar_layout.addWidget(preference_label)

        settings_button = QPushButton("Settings")
        settings_button.setObjectName("navButton")
        settings_button.setProperty("active", False)
        settings_button.setMinimumHeight(52)
        settings_button.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_button.clicked.connect(lambda: self.navigate("settings"))
        sidebar_layout.addWidget(settings_button)
        self.navigation_buttons["settings"] = settings_button

        sidebar_layout.addStretch(1)

        status_panel = QFrame()
        status_panel.setObjectName("systemStatusPanel")
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(12, 12, 12, 12)
        status_layout.setSpacing(5)
        status_heading = QLabel("WORKSPACE STATUS")
        status_heading.setObjectName("statusEyebrow")
        self.sidebar_status = QLabel("●  SYSTEM READY")
        self.sidebar_status.setObjectName("systemStatusText")
        self.sidebar_status.setWordWrap(True)
        status_layout.addWidget(status_heading)
        status_layout.addWidget(self.sidebar_status)
        sidebar_layout.addWidget(status_panel)

        root_layout.addWidget(self.sidebar)

        # Main column: persistent top bar + scrollable page stack
        self.main_column = QWidget()
        self.main_column.setObjectName("mainColumn")
        main_layout = QVBoxLayout(self.main_column)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.topbar = QFrame()
        self.topbar.setObjectName("topbar")
        topbar_layout = QHBoxLayout(self.topbar)
        topbar_layout.setContentsMargins(28, 14, 30, 14)
        topbar_layout.setSpacing(12)

        title_column = QVBoxLayout()
        title_column.setSpacing(2)
        self.topbar_eyebrow = QLabel("ATHER  /  RESEARCH WORKSPACE")
        self.topbar_eyebrow.setObjectName("topbarEyebrow")
        self.topbar_title = QLabel("Dashboard")
        self.topbar_title.setObjectName("topbarTitle")
        title_column.addWidget(self.topbar_eyebrow)
        title_column.addWidget(self.topbar_title)
        topbar_layout.addLayout(title_column, 1)

        self.topbar_status = QLabel("●  READY")
        self.topbar_status.setObjectName("topbarStatus")
        self.topbar_status.setAlignment(
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
        )
        topbar_layout.addWidget(self.topbar_status)
        main_layout.addWidget(self.topbar)

        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("pageStack")
        main_layout.addWidget(self.page_stack, 1)
        root_layout.addWidget(self.main_column, 1)

        self.build_dashboard()
        self.build_workspace()
        self.build_enhancement_lab()
        self.build_analysis()
        self.build_processing_evaluation_page()
        self.build_acp_page()
        self.build_algorithm_lab()
        self.build_benchmark_lab()
        self.build_reports_page()
        self.build_settings_page()
        self.navigate("dashboard")


    def navigate(self, key):
        if key not in self.page_indices:
            return

        self.page_stack.setCurrentIndex(self.page_indices[key])

        titles = {
            "dashboard": "Dashboard",
            "workspace": "Image Workspace",
            "enhancement": "Enhancement Lab",
            "analysis": "Image Analysis & Evaluation",
            "acp": "Adaptive Processing (ACP)",
            "algorithms": "Algorithm Lab",
            "benchmark": "Benchmark Lab",
            "reports": "Research Reports",
            "settings": "Settings",
        }
        self.topbar_title.setText(titles.get(key, key.replace("_", " ").title()))

        for nav_key, button in self.navigation_buttons.items():
            active = nav_key == key
            button.setProperty("active", active)
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()

        self.status_label_update("Viewing " + key.replace("_", " ").title())
        QTimer.singleShot(0, self.refresh_previews)


    def status_label_update(self, message):
        text = "●  " + message.upper()
        self.sidebar_status.setText(text)
        if hasattr(self, "topbar_status"):
            self.topbar_status.setText("●  " + message.upper())


    # =====================================================
    # DASHBOARD
    # =====================================================

    def build_dashboard(self):
        page, layout = self.create_page(
            "ATHER Research Dashboard",
            "A centralized research workspace for classical low-light "
            "image enhancement, image analysis and algorithm evaluation."
        )

        # Welcome panel
        hero = QFrame()
        hero.setObjectName("dashboardHero")
        hero.setStyleSheet("""
            QFrame#dashboardHero {
                background-color: #FFFFFF;
                border: 1px solid #D8CBEA;
                border-left: 5px solid #8B5CF6;
                border-radius: 15px;
            }
        """)

        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(26, 23, 26, 23)
        hero_layout.setSpacing(9)

        hero_title = QLabel(
            "Classical Image Enhancement Laboratory"
        )
        hero_title.setStyleSheet(
            "color: #261542; font-size: 22px; font-weight: 850; "
            "border: none;"
        )
        hero_title.setWordWrap(True)

        hero_description = QLabel(
            "Upload or capture a low-light image, select a mathematical "
            "enhancement method, and inspect the result directly from "
            "this dashboard. Continue to the dedicated research sections "
            "for detailed analysis, benchmarking and reporting."
        )
        hero_description.setStyleSheet(
            "color: #75658D; font-size: 12px; border: none;"
        )
        hero_description.setWordWrap(True)

        hero_note = QLabel(
            "CLASSICAL IMAGE PROCESSING  •  CPU-BASED  •  NO AI / ML"
        )
        hero_note.setStyleSheet(
            "color: #6D28D9; font-size: 10px; font-weight: 800; "
            "letter-spacing: 1px; border: none;"
        )

        hero_layout.addWidget(hero_title)
        hero_layout.addWidget(hero_description)
        hero_layout.addWidget(hero_note)

        layout.addWidget(hero)

        # Statistics
        layout.addWidget(make_section_label("Workspace Overview"))

        stats = QGridLayout()
        stats.setSpacing(12)

        card, self.dashboard_image_count = make_card(
            "Current Image", "0", "No image loaded"
        )
        stats.addWidget(card, 0, 0)

        card, self.dashboard_algorithm_count = make_card(
            "Available Algorithms", str(len(ALGORITHMS)),
            "Registered classical methods"
        )
        stats.addWidget(card, 0, 1)

        card, self.dashboard_benchmark_count = make_card(
            "Benchmark Results", "0",
            "No benchmark completed"
        )
        stats.addWidget(card, 0, 2)

        card, self.dashboard_report_status = make_card(
            "Report Status", "Ready",
            "Waiting for benchmark results"
        )
        stats.addWidget(card, 0, 3)

        layout.addLayout(stats)

        # -------------------------------------------------
        # INTEGRATED DASHBOARD WORKSPACE + ENHANCEMENT LAB
        # -------------------------------------------------

        dashboard_lab = QGroupBox(
            "Image Workspace & Enhancement Laboratory"
        )
        dashboard_lab_layout = QVBoxLayout(dashboard_lab)
        dashboard_lab_layout.setSpacing(13)

        dashboard_description = QLabel(
            "Complete the image enhancement workflow here without "
            "opening another section. The dedicated Image Workspace "
            "and Enhancement Lab remain available in the sidebar."
        )
        dashboard_description.setWordWrap(True)
        dashboard_description.setStyleSheet(
            "color: #75658D; font-size: 11px;"
        )
        dashboard_lab_layout.addWidget(dashboard_description)

        # Image input controls
        input_row = QHBoxLayout()

        self.dashboard_upload_button = QPushButton("Upload Image")
        self.dashboard_upload_button.clicked.connect(
            lambda: self.upload_image(navigate_after=False)
        )

        self.dashboard_camera_button = QPushButton("Start Webcam")
        self.dashboard_camera_button.setObjectName("secondaryButton")
        self.dashboard_camera_button.clicked.connect(
            self.toggle_camera
        )

        self.dashboard_capture_button = QPushButton("Capture Frame")
        self.dashboard_capture_button.clicked.connect(
            self.capture_camera_image
        )
        self.dashboard_capture_button.setEnabled(False)

        self.dashboard_save_button = QPushButton("Save Enhanced Image")
        self.dashboard_save_button.setObjectName("secondaryButton")
        self.dashboard_save_button.clicked.connect(
            self.save_enhanced_image
        )
        self.dashboard_save_button.setEnabled(False)

        input_row.addWidget(self.dashboard_upload_button)
        input_row.addWidget(self.dashboard_camera_button)
        input_row.addWidget(self.dashboard_capture_button)

        self.dashboard_acp_button = QPushButton("Open ACP Analysis")
        self.dashboard_acp_button.setObjectName("secondaryButton")
        self.dashboard_acp_button.clicked.connect(
            lambda: self.navigate("acp")
        )
        input_row.addWidget(self.dashboard_acp_button)
        input_row.addStretch()
        input_row.addWidget(self.dashboard_save_button)

        dashboard_lab_layout.addLayout(input_row)

        # Dashboard image previews
        preview_row = QHBoxLayout()
        preview_row.setSpacing(12)

        original_group = QGroupBox("Original Image")
        original_layout = QVBoxLayout(original_group)

        self.dashboard_original_label = make_preview_label(
            "Upload or capture an image", 240
        )
        original_layout.addWidget(self.dashboard_original_label)

        self.dashboard_original_inspect = QPushButton(
            "Inspect Original at 100%"
        )
        self.dashboard_original_inspect.setObjectName(
            "secondaryButton"
        )
        self.dashboard_original_inspect.clicked.connect(
            self.open_original_fullsize
        )
        self.dashboard_original_inspect.setEnabled(False)
        original_layout.addWidget(self.dashboard_original_inspect)

        enhanced_group = QGroupBox("Enhanced Image")
        enhanced_layout = QVBoxLayout(enhanced_group)

        self.dashboard_enhanced_label = make_preview_label(
            "Enhanced output will appear here", 240
        )
        enhanced_layout.addWidget(self.dashboard_enhanced_label)

        self.dashboard_enhanced_inspect = QPushButton(
            "Inspect Enhanced at 100%"
        )
        self.dashboard_enhanced_inspect.setObjectName(
            "secondaryButton"
        )
        self.dashboard_enhanced_inspect.clicked.connect(
            self.open_enhanced_fullsize
        )
        self.dashboard_enhanced_inspect.setEnabled(False)
        enhanced_layout.addWidget(self.dashboard_enhanced_inspect)

        preview_row.addWidget(original_group, 1)
        preview_row.addWidget(enhanced_group, 1)

        dashboard_lab_layout.addLayout(preview_row)

        self.dashboard_file_label = QLabel("No image loaded")
        self.dashboard_file_label.setObjectName("muted")
        self.dashboard_file_label.setWordWrap(True)
        dashboard_lab_layout.addWidget(self.dashboard_file_label)

        # Enhancement controls on dashboard
        configuration_group = QGroupBox(
            "Enhancement Configuration"
        )
        configuration_layout = QGridLayout(configuration_group)
        configuration_layout.setHorizontalSpacing(12)
        configuration_layout.setVerticalSpacing(10)

        configuration_layout.addWidget(
            QLabel("Select Algorithm"), 0, 0
        )

        self.dashboard_algorithm_combo = QComboBox()
        self.dashboard_algorithm_combo.setMinimumHeight(38)
        configuration_layout.addWidget(
            self.dashboard_algorithm_combo, 0, 1
        )

        configuration_layout.addWidget(
            QLabel("Gamma Value"), 0, 2
        )

        self.dashboard_gamma_spin = QDoubleSpinBox()
        self.dashboard_gamma_spin.setRange(0.1, 3.0)
        self.dashboard_gamma_spin.setSingleStep(0.1)
        self.dashboard_gamma_spin.setValue(self.default_gamma)
        self.dashboard_gamma_spin.setMinimumHeight(38)
        self.dashboard_gamma_spin.setFixedWidth(115)
        configuration_layout.addWidget(
            self.dashboard_gamma_spin, 0, 3
        )

        self.dashboard_apply_button = QPushButton(
            "Apply Enhancement"
        )
        self.dashboard_apply_button.setMinimumHeight(38)
        self.dashboard_apply_button.clicked.connect(
            self.apply_dashboard_enhancement
        )
        self.dashboard_apply_button.setEnabled(False)

        configuration_layout.addWidget(
            self.dashboard_apply_button, 0, 4
        )

        configuration_layout.setColumnStretch(1, 1)

        dashboard_lab_layout.addWidget(configuration_group)

        self.dashboard_method_information = QLabel(
            "Select an algorithm to view its purpose and mathematical formula."
        )
        self.dashboard_method_information.setWordWrap(True)
        self.dashboard_method_information.setStyleSheet(
            "color: #75658D; font-size: 11px;"
        )
        dashboard_lab_layout.addWidget(
            self.dashboard_method_information
        )

        layout.addWidget(dashboard_lab)

        # Current session information
        session_group = QGroupBox("Current Research Session")
        session_layout = QGridLayout(session_group)
        session_layout.setHorizontalSpacing(18)
        session_layout.setVerticalSpacing(12)

        session_layout.addWidget(
            make_label("Loaded image", "muted"), 0, 0
        )

        self.dashboard_current_image = QLabel("No image loaded")
        self.dashboard_current_image.setStyleSheet(
            "color: #4C1D95; font-weight: 600;"
        )
        self.dashboard_current_image.setWordWrap(True)
        session_layout.addWidget(
            self.dashboard_current_image, 0, 1
        )

        session_layout.addWidget(
            make_label("Processing state", "muted"), 1, 0
        )

        self.dashboard_processing_state = QLabel(
            "Waiting for input"
        )
        self.dashboard_processing_state.setStyleSheet(
            "color: #6D28D9; font-weight: 600;"
        )
        session_layout.addWidget(
            self.dashboard_processing_state, 1, 1
        )

        session_layout.addWidget(
            make_label("Image dimensions", "muted"), 2, 0
        )

        self.dashboard_image_dimensions = QLabel("Not available")
        self.dashboard_image_dimensions.setStyleSheet(
            "color: #4C1D95; font-weight: 600;"
        )
        session_layout.addWidget(
            self.dashboard_image_dimensions, 2, 1
        )

        session_layout.setColumnStretch(1, 1)
        layout.addWidget(session_group)

        # Workflow navigation - Algorithm Lab deliberately excluded.
        workflow_group = QGroupBox("Research Workflow")
        workflow_layout = QGridLayout(workflow_group)
        workflow_layout.setHorizontalSpacing(12)
        workflow_layout.setVerticalSpacing(12)

        workflow_steps = [
            (
                "01  IMAGE ANALYSIS",
                "Compare brightness, contrast, entropy and histograms.",
                "Open Image Analysis",
                "analysis"
            ),
            (
                "02  BENCHMARK",
                "Evaluate processing time and image statistics.",
                "Open Benchmark Lab",
                "benchmark"
            ),
            (
                "03  RESEARCH REPORT",
                "Select algorithms and generate a research PDF.",
                "Open Reports",
                "reports"
            )
        ]

        for index, (heading, description, button_text, key) in enumerate(
            workflow_steps
        ):
            step_card = QFrame()
            step_card.setStyleSheet("""
                QFrame {
                    background-color: #F3EEFC;
                    border: 1px solid #E2D8F1;
                    border-radius: 9px;
                }
            """)

            step_layout = QVBoxLayout(step_card)
            step_layout.setContentsMargins(13, 12, 13, 12)
            step_layout.setSpacing(7)

            step_heading = QLabel(heading)
            step_heading.setStyleSheet(
                "color: #4C1D95; font-weight: 800; "
                "font-size: 11px; border: none;"
            )

            step_description = QLabel(description)
            step_description.setWordWrap(True)
            step_description.setMinimumHeight(42)
            step_description.setStyleSheet(
                "color: #75658D; font-size: 11px; border: none;"
            )

            step_button = QPushButton(button_text)
            step_button.setObjectName("secondaryButton")
            step_button.clicked.connect(
                lambda checked=False, destination=key:
                self.navigate(destination)
            )

            step_layout.addWidget(step_heading)
            step_layout.addWidget(step_description)
            step_layout.addStretch()
            step_layout.addWidget(step_button)

            workflow_layout.addWidget(step_card, 0, index)

        layout.addWidget(workflow_group)
        layout.addStretch()

        self.add_page("dashboard", page)

        self.dashboard_algorithm_combo.currentIndexChanged.connect(
            self.update_dashboard_method_information
        )

    def update_dashboard_method_information(self):
        key = self.dashboard_algorithm_combo.currentData()

        if key not in ALGORITHMS:
            return

        details = ALGORITHMS[key]

        purpose = details.get(
            "purpose", "Purpose not documented."
        )
        formula = details.get(
            "formula", "Formula not documented."
        )

        self.dashboard_method_information.setText(
            f"{details.get('name', key)}\n\n"
            f"Purpose: {purpose}\n\n"
            f"Mathematical formula: {formula}"
        )

    # =====================================================
    # IMAGE WORKSPACE - SEPARATE PAGE
    # =====================================================

    def build_workspace(self):
        page, layout = self.create_page(
            "Image Workspace",
            "Import an image, capture a frame or inspect your enhancement results."
        )

        buttons = QHBoxLayout()

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)

        self.camera_button = QPushButton("Start Webcam")
        self.camera_button.clicked.connect(self.toggle_camera)

        self.stop_camera_button = QPushButton("Stop Webcam")
        self.stop_camera_button.setObjectName("secondaryButton")
        self.stop_camera_button.clicked.connect(self.stop_camera)
        self.stop_camera_button.setEnabled(False)

        self.save_button = QPushButton("Save Enhanced Image")
        self.save_button.clicked.connect(self.save_enhanced_image)
        self.save_button.setEnabled(False)

        buttons.addWidget(self.upload_button)
        buttons.addWidget(self.camera_button)
        buttons.addWidget(self.stop_camera_button)
        buttons.addStretch()
        buttons.addWidget(self.save_button)

        layout.addLayout(buttons)

        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(13)

        original_group = QGroupBox("Original Image")
        enhanced_group = QGroupBox("Enhanced Image")

        original_box = QVBoxLayout(original_group)
        enhanced_box = QVBoxLayout(enhanced_group)

        self.original_label = make_preview_label(
            "Upload an image to begin", 290
        )
        self.enhanced_label = make_preview_label(
            "Enhanced output will appear here", 290
        )

        original_box.addWidget(self.original_label)
        enhanced_box.addWidget(self.enhanced_label)

        self.original_fullsize_button = QPushButton(
            "Inspect Original at 100%"
        )
        self.original_fullsize_button.setObjectName("secondaryButton")
        self.original_fullsize_button.clicked.connect(
            self.open_original_fullsize
        )
        self.original_fullsize_button.setEnabled(False)

        self.enhanced_fullsize_button = QPushButton(
            "Inspect Enhanced at 100%"
        )
        self.enhanced_fullsize_button.setObjectName("secondaryButton")
        self.enhanced_fullsize_button.clicked.connect(
            self.open_enhanced_fullsize
        )
        self.enhanced_fullsize_button.setEnabled(False)

        original_box.addWidget(self.original_fullsize_button)
        enhanced_box.addWidget(self.enhanced_fullsize_button)

        preview_layout.addWidget(original_group, 1)
        preview_layout.addWidget(enhanced_group, 1)

        layout.addLayout(preview_layout)

        self.workspace_file_label = QLabel("No image loaded")
        self.workspace_file_label.setObjectName("muted")
        self.workspace_file_label.setWordWrap(True)
        layout.addWidget(self.workspace_file_label)

        layout.addStretch()
        self.add_page("workspace", page)

    # =====================================================
    # ENHANCEMENT LAB - SEPARATE PAGE
    # =====================================================

    def build_enhancement_lab(self):
        page, layout = self.create_page(
            "Enhancement Laboratory",
            "Select a classical algorithm, study its mathematics and process your image."
        )

        group = QGroupBox("Algorithm Configuration")
        group_layout = QVBoxLayout(group)

        controls = QGridLayout()
        controls.setHorizontalSpacing(12)
        controls.setVerticalSpacing(10)

        controls.addWidget(QLabel("Algorithm"), 0, 0)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.setMinimumHeight(38)
        self.algorithm_combo.currentIndexChanged.connect(
            self.update_algorithm_information
        )
        controls.addWidget(self.algorithm_combo, 0, 1)

        controls.addWidget(QLabel("Gamma Value"), 0, 2)

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 3.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(self.default_gamma)
        self.gamma_spin.setMinimumHeight(38)
        self.gamma_spin.setFixedWidth(120)
        controls.addWidget(self.gamma_spin, 0, 3)

        controls.setColumnStretch(1, 1)
        group_layout.addLayout(controls)

        self.purpose_label = QLabel()
        self.purpose_label.setWordWrap(True)

        self.formula_label = QLabel()
        self.formula_label.setWordWrap(True)
        self.formula_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.variables_label = QLabel()
        self.variables_label.setWordWrap(True)

        for title, widget in [
            ("Purpose", self.purpose_label),
            ("Mathematical Formula", self.formula_label),
            ("Variable Explanation", self.variables_label),
        ]:
            group_layout.addWidget(make_section_label(title))
            group_layout.addWidget(widget)

        group_layout.addWidget(
            make_section_label("Actual Python Implementation")
        )

        self.code_view = QPlainTextEdit()
        self.code_view.setReadOnly(True)
        self.code_view.setMinimumHeight(170)
        self.code_view.setMaximumHeight(250)
        group_layout.addWidget(self.code_view)

        apply_row = QHBoxLayout()
        apply_row.addStretch()

        self.apply_button = QPushButton("Apply Enhancement")
        self.apply_button.setMinimumWidth(190)
        self.apply_button.clicked.connect(
            self.apply_selected_algorithm
        )
        self.apply_button.setEnabled(False)

        apply_row.addWidget(self.apply_button)
        group_layout.addLayout(apply_row)

        layout.addWidget(group)
        layout.addStretch()
        self.add_page("enhancement", page)

    # =====================================================
    # IMAGE ANALYSIS
    # =====================================================

    def build_analysis(self):
        page, layout = self.create_page(
            "Image Analysis & Processing Evaluation",
            "A unified workspace to inspect image properties, compare "
            "original and enhanced results, evaluate classical processing "
            "components, and review their measured effects."
        )
        self.analysis_page = page
        self.analysis_content_layout = layout

        status_group = QGroupBox("Analysis Overview")
        status_layout = QHBoxLayout(status_group)

        self.analysis_original_status = QLabel(
            "ORIGINAL: Waiting for image"
        )
        self.analysis_original_status.setStyleSheet(
            "color: #6D28D9; font-weight: 700;"
        )

        self.analysis_enhanced_status = QLabel(
            "ENHANCED: No enhanced output"
        )
        self.analysis_enhanced_status.setStyleSheet(
            "color: #75658D; font-weight: 700;"
        )

        status_layout.addWidget(self.analysis_original_status)
        status_layout.addStretch()
        status_layout.addWidget(self.analysis_enhanced_status)

        layout.addWidget(status_group)

        metrics_group = QGroupBox("Image Property Comparison")
        metrics_layout = QVBoxLayout(metrics_group)

        metrics_description = QLabel(
            "Measurements are calculated from grayscale intensity. "
            "Change is Enhanced minus Original. Estimated noise is an "
            "approximate high-frequency residual measurement, not a "
            "ground-truth noise measurement."
        )
        metrics_description.setWordWrap(True)
        metrics_description.setStyleSheet(
            "color: #75658D; font-size: 11px;"
        )
        metrics_layout.addWidget(metrics_description)

        self.analysis_table = QTableWidget(9, 4)
        self.analysis_table.setHorizontalHeaderLabels([
            "Image Property",
            "Original Image",
            "Enhanced Image",
            "Change"
        ])

        self.analysis_table.setVerticalHeaderLabels([
            "01", "02", "03", "04", "05",
            "06", "07", "08", "09"
        ])

        for column in range(4):
            self.analysis_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.Stretch
            )

        self.analysis_table.setAlternatingRowColors(True)
        self.analysis_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.analysis_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.analysis_table.setMinimumHeight(360)
        self.analysis_table.verticalHeader().setDefaultSectionSize(34)

        metric_names = [
            "Dimensions",
            "Total Pixels",
            "Average Brightness",
            "Minimum Intensity",
            "Maximum Intensity",
            "Contrast (Standard Deviation)",
            "Entropy",
            "Dynamic Range",
            "Estimated Noise"
        ]

        for row, name in enumerate(metric_names):
            add_table_item(self.analysis_table, row, 0, name)
            add_table_item(self.analysis_table, row, 1, "Waiting")
            add_table_item(self.analysis_table, row, 2, "Waiting")
            add_table_item(self.analysis_table, row, 3, "—")

        metrics_layout.addWidget(self.analysis_table)
        layout.addWidget(metrics_group)

        layout.addWidget(
            make_section_label("Intensity Distribution Analysis")
        )

        histogram_description = QLabel(
            "The original and enhanced distributions are displayed "
            "separately and vertically for direct inspection."
        )
        histogram_description.setWordWrap(True)
        histogram_description.setObjectName("muted")
        layout.addWidget(histogram_description)

        original_histogram_group = QGroupBox(
            "Original Image Histogram"
        )
        original_histogram_layout = QVBoxLayout(
            original_histogram_group
        )

        self.original_histogram_canvas = HistogramCanvas()
        self.histogram_canvas = self.original_histogram_canvas
        self.original_histogram_canvas.plot_histograms(None)

        original_histogram_layout.addWidget(
            self.original_histogram_canvas
        )
        layout.addWidget(original_histogram_group)

        enhanced_histogram_group = QGroupBox(
            "Enhanced Image Histogram"
        )
        enhanced_histogram_layout = QVBoxLayout(
            enhanced_histogram_group
        )

        self.enhanced_histogram_canvas = HistogramCanvas()
        self.enhanced_histogram_canvas.plot_histograms(None)

        enhanced_histogram_layout.addWidget(
            self.enhanced_histogram_canvas
        )
        layout.addWidget(enhanced_histogram_group)

        self.add_page("analysis", page)

    # =====================================================
    # ATHER-ACP: ADAPTIVE CLASSICAL PROCESSING
    # =====================================================

    def build_acp_page(self):
        page, layout = self.create_page(
            "ATHER-ACP Adaptive Processing",
            "A deterministic, explainable analysis of the currently loaded "
            "photograph. ACP measures image conditions and recommends "
            "classical algorithms; it never enhances an image automatically."
        )

        intro = QLabel(
            "Every uploaded or webcam-captured image is analysed independently. "
            "No fixed test photograph, AI model, training data or cloud service "
            "is used. Recommendations are rule-based candidates, not guarantees "
            "of perceptual quality."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)

        action_row = QHBoxLayout()
        self.acp_analyse_button = QPushButton("Re-analyse Current Image")
        self.acp_analyse_button.clicked.connect(self.run_acp_analysis)
        action_row.addWidget(self.acp_analyse_button)

        self.acp_validate_button = QPushButton("Validate Current Photograph")
        self.acp_validate_button.setObjectName("secondaryButton")
        self.acp_validate_button.clicked.connect(self.validate_current_acp_photo)
        action_row.addWidget(self.acp_validate_button)

        self.acp_open_method_button = QPushButton(
            "Open Selected Recommendation in Enhancement Lab"
        )
        self.acp_open_method_button.setObjectName("secondaryButton")
        self.acp_open_method_button.clicked.connect(
            self.open_selected_acp_recommendation
        )
        self.acp_open_method_button.setEnabled(False)
        action_row.addWidget(self.acp_open_method_button)
        action_row.addStretch()
        layout.addLayout(action_row)

        self.acp_status_label = QLabel(
            "Waiting for an image. Upload or capture a photograph first."
        )
        self.acp_status_label.setWordWrap(True)
        self.acp_status_label.setStyleSheet(
            "color: #8B5CF6; font-weight: 700; padding: 5px 0;"
        )
        layout.addWidget(self.acp_status_label)

        metrics_group = QGroupBox("Measured Image Conditions")
        metrics_layout = QVBoxLayout(metrics_group)
        metrics_note = QLabel(
            "Values below are computed from the current original image. "
            "Noise is an approximate residual proxy, not a ground-truth estimate."
        )
        metrics_note.setWordWrap(True)
        metrics_note.setObjectName("muted")
        metrics_layout.addWidget(metrics_note)

        self.acp_metrics_table = QTableWidget(10, 2)
        self.acp_metrics_table.setHorizontalHeaderLabels([
            "Measurement", "Current Image"
        ])
        self.acp_metrics_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.acp_metrics_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.acp_metrics_table.verticalHeader().setVisible(False)
        self.acp_metrics_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.acp_metrics_table.setAlternatingRowColors(True)
        self.acp_metrics_table.setMinimumHeight(340)
        self.acp_metric_names = [
            "Image dimensions", "Total pixels", "Color mode",
            "Mean luminance (0–255)", "Median luminance (0–255)",
            "Luminance contrast (standard deviation)",
            "Dynamic range", "Dark pixels (≤ 51)",
            "Shadow / highlight clipping", "Estimated noise proxy"
        ]
        for row, name in enumerate(self.acp_metric_names):
            self.acp_metrics_table.setItem(row, 0, QTableWidgetItem(name))
            self.acp_metrics_table.setItem(row, 1, QTableWidgetItem("Waiting"))
        metrics_layout.addWidget(self.acp_metrics_table)
        layout.addWidget(metrics_group)

        conditions_group = QGroupBox("Detected Conditions and Rule Explanations")
        conditions_layout = QVBoxLayout(conditions_group)
        self.acp_conditions_text = QPlainTextEdit()
        self.acp_conditions_text.setReadOnly(True)
        self.acp_conditions_text.setMinimumHeight(170)
        self.acp_conditions_text.setPlaceholderText(
            "Detected conditions will appear after image analysis."
        )
        conditions_layout.addWidget(self.acp_conditions_text)
        layout.addWidget(conditions_group)

        recommendations_group = QGroupBox(
            "Recommended Classical Algorithms"
        )
        recommendations_layout = QVBoxLayout(recommendations_group)
        recommendation_note = QLabel(
            "Priority is a deterministic rule score, not a probability, "
            "accuracy percentage or quality ranking. Double-click a row or "
            "select it and open it in the Enhancement Lab."
        )
        recommendation_note.setWordWrap(True)
        recommendation_note.setObjectName("muted")
        recommendations_layout.addWidget(recommendation_note)

        self.acp_recommendations_table = QTableWidget(0, 3)
        self.acp_recommendations_table.setHorizontalHeaderLabels([
            "Algorithm", "Rule Priority", "Why it was suggested"
        ])
        self.acp_recommendations_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.acp_recommendations_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.acp_recommendations_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.acp_recommendations_table.verticalHeader().setVisible(False)
        self.acp_recommendations_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.acp_recommendations_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.acp_recommendations_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        self.acp_recommendations_table.setWordWrap(True)
        self.acp_recommendations_table.setMinimumHeight(230)
        self.acp_recommendations_table.itemSelectionChanged.connect(
            self.update_acp_recommendation_button
        )
        self.acp_recommendations_table.itemDoubleClicked.connect(
            lambda item: self.open_selected_acp_recommendation()
        )
        recommendations_layout.addWidget(self.acp_recommendations_table)
        layout.addWidget(recommendations_group)

        cautions_group = QGroupBox("Processing Cautions")
        cautions_layout = QVBoxLayout(cautions_group)
        self.acp_cautions_text = QPlainTextEdit()
        self.acp_cautions_text.setReadOnly(True)
        self.acp_cautions_text.setMinimumHeight(110)
        cautions_layout.addWidget(self.acp_cautions_text)
        layout.addWidget(cautions_group)

        explanation_group = QGroupBox("Complete ACP Explanation")
        explanation_layout = QVBoxLayout(explanation_group)
        self.acp_explanation_text = QPlainTextEdit()
        self.acp_explanation_text.setReadOnly(True)
        self.acp_explanation_text.setMinimumHeight(210)
        explanation_layout.addWidget(self.acp_explanation_text)
        layout.addWidget(explanation_group)

        validation_group = QGroupBox("Real-Photograph Validation")
        validation_layout = QVBoxLayout(validation_group)
        validation_note = QLabel(
            "Run repeatability and integrity checks against the actual photograph "
            "currently loaded in ATHER. This checks measurements and rule output; "
            "it does not claim that an enhancement is perceptually optimal."
        )
        validation_note.setWordWrap(True)
        validation_note.setObjectName("muted")
        validation_layout.addWidget(validation_note)
        self.acp_validation_text = QPlainTextEdit()
        self.acp_validation_text.setReadOnly(True)
        self.acp_validation_text.setMinimumHeight(190)
        self.acp_validation_text.setPlaceholderText(
            "Load a photograph, run ACP analysis, then select Validate Current Photograph."
        )
        validation_layout.addWidget(self.acp_validation_text)
        layout.addWidget(validation_group)

        layout.addStretch()
        self.add_page("acp", page)

    def validate_current_acp_photo(self):
        """Validate the ACP plan against the actual image loaded in the workspace."""
        if self.original_image is None:
            self.acp_validation_text.setPlainText(
                "No photograph is loaded. Upload or capture an image first."
            )
            return

        if not self.acp_plan:
            self.run_acp_analysis()
            if not self.acp_plan:
                self.acp_validation_text.setPlainText(
                    "ACP analysis is unavailable. Run analysis again after checking the error above."
                )
                return

        self.acp_validate_button.setEnabled(False)
        self.acp_validation_text.setPlainText(
            "Validating the current photograph and repeating ACP analysis..."
        )
        QApplication.processEvents()
        try:
            result = validate_real_image_plan(
                self.original_image.copy(),
                self.acp_plan,
                ALGORITHMS.keys(),
            )
            lines = [result.get("summary", "Validation completed."), ""]
            for index, check in enumerate(result.get("checks", []), start=1):
                marker = "PASS" if check.get("passed") else "FAIL"
                lines.append(f"{index}. [{marker}] {check.get('name', 'Check')}")
                lines.append(f"   {check.get('detail', '')}")
            lines.extend([
                "",
                "Interpretation: these checks establish input/plan consistency and "
                "repeatability for this photograph. They do not establish visual quality, "
                "recover lost detail, infer photographer intent, or replace human review."
            ])
            self.acp_validation_text.setPlainText("\n".join(lines))
            status = "PASSED" if result.get("passed") else "NEEDS REVIEW"
            self.acp_status_label.setText(
                f"ACP analysis available • Real-photograph validation: {status}"
            )
        except Exception as error:
            self.acp_validation_text.setPlainText(
                f"Validation could not complete: {type(error).__name__}: {error}"
            )
        finally:
            self.acp_validate_button.setEnabled(True)

    def run_acp_analysis(self):
        """Analyse the current original image using deterministic ACP rules."""
        if self.original_image is None:
            self.acp_status_label.setText(
                "No image loaded. Upload or capture a photograph to begin."
            )
            QMessageBox.information(
                self, "ATHER-ACP", 
                "Upload or capture an image before running ACP analysis."
            )
            return

        self.acp_analyse_button.setEnabled(False)
        self.acp_status_label.setText(
            f"ATHER-ACP v{ACP_ENGINE_VERSION}: analysing current image..."
        )
        QApplication.processEvents()

        try:
            # Always analyse a fresh copy of the current original image.
            plan = build_processing_plan(self.original_image.copy())
            analysis = plan["analysis"]
            luminance = analysis["luminance"]
            image_info = analysis["image"]
            classification = plan["classification"]
            noise = analysis["noise"]

            dark_percent = 100.0 * luminance["dark_pixel_fraction_le_51"]
            black_percent = 100.0 * luminance["black_clipping_fraction_le_5"]
            highlight_percent = 100.0 * luminance["highlight_clipping_fraction_ge_250"]

            values = [
                f"{image_info['width']} × {image_info['height']}",
                f"{image_info['total_pixels']:,}",
                image_info["color_mode"],
                f"{luminance['mean']:.2f}",
                f"{luminance['median']:.2f}",
                f"{luminance['standard_deviation']:.2f}",
                f"{luminance['dynamic_range']} / 255",
                f"{dark_percent:.2f}%",
                f"{black_percent:.2f}% shadow / {highlight_percent:.2f}% highlight",
                f"{noise['noise_estimate_proxy']:.2f} / 255 (approx.)"
            ]
            for row, value in enumerate(values):
                self.acp_metrics_table.setItem(row, 1, QTableWidgetItem(value))

            active = classification.get("active_conditions", [])
            if active:
                condition_lines = []
                explanations = classification.get("explanations", {})
                for condition in active:
                    condition_lines.append(
                        f"• {condition.replace('_', ' ').title()}\n"
                        f"  {explanations.get(condition, 'No explanation available.')}"
                    )
                self.acp_conditions_text.setPlainText("\n\n".join(condition_lines))
            else:
                self.acp_conditions_text.setPlainText(
                    "No configured condition threshold was crossed. "
                    "This does not mean the photograph is perfect; it means "
                    "the current ACP rules did not identify a strong condition."
                )

            recommendations = plan.get("recommendations", [])
            self.acp_recommendations_table.setRowCount(len(recommendations))
            for row, recommendation in enumerate(recommendations):
                name_item = QTableWidgetItem(
                    recommendation.get("algorithm_name", "Unknown algorithm")
                )
                name_item.setData(
                    Qt.ItemDataRole.UserRole,
                    recommendation.get("algorithm_key")
                )
                priority_item = QTableWidgetItem(
                    f"{recommendation.get('priority_score', 0)}/100"
                )
                reasons = recommendation.get("reasons", [])
                reason_item = QTableWidgetItem("\n".join(reasons))
                for item in (name_item, priority_item, reason_item):
                    item.setToolTip(item.text())
                self.acp_recommendations_table.setItem(row, 0, name_item)
                self.acp_recommendations_table.setItem(row, 1, priority_item)
                self.acp_recommendations_table.setItem(row, 2, reason_item)

            if not recommendations:
                self.acp_recommendations_table.setRowCount(1)
                self.acp_recommendations_table.setItem(
                    0, 0, QTableWidgetItem("No recommendation")
                )
                self.acp_recommendations_table.setItem(
                    0, 1, QTableWidgetItem("—")
                )
                self.acp_recommendations_table.setItem(
                    0, 2, QTableWidgetItem(
                        "ACP has withheld recommendations for this image profile. "
                        "You can still manually test any registered algorithm."
                    )
                )

            self.acp_cautions_text.setPlainText(
                "\n\n".join(plan.get("cautions", []))
            )
            self.acp_explanation_text.setPlainText(
                plan.get("explanation", "No explanation was generated.")
            )

            self.acp_plan = plan
            intent = classification.get(
                "exposure_interpretation", "not_assessed"
            ).replace("_", " ").title()
            self.acp_status_label.setText(
                f"Analysis complete • ATHER-ACP v{ACP_ENGINE_VERSION} • "
                f"{intent} • {len(recommendations)} recommendation(s)"
            )
            self.update_acp_recommendation_button()

        except Exception as error:
            self.acp_plan = None
            self.acp_status_label.setText(
                "ACP analysis failed. The existing enhancement workflow remains available."
            )
            self.acp_conditions_text.setPlainText(str(error))
            self.acp_recommendations_table.setRowCount(0)
            self.acp_cautions_text.clear()
            self.acp_explanation_text.clear()
            self.acp_open_method_button.setEnabled(False)
        finally:
            self.acp_analyse_button.setEnabled(True)

    def update_acp_recommendation_button(self):
        row = self.acp_recommendations_table.currentRow()
        valid = False
        if row >= 0:
            item = self.acp_recommendations_table.item(row, 0)
            valid = bool(
                item and item.data(Qt.ItemDataRole.UserRole) in ALGORITHMS
            )
        self.acp_open_method_button.setEnabled(valid)

    def open_selected_acp_recommendation(self):
        row = self.acp_recommendations_table.currentRow()
        if row < 0:
            return
        item = self.acp_recommendations_table.item(row, 0)
        key = item.data(Qt.ItemDataRole.UserRole) if item else None
        if key not in ALGORITHMS:
            return

        index = self.algorithm_combo.findData(key)
        if index >= 0:
            self.algorithm_combo.setCurrentIndex(index)
        dashboard_index = self.dashboard_algorithm_combo.findData(key)
        if dashboard_index >= 0:
            self.dashboard_algorithm_combo.setCurrentIndex(dashboard_index)
        self.navigate("enhancement")

    # =====================================================
    # PHASE 3.2: UNIVERSAL PROCESSING EVALUATION
    # =====================================================

    def build_processing_evaluation_page(self):
        # Processing Evaluation is intentionally part of Image Analysis.
        # Both stages use the same currently loaded source photograph.
        layout = self.analysis_content_layout
        layout.addSpacing(10)
        layout.addWidget(make_section_label("Classical Processing Evaluation"))

        intro = QLabel(
            f"Evaluation engine v{EVALUATION_ENGINE_VERSION}. Select one or "
            "more classical components, then evaluate the current original "
            "image. The source image remains unchanged. Results include "
            "no-reference measurements and per-component processing time."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)

        selection_group = QGroupBox("Processing Components")
        selection_layout = QVBoxLayout(selection_group)
        selection_note = QLabel(
            "Choose any combination. Components are applied in the order "
            "shown below; chroma preservation uses the original image as "
            "its colour reference."
        )
        selection_note.setWordWrap(True)
        selection_note.setObjectName("muted")
        selection_layout.addWidget(selection_note)

        component_grid = QGridLayout()
        self.evaluation_component_checks = {}
        component_options = [
            ("denoising", "Edge-aware denoising"),
            ("illumination", "Illumination correction"),
            ("sharpening", "Edge-preserving sharpening"),
            ("chroma_preservation", "Original chroma preservation"),
        ]
        for index, (key, label_text) in enumerate(component_options):
            check = QCheckBox(label_text)
            check.setChecked(True)
            self.evaluation_component_checks[key] = check
            component_grid.addWidget(check, index // 2, index % 2)
        selection_layout.addLayout(component_grid)

        action_row = QHBoxLayout()
        self.processing_evaluate_button = QPushButton(
            "Evaluate Current Image"
        )
        self.processing_evaluate_button.clicked.connect(
            self.run_processing_evaluation
        )
        self.processing_evaluate_button.setEnabled(False)
        action_row.addWidget(self.processing_evaluate_button)

        self.processing_evaluation_status = QLabel(
            "Load or capture an image in the Image Workspace to begin."
        )
        self.processing_evaluation_status.setWordWrap(True)
        self.processing_evaluation_status.setObjectName("muted")
        action_row.addWidget(self.processing_evaluation_status, 1)
        selection_layout.addLayout(action_row)
        layout.addWidget(selection_group)

        metrics_group = QGroupBox("Original vs. Final Measurements")
        metrics_layout = QVBoxLayout(metrics_group)
        metrics_note = QLabel(
            "These are image measurements, not ground-truth quality scores. "
            "PSNR and SSIM are intentionally omitted because this workflow "
            "does not have a reference ground-truth image."
        )
        metrics_note.setWordWrap(True)
        metrics_note.setObjectName("muted")
        metrics_layout.addWidget(metrics_note)

        self.processing_metrics_table = QTableWidget(0, 3)
        self.processing_metrics_table.setHorizontalHeaderLabels([
            "Metric", "Original", "Processed"
        ])
        self.processing_metrics_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.processing_metrics_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.processing_metrics_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.processing_metrics_table.verticalHeader().setVisible(False)
        self.processing_metrics_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.processing_metrics_table.setAlternatingRowColors(True)
        self.processing_metrics_table.setMinimumHeight(270)
        metrics_layout.addWidget(self.processing_metrics_table)
        layout.addWidget(metrics_group)

        preview_group = QGroupBox("Evaluation Output")
        preview_layout = QVBoxLayout(preview_group)
        self.processing_evaluation_preview = make_preview_label(
            "The evaluated image will appear here", 300
        )
        preview_layout.addWidget(self.processing_evaluation_preview)
        self.processing_evaluation_save_button = QPushButton(
            "Save Evaluated Image"
        )
        self.processing_evaluation_save_button.setObjectName(
            "secondaryButton"
        )
        self.processing_evaluation_save_button.clicked.connect(
            self.save_processing_evaluation_image
        )
        self.processing_evaluation_save_button.setEnabled(False)
        preview_layout.addWidget(
            self.processing_evaluation_save_button,
            alignment=Qt.AlignmentFlag.AlignRight
        )
        layout.addWidget(preview_group)

        details_group = QGroupBox("Per-Component Evaluation Details")
        details_layout = QVBoxLayout(details_group)
        self.processing_evaluation_details = QPlainTextEdit()
        self.processing_evaluation_details.setReadOnly(True)
        self.processing_evaluation_details.setMinimumHeight(220)
        self.processing_evaluation_details.setPlaceholderText(
            "Run an evaluation to inspect component timings, metric changes, "
            "warnings and interpretation notes."
        )
        details_layout.addWidget(self.processing_evaluation_details)
        layout.addWidget(details_group)

        layout.addStretch(1)

    @staticmethod
    def _format_evaluation_value(value):
        if value is None:
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        return str(value)

    def run_processing_evaluation(self):
        """Evaluate selected Phase 3.1 components on ATHER's loaded image."""
        if self.original_image is None:
            QMessageBox.information(
                self,
                "No Image Loaded",
                "Upload or capture an image in ATHER before running evaluation."
            )
            return

        selected = [
            key for key, checkbox in self.evaluation_component_checks.items()
            if checkbox.isChecked()
        ]
        if not selected:
            QMessageBox.information(
                self,
                "Select a Component",
                "Choose at least one processing component to evaluate."
            )
            return

        self.processing_evaluate_button.setEnabled(False)
        self.processing_evaluation_status.setText(
            "Evaluating the currently loaded image..."
        )
        self.status_label_update("Running processing evaluation")
        QApplication.processEvents()

        try:
            # The evaluation engine receives the in-memory image from ATHER.
            # It never reads a hard-coded path and must not mutate this source.
            result = evaluate_components(
                self.original_image.copy(),
                selected_components=selected
            )

            final_image = result.get("final_image")
            if final_image is None:
                raise RuntimeError(
                    "The evaluation engine did not return a final image."
                )

            self.processing_evaluation_result = result
            self.processing_evaluation_image = final_image.copy()
            self.display_image(
                self.processing_evaluation_image,
                self.processing_evaluation_preview
            )
            self.processing_evaluation_save_button.setEnabled(True)

            original_metrics = result.get("original_metrics", {})
            final_metrics = result.get("final_metrics", {})
            metric_rows = [
                ("Brightness", "brightness"),
                ("Contrast (standard deviation)", "contrast"),
                ("Dynamic range", "dynamic_range"),
                ("Entropy", "entropy"),
                ("Sharpness (Laplacian variance)", "sharpness"),
                ("Estimated noise proxy", "noise_proxy"),
                ("Width (pixels)", "width"),
                ("Height (pixels)", "height"),
                ("Channels", "channels"),
                ("Mean saturation", "saturation"),
                ("Mean LAB chroma", "chroma"),
            ]
            self.processing_metrics_table.setRowCount(len(metric_rows))
            for row, (label, key) in enumerate(metric_rows):
                values = (
                    label,
                    self._format_evaluation_value(original_metrics.get(key)),
                    self._format_evaluation_value(final_metrics.get(key)),
                )
                for column, value in enumerate(values):
                    add_table_item(
                        self.processing_metrics_table, row, column, value
                    )

            steps = result.get("steps", [])
            total_time = result.get(
                "total_processing_time_ms",
                result.get("total_time_ms")
            )
            if total_time is None:
                total_time = sum(
                    float(step.get("processing_time_ms", 0.0))
                    for step in steps
                )

            lines = [
                f"Evaluation engine: v{EVALUATION_ENGINE_VERSION}",
                f"Selected components: {', '.join(selected)}",
                f"Total processing time: {float(total_time):.3f} ms",
                f"Input dimensions: {self.original_image.shape[1]} × "
                f"{self.original_image.shape[0]}",
                "",
                "COMPONENT-BY-COMPONENT RESULTS",
                "=" * 48,
            ]
            for index, step in enumerate(steps, start=1):
                component = step.get(
                    "component_name",
                    step.get("component", step.get("name", f"Component {index}"))
                )
                step_time = step.get("processing_time_ms", 0.0)
                lines.extend([
                    f"{index}. {str(component).replace('_', ' ').title()}",
                    f"   Processing time: {float(step_time):.3f} ms",
                ])
                step_metrics = step.get("metrics", {})
                if step_metrics:
                    lines.append("   Measurements after this step:")
                    for key in (
                        "brightness", "contrast", "dynamic_range",
                        "entropy", "sharpness", "noise_proxy"
                    ):
                        if key in step_metrics:
                            lines.append(
                                f"      {key.replace('_', ' ').title()}: "
                                f"{self._format_evaluation_value(step_metrics[key])}"
                            )
                changes = step.get(
                    "changes_from_original",
                    step.get("metric_changes", {})
                )
                if changes:
                    lines.append("   Change relative to original:")
                    for key, value in changes.items():
                        lines.append(
                            f"      {str(key).replace('_', ' ').title()}: "
                            f"{self._format_evaluation_value(value)}"
                        )
                lines.append("")

            warnings = result.get("warnings", [])
            notes = result.get(
                "evaluation_notes",
                result.get("notes", [])
            )
            if warnings:
                lines.extend(["WARNINGS", "=" * 48])
                lines.extend(f"• {warning}" for warning in warnings)
                lines.append("")
            if notes:
                lines.extend(["EVALUATION NOTES", "=" * 48])
                if isinstance(notes, (list, tuple)):
                    lines.extend(f"• {note}" for note in notes)
                else:
                    lines.append(str(notes))

            self.processing_evaluation_details.setPlainText("\n".join(lines))
            self.processing_evaluation_status.setText(
                f"Evaluation complete — {len(steps)} component(s), "
                f"{float(total_time):.3f} ms total."
            )
            self.status_label_update("Processing evaluation complete")

        except Exception as exc:
            self.processing_evaluation_status.setText(
                "Evaluation failed. See the error details."
            )
            QMessageBox.critical(
                self,
                "Processing Evaluation Failed",
                f"The current image could not be evaluated.\n\n{exc}"
            )
            self.status_label_update("Processing evaluation failed")
        finally:
            self.processing_evaluate_button.setEnabled(
                self.original_image is not None
            )

    def save_processing_evaluation_image(self):
        if self.processing_evaluation_image is None:
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Evaluated Image",
            "ATHER_evaluated.png",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;TIFF Image (*.tif *.tiff)"
        )
        if not path:
            return

        if not Path(path).suffix:
            if "JPEG" in selected_filter:
                path += ".jpg"
            elif "TIFF" in selected_filter:
                path += ".tif"
            else:
                path += ".png"

        try:
            if not cv2.imwrite(path, self.processing_evaluation_image):
                raise OSError("OpenCV could not write the selected output file.")
            QMessageBox.information(
                self,
                "Image Saved",
                f"Evaluated image saved successfully:\n{path}"
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Could not save the evaluated image.\n\n{exc}"
            )

    # =====================================================
    # ALGORITHM LAB - SEPARATE SIDEBAR SECTION
    # =====================================================

    def build_algorithm_lab(self):
        page, layout = self.create_page(
            "Algorithm Laboratory",
            "Explore registered classical algorithms, their purpose, "
            "mathematical descriptions and implementation source."
        )

        catalogue_group = QGroupBox(
            "Registered Algorithm Catalogue"
        )
        catalogue_layout = QVBoxLayout(catalogue_group)

        self.algorithm_table = QTableWidget(0, 3)
        self.algorithm_table.setHorizontalHeaderLabels([
            "Algorithm", "Purpose", "Implementation"
        ])

        self.algorithm_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.algorithm_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.algorithm_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )

        self.algorithm_table.setAlternatingRowColors(True)
        self.algorithm_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.algorithm_table.setMinimumHeight(260)

        catalogue_layout.addWidget(self.algorithm_table)
        layout.addWidget(catalogue_group)

        details_group = QGroupBox(
            "Algorithm Details and Source Inspection"
        )
        details_layout = QVBoxLayout(details_group)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Select algorithm"))

        self.algorithm_lab_combo = QComboBox()
        self.algorithm_lab_combo.currentIndexChanged.connect(
            self.update_algorithm_lab_details
        )
        selector_row.addWidget(self.algorithm_lab_combo, 1)

        details_layout.addLayout(selector_row)

        self.algorithm_lab_purpose = QLabel("Select an algorithm.")
        self.algorithm_lab_purpose.setWordWrap(True)

        self.algorithm_lab_formula = QLabel()
        self.algorithm_lab_formula.setWordWrap(True)

        details_layout.addWidget(make_section_label("Purpose"))
        details_layout.addWidget(self.algorithm_lab_purpose)
        details_layout.addWidget(make_section_label("Mathematical Formula"))
        details_layout.addWidget(self.algorithm_lab_formula)
        details_layout.addWidget(
            make_section_label("Implementation Source")
        )

        self.algorithm_lab_code = QPlainTextEdit()
        self.algorithm_lab_code.setReadOnly(True)
        self.algorithm_lab_code.setMinimumHeight(240)
        details_layout.addWidget(self.algorithm_lab_code)

        layout.addWidget(details_group)
        layout.addStretch()

        self.add_page("algorithms", page)

    def populate_algorithm_catalogue(self):
        self.algorithm_table.setRowCount(len(ALGORITHMS))
        self.algorithm_lab_combo.clear()

        for row, (key, details) in enumerate(ALGORITHMS.items()):
            self.algorithm_lab_combo.addItem(details["name"], key)

            add_table_item(
                self.algorithm_table, row, 0, details["name"]
            )
            add_table_item(
                self.algorithm_table, row, 1,
                details.get("purpose", "Not documented")
            )
            add_table_item(
                self.algorithm_table, row, 2, "Available"
            )

        self.update_algorithm_lab_details()

    def update_algorithm_lab_details(self):
        key = self.algorithm_lab_combo.currentData()

        if key not in ALGORITHMS:
            return

        details = ALGORITHMS[key]

        self.algorithm_lab_purpose.setText(
            details.get("purpose", "No purpose description available.")
        )
        self.algorithm_lab_formula.setText(
            details.get("formula", "No formula description available.")
        )

        try:
            source = inspect.getsource(details["function"])
        except (OSError, TypeError, KeyError):
            source = (
                "Source code is not directly available in this environment.\n"
                "See core/enhancement.py for the implementation."
            )

        self.algorithm_lab_code.setPlainText(source)

    def open_algorithm_details(self, key):
        index = self.algorithm_lab_combo.findData(key)

        if index >= 0:
            self.algorithm_lab_combo.setCurrentIndex(index)

        self.navigate("algorithms")

    # =====================================================
    # BENCHMARK LAB
    # =====================================================

    def build_benchmark_lab(self):
        page, layout = self.create_page(
            "Benchmark Laboratory",
            "Measure and compare every registered classical enhancement algorithm."
        )

        benchmark_group = QGroupBox("Performance Benchmarking")
        benchmark_layout = QVBoxLayout(benchmark_group)

        description = QLabel(
            "Every registered algorithm is evaluated using the same original "
            "image. Processing time and image statistics are measured separately."
        )
        description.setWordWrap(True)
        description.setObjectName("muted")
        benchmark_layout.addWidget(description)

        buttons = QHBoxLayout()

        self.benchmark_button = QPushButton("Run All Benchmarks")
        self.benchmark_button.clicked.connect(self.run_all_benchmarks)
        self.benchmark_button.setEnabled(False)

        buttons.addWidget(self.benchmark_button)
        buttons.addStretch()
        benchmark_layout.addLayout(buttons)

        self.benchmark_table = QTableWidget(0, 6)
        self.benchmark_table.setHorizontalHeaderLabels([
            "Algorithm", "Time (ms)", "Brightness",
            "Contrast", "Entropy", "Dynamic Range"
        ])

        self.benchmark_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.benchmark_table.setMinimumHeight(280)
        self.benchmark_table.setAlternatingRowColors(True)
        self.benchmark_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )

        benchmark_layout.addWidget(self.benchmark_table)
        layout.addWidget(benchmark_group)

        graph_group = QGroupBox("Performance Comparison")
        graph_layout = QVBoxLayout(graph_group)

        self.benchmark_canvas = BenchmarkCanvas()
        self.benchmark_canvas.plot_results(
            [], "brightness",
            "Run benchmarks to view comparisons",
            "Value"
        )
        graph_layout.addWidget(self.benchmark_canvas)

        self.graph_metric_combo = QComboBox()
        self.graph_metric_combo.addItem("Brightness", "brightness")
        self.graph_metric_combo.addItem("Contrast", "contrast")
        self.graph_metric_combo.addItem("Entropy", "entropy")
        self.graph_metric_combo.addItem(
            "Processing Time", "processing_time_ms"
        )
        self.graph_metric_combo.currentIndexChanged.connect(
            self.update_benchmark_graph
        )
        graph_layout.addWidget(self.graph_metric_combo)

        layout.addWidget(graph_group)
        layout.addStretch()

        self.add_page("benchmark", page)

    # =====================================================
    # REPORTS PAGE
    # =====================================================

    def build_reports_page(self):
        page, layout = self.create_page(
            "Research Reports",
            "Select any combination of benchmarked algorithms and generate "
            "a research-oriented PDF report."
        )

        selection_group = QGroupBox(
            "Select Algorithms for the PDF Report"
        )
        selection_layout = QVBoxLayout(selection_group)

        description = QLabel(
            "Choose one algorithm, several algorithms or all registered "
            "algorithms. Only checked algorithms will be included."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #75658D;")
        selection_layout.addWidget(description)

        selection_buttons = QHBoxLayout()

        self.select_all_report_button = QPushButton("Select All")
        self.select_all_report_button.setObjectName("secondaryButton")
        self.select_all_report_button.clicked.connect(
            self.select_all_report_algorithms
        )

        self.clear_all_report_button = QPushButton("Clear All")
        self.clear_all_report_button.setObjectName("secondaryButton")
        self.clear_all_report_button.clicked.connect(
            self.clear_all_report_algorithms
        )

        selection_buttons.addWidget(self.select_all_report_button)
        selection_buttons.addWidget(self.clear_all_report_button)
        selection_buttons.addStretch()

        selection_layout.addLayout(selection_buttons)

        self.report_selection_scroll = QScrollArea()
        self.report_selection_scroll.setWidgetResizable(True)
        self.report_selection_scroll.setMinimumHeight(190)
        self.report_selection_scroll.setMaximumHeight(300)

        selection_content = QWidget()
        self.report_selection_grid = QGridLayout(selection_content)
        self.report_selection_grid.setContentsMargins(8, 8, 8, 8)
        self.report_selection_grid.setHorizontalSpacing(22)
        self.report_selection_grid.setVerticalSpacing(10)

        self.report_selection_scroll.setWidget(selection_content)
        selection_layout.addWidget(self.report_selection_scroll)

        self.report_selection_count = QLabel(
            "Selected algorithms: 0"
        )
        self.report_selection_count.setStyleSheet(
            "color: #4C1D95; font-weight: bold;"
        )
        selection_layout.addWidget(self.report_selection_count)

        layout.addWidget(selection_group)

        report_group = QGroupBox("IEEE-Style Performance Report")
        report_layout = QVBoxLayout(report_group)

        report_description = QLabel(
            "Generate a PDF using the selected algorithms and their actual "
            "benchmark outputs. Run the Benchmark Lab first."
        )
        report_description.setWordWrap(True)
        report_layout.addWidget(report_description)

        self.report_page_status = QLabel(
            "No report generated in this session."
        )
        self.report_page_status.setWordWrap(True)
        self.report_page_status.setObjectName("muted")
        report_layout.addWidget(self.report_page_status)

        self.report_page_button = QPushButton(
            "Generate IEEE PDF Report"
        )
        self.report_page_button.clicked.connect(
            self.download_ieee_report
        )
        self.report_page_button.setEnabled(False)
        report_layout.addWidget(self.report_page_button)

        self.open_report_button = QPushButton("Open Generated Report")
        self.open_report_button.setObjectName("secondaryButton")
        self.open_report_button.clicked.connect(
            self.open_generated_report
        )
        self.open_report_button.setEnabled(False)
        report_layout.addWidget(self.open_report_button)

        layout.addWidget(report_group)
        layout.addStretch()

        self.add_page("reports", page)

    def populate_report_algorithm_selection(self):
        self.report_algorithm_checks.clear()

        while self.report_selection_grid.count():
            item = self.report_selection_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for index, (key, details) in enumerate(ALGORITHMS.items()):
            checkbox = QCheckBox(details.get("name", key))
            checkbox.setChecked(True)
            checkbox.setProperty("algorithm_key", key)
            checkbox.stateChanged.connect(
                self.update_report_selection_count
            )

            self.report_algorithm_checks[key] = checkbox

            self.report_selection_grid.addWidget(
                checkbox, index // 2, index % 2
            )

        self.update_report_selection_count()

    def get_selected_report_algorithms(self):
        return [
            key for key in ALGORITHMS
            if key in self.report_algorithm_checks
            and self.report_algorithm_checks[key].isChecked()
        ]

    def update_report_selection_count(self, *_):
        selected = self.get_selected_report_algorithms()
        total = len(ALGORITHMS)

        self.report_selection_count.setText(
            f"Selected algorithms: {len(selected)} / {total}"
        )

        if not selected:
            self.report_selection_count.setStyleSheet(
                "color: #F59E9B; font-weight: bold;"
            )
        else:
            self.report_selection_count.setStyleSheet(
                "color: #4C1D95; font-weight: bold;"
            )

        can_generate = bool(self.benchmark_results) and bool(selected)
        self.report_page_button.setEnabled(can_generate)

    def select_all_report_algorithms(self):
        for checkbox in self.report_algorithm_checks.values():
            checkbox.setChecked(True)

        self.update_report_selection_count()

    def clear_all_report_algorithms(self):
        for checkbox in self.report_algorithm_checks.values():
            checkbox.setChecked(False)

        self.update_report_selection_count()

    def open_generated_report(self):
        if not self.last_report_path:
            return

        path = Path(self.last_report_path)

        if not path.is_file():
            QMessageBox.warning(
                self, "Report Not Found",
                "The previously generated report cannot be found."
            )
            return

        try:
            os.startfile(str(path))
        except Exception as error:
            QMessageBox.warning(
                self, "Open Report Error", str(error)
            )

    # =====================================================
    # SETTINGS
    # =====================================================

    def build_settings_page(self):
        page, layout = self.create_page(
            "ATHER Settings",
            "Configure default processing values and manage the current session."
        )

        processing_group = QGroupBox("Processing Preferences")
        processing_layout = QFormLayout(processing_group)

        self.settings_gamma_spin = QDoubleSpinBox()
        self.settings_gamma_spin.setRange(0.1, 3.0)
        self.settings_gamma_spin.setSingleStep(0.1)
        self.settings_gamma_spin.setValue(self.default_gamma)

        processing_layout.addRow(
            "Default gamma value:",
            self.settings_gamma_spin
        )

        apply_settings_button = QPushButton("Apply Settings")
        apply_settings_button.clicked.connect(self.apply_settings)
        processing_layout.addRow(apply_settings_button)

        layout.addWidget(processing_group)

        session_group = QGroupBox("Session Management")
        session_layout = QVBoxLayout(session_group)

        reset_button = QPushButton("Clear Current Session")
        reset_button.setObjectName("secondaryButton")
        reset_button.clicked.connect(self.clear_session)
        session_layout.addWidget(reset_button)

        layout.addWidget(session_group)

        info_group = QGroupBox("Application Information")
        info_layout = QFormLayout(info_group)

        info_layout.addRow("Application:", QLabel("ATHER"))
        info_layout.addRow(
            "Processing type:", QLabel("Classical image processing")
        )
        info_layout.addRow("Operating system:", QLabel(platform.system()))
        info_layout.addRow("Python:", QLabel(platform.python_version()))
        info_layout.addRow("OpenCV:", QLabel(cv2.__version__))
        info_layout.addRow(
            "Registered algorithms:", QLabel(str(len(ALGORITHMS)))
        )

        layout.addWidget(info_group)
        layout.addStretch()

        self.add_page("settings", page)

    def apply_settings(self):
        self.default_gamma = self.settings_gamma_spin.value()

        self.gamma_spin.setValue(self.default_gamma)
        self.dashboard_gamma_spin.setValue(self.default_gamma)

        QMessageBox.information(
            self,
            "Settings Applied",
            f"Default gamma value set to {self.default_gamma:.2f}."
        )
        self.status_label_update("Settings applied")

    # =====================================================
    # SESSION RESET
    # =====================================================

    def clear_session(self):
        answer = QMessageBox.question(
            self,
            "Clear Session",
            "Clear the loaded image, enhancement and benchmark results?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if answer != QMessageBox.StandardButton.Yes:
            return

        self.stop_camera()

        self.original_image = None
        self.enhanced_image = None
        self.image_path = None
        self.original_metrics = None
        self.enhanced_metrics = None
        self.benchmark_results = []
        self.last_report_path = None

        for label, message in [
            (self.original_label, "Upload an image to begin"),
            (self.dashboard_original_label, "Upload or capture an image"),
            (self.enhanced_label, "Enhanced output will appear here"),
            (self.dashboard_enhanced_label, "Enhanced output will appear here")
        ]:
            label.clear()
            label.setText(message)

        self.workspace_file_label.setText("No image loaded")
        self.dashboard_file_label.setText("No image loaded")

        self.benchmark_table.setRowCount(0)

        self.original_histogram_canvas.plot_histograms(None)
        self.enhanced_histogram_canvas.plot_histograms(None)

        self.reset_analysis_table()

        self.benchmark_canvas.plot_results(
            [], "brightness",
            "Run benchmarks to view comparisons", "Value"
        )

        self.save_button.setEnabled(False)
        self.dashboard_save_button.setEnabled(False)

        self.apply_button.setEnabled(False)
        self.dashboard_apply_button.setEnabled(False)

        self.original_fullsize_button.setEnabled(False)
        self.enhanced_fullsize_button.setEnabled(False)
        self.dashboard_original_inspect.setEnabled(False)
        self.dashboard_enhanced_inspect.setEnabled(False)

        self.benchmark_button.setEnabled(False)
        self.report_page_button.setEnabled(False)
        self.open_report_button.setEnabled(False)

        self.dashboard_image_count.setText("0")
        self.dashboard_benchmark_count.setText("0")
        self.dashboard_report_status.setText("Ready")
        self.dashboard_current_image.setText("No image loaded")
        self.dashboard_processing_state.setText("Waiting for input")
        self.dashboard_image_dimensions.setText("Not available")

        self.analysis_original_status.setText(
            "ORIGINAL: Waiting for image"
        )
        self.analysis_enhanced_status.setText(
            "ENHANCED: No enhanced output"
        )

        self.report_page_status.setText(
            "No report generated in this session."
        )

        self.update_report_selection_count()
        self.status_label_update("Session cleared")
        self.navigate("dashboard")

    # =====================================================
    # ALGORITHM INFORMATION
    # =====================================================

    def populate_algorithms(self):
        self.algorithm_combo.clear()
        self.dashboard_algorithm_combo.clear()

        for key, details in ALGORITHMS.items():
            name = details["name"]
            self.algorithm_combo.addItem(name, key)
            self.dashboard_algorithm_combo.addItem(name, key)

        self.update_algorithm_information()
        self.update_dashboard_method_information()

    def update_algorithm_information(self):
        key = self.algorithm_combo.currentData()

        if key not in ALGORITHMS:
            return

        details = ALGORITHMS[key]

        self.purpose_label.setText(
            details.get("purpose", "No purpose description available.")
        )
        self.formula_label.setText(
            details.get("formula", "No formula description available.")
        )
        variables = details.get(
            "variables", "No variable description available."
        )

        # Algorithm metadata may provide variables as a dictionary.
        # QLabel.setText() requires a string, so format each entry
        # into a readable multi-line description before displaying it.
        if isinstance(variables, dict):
            variables_text = "\n".join(
                f"{str(name).replace('_', ' ').title()}: {description}"
                for name, description in variables.items()
            )
        elif isinstance(variables, (list, tuple, set)):
            variables_text = "\n".join(
                str(item) for item in variables
            )
        else:
            variables_text = str(variables)

        self.variables_label.setText(variables_text)

        try:
            source = inspect.getsource(details["function"])
        except (OSError, TypeError, KeyError):
            source = (
                "Source code is not directly available in this "
                "packaged environment.\n\n"
                "See core/enhancement.py for the implementation."
            )

        self.code_view.setPlainText(source)
        self.gamma_spin.setEnabled(key == "gamma")

    # =====================================================
    # FULL-RESOLUTION IMAGE INSPECTION
    # =====================================================

    def open_fullsize_preview(self, image, title):
        if image is None:
            return

        pixmap = self.convert_to_pixmap(image)

        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(
                self, "Preview Error",
                "Unable to create the full-resolution image preview."
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(
            f"{title} - 100% Resolution "
            f"({image.shape[1]} × {image.shape[0]})"
        )
        dialog.resize(1150, 820)
        dialog.setMinimumSize(700, 500)

        layout = QVBoxLayout(dialog)

        information = QLabel(
            "100% view: one displayed image pixel corresponds to one "
            "source image pixel. Use the scrollbars to inspect the image."
        )
        information.setWordWrap(True)
        layout.addWidget(information)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(False)
        scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        image_label = QLabel()
        image_label.setPixmap(pixmap)
        image_label.setFixedSize(pixmap.size())
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        scroll_area.setWidget(image_label)
        layout.addWidget(scroll_area)

        close_button = QPushButton("Close")
        close_button.setMinimumWidth(120)
        close_button.clicked.connect(dialog.accept)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        dialog.exec()

    def open_original_fullsize(self):
        self.open_fullsize_preview(
            self.original_image, "Original Image"
        )

    def open_enhanced_fullsize(self):
        self.open_fullsize_preview(
            self.enhanced_image, "Enhanced Image"
        )

    # =====================================================
    # IMAGE DISPLAY
    # =====================================================

    def convert_to_pixmap(self, image):
        if image is None:
            return None

        height, width = image.shape[:2]

        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        q_image = QImage(
            rgb.data,
            width,
            height,
            rgb.strides[0],
            QImage.Format.Format_RGB888
        ).copy()

        return QPixmap.fromImage(q_image)

    def display_image(self, image, target_label):
        if image is None:
            return

        pixmap = self.convert_to_pixmap(image)

        if pixmap is None:
            return

        available_size = target_label.contentsRect().size()

        if available_size.width() <= 0 or available_size.height() <= 0:
            return

        scaled_pixmap = pixmap.scaled(
            available_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        target_label.setPixmap(scaled_pixmap)

    def refresh_previews(self):
        preview_pairs = [
            (self.original_image, self.original_label),
            (self.enhanced_image, self.enhanced_label),
            (self.original_image, self.dashboard_original_label),
            (self.enhanced_image, self.dashboard_enhanced_label)
        ]

        for image, label in preview_pairs:
            if image is not None:
                self.display_image(image, label)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.refresh_previews)

    # =====================================================
    # IMAGE UPLOAD
    # =====================================================

    def upload_image(self, navigate_after=True):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )

        if not path:
            return

        self.stop_camera()

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if image is None:
            QMessageBox.warning(
                self, "Image Error",
                "Unable to open the selected image."
            )
            return

        if image.dtype != np.uint8:
            QMessageBox.warning(
                self, "Unsupported Image",
                "Please select an 8-bit image."
            )
            return

        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]

        if image.ndim not in (2, 3):
            QMessageBox.warning(
                self, "Unsupported Image",
                "The selected image format is not supported."
            )
            return

        if image.ndim == 3 and image.shape[2] != 3:
            QMessageBox.warning(
                self, "Unsupported Image",
                "Please select a grayscale or three-channel color image."
            )
            return

        self.set_new_original_image(image, path)

        if navigate_after:
            self.navigate("workspace")

    def set_new_original_image(self, image, path=None):
        self.original_image = image.copy()
        self.enhanced_image = None
        self.image_path = path

        self.original_metrics = None
        self.enhanced_metrics = None

        self.benchmark_results = []
        self.last_report_path = None
        self.processing_evaluation_result = None
        self.processing_evaluation_image = None

        if hasattr(self, "processing_evaluation_status"):
            self.processing_evaluation_status.setText(
                "Image changed. Run evaluation on the newly loaded image."
            )
            self.processing_evaluation_status.setEnabled(True)
            self.processing_evaluate_button.setEnabled(True)
            self.processing_evaluation_save_button.setEnabled(False)
            self.processing_evaluation_preview.clear()
            self.processing_evaluation_preview.setText(
                "The evaluated image will appear here"
            )
            self.processing_metrics_table.setRowCount(0)
            self.processing_evaluation_details.clear()

        filename = Path(path).name if path else "Webcam Captured Image"

        # Reset every image preview.
        for label in [
            self.original_label,
            self.dashboard_original_label
        ]:
            label.clear()

        for label in [
            self.enhanced_label,
            self.dashboard_enhanced_label
        ]:
            label.clear()
            label.setText("Enhanced output will appear here")

        self.workspace_file_label.setText(
            f"Loaded image: {filename}"
        )
        self.dashboard_file_label.setText(
            f"Loaded image: {filename}"
        )

        self.benchmark_table.setRowCount(0)
        self.report_button_state(False)

        self.save_button.setEnabled(False)
        self.dashboard_save_button.setEnabled(False)

        self.original_fullsize_button.setEnabled(True)
        self.dashboard_original_inspect.setEnabled(True)

        self.enhanced_fullsize_button.setEnabled(False)
        self.dashboard_enhanced_inspect.setEnabled(False)

        self.apply_button.setEnabled(True)
        self.dashboard_apply_button.setEnabled(True)
        self.benchmark_button.setEnabled(True)

        self.open_report_button.setEnabled(False)

        self.original_metrics = self.calculate_image_metrics(image)
        self.enhanced_metrics = None

        self.update_analysis_comparison()

        self.original_histogram_canvas.plot_histograms(image)
        self.enhanced_histogram_canvas.plot_histograms(None)

        self.dashboard_image_count.setText("1")
        self.dashboard_benchmark_count.setText("0")
        self.dashboard_report_status.setText("Ready")
        self.dashboard_current_image.setText(filename)
        self.dashboard_processing_state.setText(
            "Image loaded — ready for enhancement"
        )
        self.dashboard_image_dimensions.setText(
            f"{image.shape[1]} × {image.shape[0]}"
        )

        self.analysis_original_status.setText(
            "ORIGINAL: Image loaded"
        )
        self.analysis_enhanced_status.setText(
            "ENHANCED: Awaiting enhancement"
        )

        self.report_page_status.setText(
            "Image changed. Run benchmarks to enable report generation."
        )

        self.status_label_update("Image loaded successfully")

        # A new upload or webcam capture always receives its own fresh ACP plan.
        self.run_acp_analysis()

        self.dashboard_capture_button.setEnabled(
            self.camera is not None
        )

        QTimer.singleShot(0, self.refresh_previews)

    def report_button_state(self, enabled):
        self.report_page_button.setEnabled(
            bool(enabled)
            and bool(self.get_selected_report_algorithms())
        )

    # =====================================================
    # IMAGE METRICS AND COMPARISON
    # =====================================================

    def calculate_image_metrics(self, image):
        if image is None:
            return None

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        gray_float = gray.astype(np.float32)

        height, width = gray.shape
        minimum = int(np.min(gray))
        maximum = int(np.max(gray))

        histogram = cv2.calcHist(
            [gray], [0], None, [256], [0, 256]
        ).ravel()

        probability = histogram / max(
            float(histogram.sum()), 1.0
        )
        nonzero_probability = probability[probability > 0]

        entropy = float(
            -np.sum(
                nonzero_probability * np.log2(nonzero_probability)
            )
        )

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        residual = gray_float - blurred.astype(np.float32)

        noise = float(
            np.median(
                np.abs(residual - np.median(residual))
            ) / 0.6745
        )

        return {
            "dimensions": f"{width} × {height}",
            "pixels": width * height,
            "brightness": float(np.mean(gray_float)),
            "minimum": minimum,
            "maximum": maximum,
            "contrast": float(np.std(gray_float)),
            "entropy": entropy,
            "dynamic_range": maximum - minimum,
            "noise": noise
        }

    def reset_analysis_table(self):
        if not hasattr(self, "analysis_table"):
            return

        for row in range(self.analysis_table.rowCount()):
            for column in range(1, 4):
                item = self.analysis_table.item(row, column)
                if item is not None:
                    item.setText(
                        "Waiting" if column < 3 else "—"
                    )

    def update_analysis_comparison(self):
        if not hasattr(self, "analysis_table"):
            return

        rows = [
            ("Dimensions", "dimensions", "text"),
            ("Total Pixels", "pixels", "integer"),
            ("Average Brightness", "brightness", "float"),
            ("Minimum Intensity", "minimum", "integer"),
            ("Maximum Intensity", "maximum", "integer"),
            ("Contrast (Standard Deviation)", "contrast", "float"),
            ("Entropy", "entropy", "float"),
            ("Dynamic Range", "dynamic_range", "integer"),
            ("Estimated Noise", "noise", "noise")
        ]

        for row, (label, key, value_type) in enumerate(rows):
            original = (
                self.original_metrics.get(key)
                if self.original_metrics else None
            )

            enhanced = (
                self.enhanced_metrics.get(key)
                if self.enhanced_metrics else None
            )

            original_text = self.format_metric(
                original, value_type
            )
            enhanced_text = self.format_metric(
                enhanced, value_type
            )

            if original is None or enhanced is None:
                change_text = "—"
            elif value_type == "text":
                change_text = "—"
            else:
                difference = float(enhanced) - float(original)
                change_text = f"{difference:+.3f}"

            add_table_item(
                self.analysis_table, row, 0, label
            )
            add_table_item(
                self.analysis_table, row, 1, original_text
            )
            add_table_item(
                self.analysis_table, row, 2, enhanced_text
            )
            add_table_item(
                self.analysis_table, row, 3, change_text
            )

    @staticmethod
    def format_metric(value, value_type):
        if value is None:
            return "Waiting"

        if value_type == "text":
            return str(value)

        if value_type == "integer":
            return f"{int(value):,}"

        if value_type == "noise":
            return f"{float(value):.3f} (approx.)"

        return f"{float(value):.3f}"

    def analyse_image(self, image):
        self.original_metrics = self.calculate_image_metrics(image)
        self.update_analysis_comparison()

    # =====================================================
    # ENHANCEMENT
    # =====================================================

    def apply_selected_algorithm(self):
        if self.original_image is None:
            QMessageBox.warning(
                self, "No Image",
                "Upload an image or capture one first."
            )
            return

        self.perform_enhancement(
            self.algorithm_combo.currentData(),
            self.gamma_spin.value()
        )

    def apply_dashboard_enhancement(self):
        if self.original_image is None:
            QMessageBox.warning(
                self, "No Image",
                "Upload or capture an image on the Dashboard first."
            )
            return

        self.perform_enhancement(
            self.dashboard_algorithm_combo.currentData(),
            self.dashboard_gamma_spin.value()
        )

    def perform_enhancement(self, key, gamma_value):
        if self.original_image is None:
            return

        if key not in ALGORITHMS:
            QMessageBox.warning(
                self, "Algorithm Error",
                "Select a valid algorithm."
            )
            return

        try:
            self.status_label_update("Processing image")
            self.dashboard_processing_state.setText(
                "Enhancement in progress"
            )

            QApplication.processEvents()

            result = apply_enhancement(
                key,
                self.original_image.copy(),
                gamma_value
            )

            if result is None or result.size == 0:
                raise ValueError(
                    "Enhancement returned an empty image."
                )

            if result.shape != self.original_image.shape:
                raise ValueError(
                    "Enhancement changed the original image dimensions."
                )

            self.enhanced_image = result.copy()
            self.enhanced_metrics = self.calculate_image_metrics(
                result
            )

            self.save_button.setEnabled(True)
            self.dashboard_save_button.setEnabled(True)

            self.enhanced_fullsize_button.setEnabled(True)
            self.dashboard_enhanced_inspect.setEnabled(True)

            # Update only the enhanced histogram.
            # Original histogram remains unchanged.
            self.enhanced_histogram_canvas.plot_histograms(result)

            self.update_analysis_comparison()

            algorithm_name = ALGORITHMS[key]["name"]

            self.analysis_enhanced_status.setText(
                "ENHANCED: " + algorithm_name
            )

            self.dashboard_processing_state.setText(
                algorithm_name + " completed"
            )

            self.status_label_update(
                algorithm_name + " completed"
            )

            QTimer.singleShot(0, self.refresh_previews)

        except Exception as error:
            self.dashboard_processing_state.setText(
                "Enhancement failed"
            )

            QMessageBox.critical(
                self, "Enhancement Error", str(error)
            )
            self.status_label_update("Enhancement failed")

    # =====================================================
    # SAVE ENHANCED IMAGE
    # =====================================================

    def save_enhanced_image(self):
        if self.enhanced_image is None:
            QMessageBox.warning(
                self, "No Enhanced Image",
                "Apply an enhancement algorithm before saving."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Enhanced Image",
            "ATHER_enhanced.png",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;BMP Image (*.bmp)"
        )

        if not path:
            return

        success = cv2.imwrite(path, self.enhanced_image)

        if success:
            QMessageBox.information(
                self, "Saved",
                "Enhanced image saved successfully."
            )
            self.status_label_update("Enhanced image saved")
        else:
            QMessageBox.warning(
                self, "Save Error",
                "Unable to save the image."
            )

    # =====================================================
    # BENCHMARKS
    # =====================================================

    def run_all_benchmarks(self):
        if self.original_image is None:
            QMessageBox.warning(
                self, "No Image",
                "Upload or capture an image before benchmarking."
            )
            return

        algorithms = list(ALGORITHMS.keys())

        if not algorithms:
            QMessageBox.warning(
                self, "No Algorithms",
                "No algorithms are registered in core/enhancement.py."
            )
            return

        self.benchmark_results = []
        self.benchmark_button.setEnabled(False)
        self.report_button_state(False)

        self.status_label_update("Running benchmarks")
        self.dashboard_processing_state.setText(
            "Benchmarking algorithms"
        )
        QApplication.processEvents()

        try:
            for algorithm in algorithms:
                self.status_label_update(
                    "Benchmarking " + ALGORITHMS[algorithm]["name"]
                )
                QApplication.processEvents()

                result = benchmark_algorithm(
                    algorithm,
                    self.original_image.copy(),
                    self.gamma_spin.value()
                )

                if not isinstance(result, dict):
                    raise TypeError(
                        f"Benchmark for {algorithm} returned an invalid result."
                    )

                if result.get("algorithm") != algorithm:
                    raise ValueError(
                        f"Benchmark result key mismatch for {algorithm}."
                    )

                if "enhanced_image" not in result:
                    raise KeyError(
                        f"Benchmark result for {algorithm} has no enhanced image."
                    )

                self.benchmark_results.append(result)

            self.populate_benchmark_table()
            self.update_benchmark_graph()

            self.report_button_state(True)

            self.dashboard_benchmark_count.setText(
                str(len(self.benchmark_results))
            )
            self.dashboard_report_status.setText("Available")
            self.dashboard_processing_state.setText(
                "Benchmarking completed"
            )

            self.report_page_status.setText(
                f"{len(self.benchmark_results)} algorithms benchmarked. "
                "Select the algorithms you want in your PDF report."
            )

            self.status_label_update("All benchmarks completed")

            QMessageBox.information(
                self,
                "Benchmark Complete",
                f"Successfully benchmarked {len(self.benchmark_results)} "
                "registered algorithms."
            )

        except Exception as error:
            self.benchmark_results = []
            self.benchmark_table.setRowCount(0)
            self.report_button_state(False)

            self.dashboard_benchmark_count.setText("0")
            self.dashboard_report_status.setText("Ready")
            self.dashboard_processing_state.setText(
                "Benchmarking failed"
            )

            QMessageBox.critical(
                self, "Benchmark Error", str(error)
            )
            self.status_label_update("Benchmark failed")

        finally:
            self.benchmark_button.setEnabled(
                self.original_image is not None
            )
            self.update_report_selection_count()

    def populate_benchmark_table(self):
        self.benchmark_table.setRowCount(
            len(self.benchmark_results)
        )

        for row, result in enumerate(self.benchmark_results):
            metrics = result["enhanced_metrics"]

            values = [
                result["algorithm_name"],
                f"{result['processing_time_ms']:.3f}",
                f"{metrics['brightness']:.2f}",
                f"{metrics['contrast']:.2f}",
                f"{metrics['entropy']:.3f}",
                str(metrics["dynamic_range"])
            ]

            for column, value in enumerate(values):
                add_table_item(
                    self.benchmark_table, row, column, value
                )

    def update_benchmark_graph(self):
        if not self.benchmark_results:
            return

        metric = self.graph_metric_combo.currentData()

        titles = {
            "brightness": (
                "Mean Brightness Comparison", "Mean Intensity"
            ),
            "contrast": (
                "Contrast Comparison", "Standard Deviation"
            ),
            "entropy": (
                "Entropy Comparison", "Entropy (bits)"
            ),
            "processing_time_ms": (
                "Processing Time Comparison", "Time (milliseconds)"
            )
        }

        title, ylabel = titles[metric]

        self.benchmark_canvas.plot_results(
            self.benchmark_results,
            metric,
            title,
            ylabel
        )

    # =====================================================
    # IEEE REPORT GENERATION
    # =====================================================

    def download_ieee_report(self):
        if not self.benchmark_results:
            QMessageBox.warning(
                self, "No Benchmark Results",
                "Run all benchmarks before generating the report."
            )
            return

        selected_algorithms = self.get_selected_report_algorithms()

        if not selected_algorithms:
            QMessageBox.warning(
                self, "No Algorithms Selected",
                "Select at least one algorithm for the PDF report."
            )
            return

        default_name = (
            "ATHER_Performance_Report_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".pdf"
        )

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save IEEE-Style Performance Report",
            default_name,
            "PDF Files (*.pdf)"
        )

        if not path:
            return

        if not path.lower().endswith(".pdf"):
            path += ".pdf"

        image_name = (
            Path(self.image_path).name
            if self.image_path
            else "Webcam Captured Image"
        )

        results_by_key = {
            result["algorithm"]: result
            for result in self.benchmark_results
        }

        missing = [
            key for key in selected_algorithms
            if key not in results_by_key
        ]

        if missing:
            QMessageBox.warning(
                self,
                "Report Data Missing",
                "Benchmark results are missing for: "
                + ", ".join(missing)
                + ". Run the benchmarks again."
            )
            return

        report_results = [
            results_by_key[key]
            for key in selected_algorithms
        ]

        try:
            self.report_page_button.setEnabled(False)
            self.status_label_update("Preparing report images")
            QApplication.processEvents()

            with tempfile.TemporaryDirectory(
                prefix="ATHER_Report_"
            ) as temporary_directory:

                temporary_path = Path(temporary_directory)

                original_report_path = (
                    temporary_path / "original.png"
                )

                if not cv2.imwrite(
                    str(original_report_path),
                    self.original_image
                ):
                    raise RuntimeError(
                        "Unable to prepare the original image for the report."
                    )

                enhanced_image_paths = {}

                for algorithm in selected_algorithms:
                    self.status_label_update(
                        "Preparing " + ALGORITHMS[algorithm]["name"]
                    )
                    QApplication.processEvents()

                    enhanced = results_by_key[algorithm]["enhanced_image"]
                    enhanced_path = (
                        temporary_path / f"{algorithm}.png"
                    )

                    if not cv2.imwrite(str(enhanced_path), enhanced):
                        raise RuntimeError(
                            "Unable to save report image for "
                            + ALGORITHMS[algorithm]["name"]
                        )

                    enhanced_image_paths[algorithm] = str(
                        enhanced_path
                    )

                self.status_label_update("Generating IEEE report")
                QApplication.processEvents()

                system_information = {
                    "Operating System": platform.platform(),
                    "Python Version": platform.python_version(),
                    "Processor": platform.processor() or "Not reported",
                    "OpenCV Version": cv2.__version__,
                    "NumPy Version": np.__version__
                }

                algorithm_parameters = {
                    "gamma": {
                        "gamma_value": self.gamma_spin.value()
                    }
                }

                generated_path = generate_ieee_report(
                    report_results,
                    path,
                    image_name,
                    self.gamma_spin.value(),
                    original_image_path=str(original_report_path),
                    enhanced_image_paths=enhanced_image_paths,
                    algorithm_parameters=algorithm_parameters,
                    system_info=system_information
                )

            self.last_report_path = str(generated_path)

            self.dashboard_report_status.setText("Generated")

            selected_names = [
                ALGORITHMS[key]["name"]
                for key in selected_algorithms
            ]

            self.report_page_status.setText(
                "Report generated successfully.\n"
                f"Algorithms included: {len(selected_algorithms)}\n"
                + ", ".join(selected_names)
                + "\n\nSaved to:\n"
                + str(generated_path)
            )

            self.open_report_button.setEnabled(True)

            QMessageBox.information(
                self,
                "Report Generated",
                "Your performance report has been generated successfully.\n\n"
                f"Algorithms included: {len(selected_algorithms)}\n\n"
                + str(generated_path)
            )

            self.status_label_update("IEEE report generated")

        except Exception as error:
            QMessageBox.critical(
                self, "Report Generation Error", str(error)
            )
            self.status_label_update("Report generation failed")

        finally:
            self.update_report_selection_count()

    # =====================================================
    # WEBCAM
    # =====================================================

    def toggle_camera(self):
        if self.camera is not None:
            self.capture_camera_image()
            return

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None

            QMessageBox.warning(
                self, "Camera Error",
                "Unable to access webcam."
            )
            return

        self.camera_button.setText("Capture Image")
        self.dashboard_camera_button.setText("Capture Image")

        self.stop_camera_button.setEnabled(True)
        self.dashboard_capture_button.setEnabled(True)

        self.camera_timer.start(30)
        self.status_label_update("Webcam active")

    def update_camera_frame(self):
        if self.camera is None:
            return

        success, frame = self.camera.read()

        if success:
            self.display_image(frame, self.original_label)
            self.display_image(frame, self.dashboard_original_label)

    def capture_camera_image(self):
        if self.camera is None:
            QMessageBox.information(
                self, "Webcam Not Started",
                "Start the webcam before capturing an image."
            )
            return

        success, frame = self.camera.read()

        if not success:
            QMessageBox.warning(
                self, "Camera Error",
                "Unable to capture image."
            )
            return

        self.stop_camera()
        self.set_new_original_image(frame, None)

        # Keep the user on the page where capture was started.
        current_key = next(
            (
                key for key, index in self.page_indices.items()
                if index == self.page_stack.currentIndex()
            ),
            "dashboard"
        )

        self.navigate(current_key)

    def stop_camera(self):
        self.camera_timer.stop()

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        if hasattr(self, "camera_button"):
            self.camera_button.setText("Start Webcam")

        if hasattr(self, "dashboard_camera_button"):
            self.dashboard_camera_button.setText("Start Webcam")

        if hasattr(self, "stop_camera_button"):
            self.stop_camera_button.setEnabled(False)

        if hasattr(self, "dashboard_capture_button"):
            self.dashboard_capture_button.setEnabled(False)

        QTimer.singleShot(0, self.refresh_previews)

    # =====================================================
    # APPLICATION SHUTDOWN
    # =====================================================

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================

if __name__ == "__main__":
    print("Starting ATHER...")

    app = QApplication(sys.argv)
    app.setApplicationName("ATHER")
    app.setOrganizationName("ATHER Research System")

    window = ATHERApp()
    window.show()

    print("ATHER launched successfully!")

    sys.exit(app.exec())
