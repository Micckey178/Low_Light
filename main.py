
"""
ATHER - Image Enhancement Laboratory

Desktop application with:
- Image upload and webcam capture
- Classical image enhancement
- Mathematical and code explanations
- Image analysis and histograms
- 100% full-resolution image inspection
- Performance benchmarking
- IEEE-style PDF report generation
"""

import sys
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
    QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog,
    QGroupBox, QMessageBox, QComboBox, QDoubleSpinBox, QDialog,
    QPlainTextEdit, QSizePolicy, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView
)

from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer

from core.enhancement import ALGORITHMS, apply_enhancement
from core.performance import benchmark_algorithm
from core.report_generator import generate_ieee_report


# =========================================================
# HISTOGRAM CANVAS
# =========================================================

class HistogramCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None):

        self.figure = Figure(figsize=(7, 3))
        self.ax = self.figure.add_subplot(111)

        super().__init__(self.figure)

        self.setParent(parent)
        self.setMinimumHeight(260)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

    def plot_histograms(self, image):

        self.ax.clear()

        if image is None:

            self.ax.set_title("No image loaded")
            self.draw()
            return

        if len(image.shape) == 2:

            self.ax.hist(
                image.ravel(),
                bins=256,
                range=(0, 256),
                color="gray"
            )

            self.ax.set_title("Grayscale Histogram")

        else:

            for channel, color, label in zip(
                cv2.split(image),
                ("b", "g", "r"),
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

            self.ax.set_title("RGB Histogram")
            self.ax.legend()

        self.ax.set_xlabel("Pixel Intensity")
        self.ax.set_ylabel("Number of Pixels")
        self.ax.set_xlim(0, 255)
        self.ax.grid(alpha=0.2)

        self.figure.tight_layout()
        self.draw()


# =========================================================
# BENCHMARK CANVAS
# =========================================================

class BenchmarkCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None):

        self.figure = Figure(figsize=(8, 3))
        self.ax = self.figure.add_subplot(111)

        super().__init__(self.figure)

        self.setParent(parent)
        self.setMinimumHeight(260)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

    def plot_results(self, results, metric, title, ylabel):

        self.ax.clear()

        if not results:

            self.ax.set_title(title)
            self.draw()
            return

        names = [
            result["algorithm"].replace("_", " ").title()
            for result in results
        ]

        values = [
            result["enhanced_metrics"][metric]
            for result in results
        ]

        bars = self.ax.bar(names, values)

        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)

        self.ax.tick_params(
            axis="x",
            rotation=15
        )

        self.ax.grid(
            axis="y",
            alpha=0.2
        )

        for bar, value in zip(bars, values):

            self.ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

        self.figure.tight_layout()
        self.draw()


# =========================================================
# MAIN ATHER APPLICATION
# =========================================================

class ATHERApp(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "ATHER - Image Enhancement Laboratory"
        )

        self.resize(1350, 950)
        self.setMinimumSize(950, 650)

        self.original_image = None
        self.enhanced_image = None
        self.image_path = None

        self.camera = None
        self.benchmark_results = []

        self.camera_timer = QTimer(self)

        self.camera_timer.timeout.connect(
            self.update_camera_frame
        )

        self.setStyleSheet("""
            QMainWindow {
                background-color: #f3f7f8;
            }

            QWidget#contentWidget {
                background-color: #f3f7f8;
            }

            QLabel {
                color: #18343a;
                font-size: 13px;
            }

            QPushButton {
                background-color: #087f8c;
                color: white;
                border: none;
                border-radius: 7px;
                padding: 10px 16px;
                font-size: 13px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #066570;
            }

            QPushButton:pressed {
                background-color: #044e57;
            }

            QPushButton:disabled {
                background-color: #aabfc2;
                color: #edf3f4;
            }

            QGroupBox {
                background-color: white;
                border: 1px solid #d6e3e5;
                border-radius: 9px;
                margin-top: 14px;
                padding: 14px;
                font-size: 14px;
                font-weight: bold;
                color: #087f8c;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }

            QComboBox, QDoubleSpinBox {
                background-color: white;
                color: #18343a;
                border: 1px solid #b9cccf;
                border-radius: 5px;
                padding: 8px;
                min-height: 24px;
                font-size: 13px;
            }

            QComboBox QAbstractItemView {
                background-color: white;
                color: #18343a;
                selection-background-color: #087f8c;
                selection-color: white;
            }

            QPlainTextEdit {
                background-color: #182c32;
                color: #e3f3f4;
                border: none;
                border-radius: 7px;
                padding: 10px;
                font-family: Consolas;
                font-size: 12px;
            }

            QTableWidget {
                background-color: white;
                color: #18343a;
                gridline-color: #d6e3e5;
                border: 1px solid #d6e3e5;
            }

            QHeaderView::section {
                background-color: #087f8c;
                color: white;
                padding: 7px;
                font-weight: bold;
                border: none;
            }

            QScrollArea {
                border: none;
                background-color: #f3f7f8;
            }

            /* Main application vertical scrollbar */

            QScrollBar:vertical {
                background: #e2ebed;
                width: 15px;
                margin: 2px;
                border-radius: 7px;
            }

            QScrollBar::handle:vertical {
                background: #87aeb4;
                min-height: 55px;
                border-radius: 6px;
                border: 2px solid #e2ebed;
            }

            QScrollBar::handle:vertical:hover {
                background: #087f8c;
            }

            QScrollBar::handle:vertical:pressed {
                background: #066570;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
                background: transparent;
            }

            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
            }

            /* Main application horizontal scrollbar */

            QScrollBar:horizontal {
                background: #e2ebed;
                height: 15px;
                margin: 2px;
                border-radius: 7px;
            }

            QScrollBar::handle:horizontal {
                background: #87aeb4;
                min-width: 55px;
                border-radius: 6px;
                border: 2px solid #e2ebed;
            }

            QScrollBar::handle:horizontal:hover {
                background: #087f8c;
            }

            QScrollBar::handle:horizontal:pressed {
                background: #066570;
            }

            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0px;
                background: transparent;
            }

            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: transparent;
            }
        """)

        self.setup_ui()
        self.populate_algorithms()

    # =====================================================
    # USER INTERFACE
    # =====================================================

    def setup_ui(self):

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        self.content_widget = QWidget()
        self.content_widget.setObjectName("contentWidget")
        self.content_widget.setMinimumWidth(850)

        self.main_layout = QVBoxLayout(
            self.content_widget
        )

        self.main_layout.setContentsMargins(
            22, 18, 22, 22
        )

        self.main_layout.setSpacing(14)

        self.scroll_area.setWidget(
            self.content_widget
        )

        self.setCentralWidget(
            self.scroll_area
        )

        main = self.main_layout

        # HEADER

        heading = QLabel("ATHER")

        heading.setStyleSheet(
            "font-size: 32px; font-weight: bold; color: #087f8c;"
        )

        subtitle = QLabel(
            "Classical Low-Light Image Enhancement and Analysis System"
        )

        subtitle.setStyleSheet(
            "font-size: 14px; color: #526b70;"
        )

        main.addWidget(heading)
        main.addWidget(subtitle)

        # IMAGE INPUT BUTTONS

        buttons = QHBoxLayout()
        buttons.setSpacing(10)

        self.upload_button = QPushButton("Upload Image")

        self.upload_button.clicked.connect(
            self.upload_image
        )

        self.camera_button = QPushButton("Start Webcam")

        self.camera_button.clicked.connect(
            self.toggle_camera
        )

        self.stop_camera_button = QPushButton("Stop Webcam")

        self.stop_camera_button.clicked.connect(
            self.stop_camera
        )

        self.stop_camera_button.setEnabled(False)

        self.save_button = QPushButton(
            "Download Enhanced Image"
        )

        self.save_button.clicked.connect(
            self.save_enhanced_image
        )

        self.save_button.setEnabled(False)

        buttons.addWidget(self.upload_button)
        buttons.addWidget(self.camera_button)
        buttons.addWidget(self.stop_camera_button)
        buttons.addStretch()
        buttons.addWidget(self.save_button)

        main.addLayout(buttons)

        # ALGORITHM SELECTION

        algorithm_group = QGroupBox(
            "Enhancement Algorithm Selection"
        )

        algorithm_layout = QVBoxLayout(
            algorithm_group
        )

        algorithm_layout.setContentsMargins(
            18, 22, 18, 18
        )

        algorithm_layout.setSpacing(14)

        controls = QGridLayout()

        controls.setHorizontalSpacing(14)
        controls.setVerticalSpacing(12)

        controls.addWidget(
            QLabel("Select Algorithm"), 0, 0
        )

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.setMinimumHeight(38)

        self.algorithm_combo.currentIndexChanged.connect(
            self.update_algorithm_information
        )

        controls.addWidget(
            self.algorithm_combo, 0, 1
        )

        controls.addWidget(
            QLabel("Gamma Value"), 0, 2
        )

        self.gamma_spin = QDoubleSpinBox()

        self.gamma_spin.setRange(0.1, 3.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(0.6)
        self.gamma_spin.setFixedWidth(120)
        self.gamma_spin.setMinimumHeight(38)

        controls.addWidget(
            self.gamma_spin, 0, 3
        )

        controls.setColumnStretch(1, 1)

        algorithm_layout.addLayout(controls)

        purpose_title = QLabel("Purpose")
        purpose_title.setStyleSheet("font-weight: bold;")

        self.purpose_label = QLabel()
        self.purpose_label.setWordWrap(True)

        algorithm_layout.addWidget(purpose_title)
        algorithm_layout.addWidget(self.purpose_label)

        formula_title = QLabel("Mathematical Formula")
        formula_title.setStyleSheet("font-weight: bold;")

        self.formula_label = QLabel()
        self.formula_label.setWordWrap(True)

        algorithm_layout.addWidget(formula_title)
        algorithm_layout.addWidget(self.formula_label)

        variables_title = QLabel("Variable Explanation")
        variables_title.setStyleSheet("font-weight: bold;")

        self.variables_label = QLabel()
        self.variables_label.setWordWrap(True)

        algorithm_layout.addWidget(variables_title)
        algorithm_layout.addWidget(self.variables_label)

        code_title = QLabel("Actual Python Implementation")
        code_title.setStyleSheet("font-weight: bold;")

        self.code_view = QPlainTextEdit()
        self.code_view.setReadOnly(True)
        self.code_view.setMinimumHeight(170)
        self.code_view.setMaximumHeight(190)

        algorithm_layout.addWidget(code_title)
        algorithm_layout.addWidget(self.code_view)

        apply_layout = QHBoxLayout()
        apply_layout.addStretch()

        self.apply_button = QPushButton(
            "Apply Enhancement"
        )

        self.apply_button.setMinimumWidth(190)

        self.apply_button.clicked.connect(
            self.apply_selected_algorithm
        )

        self.apply_button.setEnabled(False)

        apply_layout.addWidget(self.apply_button)
        algorithm_layout.addLayout(apply_layout)

        main.addWidget(algorithm_group)

        # ORIGINAL AND ENHANCED IMAGE PANELS

        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(16)

        original_group = QGroupBox("Original Image")
        enhanced_group = QGroupBox("Enhanced Image")

        original_group.setMinimumWidth(0)
        enhanced_group.setMinimumWidth(0)

        original_group.setMinimumHeight(350)
        enhanced_group.setMinimumHeight(350)

        original_box = QVBoxLayout(original_group)
        enhanced_box = QVBoxLayout(enhanced_group)

        self.original_label = QLabel(
            "Upload an image to begin"
        )

        self.original_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.original_label.setMinimumSize(0, 280)

        self.original_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding
        )

        self.original_label.setStyleSheet("""
            background-color: #e4edef;
            color: #34565c;
            border: 2px dashed #91b6bc;
            border-radius: 8px;
        """)

        self.enhanced_label = QLabel(
            "Enhanced output will appear here"
        )

        self.enhanced_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.enhanced_label.setMinimumSize(0, 280)

        self.enhanced_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding
        )

        self.enhanced_label.setStyleSheet("""
            background-color: #e4edef;
            color: #34565c;
            border: 2px dashed #91b6bc;
            border-radius: 8px;
        """)

        original_box.addWidget(self.original_label)

        self.original_fullsize_button = QPushButton(
            "Inspect Original at 100%"
        )

        self.original_fullsize_button.clicked.connect(
            self.open_original_fullsize
        )

        self.original_fullsize_button.setEnabled(False)

        original_box.addWidget(
            self.original_fullsize_button
        )

        enhanced_box.addWidget(self.enhanced_label)

        self.enhanced_fullsize_button = QPushButton(
            "Inspect Enhanced at 100%"
        )

        self.enhanced_fullsize_button.clicked.connect(
            self.open_enhanced_fullsize
        )

        self.enhanced_fullsize_button.setEnabled(False)

        enhanced_box.addWidget(
            self.enhanced_fullsize_button
        )

        preview_layout.addWidget(original_group, 1)
        preview_layout.addWidget(enhanced_group, 1)

        main.addLayout(preview_layout)

        # IMAGE ANALYSIS AND HISTOGRAM

        analysis_layout = QHBoxLayout()
        analysis_layout.setSpacing(16)

        info_group = QGroupBox(
            "Original Image Analysis"
        )

        info_group.setMinimumWidth(300)

        info_grid = QGridLayout(info_group)

        self.info_labels = {}

        information = [
            ("Dimensions", "dimensions"),
            ("Total Pixels", "pixels"),
            ("Average Brightness", "brightness"),
            ("Minimum Intensity", "minimum"),
            ("Maximum Intensity", "maximum"),
            ("Contrast (Std. Dev.)", "contrast"),
            ("Estimated Noise", "noise")
        ]

        for row, (title, key) in enumerate(information):

            info_grid.addWidget(
                QLabel(title + ":"), row, 0
            )

            value = QLabel("Waiting")

            value.setStyleSheet(
                "color: #087f8c; font-weight: bold;"
            )

            info_grid.addWidget(value, row, 1)

            self.info_labels[key] = value

        histogram_group = QGroupBox(
            "Intensity Distribution"
        )

        histogram_layout = QVBoxLayout(
            histogram_group
        )

        self.histogram_canvas = HistogramCanvas()

        histogram_layout.addWidget(
            self.histogram_canvas
        )

        analysis_layout.addWidget(info_group, 1)
        analysis_layout.addWidget(histogram_group, 2)

        main.addLayout(analysis_layout)

        # PERFORMANCE BENCHMARK SECTION

        benchmark_group = QGroupBox(
            "Performance Benchmarking"
        )

        benchmark_layout = QVBoxLayout(
            benchmark_group
        )

        benchmark_description = QLabel(
            "Compare all four enhancement algorithms using "
            "the same original image."
        )

        benchmark_description.setWordWrap(True)

        benchmark_layout.addWidget(
            benchmark_description
        )

        benchmark_buttons = QHBoxLayout()

        self.benchmark_button = QPushButton(
            "Run All Benchmarks"
        )

        self.benchmark_button.clicked.connect(
            self.run_all_benchmarks
        )

        self.report_button = QPushButton(
            "Download IEEE Performance Report"
        )

        self.report_button.clicked.connect(
            self.download_ieee_report
        )

        self.report_button.setEnabled(False)

        benchmark_buttons.addWidget(
            self.benchmark_button
        )

        benchmark_buttons.addWidget(
            self.report_button
        )

        benchmark_layout.addLayout(
            benchmark_buttons
        )

        self.benchmark_table = QTableWidget(0, 6)

        self.benchmark_table.setHorizontalHeaderLabels([
            "Algorithm",
            "Time (ms)",
            "Brightness",
            "Contrast",
            "Entropy",
            "Dynamic Range"
        ])

        self.benchmark_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.benchmark_table.setMinimumHeight(190)

        self.benchmark_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )

        benchmark_layout.addWidget(
            self.benchmark_table
        )

        main.addWidget(benchmark_group)

        # BENCHMARK GRAPHS

        graph_group = QGroupBox(
            "Performance Comparison Graphs"
        )

        graph_layout = QVBoxLayout(graph_group)

        self.benchmark_canvas = BenchmarkCanvas()

        self.benchmark_canvas.plot_results(
            [],
            "brightness",
            "Run benchmarks to view comparisons",
            "Value"
        )

        graph_layout.addWidget(
            self.benchmark_canvas
        )

        self.graph_metric_combo = QComboBox()

        self.graph_metric_combo.addItem(
            "Brightness", "brightness"
        )

        self.graph_metric_combo.addItem(
            "Contrast", "contrast"
        )

        self.graph_metric_combo.addItem(
            "Entropy", "entropy"
        )

        self.graph_metric_combo.addItem(
            "Processing Time", "processing_time_ms"
        )

        self.graph_metric_combo.currentIndexChanged.connect(
            self.update_benchmark_graph
        )

        graph_layout.addWidget(
            self.graph_metric_combo
        )

        main.addWidget(graph_group)

        # STATUS

        self.status_label = QLabel(
            "Status: Ready"
        )

        self.status_label.setStyleSheet(
            "font-size: 12px; color: #526b70;"
        )

        main.addWidget(self.status_label)
        main.addSpacing(15)

    # =====================================================
    # ALGORITHM INFORMATION
    # =====================================================

    def populate_algorithms(self):

        for key, details in ALGORITHMS.items():

            self.algorithm_combo.addItem(
                details["name"],
                key
            )

        self.update_algorithm_information()

    def update_algorithm_information(self):

        key = self.algorithm_combo.currentData()

        if key not in ALGORITHMS:
            return

        details = ALGORITHMS[key]

        self.purpose_label.setText(
            details["purpose"]
        )

        self.formula_label.setText(
            details["formula"]
        )

        self.variables_label.setText(
            details["variables"]
        )

        self.code_view.setPlainText(
            inspect.getsource(details["function"])
        )

        self.gamma_spin.setEnabled(
            key == "gamma"
        )

    # =====================================================
    # FULL-RESOLUTION IMAGE INSPECTION
    # =====================================================

    def open_fullsize_preview(self, image, title):
        """
        Display the original pixel data at 100% scale.

        One displayed image pixel corresponds to one source
        image pixel. Scrollbars allow inspection of the
        entire image without reducing its resolution.
        """

        if image is None:
            return

        pixmap = self.convert_to_pixmap(image)

        if pixmap is None or pixmap.isNull():

            QMessageBox.warning(
                self,
                "Preview Error",
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

        dialog.setStyleSheet("""
            QDialog {
                background-color: #f3f7f8;
            }

            QLabel {
                color: #18343a;
                font-size: 12px;
            }

            QScrollArea {
                background-color: #dce7e9;
                border: 1px solid #a9bec2;
            }

            QScrollBar:vertical {
                background: #dce7e9;
                width: 15px;
                margin: 2px;
                border-radius: 7px;
            }

            QScrollBar::handle:vertical {
                background: #87aeb4;
                min-height: 55px;
                border-radius: 6px;
                border: 2px solid #dce7e9;
            }

            QScrollBar::handle:vertical:hover {
                background: #087f8c;
            }

            QScrollBar:horizontal {
                background: #dce7e9;
                height: 15px;
                margin: 2px;
                border-radius: 7px;
            }

            QScrollBar::handle:horizontal {
                background: #87aeb4;
                min-width: 55px;
                border-radius: 6px;
                border: 2px solid #dce7e9;
            }

            QScrollBar::handle:horizontal:hover {
                background: #087f8c;
            }

            QPushButton {
                background-color: #087f8c;
                color: white;
                border: none;
                border-radius: 7px;
                padding: 10px 24px;
                font-size: 13px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #066570;
            }
        """)

        layout = QVBoxLayout(dialog)

        information = QLabel(
            "100% view: one displayed image pixel corresponds "
            "to one source image pixel. Use the scrollbars "
            "to inspect the entire image."
        )

        information.setWordWrap(True)

        layout.addWidget(information)

        scroll_area = QScrollArea()

        scroll_area.setWidgetResizable(False)

        scroll_area.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        image_label = QLabel()

        image_label.setPixmap(pixmap)

        image_label.setFixedSize(
            pixmap.size()
        )

        image_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        scroll_area.setWidget(image_label)

        layout.addWidget(scroll_area)

        close_button = QPushButton("Close")

        close_button.setMinimumWidth(120)

        close_button.clicked.connect(
            dialog.accept
        )

        button_layout = QHBoxLayout()

        button_layout.addStretch()
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        dialog.exec()

    def open_original_fullsize(self):

        self.open_fullsize_preview(
            self.original_image,
            "Original Image"
        )

    def open_enhanced_fullsize(self):

        self.open_fullsize_preview(
            self.enhanced_image,
            "Enhanced Image"
        )

    # =====================================================
    # IMAGE DISPLAY
    # =====================================================

    def convert_to_pixmap(self, image):

        if image is None:
            return None

        height, width = image.shape[:2]

        if len(image.shape) == 2:

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

        else:

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

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

        if (
            available_size.width() <= 0
            or available_size.height() <= 0
        ):
            return

        # Only the preview is resized.
        # The actual image array remains full resolution.

        scaled_pixmap = pixmap.scaled(
            available_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        target_label.setPixmap(
            scaled_pixmap
        )

    def refresh_previews(self):

        if self.original_image is not None:

            self.display_image(
                self.original_image,
                self.original_label
            )

        if self.enhanced_image is not None:

            self.display_image(
                self.enhanced_image,
                self.enhanced_label
            )

    def resizeEvent(self, event):

        super().resizeEvent(event)

        QTimer.singleShot(
            0,
            self.refresh_previews
        )

    # =====================================================
    # IMAGE UPLOAD
    # =====================================================

    def upload_image(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )

        if not path:
            return

        self.stop_camera()

        image = cv2.imread(path)

        if image is None:

            QMessageBox.warning(
                self,
                "Image Error",
                "Unable to open the selected image."
            )

            return

        self.original_image = image.copy()
        self.enhanced_image = None
        self.image_path = path

        self.benchmark_results = []

        self.original_label.clear()
        self.enhanced_label.clear()

        self.enhanced_label.setText(
            "Enhanced output will appear here"
        )

        self.benchmark_table.setRowCount(0)

        self.report_button.setEnabled(False)

        self.save_button.setEnabled(False)

        self.original_fullsize_button.setEnabled(True)
        self.enhanced_fullsize_button.setEnabled(False)

        self.apply_button.setEnabled(True)
        self.benchmark_button.setEnabled(True)

        self.analyse_image(image)

        self.histogram_canvas.plot_histograms(image)

        self.status_label.setText(
            "Status: Image uploaded successfully"
        )

        QTimer.singleShot(
            0,
            self.refresh_previews
        )

    # =====================================================
    # IMAGE ANALYSIS
    # =====================================================

    def analyse_image(self, image):

        gray = (
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )
            if len(image.shape) == 3
            else image
        )

        height, width = gray.shape

        blurred = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        residual = (
            gray.astype(np.float32)
            - blurred.astype(np.float32)
        )

        noise = np.median(
            np.abs(
                residual - np.median(residual)
            )
        ) / 0.6745

        values = {
            "dimensions": f"{width} × {height}",
            "pixels": f"{width * height:,}",
            "brightness": f"{np.mean(gray):.2f} / 255",
            "minimum": str(int(np.min(gray))),
            "maximum": str(int(np.max(gray))),
            "contrast": f"{np.std(gray):.2f}",
            "noise": f"{noise:.2f} (approx.)"
        }

        for key, value in values.items():

            self.info_labels[key].setText(
                value
            )

    # =====================================================
    # APPLY ONE ENHANCEMENT
    # =====================================================

    def apply_selected_algorithm(self):

        if self.original_image is None:

            QMessageBox.warning(
                self,
                "No Image",
                "Upload an image or capture one first."
            )

            return

        key = self.algorithm_combo.currentData()

        try:

            self.status_label.setText(
                "Status: Processing image..."
            )

            QApplication.processEvents()

            result = apply_enhancement(
                key,
                self.original_image.copy(),
                self.gamma_spin.value()
            )

            if result is None or result.size == 0:

                raise ValueError(
                    "Enhancement returned an empty image."
                )

            self.enhanced_image = result.copy()

            self.save_button.setEnabled(True)

            self.enhanced_fullsize_button.setEnabled(True)

            self.histogram_canvas.plot_histograms(
                result
            )

            self.status_label.setText(
                "Status: "
                + ALGORITHMS[key]["name"]
                + " completed successfully"
            )

            QTimer.singleShot(
                0,
                self.refresh_previews
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Enhancement Error",
                str(error)
            )

            self.status_label.setText(
                "Status: Enhancement failed"
            )

    # =====================================================
    # SAVE ENHANCED IMAGE
    # =====================================================

    def save_enhanced_image(self):

        if self.enhanced_image is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Enhanced Image",
            "ATHER_enhanced.png",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;BMP Image (*.bmp)"
        )

        if not path:
            return

        success = cv2.imwrite(
            path,
            self.enhanced_image
        )

        if success:

            QMessageBox.information(
                self,
                "Saved",
                "Enhanced image saved successfully."
            )

            self.status_label.setText(
                "Status: Enhanced image saved"
            )

        else:

            QMessageBox.warning(
                self,
                "Save Error",
                "Unable to save the image."
            )

    # =====================================================
    # RUN ALL BENCHMARKS
    # =====================================================

    def run_all_benchmarks(self):

        if self.original_image is None:

            QMessageBox.warning(
                self,
                "No Image",
                "Upload or capture an image before benchmarking."
            )

            return

        algorithms = [
            "gamma",
            "contrast_stretching",
            "histogram_equalization",
            "clahe"
        ]

        self.benchmark_results = []

        self.benchmark_button.setEnabled(False)
        self.report_button.setEnabled(False)

        self.status_label.setText(
            "Status: Running performance benchmarks..."
        )

        QApplication.processEvents()

        try:

            for algorithm in algorithms:

                self.status_label.setText(
                    f"Status: Benchmarking {algorithm}..."
                )

                QApplication.processEvents()

                gamma = self.gamma_spin.value()

                result = benchmark_algorithm(
                    algorithm,
                    self.original_image.copy(),
                    gamma
                )

                self.benchmark_results.append(
                    result
                )

            self.populate_benchmark_table()

            self.update_benchmark_graph()

            self.report_button.setEnabled(True)

            self.status_label.setText(
                "Status: All four benchmarks completed successfully"
            )

            QMessageBox.information(
                self,
                "Benchmark Complete",
                "All four algorithms have been benchmarked successfully."
            )

        except Exception as error:

            self.benchmark_results = []

            self.report_button.setEnabled(False)

            QMessageBox.critical(
                self,
                "Benchmark Error",
                str(error)
            )

            self.status_label.setText(
                "Status: Benchmark failed"
            )

        finally:

            self.benchmark_button.setEnabled(True)

    # =====================================================
    # DISPLAY BENCHMARK TABLE
    # =====================================================

    def populate_benchmark_table(self):

        self.benchmark_table.setRowCount(
            len(self.benchmark_results)
        )

        for row, result in enumerate(
            self.benchmark_results
        ):

            metrics = result["enhanced_metrics"]

            values = [
                result["algorithm"].replace(
                    "_", " "
                ).title(),

                f"{result['processing_time_ms']:.3f}",

                f"{metrics['brightness']:.2f}",

                f"{metrics['contrast']:.2f}",

                f"{metrics['entropy']:.3f}",

                str(metrics["dynamic_range"])
            ]

            for column, value in enumerate(values):

                item = QTableWidgetItem(value)

                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter
                )

                self.benchmark_table.setItem(
                    row,
                    column,
                    item
                )

    # =====================================================
    # UPDATE BENCHMARK GRAPH
    # =====================================================

    def update_benchmark_graph(self):

        if not self.benchmark_results:
            return

        metric = self.graph_metric_combo.currentData()

        if metric == "processing_time_ms":

            names = [
                result["algorithm"].replace(
                    "_", " "
                ).title()
                for result in self.benchmark_results
            ]

            values = [
                result["processing_time_ms"]
                for result in self.benchmark_results
            ]

            self.benchmark_canvas.ax.clear()

            bars = self.benchmark_canvas.ax.bar(
                names,
                values
            )

            self.benchmark_canvas.ax.set_title(
                "Processing Time Comparison"
            )

            self.benchmark_canvas.ax.set_ylabel(
                "Time (milliseconds)"
            )

            self.benchmark_canvas.ax.tick_params(
                axis="x",
                rotation=15
            )

            self.benchmark_canvas.ax.grid(
                axis="y",
                alpha=0.2
            )

            for bar, value in zip(bars, values):

                self.benchmark_canvas.ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

            self.benchmark_canvas.figure.tight_layout()
            self.benchmark_canvas.draw()

            return

        titles = {
            "brightness": (
                "Mean Brightness Comparison",
                "Mean Intensity"
            ),

            "contrast": (
                "Contrast Comparison",
                "Standard Deviation"
            ),

            "entropy": (
                "Entropy Comparison",
                "Entropy (bits)"
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
    # DOWNLOAD IEEE-STYLE REPORT
    # =====================================================

    def download_ieee_report(self):

        if not self.benchmark_results:

            QMessageBox.warning(
                self,
                "No Benchmark Results",
                "Run all four benchmarks before generating the report."
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

        try:

            self.status_label.setText(
                "Status: Preparing report images..."
            )

            QApplication.processEvents()

            # A temporary directory holds report images.
            # The files remain available until PDF generation
            # has completed.

            with tempfile.TemporaryDirectory(
                prefix="ATHER_Report_"
            ) as temporary_directory:

                temporary_path = Path(
                    temporary_directory
                )

                original_report_path = (
                    temporary_path / "original.png"
                )

                original_saved = cv2.imwrite(
                    str(original_report_path),
                    self.original_image
                )

                if not original_saved:

                    raise RuntimeError(
                        "Unable to prepare the original image "
                        "for the report."
                    )

                algorithms = [
                    "gamma",
                    "contrast_stretching",
                    "histogram_equalization",
                    "clahe"
                ]

                enhanced_image_paths = {}

                for algorithm in algorithms:

                    self.status_label.setText(
                        "Status: Preparing "
                        + algorithm.replace("_", " ")
                        + " comparison image..."
                    )

                    QApplication.processEvents()

                    enhanced = apply_enhancement(
                        algorithm,
                        self.original_image.copy(),
                        self.gamma_spin.value()
                    )

                    if enhanced is None or enhanced.size == 0:

                        raise RuntimeError(
                            "Unable to generate report image for "
                            + algorithm
                        )

                    enhanced_path = (
                        temporary_path
                        / f"{algorithm}.png"
                    )

                    saved = cv2.imwrite(
                        str(enhanced_path),
                        enhanced
                    )

                    if not saved:

                        raise RuntimeError(
                            "Unable to save report comparison image for "
                            + algorithm
                        )

                    enhanced_image_paths[algorithm] = str(
                        enhanced_path
                    )

                self.status_label.setText(
                    "Status: Generating IEEE performance report..."
                )

                QApplication.processEvents()

                system_information = {
                    "Operating System": platform.platform(),
                    "Python Version": platform.python_version(),
                    "Processor": platform.processor() or "Not reported",
                    "OpenCV Version": cv2.__version__,
                    "NumPy Version": np.__version__
                }

                generated_path = generate_ieee_report(
                    self.benchmark_results,
                    path,
                    image_name,
                    self.gamma_spin.value(),

                    original_image_path=str(
                        original_report_path
                    ),

                    enhanced_image_paths=enhanced_image_paths,

                    system_info=system_information
                )

            QMessageBox.information(
                self,
                "Report Generated",
                "Your performance report has been generated successfully.\n\n"
                + generated_path
            )

            self.status_label.setText(
                "Status: IEEE-style PDF report generated"
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Report Generation Error",
                str(error)
            )

            self.status_label.setText(
                "Status: Report generation failed"
            )

    # =====================================================
    # WEBCAM
    # =====================================================

    def toggle_camera(self):

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():

            self.camera.release()
            self.camera = None

            QMessageBox.warning(
                self,
                "Camera Error",
                "Unable to access webcam."
            )

            return

        self.camera_button.setText(
            "Capture Image"
        )

        try:

            self.camera_button.clicked.disconnect()

        except RuntimeError:
            pass

        self.camera_button.clicked.connect(
            self.capture_camera_image
        )

        self.stop_camera_button.setEnabled(True)

        self.camera_timer.start(30)

        self.status_label.setText(
            "Status: Webcam active"
        )

    def update_camera_frame(self):

        if self.camera is None:
            return

        success, frame = self.camera.read()

        if success:

            self.display_image(
                frame,
                self.original_label
            )

    def capture_camera_image(self):

        if self.camera is None:
            return

        success, frame = self.camera.read()

        if not success:

            QMessageBox.warning(
                self,
                "Camera Error",
                "Unable to capture image."
            )

            return

        self.original_image = frame.copy()
        self.enhanced_image = None
        self.image_path = None

        self.stop_camera()

        self.enhanced_label.clear()

        self.enhanced_label.setText(
            "Enhanced output will appear here"
        )

        self.save_button.setEnabled(False)

        self.original_fullsize_button.setEnabled(True)
        self.enhanced_fullsize_button.setEnabled(False)

        self.apply_button.setEnabled(True)
        self.benchmark_button.setEnabled(True)

        self.benchmark_results = []

        self.benchmark_table.setRowCount(0)
        self.report_button.setEnabled(False)

        self.analyse_image(
            self.original_image
        )

        self.histogram_canvas.plot_histograms(
            self.original_image
        )

        self.status_label.setText(
            "Status: Webcam image captured successfully"
        )

        QTimer.singleShot(
            0,
            self.refresh_previews
        )

    def stop_camera(self):

        self.camera_timer.stop()

        if self.camera is not None:

            self.camera.release()
            self.camera = None

        self.camera_button.setText(
            "Start Webcam"
        )

        try:

            self.camera_button.clicked.disconnect()

        except RuntimeError:
            pass

        self.camera_button.clicked.connect(
            self.toggle_camera
        )

        self.stop_camera_button.setEnabled(False)

    def closeEvent(self, event):

        self.stop_camera()
        event.accept()


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================

if __name__ == "__main__":

    print("Starting ATHER...")

    app = QApplication(sys.argv)

    window = ATHERApp()
    window.show()

    print("ATHER launched successfully!")

    sys.exit(app.exec())
