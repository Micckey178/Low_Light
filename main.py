
import sys
import cv2
import numpy as np
import inspect

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog,
    QGroupBox, QMessageBox, QComboBox, QDoubleSpinBox,
    QPlainTextEdit, QSizePolicy, QScrollArea
)

from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer

from core.enhancement import ALGORITHMS, apply_enhancement


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
                image.ravel(), bins=256,
                range=(0, 256), color="gray"
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


class ATHERApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "ATHER - Image Enhancement Laboratory"
        )
        self.resize(1350, 900)
        self.setMinimumSize(950, 650)

        self.original_image = None
        self.enhanced_image = None
        self.camera = None

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

            QPushButton:disabled {
                background-color: #aabfc2;
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
                border: 1px solid #b9cccf;
                outline: none;
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

            QScrollArea {
                border: none;
                background-color: #f3f7f8;
            }

            QScrollBar:vertical {
                background: #e2ebed;
                width: 12px;
                margin: 2px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical {
                background: #87aeb4;
                min-height: 35px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical:hover {
                background: #087f8c;
            }
        """)

        self.setup_ui()
        self.populate_algorithms()

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
        self.setCentralWidget(self.scroll_area)

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

        # TOP BUTTONS

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

        # ALGORITHM SECTION

        algorithm_group = QGroupBox(
            "Enhancement Algorithm Selection"
        )

        algorithm_layout = QVBoxLayout(algorithm_group)
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
        self.purpose_label.setMinimumHeight(35)

        algorithm_layout.addWidget(purpose_title)
        algorithm_layout.addWidget(self.purpose_label)

        formula_title = QLabel("Mathematical Formula")
        formula_title.setStyleSheet("font-weight: bold;")

        self.formula_label = QLabel()
        self.formula_label.setWordWrap(True)
        self.formula_label.setMinimumHeight(30)

        algorithm_layout.addWidget(formula_title)
        algorithm_layout.addWidget(self.formula_label)

        variables_title = QLabel("Variable Explanation")
        variables_title.setStyleSheet("font-weight: bold;")

        self.variables_label = QLabel()
        self.variables_label.setWordWrap(True)
        self.variables_label.setMinimumHeight(65)

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

        self.apply_button = QPushButton("Apply Enhancement")
        self.apply_button.setMinimumWidth(190)
        self.apply_button.clicked.connect(
            self.apply_selected_algorithm
        )
        self.apply_button.setEnabled(False)

        apply_layout.addWidget(self.apply_button)
        algorithm_layout.addLayout(apply_layout)

        main.addWidget(algorithm_group)

        # IMAGE PREVIEW SECTION

        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(16)

        # Both panels have equal stretch and equal minimum size.

        original_group = QGroupBox("Original Image")
        enhanced_group = QGroupBox("Enhanced Image")

        original_group.setMinimumWidth(0)
        enhanced_group.setMinimumWidth(0)

        original_group.setMinimumHeight(350)
        enhanced_group.setMinimumHeight(350)

        original_box = QVBoxLayout(original_group)
        enhanced_box = QVBoxLayout(enhanced_group)

        original_box.setContentsMargins(12, 16, 12, 12)
        enhanced_box.setContentsMargins(12, 16, 12, 12)

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
        enhanced_box.addWidget(self.enhanced_label)

        # Equal stretch factors ensure equal panel widths.
        preview_layout.addWidget(original_group, 1)
        preview_layout.addWidget(enhanced_group, 1)

        main.addLayout(preview_layout)

        # ANALYSIS AND HISTOGRAM

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

        histogram_layout = QVBoxLayout(histogram_group)

        self.histogram_canvas = HistogramCanvas()
        histogram_layout.addWidget(
            self.histogram_canvas
        )

        analysis_layout.addWidget(info_group, 1)
        analysis_layout.addWidget(histogram_group, 2)

        main.addLayout(analysis_layout)

        # STATUS

        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet(
            "font-size: 12px; color: #526b70;"
        )

        main.addWidget(self.status_label)
        main.addSpacing(15)

    # ALGORITHM LIST

    def populate_algorithms(self):

        for key, details in ALGORITHMS.items():
            self.algorithm_combo.addItem(
                details["name"], key
            )

        self.update_algorithm_information()

    # ALGORITHM INFORMATION

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

    # IMAGE CONVERSION AND DISPLAY

    def convert_to_pixmap(self, image):

        if image is None:
            return None

        height, width = image.shape[:2]

        if len(image.shape) == 2:
            rgb = cv2.cvtColor(
                image, cv2.COLOR_GRAY2RGB
            )
        else:
            rgb = cv2.cvtColor(
                image, cv2.COLOR_BGR2RGB
            )

        # Use the actual row stride for safe QImage conversion.
        bytes_per_line = rgb.strides[0]

        q_image = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
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

        # Refit both images whenever the application is resized.
        QTimer.singleShot(
            0, self.refresh_previews
        )

    # UPLOAD IMAGE

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

        self.original_label.clear()
        self.enhanced_label.clear()

        self.enhanced_label.setText(
            "Enhanced output will appear here"
        )

        self.save_button.setEnabled(False)
        self.apply_button.setEnabled(True)

        self.analyse_image(image)
        self.histogram_canvas.plot_histograms(image)

        self.status_label.setText(
            "Status: Image uploaded successfully"
        )

        QTimer.singleShot(
            0, self.refresh_previews
        )

    # IMAGE ANALYSIS

    def analyse_image(self, image):

        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image
        )

        height, width = gray.shape

        blurred = cv2.GaussianBlur(
            gray, (3, 3), 0
        )

        residual = (
            gray.astype(np.float32)
            - blurred.astype(np.float32)
        )

        noise = np.median(
            np.abs(residual - np.median(residual))
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
            self.info_labels[key].setText(value)

    # APPLY ENHANCEMENT

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

            self.histogram_canvas.plot_histograms(
                result
            )

            self.status_label.setText(
                "Status: "
                + ALGORITHMS[key]["name"]
                + " completed successfully"
            )

            QTimer.singleShot(
                0, self.refresh_previews
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

    # SAVE ENHANCED IMAGE

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
            path, self.enhanced_image
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

    # WEBCAM

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

        self.camera_button.setText("Capture Image")

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
                frame, self.original_label
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

        self.stop_camera()

        self.enhanced_label.clear()
        self.enhanced_label.setText(
            "Enhanced output will appear here"
        )

        self.save_button.setEnabled(False)
        self.apply_button.setEnabled(True)

        self.analyse_image(self.original_image)

        self.histogram_canvas.plot_histograms(
            self.original_image
        )

        self.status_label.setText(
            "Status: Webcam image captured successfully"
        )

        QTimer.singleShot(
            0, self.refresh_previews
        )

    def stop_camera(self):

        self.camera_timer.stop()

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.camera_button.setText("Start Webcam")

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


if __name__ == "__main__":

    print("Starting ATHER...")

    app = QApplication(sys.argv)

    window = ATHERApp()
    window.show()

    print("ATHER launched successfully!")

    sys.exit(app.exec())
