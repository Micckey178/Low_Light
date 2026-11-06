
import sys
import cv2
import numpy as np
import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QGroupBox,
    QGridLayout,
    QMessageBox,
    QScrollArea,
    QSizePolicy
)

from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QTimer


class HistogramCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(7, 3))
        self.ax = self.figure.add_subplot(111)

        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.updateGeometry()

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
                color="gray",
                alpha=0.8
            )
            self.ax.set_title("Grayscale Histogram")
        else:
            colors = ("b", "g", "r")
            labels = ("Blue", "Green", "Red")

            for channel, color, label in zip(
                cv2.split(image), colors, labels
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

        self.setWindowTitle("ATHER - Image Enhancement Laboratory")
        self.setMinimumSize(1100, 750)

        self.original_image = None
        self.current_image_path = None
        self.camera = None
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_camera_frame)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #f3f7f8;
            }

            QLabel {
                color: #18343a;
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

            QGroupBox {
                background-color: white;
                border: 1px solid #d6e3e5;
                border-radius: 9px;
                margin-top: 12px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                color: #087f8c;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }
        """)

        self.setup_ui()

    def setup_ui(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(22, 18, 22, 18)
        main_layout.setSpacing(14)

        # HEADER
        heading = QLabel("ATHER")
        heading.setStyleSheet(
            "font-size: 30px; font-weight: bold; color: #087f8c;"
        )

        subtitle = QLabel(
            "Classical Low-Light Image Enhancement and Analysis System"
        )
        subtitle.setStyleSheet("font-size: 14px; color: #526b70;")

        main_layout.addWidget(heading)
        main_layout.addWidget(subtitle)

        # INPUT BUTTONS
        button_layout = QHBoxLayout()

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)

        self.camera_button = QPushButton("Start Webcam")
        self.camera_button.clicked.connect(self.toggle_camera)

        self.stop_camera_button = QPushButton("Stop Webcam")
        self.stop_camera_button.clicked.connect(self.stop_camera)
        self.stop_camera_button.setEnabled(False)

        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.camera_button)
        button_layout.addWidget(self.stop_camera_button)
        button_layout.addStretch()

        main_layout.addLayout(button_layout)

        # IMAGE AND ANALYSIS AREA
        content_layout = QHBoxLayout()

        # IMAGE PREVIEW
        image_group = QGroupBox("Image Preview")
        image_layout = QVBoxLayout(image_group)

        self.image_label = QLabel(
            "Welcome to ATHER\n\nUpload an image or start your webcam."
        )
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(420, 330)
        self.image_label.setStyleSheet("""
            background-color: #e4edef;
            border: 2px dashed #91b6bc;
            border-radius: 8px;
            font-size: 16px;
            color: #34565c;
        """)

        image_layout.addWidget(self.image_label)

        # IMAGE INFORMATION
        info_group = QGroupBox("Image Analysis")
        info_layout = QGridLayout(info_group)

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
            name_label = QLabel(title + ":")
            name_label.setStyleSheet("font-weight: bold;")

            value_label = QLabel("Waiting for image")
            value_label.setStyleSheet("color: #087f8c;")

            info_layout.addWidget(name_label, row, 0)
            info_layout.addWidget(value_label, row, 1)

            self.info_labels[key] = value_label

        image_scroll = QScrollArea()
        image_scroll.setWidgetResizable(True)
        image_scroll.setWidget(info_group)
        image_scroll.setMinimumWidth(330)
        image_scroll.setMaximumWidth(430)

        content_layout.addWidget(image_group, 3)
        content_layout.addWidget(image_scroll, 2)

        main_layout.addLayout(content_layout, 3)

        # HISTOGRAM
        histogram_group = QGroupBox("Intensity Distribution")
        histogram_layout = QVBoxLayout(histogram_group)

        self.histogram_canvas = HistogramCanvas()
        histogram_layout.addWidget(self.histogram_canvas)

        main_layout.addWidget(histogram_group, 2)

        # STATUS
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet(
            "font-size: 12px; color: #526b70;"
        )

        main_layout.addWidget(self.status_label)

    # IMAGE UPLOAD
    def upload_image(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )

        if not file_path:
            return

        self.stop_camera()

        image = cv2.imread(file_path)

        if image is None:
            QMessageBox.warning(
                self,
                "Image Error",
                "The selected image could not be opened."
            )
            return

        self.original_image = image
        self.current_image_path = file_path

        self.display_image(image)
        self.analyse_image(image)

        self.status_label.setText(
            "Status: Image uploaded successfully"
        )

    # DISPLAY IMAGE
    def display_image(self, image):

        if image is None:
            return

        height, width = image.shape[:2]

        if len(image.shape) == 2:
            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )
        else:
            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        bytes_per_line = 3 * width

        from PySide6.QtGui import QImage

        q_image = QImage(
            rgb_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(q_image)

        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)

    # IMAGE ANALYSIS
    def analyse_image(self, image):

        if image is None:
            return

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        ) if len(image.shape) == 3 else image

        height, width = gray.shape

        total_pixels = width * height

        average_brightness = np.mean(gray)
        minimum_intensity = np.min(gray)
        maximum_intensity = np.max(gray)

        contrast = np.std(gray)

        # Simple high-frequency residual noise estimate.
        # This is an approximate indicator, not ground truth.
        blurred = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        residual = gray.astype(np.float32) - blurred.astype(np.float32)

        noise_estimate = np.median(
            np.abs(residual - np.median(residual))
        ) / 0.6745

        self.info_labels["dimensions"].setText(
            f"{width} × {height}"
        )

        self.info_labels["pixels"].setText(
            f"{total_pixels:,}"
        )

        self.info_labels["brightness"].setText(
            f"{average_brightness:.2f} / 255"
        )

        self.info_labels["minimum"].setText(
            str(int(minimum_intensity))
        )

        self.info_labels["maximum"].setText(
            str(int(maximum_intensity))
        )

        self.info_labels["contrast"].setText(
            f"{contrast:.2f}"
        )

        self.info_labels["noise"].setText(
            f"{noise_estimate:.2f} (approx.)"
        )

        self.histogram_canvas.plot_histograms(image)

    # WEBCAM
    def toggle_camera(self):

        if self.camera is not None:
            self.stop_camera()
            return

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None

            QMessageBox.warning(
                self,
                "Camera Error",
                "Unable to access the webcam. Check your camera connection and permissions."
            )
            return

        self.camera_button.setText("Capture Image")
        self.camera_button.clicked.disconnect()
        self.camera_button.clicked.connect(self.capture_camera_image)

        self.stop_camera_button.setEnabled(True)

        self.camera_timer.start(30)

        self.status_label.setText(
            "Status: Webcam active. Click Capture Image to take a photo."
        )

    def update_camera_frame(self):

        if self.camera is None:
            return

        success, frame = self.camera.read()

        if success:
            self.display_image(frame)

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

        self.stop_camera()

        self.display_image(self.original_image)
        self.analyse_image(self.original_image)

        self.status_label.setText(
            "Status: Webcam image captured and analysed"
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

        self.camera_button.clicked.connect(self.toggle_camera)

        self.stop_camera_button.setEnabled(False)

    def closeEvent(self, event):

        self.stop_camera()
        event.accept()


if __name__ == "__main__":

    print("Starting ATHER...")

    app = QApplication(sys.argv)

    window = ATHERApp()
    window.show()

    print("ATHER Phase 2 launched successfully!")

    sys.exit(app.exec())
