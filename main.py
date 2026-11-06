
import sys

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog
)

from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


class ATHERApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("ATHER - Image Enhancement Laboratory")
        self.setMinimumSize(1000, 650)

        self.image_path = None

        # Main widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 25, 30, 25)
        main_layout.setSpacing(20)

        # Application title
        title = QLabel("ATHER")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size: 36px;"
            "font-weight: bold;"
            "color: #167D78;"
        )

        # Application subtitle
        subtitle = QLabel(
            "CLASSICAL IMAGE ENHANCEMENT LABORATORY"
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(
            "font-size: 13px;"
            "color: #667777;"
        )

        # Welcome message
        welcome = QLabel(
            "Bring your images back to life."
        )
        welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome.setStyleSheet(
            "font-size: 23px;"
            "font-weight: bold;"
            "color: #263B3A;"
        )

        # Description
        description = QLabel(
            "Upload or capture an image, analyse it and enhance it "
            "using classical image-processing techniques."
        )
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        description.setStyleSheet(
            "font-size: 14px;"
            "color: #526563;"
        )

        # Upload image button
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)

        # Camera button
        self.camera_button = QPushButton("Capture Image")
        self.camera_button.clicked.connect(self.capture_image)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.camera_button)

        # Image preview
        self.preview = QLabel(
            "Your original image will appear here"
        )

        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(500, 300)

        self.preview.setStyleSheet("""
            background-color: #F1F5F4;
            border: 2px dashed #9ABBB5;
            border-radius: 12px;
            font-size: 16px;
            color: #667777;
        """)

        # Status message
        self.status = QLabel("Ready to begin")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet(
            "font-size: 13px;"
            "color: #526563;"
        )

        # Add widgets to layout
        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)
        main_layout.addWidget(welcome)
        main_layout.addWidget(description)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.preview, stretch=1)
        main_layout.addWidget(self.status)

        # Application styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FAFCFA;
            }

            QPushButton {
                background-color: #167D78;
                color: white;
                border: none;
                padding: 13px 25px;
                border-radius: 8px;
                font-size: 15px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #105F5B;
            }

            QPushButton:pressed {
                background-color: #0B4D49;
            }
        """)

    # Upload image function
    def upload_image(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )

        if path:

            pixmap = QPixmap(path)

            if pixmap.isNull():
                self.status.setText(
                    "Unable to open this image."
                )
                return

            self.image_path = path

            self.display_image(pixmap)

            self.status.setText(
                "Image loaded successfully!"
            )

    # Display image function
    def display_image(self, pixmap):

        scaled_pixmap = pixmap.scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.preview.setPixmap(scaled_pixmap)

    # Camera function (will be implemented next)
    def capture_image(self):

        self.status.setText(
            "Camera capture will be implemented in the next stage."
        )


# Main application execution
if __name__ == "__main__":

    print("Starting ATHER...")

    app = QApplication(sys.argv)

    window = ATHERApp()
    window.show()

    print("ATHER window launched successfully!")

    sys.exit(app.exec())
