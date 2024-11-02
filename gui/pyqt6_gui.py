import sys
import cv2
import numpy as np
import logging
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QSlider, QComboBox, QFileDialog, QSizePolicy, QPlainTextEdit, QSpinBox, QCheckBox, QDialog, QScrollArea, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QInputDialog
import threading
from ultralytics import YOLO
from PyQt6.QtWidgets import QInputDialog, QMessageBox

class QTextEditLogger(logging.Handler):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        msg = self.format(record)
        self.text_edit.appendPlainText(msg)

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep Learning Model Analyzer")
        self.init_ui()
        self.init_video_settings()
        self.init_timer()
        self.init_logger()
        self.classes_to_detect = []


    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.init_top_layout()
        self.init_bottom_layout()
        self.adjust_window_size()

    def stop_recording(self):
        self.is_recording = False
        self.logger.info("Recording stopped")

    def adjust_window_size(self):
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        width = min(1200, screen_size.width() - 100)
        height = min(600, screen_size.height() - 100)
        self.setGeometry(100, 100, width, height)

    def init_top_layout(self):
        self.top_layout = QHBoxLayout()
        self.init_control_layout()
        self.init_video_display()
        self.main_layout.addLayout(self.top_layout)

    def init_control_layout(self):
        self.control_scroll_area = QScrollArea()
        self.control_scroll_area.setWidgetResizable(True)
        self.control_widget = QWidget()
        self.control_layout = QVBoxLayout(self.control_widget)
        # self.load_model_button = self.create_button("Load YOLOv8 Model", self.load_model)
        self.add_control_widgets()
        self.control_scroll_area.setWidget(self.control_widget)
        self.top_layout.addWidget(self.control_scroll_area, 1)  # 20% of the width

    def load_model(self):
        model_file, _ = QFileDialog.getOpenFileName(self, "Select YOLOv8 Model File", "", "Model Files (*.pt *.onnx)")
        if model_file:
            self.logger.info(f"Model loaded: {model_file}")
            self.model_path = model_file
            self.model = YOLO(self.model_path)
            self.select_classes()
            self.model_prediction_threshold()

    def model_prediction_threshold(self):
        while True:
            threshold, ok = QInputDialog.getText(self, "confidence threshold",
                                                 "Enter confidence threshold between 0 - 1, example 0.70")

            if ok and threshold:
                try:
                    # Convert value to float
                    self.confidence_threshold = float(threshold)

                    # Check if the number is between 0 and 1 (inclusive)
                    if 0 <= self.confidence_threshold <= 1:
                        self.logger.info(f"Threshold value: {self.confidence_threshold}")
                        return True
                    else:
                        # Show error message and re-prompt without closing the dialog
                        QMessageBox.warning(self, "Invalid Threshold", "Please enter a value between 0 and 1.")
                except ValueError:
                    # Show error message for non-float input and re-prompt
                    QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")
            else:
                # Log and return if the dialog is canceled
                self.logger.info("Threshold selection canceled")
                return False

    def select_classes(self):
        classes, ok = QInputDialog.getText(self, "Select Classes", "Enter classes to detect (comma-separated):")
        if ok and classes:
            self.classes_to_detect = [cls.strip() for cls in classes.split(',')]

            for i in range(len(self.classes_to_detect)):
                # Try to convert the string to a float/int
                try:
                    # Check if it's a number and replace it in the list
                    self.classes_to_detect[i] = float(self.classes_to_detect[i]) if '.' in self.classes_to_detect[i] else int(self.classes_to_detect[i])
                except ValueError:
                    # If conversion fails, it's a non-numeric string, so we leave it
                    continue

            self.logger.info(f"Classes to detect: {self.classes_to_detect}")
        else:
            self.logger.info("Class selection canceled")


    def add_control_widgets(self):
        # Video Selection Section
        video_selection_group = QGroupBox("Video Selection")
        video_selection_layout = QVBoxLayout()
        self.add_video_button = self.create_button("Add Video MP4", self.add_video)
        video_selection_layout.addWidget(self.add_video_button)
        video_selection_group.setLayout(video_selection_layout)
        self.control_layout.addWidget(video_selection_group)

        # Algorithm Selection Section
        algorithm_group = QGroupBox("Algorithm Selection")
        algorithm_layout = QVBoxLayout()
        self.algorithm_combo = self.create_combo_box(
            ["None", "Sobel", "Canny", "Laplace of Gaussian", "Load YOLOv8 Model"],
            self.algorithm_changed
        )
        algorithm_layout.addWidget(self.algorithm_combo)

        algorithm_group.setLayout(algorithm_layout)
        self.control_layout.addWidget(algorithm_group)

        # Playback Controls Section
        playback_controls_group = QGroupBox("Playback Controls")
        playback_controls_layout = QVBoxLayout()
        self.start_button = self.create_button("Play", self.play_video)
        self.pause_button = self.create_button("Pause", self.pause_video)
        self.stop_button = self.create_button("Stop", self.stop_video)
        self.repeat_checkbox = QCheckBox("Repeat Video")
        playback_controls_layout.addWidget(self.start_button)
        playback_controls_layout.addWidget(self.pause_button)
        playback_controls_layout.addWidget(self.stop_button)
        playback_controls_layout.addWidget(self.repeat_checkbox)
        playback_controls_group.setLayout(playback_controls_layout)
        self.control_layout.addWidget(playback_controls_group)

        # Snapshot and Camera Controls Section
        camera_controls_group = QGroupBox(" Camera Controls")
        snapshot_camera_controls_layout = QVBoxLayout()
        self.start_camera_button = self.create_button("Start Camera", self.start_camera)
        self.stop_camera_button = self.create_button("Stop Camera", self.stop_camera)
        snapshot_camera_controls_layout.addWidget(self.start_camera_button)
        snapshot_camera_controls_layout.addWidget(self.stop_camera_button)
        camera_controls_group.setLayout(snapshot_camera_controls_layout)
        self.control_layout.addWidget(camera_controls_group)

        # Recording Controls Section
        recording_controls_group = QGroupBox("Recording Controls")
        recording_controls_layout = QVBoxLayout()
        self.save_video_button = self.create_button("Save Recording", self.save_video)
        self.save_camera_recording_button = self.create_button("Save Camera Recording", self.save_camera_recording)
        self.stop_recording_button = self.create_button("Stop Recording", self.stop_recording)
        self.snapshot_button = self.create_button("Snapshot", self.capture_snapshot)
        recording_controls_layout.addWidget(self.snapshot_button)
        recording_controls_layout.addWidget(self.save_video_button)
        recording_controls_layout.addWidget(self.save_camera_recording_button)
        recording_controls_layout.addWidget(self.stop_recording_button)
        recording_controls_group.setLayout(recording_controls_layout)
        self.control_layout.addWidget(recording_controls_group)

        # Brightness and Contrast Section
        brightness_contrast_group = QGroupBox("Brightness and Contrast")
        brightness_contrast_layout = QVBoxLayout()
        brightness_label = QLabel("Adjust Brightness")
        self.brightness_slider = self.create_slider(-100, 100, 0, self.update_brightness_contrast)
        self.brightness_value_label = QLabel(f"Brightness: {self.brightness_slider.value()}")
        contrast_label = QLabel("Adjust Contrast")
        self.contrast_slider = self.create_slider(0, 200, 100, self.update_brightness_contrast)
        self.contrast_value_label = QLabel(f"Contrast: {self.contrast_slider.value() / 100.0:.1f}")
        brightness_contrast_layout.addWidget(brightness_label)
        brightness_contrast_layout.addWidget(self.brightness_slider)
        brightness_contrast_layout.addWidget(self.brightness_value_label)
        brightness_contrast_layout.addWidget(contrast_label)
        brightness_contrast_layout.addWidget(self.contrast_slider)
        brightness_contrast_layout.addWidget(self.contrast_value_label)
        brightness_contrast_group.setLayout(brightness_contrast_layout)
        self.control_layout.addWidget(brightness_contrast_group)

        # Frame Rate Section
        frame_rate_group = QGroupBox("Frame Rate")
        frame_rate_layout = QVBoxLayout()
        frame_rate_label = QLabel("Frame Rate")
        self.frame_rate_slider = self.create_slider(1, 60, 30, self.update_frame_rate)
        self.frame_rate_value_label = QLabel(f"Frame Rate: {self.frame_rate_slider.value()} FPS")
        frame_rate_layout.addWidget(frame_rate_label)
        frame_rate_layout.addWidget(self.frame_rate_slider)
        frame_rate_layout.addWidget(self.frame_rate_value_label)
        frame_rate_group.setLayout(frame_rate_layout)
        self.control_layout.addWidget(frame_rate_group)

        # Crop Controls Section
        self.init_crop_controls()

        # Resize Controls Section
        self.init_resize_controls()

    def save_video(self):
        if self.cap:
            formats = ["mp4", "avi", "mov"]
            format, ok = QInputDialog.getItem(self, "Select Video Format", "Format:", formats, 0, False)
            if ok and format:
                default_filename = f"output_video.{format}"
                file_name, _ = QFileDialog.getSaveFileName(self, "Save Video File", default_filename,
                                                           f"Video Files (*.{format})")
                if file_name:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if format == "mp4" else cv2.VideoWriter_fourcc(*'XVID')
                    frame_rate = self.frame_rate_slider.value()  # Ensure frame rate is updated
                    out = cv2.VideoWriter(file_name, fourcc, frame_rate,
                                          (self.resize_width, self.resize_height))

                    if not out.isOpened():
                        self.logger.error("Failed to open video writer")
                        return

                    self.is_recording = True  # Set recording flag to True

                    def record():
                        self.logger.info("Starting video recording...")
                        while self.cap.isOpened() and self.is_recording:
                            ret, frame = self.cap.read()
                            if not ret:
                                break
                            frame = self.process_frame(frame)
                            out.write(frame)
                        out.release()
                        self.logger.info(f"Video saved: {file_name}")

                    recording_thread = threading.Thread(target=record)
                    recording_thread.start()
                else:
                    self.logger.info("Save video canceled")
            else:
                self.logger.info("Save video format selection canceled")
        else:
            self.logger.error("No video loaded to save")

    def save_camera_recording(self):
        if self.cap and self.cap.isOpened():
            formats = ["mp4", "avi", "mov"]
            format, ok = QInputDialog.getItem(self, "Select Video Format", "Format:", formats, 0, False)
            if ok and format:
                default_filename = f"camera_recording.{format}"
                file_name, _ = QFileDialog.getSaveFileName(self, "Save Camera Recording", default_filename,
                                                           f"Video Files (*.{format})")
                if file_name:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if format == "mp4" else cv2.VideoWriter_fourcc(*'XVID')
                    frame_rate = self.frame_rate_slider.value()  # Ensure frame rate is updated
                    out = cv2.VideoWriter(file_name, fourcc, frame_rate,
                                          (self.resize_width, self.resize_height))

                    if not out.isOpened():
                        self.logger.error("Failed to open video writer for camera recording")
                        return

                    self.is_recording = True  # Set recording flag to True

                    def record():
                        self.logger.info("Starting camera recording...")
                        while self.cap.isOpened() and self.is_recording:
                            ret, frame = self.cap.read()
                            if not ret:
                                break
                            frame = self.process_frame(frame)
                            out.write(frame)
                        out.release()
                        self.logger.info(f"Camera recording saved: {file_name}")

                    recording_thread = threading.Thread(target=record)
                    recording_thread.start()
                else:
                    self.logger.info("Save camera recording canceled")
            else:
                self.logger.info("Save camera recording format selection canceled")
        else:
            self.logger.error("Camera is not started or not available")

    # Other methods remain unchanged

    def create_button(self, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        self.control_layout.addWidget(button)
        return button

    def create_combo_box(self, items, callback):
        combo_box = QComboBox()
        combo_box.addItems(items)
        combo_box.currentIndexChanged.connect(callback)
        self.control_layout.addWidget(combo_box)
        return combo_box

    def create_slider(self, min_val, max_val, default_val, callback):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(callback)
        return slider

    def init_crop_controls(self):
        crop_controls_group = QGroupBox("Crop Controls")
        self.crop_controls_layout = QVBoxLayout()  # Define the layout here

        self.crop_start_x_slider = self.create_crop_slider("Crop Start X", 0, 1920, 640)
        self.crop_start_y_slider = self.create_crop_slider("Crop Start Y", 0, 1080, 480)
        self.crop_width_slider = self.create_crop_slider("Crop Width", 1, 1920, 640)
        self.crop_height_slider = self.create_crop_slider("Crop Height", 1, 1080, 480)
        self.reset_crop_button = self.create_button("Reset Crop", self.reset_crop)

        self.crop_controls_layout.addWidget(QLabel("Crop Start X"))
        self.crop_controls_layout.addWidget(self.crop_start_x_slider)
        self.crop_controls_layout.addWidget(QLabel("Crop Start Y"))
        self.crop_controls_layout.addWidget(self.crop_start_y_slider)
        self.crop_controls_layout.addWidget(QLabel("Crop Width"))
        self.crop_controls_layout.addWidget(self.crop_width_slider)
        self.crop_controls_layout.addWidget(QLabel("Crop Height"))
        self.crop_controls_layout.addWidget(self.crop_height_slider)
        self.crop_controls_layout.addWidget(self.reset_crop_button)

        crop_controls_group.setLayout(self.crop_controls_layout)
        self.control_layout.addWidget(crop_controls_group)

    def create_crop_slider(self, label_text, min_val, max_val, default_val):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(self.update_crop)
        return slider

    def init_resize_controls(self):
        resize_controls_group = QGroupBox("Resize Controls")
        self.resize_controls_layout = QVBoxLayout()  # Define the layout here

        self.resize_width_input = self.create_resize_input("Resize Width", 1, 1920, 640)
        self.resize_height_input = self.create_resize_input("Resize Height", 1, 1080, 480)
        self.reset_resize_button = self.create_button("Reset Resize", self.reset_resize)

        self.resize_controls_layout.addWidget(QLabel("Resize Width"))
        self.resize_controls_layout.addWidget(self.resize_width_input)
        self.resize_controls_layout.addWidget(QLabel("Resize Height"))
        self.resize_controls_layout.addWidget(self.resize_height_input)
        self.resize_controls_layout.addWidget(self.reset_resize_button)

        resize_controls_group.setLayout(self.resize_controls_layout)
        self.control_layout.addWidget(resize_controls_group)

    def create_resize_input(self, label_text, min_val, max_val, default_val):
        spin_box = QSpinBox()
        spin_box.setMinimum(min_val)
        spin_box.setMaximum(max_val)
        spin_box.setValue(default_val)
        spin_box.valueChanged.connect(self.update_resize)
        return spin_box



    def init_video_display(self):
        self.video_display = QLabel()
        self.video_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_display.setStyleSheet("background: black;")

        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(100)
        self.video_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.video_slider.sliderMoved.connect(self.set_video_position)
        self.video_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #444444;
                height: 8px;
            }
            QSlider::handle:horizontal {
                background: #00ff00;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -5px 0;
                border-radius: 3px;
            }
        """)

        self.video_container = QWidget()
        self.video_container.setStyleSheet("background: black;")
        self.video_container_layout = QVBoxLayout(self.video_container)
        self.video_container_layout.addWidget(self.video_display, alignment=Qt.AlignmentFlag.AlignCenter)
        self.video_container_layout.addWidget(self.video_slider)

        self.top_layout.addWidget(self.video_container, 4)  # 80% of the width

    def init_bottom_layout(self):
        self.bottom_layout = QVBoxLayout()
        self.log_window = QPlainTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setFixedHeight(100)
        self.bottom_layout.addWidget(self.log_window)
        self.toggle_log_button = QPushButton("Hide Log")
        self.toggle_log_button.clicked.connect(self.toggle_log_window)
        self.bottom_layout.addWidget(self.toggle_log_button)
        self.main_layout.addLayout(self.bottom_layout)

    def toggle_log_window(self):
        if self.log_window.isVisible():
            self.log_window.hide()
            self.toggle_log_button.setText("Show Log")
        else:
            self.log_window.show()
            self.toggle_log_button.setText("Hide Log")

    def init_video_settings(self):
        self.cap = None
        self.current_frame = None
        self.is_paused = False
        self.is_playing = False
        self.algorithm = None
        self.brightness = 0
        self.contrast = 1.0
        self.crop_x = 0
        self.crop_y = 0
        self.crop_width = 640
        self.crop_height = 480
        self.resize_width = 640
        self.resize_height = 480
        self.snapshot_counter = 0

    def init_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer_interval = 1000 / self.frame_rate_slider.value()
        self.timer.setInterval(int(self.timer_interval))

    def init_logger(self):
        self.logger = logging.getLogger("VideoPlayerLogger")
        self.logger.setLevel(logging.DEBUG)
        log_handler = QTextEditLogger(self.log_window)
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(log_handler)

    def add_video(self):
        self.video_file, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if self.video_file:
            self.cap = cv2.VideoCapture(self.video_file)
            self.video_slider.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
            self.logger.info(f"Video loaded: {self.video_file}")

    def play_video(self):
        if self.cap:
            self.is_paused = False
            self.is_playing = True
            self.timer.start()
            self.logger.info("Video playback started")

    def pause_video(self):
        self.is_paused = True
        self.is_playing = False
        self.timer.stop()
        self.logger.info("Video playback paused")

    def stop_video(self):
        self.is_playing = False
        self.is_paused = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_display.clear()
        self.logger.info("Video playback stopped")

    def update_brightness_contrast(self):
        self.brightness = self.brightness_slider.value()
        self.contrast = self.contrast_slider.value() / 100.0
        self.brightness_value_label.setText(f"Brightness: {self.brightness}")
        self.contrast_value_label.setText(f"Contrast: {self.contrast:.1f}")
        self.logger.info(f"Brightness set to {self.brightness}, Contrast set to {self.contrast:.1f}")

    def update_frame_rate(self):
        frame_rate = self.frame_rate_slider.value()
        if frame_rate > 0:
            self.timer_interval = 1000 / frame_rate
            self.timer.setInterval(int(self.timer_interval))
            self.frame_rate_value_label.setText(f"Frame Rate: {frame_rate} FPS")
            if self.cap:
                self.video_slider.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
            self.logger.info(f"Frame rate set to {frame_rate} FPS")
        else:
            self.logger.error("Frame rate must be greater than 0")

    def update_frame(self):
        if self.cap and self.is_playing:
            ret, frame = self.cap.read()
            if not ret:
                if self.repeat_checkbox.isChecked():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                else:
                    self.stop_video()
                    return
            frame = self.process_frame(frame)
            self.display_frame(frame)
            self.video_slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))

    def process_frame(self, frame):
        frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)
        frame = frame[self.crop_y:self.crop_y + self.crop_height, self.crop_x:self.crop_x + self.crop_width]
        frame = cv2.resize(frame, (self.resize_width, self.resize_height))
        if self.algorithm:
            frame = self.apply_edge_detection(frame)
        return frame

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_image = QImage(rgb_image.data, w, h, w * ch, QImage.Format.Format_RGB888)
        self.video_display.setPixmap(QPixmap.fromImage(q_image))

    def set_video_position(self, position):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            self.update_frame()

    def apply_edge_detection(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.algorithm == "Sobel":
            grad_x = cv2.Sobel(gray_frame, cv2.CV_16S, 1, 0)
            grad_y = cv2.Sobel(gray_frame, cv2.CV_16S, 0, 1)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            edge_frame = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        elif self.algorithm == "Canny":
            edge_frame = cv2.Canny(gray_frame, 100, 200)
            edge_frame = cv2.cvtColor(edge_frame, cv2.COLOR_GRAY2BGR)
        elif self.algorithm == "Laplace of Gaussian":
            blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
            edge_frame = cv2.Laplacian(blurred_frame, cv2.CV_16S)
            edge_frame = cv2.convertScaleAbs(edge_frame)
            edge_frame = cv2.cvtColor(edge_frame, cv2.COLOR_GRAY2BGR)
        elif self.algorithm == "Load YOLOv8 Model":
            # Perform detection
            results = self.model(frame)

            # Filter detections to only selected classes and above the confidence threshold
            filtered_boxes = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    # Apply class filter and confidence threshold
                    if cls_id in self.classes_to_detect and box.conf.item() > self.confidence_threshold:
                        filtered_boxes.append(box)

            # Draw the filtered detections on the frame
            for box in filtered_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integer coordinates
                label = f"{self.model.names[box.cls.item()]}: {box.conf.item():.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            edge_frame = frame
        else:
            edge_frame = frame
        return edge_frame

    def algorithm_changed(self):
        self.algorithm = self.algorithm_combo.currentText()
        if self.algorithm == "Load YOLOv8 Model":
            self.load_model()
        else:
            self.logger.info(f"Algorithm changed to {self.algorithm}")

    def capture_snapshot(self):
        if self.cap:
            self.pause_video()
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
            ret, frame = self.cap.read()
            if ret:
                frame = self.process_frame(frame)
                self.snapshot_counter += 1
                default_filename = f"snapshot_{self.snapshot_counter}.png"
                dialog = QDialog(self)
                dialog.setWindowTitle("Save Snapshot")
                dialog.setLayout(QVBoxLayout())
                file_name, _ = QFileDialog.getSaveFileName(dialog, "Save Snapshot", default_filename, "Image Files (*.png *.jpg *.bmp)")
                if file_name:
                    cv2.imwrite(file_name, frame)
                    self.logger.info(f"Snapshot saved: {file_name}")
                    self.show_snapshot(frame)
                else:
                    self.logger.info("Snapshot save canceled")
            self.play_video()

    def show_snapshot(self, frame):
        snapshot_dialog = QDialog(self)
        snapshot_dialog.setWindowTitle("Snapshot")
        snapshot_layout = QVBoxLayout(snapshot_dialog)
        snapshot_label = QLabel()
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_image = QImage(rgb_image.data, w, h, w * ch, QImage.Format.Format_RGB888)
        snapshot_label.setPixmap(QPixmap.fromImage(q_image))
        snapshot_layout.addWidget(snapshot_label)
        snapshot_dialog.exec()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.play_video()
        self.logger.info("Camera started")

    def stop_camera(self):
        if self.cap:
            self.cap.release()
        self.video_display.clear()
        self.logger.info("Camera stopped")

    def update_crop(self):
        self.crop_x = self.crop_start_x_slider.value()
        self.crop_y = self.crop_start_y_slider.value()
        self.crop_width = self.crop_width_slider.value()
        self.crop_height = self.crop_height_slider.value()
        self.logger.info(f"Crop updated: x={self.crop_x}, y={self.crop_y}, width={self.crop_width}, height={self.crop_height}")

    def reset_crop(self):
        self.crop_start_x_slider.setValue(0)
        self.crop_start_y_slider.setValue(0)
        self.crop_width_slider.setValue(640)
        self.crop_height_slider.setValue(480)
        self.update_crop()
        self.logger.info("Crop reset to default values")

    def update_resize(self):
        self.resize_width = self.resize_width_input.value()
        self.resize_height = self.resize_height_input.value()
        self.logger.info(f"Resize updated: width={self.resize_width}, height={self.resize_height}")

    def reset_resize(self):
        self.resize_width_input.setValue(640)
        self.resize_height_input.setValue(480)
        self.update_resize()
        self.logger.info("Resize reset to default values")

