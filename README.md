# Deep Learning Model Analyzer

This is a Python-based video analysis tool that uses PyQt6 for the graphical user interface (GUI) and OpenCV for video processing. The tool allows users to load videos, apply various edge detection algorithms, and use YOLOv8 models for object detection.

## Features

- **Video Playback**: Load, play, pause, and stop video files.
- **Camera Input**: Start and stop camera input for real-time video analysis.
- **Edge Detection**: Apply Sobel, Canny, and Laplace of Gaussian edge detection algorithms.
- **YOLOv8 Model**: Load YOLOv8 models for object detection and filter detections based on selected classes and confidence thresholds.
- **Brightness and Contrast Adjustment**: Adjust the brightness and contrast of the video frames.
- **Cropping and Resizing**: Crop and resize video frames.
- **Snapshot and Recording**: Capture snapshots and save video recordings.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/deep-learning-model-analyzer.git
    cd deep-learning-model-analyzer
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the application**:
    ```bash
    python yolo8.py
    ```

2. **Load a video**:
    - Click on "Add Video MP4" to load a video file.

3. **Select an algorithm**:
    - Choose an edge detection algorithm or "Load YOLOv8 Model" from the "Algorithm Selection" dropdown.

4. **Load a YOLOv8 model**:
    - If "Load YOLOv8 Model" is selected, click on "Load YOLOv8 Model" to load a model file.
    - Enter the classes to detect and the confidence threshold when prompted.

5. **Adjust settings**:
    - Use the sliders to adjust brightness, contrast, cropping, and resizing settings.

6. **Control playback**:
    - Use the "Play", "Pause", and "Stop" buttons to control video playback.
    - Check "Repeat Video" to loop the video.

7. **Capture snapshots and recordings**:
    - Click on "Snapshot" to capture a snapshot of the current frame.
    - Click on "Save Recording" or "Save Camera Recording" to save video recordings.

## Code Overview

### `QTextEditLogger`
A custom logging handler that logs messages to a `QPlainTextEdit` widget.

### `VideoPlayer`
The main class that extends `QMainWindow` and contains all the functionalities of the video analysis tool.

#### Key Methods

- `load_model`: Loads a YOLOv8 model file and initializes it for use.
- `model_prediction_threshold`: Prompts the user to enter a confidence threshold for YOLO model predictions.
- `select_classes`: Prompts the user to enter classes to detect, which are then used to filter YOLO model predictions.
- `apply_edge_detection`: Applies the selected edge detection algorithm or YOLO model to the video frames.
- `process_frame`: Processes each video frame by adjusting brightness, contrast, cropping, resizing, and applying the selected algorithm.
- `update_frame`: Updates the video frame displayed in the GUI.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgements

- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/intro) for the GUI components.
- [OpenCV](https://opencv.org/) for video processing.
- [Ultralytics YOLO](https://github.com/ultralytics/yolov5) for the YOLOv8 model.
