
# Face Morpher

A sophisticated desktop application for creating smooth face morphing animations with an intuitive timeline interface.

## Features

- Interactive timeline-based face morphing
- Support for multiple facial transition effects
- Customizable morphing settings
- Real-time preview
- Export to MP4 video
- Adjustable FPS and landmark density
- Face landmark detection and visualization

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- PyQt5
- NumPy
- SciPy

## Installation

1. Clone this repository
2. Install required dependencies:
```bash
pip install opencv-python mediapipe PyQt5 numpy scipy
```

## Usage

1. Run the application:
```bash
python image_morph_app.py
```

2. Add faces to the timeline using the "Add Face" button
3. Adjust duration and transition settings for each face
4. Click "Create Video" to generate the morphing animation
5. Use "Preview Video" to watch the result

## Control Features

- **Timeline Management**: Add, remove, and reorder faces
- **Face Settings**: Adjust individual face duration and transition times
- **Morphing Settings**: Configure FPS and landmark density
- **Preview**: Real-time preview of morphing effects
- **Export**: Save animations as MP4 files

## Troubleshooting

- Ensure all dependencies are properly installed
- Check that input images contain clear, single faces
- Verify sufficient system memory for processing
- For video export issues, check write permissions in output directory

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgments

- Built with MediaPipe for face landmark detection
- Uses PyQt5 for the graphical interface
- Leverages OpenCV for image processing

---

For bug reports and feature requests, please open an issue on the project repository.