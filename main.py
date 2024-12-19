import sys
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import Delaunay
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                           QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog,
                           QProgressBar, QSpinBox, QComboBox, QMessageBox,
                           QMenuBar, QMenu, QAction, QDialog, QFormLayout, 
                           QDoubleSpinBox, QDialogButtonBox, QListWidget,
                           QListWidgetItem, QFrame, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon
import os
import logging
import absl.logging

# Disable logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)

# Define landmark presets (same as before)
LANDMARK_PRESETS = {
    'minimal': [33, 133, 362, 61, 291, 0, 17, 267, 397],
    'moderate': [33, 133, 362, 130, 359, 61, 291, 0, 17, 267, 397,
                61, 291, 306, 375, 4, 152, 282, 405, 70, 63, 105,
                334, 336, 296, 14, 54, 284, 324],
    'full': list(range(468))
}

class ImageProcessor:
    @staticmethod
    def resize_strategy(img1, img2, strategy='resize'):
        if strategy == 'resize':
            # Get target size (use larger dimensions)
            target_w = max(img1.shape[1], img2.shape[1])
            target_h = max(img1.shape[0], img2.shape[0])
            
            # Resize both images
            img1_resized = cv2.resize(img1, (target_w, target_h))
            img2_resized = cv2.resize(img2, (target_w, target_h))
            return img1_resized, img2_resized
            
        elif strategy == 'crop':
            # Get target size (use smaller dimensions)
            target_w = min(img1.shape[1], img2.shape[1])
            target_h = min(img1.shape[0], img2.shape[0])
            
            # Center crop both images
            def center_crop(img, th, tw):
                h, w = img.shape[:2]
                x = w//2 - tw//2
                y = h//2 - th//2
                return img[y:y+th, x:x+tw]
            
            return center_crop(img1, target_h, target_w), center_crop(img2, target_h, target_w)

class TimelineItem:
    def __init__(self):
        self.image = None
        self.points = None
        self.duration = 3.0
        self.transition_duration = 1.0
        self.name = "Face"
# Modified TimelineItemWidget for horizontal layout
class TimelineItemWidget(QFrame):

    deleted = pyqtSignal(int)
    duration_changed = pyqtSignal(int, float)
    transition_changed = pyqtSignal(int, float)
    def __init__(self, index, timeline_item):
        super().__init__()
        self.index = index
        self.item = timeline_item
        self.setAcceptDrops(True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            TimelineItemWidget {
                background: #f0f0f0;
                border-radius: 8px;
                padding: 8px;
                margin: 4px;
            }
            TimelineItemWidget:hover {
                background: #e0e0e0;
            }
        """)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Thumbnail with better styling
        self.thumbnail = QLabel()
        self.thumbnail.setFixedSize(120, 120)
        self.thumbnail.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 4px;
                background: white;
            }
        """)
        layout.addWidget(self.thumbnail)
        
        # Name label
        self.name_label = QLabel(self.item.name)
        self.name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.name_label)
        
        # Settings button
        settings_btn = QPushButton("⚙")
        settings_btn.clicked.connect(self.show_settings)
        layout.addWidget(settings_btn)
        
    def update_thumbnail(self):
        if self.item.image is not None:
            img = cv2.cvtColor(self.item.image, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            aspect = w / h
            if aspect > 1:
                new_w = 80
                new_h = int(80 / aspect)
            else:
                new_h = 80
                new_w = int(80 * aspect)
            img = cv2.resize(img, (new_w, new_h))
            h, w = img.shape[:2]
            qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
            self.thumbnail.setPixmap(QPixmap.fromImage(qimg))

    def show_settings(self):
        dialog = FaceSettingsDialog(self.item, self)
        dialog.exec_()

# New FaceSettingsDialog
class FaceSettingsDialog(QDialog):
    def __init__(self, face_item, parent=None):
        super().__init__(parent)
        self.face_item = face_item
        self.setWindowTitle("Face Settings")
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Image preview with landmarks
        preview = QLabel()
        preview.setFixedSize(400, 400)
        layout.addWidget(preview)
        
        # Settings form
        form = QFormLayout()
        
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 30.0)
        self.duration_spin.setValue(self.face_item.duration)
        form.addRow("Duration (s):", self.duration_spin)
        
        self.transition_spin = QDoubleSpinBox()
        self.transition_spin.setRange(0.1, 10.0)
        self.transition_spin.setValue(self.face_item.transition_duration)
        form.addRow("Transition (s):", self.transition_spin)
        
        layout.addLayout(form)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        
class TimelineWidget(QWidget):
    item_selected = pyqtSignal(int)
    items_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.items = []
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Horizontal scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(200)
        
        container = QWidget()
        self.timeline_layout = QHBoxLayout(container)
        self.timeline_layout.setAlignment(Qt.AlignLeft)
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
    def add_face(self):
        item = TimelineItem()
        self.items.append(item)
        self.add_timeline_item_widget(len(self.items) - 1, item)
        self.items_changed.emit()        

    def dragEnterEvent(self, e):
        e.accept()
    
    def dropEvent(self, e):
        pos = e.pos()
        widget = e.source()
        for i in range(self.timeline_layout.count()):
            w = self.timeline_layout.itemAt(i).widget()
            if w.geometry().contains(pos):
                self.reorder_items(widget.index, i)
                break

        
    def add_timeline_item_widget(self, index, item):
        widget = TimelineItemWidget(index, item)
        widget.deleted.connect(self.remove_item)
        widget.duration_changed.connect(self.update_duration)
        widget.transition_changed.connect(self.update_transition)
        self.timeline_layout.addWidget(widget)
        
    def remove_item(self, index):
        widget = self.timeline_layout.itemAt(index).widget()
        self.timeline_layout.removeWidget(widget)
        widget.deleteLater()
        self.items.pop(index)
        self.items_changed.emit()
        self.update_indices()
        
    def update_indices(self):
        for i in range(self.timeline_layout.count()):
            widget = self.timeline_layout.itemAt(i).widget()
            if isinstance(widget, TimelineItemWidget):
                widget.index = i
                
    def update_duration(self, index, value):
        self.items[index].duration = value
        
    def update_transition(self, index, value):
        self.items[index].transition_duration = value
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Morphing Settings")
        self.setModal(True)
        
        layout = QFormLayout(self)
        
        # FPS settings
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(15, 60)
        self.fps_spin.setValue(30)
        layout.addRow("FPS:", self.fps_spin)
        
        # Landmark density
        self.landmark_combo = QComboBox()
        self.landmark_combo.addItems(['minimal', 'moderate', 'full'])
        self.landmark_combo.setCurrentText('moderate')
        layout.addRow("Landmark Density:", self.landmark_combo)
        
        # Add OK/Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

class TimelineMorphThread(QThread):
    progress = pyqtSignal(int)
    image_update = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, timeline_items, output_path, fps):
        super().__init__()
        self.timeline_items = timeline_items
        self.output_path = output_path
        self.fps = fps
        self.out = None

    def transform_triangle(self, img, transformed, src_tri, dst_tri):
        """Transform a triangle from source to destination"""
        rect = cv2.boundingRect(np.float32([dst_tri]))
        (x, y, w, h) = rect

        mask = np.zeros((h, w), np.uint8)
        dst_tri_shifted = dst_tri - [x, y]
        cv2.fillConvexPoly(mask, np.int32(dst_tri_shifted), (1, 1, 1), 16, 0)

        matrix = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri_shifted))
        warped = cv2.warpAffine(img, matrix, (w, h))
        warped_masked = warped * mask[:, :, np.newaxis]
        transformed[y:y+h, x:x+w] = transformed[y:y+h, x:x+w] * (1 - mask[:, :, np.newaxis]) + warped_masked

    def transform_image(self, img, src_points, dst_points, triangulation):
        """Transform entire image using triangulation"""
        transformed = np.zeros_like(img)
        
        for triangle in triangulation.simplices:
            src_tri = src_points[triangle]
            dst_tri = dst_points[triangle]
            self.transform_triangle(img, transformed, src_tri, dst_tri)
            
        return transformed

    def morph_and_blend(self, img1, img2, points1, points2, triangulation, alpha):
        """Morph and blend images at given alpha value"""
        current_points = (1 - alpha) * points1 + alpha * points2
        
        img1_morphed = self.transform_image(img1, points1, current_points, triangulation)
        img2_morphed = self.transform_image(img2, points2, current_points, triangulation)
        
        result = cv2.addWeighted(img1_morphed, 1-alpha, img2_morphed, alpha, 0)
        return result

    def run(self):
        try:
            if not self.timeline_items:
                raise Exception("No faces in timeline")

            h, w = self.timeline_items[0].image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))

            total_frames = 0
            frame_counts = []

            # Calculate total frames
            for i, item in enumerate(self.timeline_items):
                hold_frames = int(item.duration * self.fps)
                if i < len(self.timeline_items) - 1:
                    transition_frames = int(item.transition_duration * self.fps)
                    total_frames += hold_frames + transition_frames
                else:
                    total_frames += hold_frames
                frame_counts.append((hold_frames, transition_frames if i < len(self.timeline_items) - 1 else 0))

            current_frame = 0

            # Process each face in timeline
            for i, (item, (hold_frames, transition_frames)) in enumerate(zip(self.timeline_items, frame_counts)):
                # Hold current face
                for _ in range(hold_frames):
                    self.out.write(item.image)
                    self.image_update.emit(item.image)
                    current_frame += 1
                    self.progress.emit(int((current_frame / total_frames) * 100))

                # Transition to next face if not last
                if i < len(self.timeline_items) - 1:
                    next_item = self.timeline_items[i + 1]
                    mid_points = (item.points + next_item.points) * 0.5
                    triangulation = Delaunay(mid_points)

                    for frame in range(transition_frames):
                        alpha = frame / transition_frames
                        morphed = self.morph_and_blend(
                            item.image, next_item.image,
                            item.points, next_item.points,
                            triangulation, alpha
                        )
                        self.out.write(morphed)
                        self.image_update.emit(morphed)
                        current_frame += 1
                        self.progress.emit(int((current_frame / total_frames) * 100))

            if self.out is not None:
                self.out.release()

            self.progress.emit(100)
            self.finished.emit()

        except Exception as e:
            if self.out is not None:
                self.out.release()
            self.error.emit(str(e))

class ImageMorphApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Timeline Face Morphing App')
        self.setGeometry(100, 100, 1200, 800)
        self.last_video_path = None
        
        # Initialize settings
        self.settings = {
            'fps': 30,
            'landmark_preset': 'moderate'
        }
        
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        self.create_menu_bar()
        self.init_ui()

    def create_menu_bar(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('File')
        
        new_action = QAction('New Timeline', self)
        new_action.triggered.connect(self.new_timeline)
        file_menu.addAction(new_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        settings_menu = menubar.addMenu('Settings')
        
        morph_settings_action = QAction('Morphing Settings', self)
        morph_settings_action.triggered.connect(self.show_settings_dialog)
        settings_menu.addAction(morph_settings_action)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Preview area
        self.preview_label = QLabel('Preview')
        self.preview_label.setMinimumSize(800, 600)
        layout.addWidget(self.preview_label)
        
        # Timeline
        self.timeline = TimelineWidget()
        self.timeline.items_changed.connect(self.update_ui_state)
        layout.addWidget(self.timeline)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.btn_add_face = QPushButton('Add Face')
        self.btn_add_face.clicked.connect(self.load_face)
        controls_layout.addWidget(self.btn_add_face)
        
        self.btn_create_video = QPushButton('Create Video')
        self.btn_create_video.setEnabled(False)
        self.btn_create_video.clicked.connect(self.create_video)
        controls_layout.addWidget(self.btn_create_video)
        
        self.btn_preview = QPushButton('Preview Video')
        self.btn_preview.setEnabled(False)
        self.btn_preview.clicked.connect(self.preview_video)
        controls_layout.addWidget(self.btn_preview)
        
        layout.addLayout(controls_layout)
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

    # Continue with the methods in the next part...
    def new_timeline(self):
        """Clear the current timeline"""
        reply = QMessageBox.question(
            self, 'New Timeline',
            'Are you sure you want to clear the current timeline?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.timeline.items.clear()
            for i in reversed(range(self.timeline.timeline_layout.count())):
                self.timeline.timeline_layout.itemAt(i).widget().deleteLater()
            self.update_ui_state()

    def detect_face_landmarks(self, image):
        """Detect facial landmarks on the given image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            raise Exception("No face detected in image")
            
        h, w = image.shape[:2]
        landmarks = []
        
        for facial_landmarks in results.multi_face_landmarks:
            for idx in LANDMARK_PRESETS[self.settings['landmark_preset']]:
                landmark = facial_landmarks.landmark[idx]
                x = landmark.x * w
                y = landmark.y * h
                landmarks.append([x, y])
                
        corners = [[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]]
        landmarks.extend(corners)
        
        return np.float32(landmarks)

    def draw_landmarks(self, img, points):
        """Draw landmarks overlay on image"""
        overlay = img.copy()
        
        # Draw landmarks
        for (x, y) in points:
            cv2.circle(overlay, (int(x), int(y)), 3, (0, 255, 0), -1)
            
        alpha = 0.7
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    def update_preview(self, img):
        """Update the preview label with the given image"""
        if img is None:
            return
            
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = rgb_img.shape
        bytes_per_line = 3 * w
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio)
        self.preview_label.setPixmap(scaled_pixmap)

    def show_settings_dialog(self):
        """Show the settings dialog"""
        dialog = SettingsDialog(self)
        
        # Set current values
        dialog.fps_spin.setValue(self.settings['fps'])
        dialog.landmark_combo.setCurrentText(self.settings['landmark_preset'])
        
        if dialog.exec_() == QDialog.Accepted:
            # Update settings
            old_preset = self.settings['landmark_preset']
            self.settings.update({
                'fps': dialog.fps_spin.value(),
                'landmark_preset': dialog.landmark_combo.currentText()
            })
            
            # If landmark preset changed, recompute landmarks for all faces
            if old_preset != self.settings['landmark_preset']:
                try:
                    for item in self.timeline.items:
                        if item.image is not None:
                            item.points = self.detect_face_landmarks(item.image)
                except Exception as e:
                    QMessageBox.warning(self, "Error", str(e))

    def load_face(self):
        """Load a new face image into the timeline"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Face Image",
            "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_name:
            try:
                img = cv2.imread(file_name)
                if img is None:
                    raise Exception("Failed to load image")
                
                # Resize to match first image if exists
                if self.timeline.items:
                    first_img = self.timeline.items[0].image
                    img = cv2.resize(img, (first_img.shape[1], first_img.shape[0]))
                
                # Create new timeline item
                item = TimelineItem()
                item.image = img
                item.points = self.detect_face_landmarks(img)
                item.name = os.path.basename(file_name)
                
                self.timeline.add_face()
                self.timeline.items[-1] = item
                self.timeline.timeline_layout.itemAt(len(self.timeline.items)-1).widget().update_thumbnail()
                
                # Update preview
                self.update_preview(self.draw_landmarks(img, item.points))
                
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))

    def update_ui_state(self):
        """Update UI elements based on current state"""
        has_faces = len(self.timeline.items) > 0
        self.btn_create_video.setEnabled(has_faces)

    def handle_error(self, error_message):
        """Handle errors from the morphing thread"""
        QMessageBox.critical(self, "Error", f"Morphing failed: {error_message}")
        self.btn_create_video.setEnabled(True)
        self.progress_bar.setValue(0)
    
    def create_video(self):
        """Start the video creation process"""
        if len(self.timeline.items) < 1:
            QMessageBox.warning(self, "Error", "Add at least one face to the timeline")
            return
            
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Morphing Video",
            "",
            "Video Files (*.mp4)"
        )
        
        if output_path:
            self.last_video_path = output_path
            self.btn_create_video.setEnabled(False)
            
            self.morph_thread = TimelineMorphThread(
                self.timeline.items,
                output_path,
                self.settings['fps']
            )
            
            self.morph_thread.progress.connect(self.progress_bar.setValue)
            self.morph_thread.image_update.connect(self.update_preview)
            self.morph_thread.error.connect(self.handle_error)
            self.morph_thread.finished.connect(self.morphing_finished)
            self.morph_thread.start()

    def morphing_finished(self):
        """Handle completion of the morphing process"""
        self.btn_create_video.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.progress_bar.setValue(0)
        QMessageBox.information(self, "Success", "Video created successfully!")

    def preview_video(self):
        """Preview the created video"""
        if self.last_video_path and os.path.exists(self.last_video_path):
            if sys.platform == 'win32':
                os.startfile(self.last_video_path)
            elif sys.platform == 'darwin':  # macOS
                subprocess.call(('open', self.last_video_path))
            else:  # linux variants
                subprocess.call(('xdg-open', self.last_video_path))
        else:
            QMessageBox.warning(self, "Error", "No video available for preview")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageMorphApp()
    window.show()
    sys.exit(app.exec_())
