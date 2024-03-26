import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, QRect, QTimer
import threading
import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtCore import Qt


class RecordingControlWindow(QWidget):
    def __init__(self, panning_window):
        super().__init__()

        self.setWindowTitle('Recording Control')
        self.setGeometry(200, 200, 200, 100)

        self.panning_window = panning_window

        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.start_recording)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_recording)

        layout = QVBoxLayout(self)
        layout.addWidget(self.play_button)
        layout.addWidget(self.stop_button)

    def start_recording(self):
        self.panning_window.start_recording()

    def stop_recording(self):
        self.panning_window.stop_recording()

class PanningWindow(QWidget):
    def __init__(self, control_window):
        super().__init__()

        self.setWindowTitle('Panning, Stretching, and Timer Window')
        self.setGeometry(100, 100, 800, 600)

        self.control_window = control_window

        self.original_pos = None
        self.last_stretch_pos = None
        self.panning = False
        self.stretching = False

        self.recording = False
        self.frames = []

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.timer_value = 0

        # Set flags to make the window frameless
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Border properties
        self.border_color = Qt.GlobalColor.white
        self.border_width = 7

    def record_screen(self):
        self.recording = True
        screen = gw.getWindowsWithTitle(self.windowTitle())[0]
        screen_rect = (screen.left, screen.top, screen.width, screen.height)

        while self.recording:
            screenshot = pyautogui.screenshot(region=screen_rect)
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            # Crop the frame to the width of the window
            frame = frame[7:self.height(), 7:self.width()]

            self.frames.append(frame)

    def stop_recording(self):
        self.recording = False
        self.timer.stop()
        self.save_video()
        self.hide()

    def save_video(self):
        if not self.frames:
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('generated_vids/CAPTURE.mp4', fourcc, 30.0, (self.frames[0].shape[1], self.frames[0].shape[0]))

        for frame in self.frames:
            video_writer.write(frame)

        video_writer.release()

    def start_recording(self):
        self.timer.start(1000)
        recording_thread = threading.Thread(target=self.record_screen)
        recording_thread.start()

    def update_timer(self):
        self.timer_value += 1
        print(f"Timer: {self.timer_value}")

    def paintEvent(self, event):
        # Custom paint event to handle transparency and draw a border
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Fill the window with a transparent color
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)

        # Draw a border only if the window is not minimized
        if not self.isMinimized() and not self.recording:
            pen = QPen(self.border_color)
            pen.setWidth(self.border_width)
            painter.setPen(pen)
            painter.drawRect(self.rect())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.panning = True
            self.original_pos = event.globalPosition()
            self.last_stretch_pos = self.original_pos

            corner_size = 20
            top_left_corner = QRect(0, 0, corner_size, corner_size)
            top_right_corner = QRect(self.width() - corner_size, 0, corner_size, corner_size)
            bottom_left_corner = QRect(0, self.height() - corner_size, corner_size, corner_size)
            bottom_right_corner = QRect(self.width() - corner_size, self.height() - corner_size, corner_size, corner_size)

            if top_left_corner.contains(event.pos()):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif top_right_corner.contains(event.pos()):
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif bottom_left_corner.contains(event.pos()):
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif bottom_right_corner.contains(event.pos()):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            self.stretching = True

    def mouseMoveEvent(self, event):
        if self.panning and self.stretching:
            delta = event.globalPosition() - self.last_stretch_pos

            new_x = self.x()
            new_y = self.y()
            new_width = self.width()
            new_height = self.height()

            if self.cursor().shape() == Qt.CursorShape.SizeBDiagCursor:
                new_width = max(self.width() + delta.x(), self.minimumWidth())
                new_height = max(self.height() + delta.y(), self.minimumHeight())
            elif self.cursor().shape() == Qt.CursorShape.SizeFDiagCursor:
                new_x = self.x() + delta.x()
                new_y = self.y() + delta.y()
                new_width = max(self.width() - delta.x(), self.minimumWidth())
                new_height = max(self.height() - delta.y(), self.minimumHeight())

            self.setGeometry(int(new_x), int(new_y), int(new_width), int(new_height))
            self.last_stretch_pos = event.globalPosition()

        elif self.panning:
            delta = event.globalPosition() - self.original_pos
            self.move(int(self.x() + delta.x()), int(self.y() + delta.y()))
            self.original_pos = event.globalPosition()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.panning = False
            self.stretching = False
            self.setCursor(Qt.CursorShape.ArrowCursor)






