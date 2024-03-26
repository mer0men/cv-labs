
from PyQt6.QtWidgets import QWidget, QPushButton, QLineEdit, QGridLayout, QFileDialog
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont

import cv2
import numpy as np

from gui.windows.contentWindow import config
import random

class ModelingConfigWindow(QWidget):
    image_generated = pyqtSignal(np.ndarray)
    video_generate = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.image_m = None
        self.image_v = None

        self.width, self.height = config.screen_w, config.screen_h 

        self.input_model = QLineEdit(self)
        self.input_model.setPlaceholderText(f"Model")

        font = QFont('Arial', 14)

        self.model_1 = QPushButton("Задержанный единичный импульс")
        self.model_1_params = QLineEdit(self)
        self.model_1.setFont(font)
        self.model_1_params.setFont(font)
        self.model_1_params.setText("0 0")

        self.model_2_params = QLineEdit(self)
        self.model_2 = QPushButton("Задержанный единичный скачок")
        self.model_2.setFont(font)
        self.model_2_params.setFont(font)
        self.model_2_params.setText("0 0")

        self.model_3_params = QLineEdit(self)
        self.model_3 = QPushButton("Одиночный круг радиуса R")
        self.model_3.setFont(font)
        self.model_3_params.setFont(font)
        self.model_3_params.setText("0 1")

        self.model_4_params = QLineEdit(self)
        self.model_4 = QPushButton("Одиночный квадрат со стороной a")
        self.model_4.setFont(font)
        self.model_4_params.setFont(font)
        self.model_4_params.setText("0 1")
        
        self.model_5_params = QLineEdit(self)
        self.model_5 = QPushButton("Шахматная доска")
        self.model_5.setFont(font)
        self.model_5_params.setFont(font)
        self.model_5_params.setText("0 0")

        self.model_6_params = QLineEdit(self)
        self.model_6 = QPushButton("Круги в узлах прямоугольной решетки")
        self.model_6.setFont(font)
        self.model_6_params.setFont(font)

        self.model_7_params = QLineEdit(self)
        self.model_7 = QPushButton("Случайные  круги")
        self.model_7.setFont(font)
        self.model_7_params.setFont(font)

        self.model_8_params = QLineEdit(self)
        self.model_8 = QPushButton("Белый шум с равномерным распределением")
        self.model_8.setFont(font)
        self.model_8_params.setFont(font)

        self.model_9_params = QLineEdit(self)
        self.model_9 = QPushButton("Белый шум с нормальным распределением")
        self.model_9.setFont(font)
        self.model_9_params.setFont(font)

        self.size = QLineEdit(self)
        self.size.setPlaceholderText(f"Ширина кадра")
        self.save_bnt = QPushButton("Сохранить")
        self.save_bnt.setFont(font)
        self.size.setFont(font)


        self.main_layout = QGridLayout()
        self.setWindowTitle('Describe model')

        self.initUI()

    def initUI(self):
        super().__init__()

        self.model_1.clicked.connect(self.generate_model_1)  
        self.model_2.clicked.connect(self.generate_model_2)  
        self.model_3.clicked.connect(self.generate_model_3)  
        self.model_4.clicked.connect(self.generate_model_4)  
        self.model_5.clicked.connect(self.generate_model_5)  
        self.model_7.clicked.connect(self.generate_model_7)  
        self.model_8.clicked.connect(self.generate_model_8)
        self.model_9.clicked.connect(self.generate_model_9) 
        self.save_bnt.clicked.connect(self.save)  


        self.main_layout.addWidget(self.model_1_params, 1, 0, 1, 1)
        self.main_layout.addWidget(self.model_1, 1, 1, 1, 1)

        self.main_layout.addWidget(self.model_2_params, 2, 0, 1, 1)
        self.main_layout.addWidget(self.model_2, 2, 1, 1, 1)
        
        self.main_layout.addWidget(self.model_3_params, 3, 0, 1, 1)
        self.main_layout.addWidget(self.model_3, 3, 1, 1, 1)

        self.main_layout.addWidget(self.model_4_params, 4, 0, 1, 1)
        self.main_layout.addWidget(self.model_4, 4, 1, 1, 1)

        self.main_layout.addWidget(self.model_5_params, 5, 0, 1, 1)
        self.main_layout.addWidget(self.model_5, 5, 1, 1, 1)

        self.main_layout.addWidget(self.model_6_params, 6, 0, 1, 1)
        self.main_layout.addWidget(self.model_6, 6, 1, 1, 1)

        self.main_layout.addWidget(self.model_7_params, 7, 0, 1, 1)
        self.main_layout.addWidget(self.model_7, 7, 1, 1, 1)

        self.main_layout.addWidget(self.model_8_params, 8, 0, 1, 1)
        self.main_layout.addWidget(self.model_8, 8, 1, 1, 1)

        self.main_layout.addWidget(self.model_9_params, 9, 0, 1, 1)
        self.main_layout.addWidget(self.model_9, 9, 1, 1, 1)

        self.main_layout.addWidget(self.size, 10, 0, 1, 1)
        self.main_layout.addWidget(self.save_bnt, 10, 1, 1, 1)


        
        self.setLayout(self.main_layout)
    
    def parse_params(self):
        params = self.input_model.text().split(" ")
        return params
    
    def generate_model_1(self):
        params = self.model_1_params.text().split(" ")

        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        cv2.line(img, (self.width // 2, 0), (self.width // 2, self.height), 0, 1)
        cv2.line(img, (0, self.height // 2), (self.width, self.height // 2), 0, 1)

        rectangle_height = 3
        rectangle_width = 20
        i0 = int(params[0]) # x
        j0 = int(params[1]) # y
        
        center_x, center_y = self.width // 2, self.height // 2
        x_position, y_position = center_x + i0, center_y + j0

        rec = cv2.rectangle(img, (x_position, y_position), (x_position + rectangle_width, y_position + rectangle_height), 0, -1)
        self.image_m = rec
        self.image_v = None
        self.image_generated.emit(img)

    def generate_model_2(self):
        params = self.model_2_params.text().split(" ")

        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255


        cv2.line(img, (self.width // 2, 0), (self.width // 2, self.height), 0, 1)
        cv2.line(img, (0, self.height // 2), (self.width, self.height // 2), 0, 1)

        i0 = int(params[0]) # x
        j0 = int(params[1]) # y

        center_x, center_y = self.width // 2, self.height // 2
        x_position, y_position = center_x + i0, center_y + j0

        start_x, start_y = center_x + i0, center_y + j0
        end_x, end_y = self.width - 1, start_y
        thickness = 2
        line = cv2.line(img, (start_x, start_y), (end_x, end_y), 0, thickness)

        self.image_m = line
        self.image_v = None
        self.image_generated.emit(img)

    def generate_model_3(self):
        def generate_frame(radius, center, frame_size, deltaR, T0, t):
            frame = 255 * np.ones(frame_size, dtype=np.uint8)  
            changing_radius = int(radius + deltaR * np.cos(2 * np.pi * t / T0))
            cv2.circle(frame, center, changing_radius, 0, thickness=-1)

            return frame
        
        output_video_path  = 'generated_vids/changing_radius_circle_formula.mp4'
        frame_size = (self.height, self.width)
        fps = config.fps
        duration_sec = 5
        center = (frame_size[0] // 2, frame_size[1] // 2)

        params = self.model_3_params.text().split(" ")

        radius = 100
        deltaR = float(params[0])
        T0 = float(params[1])

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size, isColor=False)
        total_frames = int(fps * duration_sec)
        for frame_num in range(total_frames):
            t = frame_num / fps 
            frame = generate_frame(radius, center, frame_size, deltaR, T0, t)
            out.write(frame)

        self.image_m = None
        self.image_v = output_video_path
        out.release()
        self.video_generate.emit(output_video_path)

    def generate_model_4(self):
        def generate_frame(side_length, frame_size, deltaR, T0, t):
            frame = 255 * np.ones(frame_size, dtype=np.uint8)  
            changing_side = int(side_length + deltaR * np.cos(2 * np.pi * t / T0))
            x = (frame_size[0] - changing_side) // 2
            y = (frame_size[1] - changing_side) // 2
            cv2.rectangle(frame, (x, y), (x + changing_side, y + changing_side), 0, thickness=-1)

            return frame
        
        output_video_path  = 'generated_vids/changing_radius_circle_formula.mp4'
        frame_size = (self.height, self.width)
        fps = config.fps
        duration_sec = 5

        params = self.model_4_params.text().split(" ")

        side_length = 100
        delta_a = int(params[0])
        T0 = float(params[1])

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size, isColor=False)
        total_frames = int(fps * duration_sec)
        for frame_num in range(total_frames):
            t = frame_num / fps  
            frame = generate_frame(side_length, frame_size, delta_a, T0, t)
            out.write(frame)

        self.image_m = None
        self.image_v = output_video_path
        out.release()
        self.video_generate.emit(output_video_path)
        
    def generate_model_5(self):
        params = self.model_5_params.text().split(" ")
        p1 = int(params[0])
        p2 = int(params[1])

        total_size = p1 * p2

        # Create a white background
        chessboard = np.ones((total_size, total_size, 3), dtype=np.uint8) * 255

        for i in range(p1):
            for j in range(p1):
                # Calculate the starting point of each cell
                x = j * p2
                y = i * p2

                # Draw a black or white square based on the cell indices
                if (i + j) % 2 == 0:
                    cv2.rectangle(chessboard, (x, y), (x + p2, y + p2), 0, thickness=-1)

        self.image_m = chessboard
        self.image_v = None
        self.image_generated.emit(chessboard)
    
    def generate_model_7(self):
        def generate_random_circles(num_circles, max_radius, image_size):
            circles = []
            for _ in range(num_circles):
                radius = random.randint(5, max_radius)
                center = (random.randint(radius, image_size[1] - radius), random.randint(radius, image_size[0] - radius))
                direction = random.uniform(0, 2 * np.pi)
                circles.append({'center': center, 'radius': radius, 'direction': direction})
            return circles

        def update_circle_positions(circles, speed, repulsion_factor, image_size):
            for i in range(len(circles)):
                x, y = circles[i]['center']
                direction = circles[i]['direction']

                # Update circle position based on linear movement
                x += speed * np.cos(direction)
                y += speed * np.sin(direction)

                # Repulsion from other circles
                for j in range(len(circles)):
                    if i != j:
                        distance = np.sqrt((x - circles[j]['center'][0]) ** 2 + (y - circles[j]['center'][1]) ** 2)
                        if distance < circles[i]['radius'] + circles[j]['radius']:
                            # Apply repulsion force
                            repulsion_force = repulsion_factor / distance
                            direction += repulsion_force * np.arctan2(y - circles[j]['center'][1], x - circles[j]['center'][0])

                # Check and reflect at boundaries
                if x - circles[i]['radius'] < 0 or x + circles[i]['radius'] >= image_size[1]:
                    direction = np.pi - direction
                if y - circles[i]['radius'] < 0 or y + circles[i]['radius'] >= image_size[0]:
                    direction = -direction

                circles[i]['center'] = (x, y)
                circles[i]['direction'] = direction

        def draw_circles(image, circles):
            for circle in circles:
                cv2.circle(image, (int(circle['center'][0]), int(circle['center'][1])), circle['radius'], (0, 0, 0), -1)

        # Video parameters
        image_size = (self.width, self.height)
        num_circles = 5
        max_radius = 30
        speed = 2.0
        repulsion_factor = 100.0  # Adjust the repulsion factor as needed
        total_frames = 200
        fps = 30
        output_video_path = 'generated_vids/random_circles_repulsion_video.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, image_size, isColor=True)

        # Generate random circles
        circles = generate_random_circles(num_circles, max_radius, image_size)

        # Simulation
        for frame_num in range(total_frames):
            # Update circle positions with repulsion
            update_circle_positions(circles, speed, repulsion_factor, image_size)

            # Draw circles on the image
            frame = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255  # White background
            draw_circles(frame, circles)

            # Write the frame to the video
            out.write(frame)

        # Release the VideoWriter
        self.image_m = None
        self.image_v = output_video_path
        out.release()
        self.video_generate.emit(output_video_path)

    
    def generate_model_8(self):
        params = self.model_8_params.text().split(" ")
        p1 = 0
        p2 = 255 
        if params[0] != "":
            p1 = float(params[0])
            p2 = float(params[1])
        print(p1, p2)

        image_size = (self.height, self.width)

        def generate_uniform_noise_frame(height, width, low=0, high=255):
            # Generate white noise with uniform distribution for a single frame
            noise_frame = np.random.uniform(low, high, size=(height, width, 3)).astype(np.uint8)
            return noise_frame

        total_frames = 100  
        fps = 30  
        output_video_path = 'generated_vids/uniform_noise_videoo.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, image_size)

        for frame_num in range(total_frames):
            noise_frame = generate_uniform_noise_frame(image_size[0], image_size[1], p1, p2)
            out.write(noise_frame)

        self.image_m = None
        self.image_v = output_video_path
        out.release()
        self.video_generate.emit(output_video_path)


    def generate_model_9(self):
        image_size = (self.height, self.width)

        params = self.model_9_params.text().split(" ")


        if params[0] != "":
            p1 = int(params[0])
            p2 = int(params[1])

        def generate_normal_noise(mean=0, variance=1, image_size=(512, 512)):
            noise_field = np.random.normal(mean, np.sqrt(variance), image_size)
            noise_field_scaled = np.clip(noise_field, 0, 255).astype(np.uint8)

            return noise_field_scaled
        
        mean = 128  # Mean of the normal distribution
        variance = 100  # Variance of the normal distribution
        if params[0] != "":
            mean = int(params[0])
            variance = int(params[1])

        total_frames = 100  # Number of frames in the video
        fps = 30  # Frames per second
        output_video_path = 'generated_vids/normal_noise_video.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, image_size, isColor=False)

        for frame_num in range(total_frames):
            noise_frame = generate_normal_noise(mean, variance, image_size)
            out.write(noise_frame)

        
        self.image_m = None
        self.image_v = output_video_path
        out.release()
        self.video_generate.emit(output_video_path)

    def resize_with_aspect_ratio(self, image, target_width=None, target_height=None):
        current_height, current_width = image.shape[:2]

        aspect_ratio = current_width / float(current_height)

        if target_width is not None and target_height is not None:
            new_width = target_width
            new_height = target_height

        elif target_width is not None:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        elif target_height is not None:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        else:
            return image
        print(new_width, new_height)
        resized_image = cv2.resize(image, (int(new_width), int(new_height)))
        return resized_image
    
    def resize_video(self, input_path, output_path, target_width=None, target_height=None):
        video_capture = cv2.VideoCapture(input_path)

        original_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect_ratio = original_width / float(original_height)
        if target_width is not None and target_height is not None:
            new_width = target_width
            new_height = target_height
        elif target_width is not None:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        elif target_height is not None:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        else:
            new_width, new_height = original_width, original_height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (int(new_width), int(new_height)))

        while True:
            ret, frame = video_capture.read()

            if not ret:
                break

            resized_frame = cv2.resize(frame, (int(new_width), int(new_height)))

            video_writer.write(resized_frame)

        video_capture.release()
        video_writer.release()

        print(f"Resized video saved to {output_path}")

    def save(self):
        if self.image_m is None and self.image_v is None:
            return
        
        x = float(self.size.text().split()[0])
        if self.image_m is not None:

            file_dialog = QFileDialog()
            file_dialog.setDefaultSuffix("jpg")

            file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            selected_file, _ = file_dialog.getSaveFileName(self, 'Save data', '', 'Jpg Files (*.jpg)')
            
            if selected_file:
                resized_image = self.resize_with_aspect_ratio(self.image_m, x)
                cv2.imwrite(selected_file, resized_image)
        else:
            file_dialog = QFileDialog()
            file_dialog.setDefaultSuffix("mp4")

            file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            selected_file, _ = file_dialog.getSaveFileName(self, 'Save data', '', 'mp4 Files (*.mp4)')
            
            if selected_file:
                self.resize_video(self.image_v, selected_file, target_width=x)
