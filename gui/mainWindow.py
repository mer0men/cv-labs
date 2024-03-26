from PyQt6.QtWidgets import QWidget, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt6.QtGui import QAction, QIcon

from gui.windows.contentWindow import ContentWindow
from gui.windows.colorWindow import ColorWindow
from gui.windows.modelingConfigWindow import ModelingConfigWindow



class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()

        open_button = QPushButton(icon=QIcon('icons/folder-open.png'))
        open_button.setToolTip("Open file") 
        open_button.clicked.connect(self.open_content_window)


        filter_button = QPushButton(icon=QIcon('icons/settings-sliders.png'))
        filter_button.setToolTip("Filters")
        filter_button.clicked.connect(self.open_color_window)

        modeling_button = QPushButton(icon=QIcon('icons/modeling.png'))
        modeling_button.setToolTip("MOdeling")
        modeling_button.clicked.connect(self.open_modeling_window)


        layout.addWidget(open_button)
        layout.addWidget(filter_button)
        layout.addWidget(modeling_button)

        self.content_screen = ContentWindow()
        self.color_screen = ColorWindow()
        self.modeling_window = ModelingConfigWindow()

        self.modeling_window.image_generated.connect(self.content_screen.load_image)
        self.modeling_window.video_generate.connect(self.content_screen.load_video)


        self.setLayout(layout)
        self.setGeometry(200,100, 150, 50)
        self.setWindowTitle('IVP')
        self.show()

    def open_content_window(self):
        self.content_screen.show()
    
    def open_color_window(self):
        self.color_screen.setGeometry(620, 200, 300, 50)
        self.color_screen.show()

    def open_modeling_window(self):
        self.modeling_window.setGeometry(620, 200, 600, 50)
        self.modeling_window.show()