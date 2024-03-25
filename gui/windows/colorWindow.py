from PyQt6.QtWidgets import QWidget, QPushButton, QMenu
from PyQt6.QtGui import QAction

from ..state import state

class ColorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        super().__init__()
        
        self.color_button = QPushButton(state.selected_filter, self)
        
        menu = QMenu()
        color_schemes = state.filters

        for scheme in color_schemes:
            action = QAction(scheme, self)
            action.triggered.connect(self.update_color_scheme)
            menu.addAction(action)
        self.color_button.setMenu(menu)

    def update_color_scheme(self):
        action = self.sender()
        if action:
            self.color_scheme = action.text()
            state.selected_filter = action.text()
            print(state.selected_filter)
            self.color_button.setText(state.selected_filter)