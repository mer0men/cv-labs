import sys
from PyQt6.QtWidgets import QApplication

from gui.mainWindow import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec()
