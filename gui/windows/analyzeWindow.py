from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QTextEdit, QPushButton, QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

from gui.config import config

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd

class AnalyzeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.screen_w = config.screen_w
        self.screen_h = config.screen_h
        self.datas = pd.DataFrame(columns=["mean", "dispersion", "std", "skew", "kurtosis", 
                                           "variance", "min", "max", "percentile5", "percentile95", "median", "hist"])

        self.histogram_label = QLabel(self)
        
        statistics_h, statistics_w = 60, 100
        self.means_label = QTextEdit(self)
        self.means_label.setReadOnly(True)  
        self.means_label.setFixedHeight(statistics_h)
        self.means_label.setFixedWidth(statistics_w)
        self.means_label.setStyleSheet("border: none;")

        self.dispersion_label = QTextEdit(self)
        self.dispersion_label.setReadOnly(True)  
        self.dispersion_label.setFixedHeight(statistics_h)
        self.dispersion_label.setFixedWidth(statistics_w)
        self.dispersion_label.setStyleSheet("border: none;")

        self.std_label = QTextEdit(self)
        self.std_label.setReadOnly(True)  
        self.std_label.setFixedHeight(statistics_h)
        self.std_label.setFixedWidth(statistics_w)
        self.std_label.setStyleSheet("border: none;")

        self.skew_label = QTextEdit(self)
        self.skew_label.setReadOnly(True)  
        self.skew_label.setFixedHeight(statistics_h)
        self.skew_label.setFixedWidth(statistics_w)
        self.skew_label.setStyleSheet("border: none;")

        self.kurtosis_label = QTextEdit(self)
        self.kurtosis_label.setReadOnly(True)  
        self.kurtosis_label.setFixedHeight(statistics_h)
        self.kurtosis_label.setFixedWidth(statistics_w)
        self.kurtosis_label.setStyleSheet("border: none;")

        self.variance_label = QTextEdit(self)
        self.variance_label.setReadOnly(True)  
        self.variance_label.setFixedHeight(statistics_h)
        self.variance_label.setFixedWidth(statistics_w)
        self.variance_label.setStyleSheet("border: none;")

        self.mins_label = QTextEdit(self)
        self.mins_label.setReadOnly(True)  
        self.mins_label.setFixedHeight(statistics_h)
        self.mins_label.setFixedWidth(statistics_w)
        self.mins_label.setStyleSheet("border: none;")

        self.maxs_label = QTextEdit(self)
        self.maxs_label.setReadOnly(True)  
        self.maxs_label.setFixedHeight(statistics_h)
        self.maxs_label.setFixedWidth(statistics_w)
        self.maxs_label.setStyleSheet("border: none;")

        self.percentil5_label = QTextEdit(self)
        self.percentil5_label.setReadOnly(True)  
        self.percentil5_label.setFixedHeight(statistics_h)
        self.percentil5_label.setFixedWidth(statistics_w)
        self.percentil5_label.setStyleSheet("border: none;")
        
        self.percentil95_label = QTextEdit(self)
        self.percentil95_label.setReadOnly(True)  
        self.percentil95_label.setFixedHeight(statistics_h)
        self.percentil95_label.setFixedWidth(statistics_w)
        self.percentil95_label.setStyleSheet("border: none;")

        self.median_label = QTextEdit(self)
        self.median_label.setReadOnly(True)  
        self.median_label.setFixedHeight(statistics_h)
        self.median_label.setFixedWidth(statistics_w)
        self.median_label.setStyleSheet("border: none;")
        
        self.save_statistics = self.button = QPushButton('Save statistics', self)

        self.main_layout = QGridLayout()
        self.initUI()

    def initUI(self):
        super().__init__()
        
        self.save_statistics.clicked.connect(self.show_file_dialog)

        self.setWindowTitle('Statistics')
        self.setGeometry(600, 200, self.screen_w, self.screen_h)

        self.setLayout(self.main_layout)

    def convert_cv_to_pixmap(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap
    
    def convert_f(self, elem):
        return str(int(elem))
    
    def display_params(self, hist, means, dispersion, std, skew, kurtosis,
                       variance, mins, maxs, percentil5, percentil95,
                       median, hist_path, freq):
        

        self.hist = hist
        self.means = means[0]["val"]
        self.dispersion = dispersion[0]["val"]
        self.std = std[0]["val"]
        self.skew = skew[0]["val"]
        self.kurtosis = kurtosis[0]["val"]
        self.variance = variance[0]["val"]
        self.mins = mins[0]["val"]
        self.maxs = maxs[0]["val"]
        self.percentil5 = percentil5[0]["val"]
        self.percentil95 = percentil95[0]["val"]
        self.median = median[0]["val"]
        self.hist_path = hist_path
        

        data = {
            "mean": self.means , 
            "dispersion": self.dispersion, "std": self.std, "skew": self.skew, "kurtosis": self.kurtosis,
            "variance": self.variance, "min": self.mins, "max": self.maxs, "percentile5": self.percentil5, 
            "percentile95": self.percentil95, "median": self.median , "hist": ", ".join(map(self.convert_f, freq))}
        
        self.datas.loc[len(self.datas)] = data 

        text = "Среднее\n"
        for m in means:
            text += f''' {m["name"]}: {m["val"]}\n'''
        self.means_label.setPlainText(text)
        
        text = ""
        text += "Дисперсия\n"
        for v in dispersion:
            text += f''' {v["name"]}: {v["val"]}\n'''        
        self.dispersion_label.setPlainText(text)

        text = ""
        text += "Стд отклонение\n"
        for v in std:
            text += f''' {v["name"]}: {v["val"]}\n'''        
        self.std_label.setPlainText(text)

        text = ""
        text += "Ассиметрия\n"
        for v in skew:
            text += f''' {v["name"]}: {v["val"]}\n'''        
        self.skew_label.setPlainText(text)

        text = ""
        text += "Эксцесс\n"
        for v in kurtosis:
            text += f''' {v["name"]}: {v["val"]}\n'''        
        self.kurtosis_label.setPlainText(text)

        text = ""
        text += "Вариация\n"
        for v in variance:
            text += f''' {v["name"]}: {v["val"]}\n'''        
        self.variance_label.setPlainText(text)

        text = ""
        text += "Минимум\n"
        for v in mins:
            text += f''' {v["name"]}: {v["val"]}\n'''        
        self.mins_label.setPlainText(text)

        text = ""
        text += "Максимум\n"
        for v in maxs:
            text += f''' {v["name"]}: {v["val"]}\n'''        
        self.maxs_label.setPlainText(text)
        
        text = ""
        text += "Квартиль 0.05\n"
        for v in percentil5:
            text += f''' {v["name"]}: {v["val"]}\n'''        
        self.percentil5_label.setPlainText(text)

        text = ""
        text += "Квартиль 0.95\n"
        for v in percentil95:
            text += f''' {v["name"]}: {v["val"]}\n'''        
        self.percentil95_label.setPlainText(text)

        text = ""
        text += "Медиана\n"
        for v in median:
            text += f''' {v["name"]}: {v["val"]}\n'''        
        self.median_label.setPlainText(text)

        self.histogram_label.setPixmap(self.convert_cv_to_pixmap(hist))

        self.main_layout.setSpacing(0)

        
        self.main_layout.addWidget(self.means_label, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addWidget(self.dispersion_label, 0, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addWidget(self.std_label, 0, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addWidget(self.skew_label, 0, 3, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addWidget(self.kurtosis_label, 1, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addWidget(self.variance_label, 1, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addWidget(self.mins_label, 1, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addWidget(self.maxs_label, 1, 3, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addWidget(self.percentil5_label, 2, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addWidget(self.percentil95_label, 2, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addWidget(self.median_label, 2, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addWidget(self.save_statistics, 2, 3, alignment=Qt.AlignmentFlag.AlignLeft)


        self.main_layout.setRowStretch(0, 1)

        self.main_layout.addWidget(self.histogram_label, 10, 0, 1, 10)


    def show_file_dialog(self):

        file_dialog = QFileDialog()
        # file_dialog.setDefaultSuffix('pdf')
        file_dialog.setDefaultSuffix("xlsx")

        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        selected_file, _ = file_dialog.getSaveFileName(self, 'Save data', '', 'Xlsx Files (*.xlsx)')
        
        if selected_file:
            self.datas.to_excel(selected_file, index=False)
            print("saved")
        else:
            return