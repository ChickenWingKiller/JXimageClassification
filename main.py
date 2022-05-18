from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from logic import classify_image, main_window, show_records, show_statistic

class mainWindow(main_window.MainWindow):
    def openClassifierWindow(self):
        window_1.setui(window)
    def openRecordsWindow(self):
        window_2.setui(window)
    def openStatisticWindow(self):
        window_3.setupUi(window)
    def exit(self):
        window.close()

class classifyWindow(classify_image.MainWindow):
    def back(self):
        window.setui(window)

class recordsWindow(show_records.MainWindow):
    def back(self):
        window.setui(window)

class statisticWindow(show_statistic.MainWindow):
    def back(self):
        window.setui(window)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = mainWindow()
    window_1 = classifyWindow()
    window_2 = recordsWindow()
    window_3 = statisticWindow()
    window.show()
    sys.exit(app.exec_())