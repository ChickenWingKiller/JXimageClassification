from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from logic import classify_image, main_window, show_records

class mainWindow(main_window.MainWindow):
    def openClassifierWindow(self):
        window_1.show()
    def openRecordsWindow(self):
        window_2.show()

class classifyWindow(classify_image.MainWindow):
    pass

class recordsWindow(show_records.MainWindow):
    pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = mainWindow()
    window_1 = classifyWindow()
    window_2 = recordsWindow()
    window.show()
    sys.exit(app.exec_())