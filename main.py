from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from logic import classify_image, main_window, show_records

class mainWindow(main_window.MainWindow):
    def openClassifierWindow(self):
        # self.m = classifyWindow()
        # self.m.show()
        # window_1.setupUi(windowMain)
        # window_1.update()
        # window_1.setupUi(window)
        # window_1.__init__()
        window_1.show()
        window.hide()
    def openRecordsWindow(self):
        # window_2.setupUi(window)
        window_2.show()
        window.hide()
    def exit(self):
        window.close()
        # windowMain.close()

class classifyWindow(classify_image.MainWindow):
    # def show(self):
    #     self.setupUi(window)
    # def setui(self, window):
    #     self.setupUi(window)
    def back(self):
        # window.setupUi(window)
        window.show()
        window_1.hide()


class recordsWindow(show_records.MainWindow):
    def back(self):
        window.show()
        window_2.hide()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    # windowMain = QtWidgets.QMainWindow()
    window = mainWindow()
    window_1 = classifyWindow()
    window_2 = recordsWindow()
    # window.__init__(windowMain)
    # window.setupUi(windowMain)
    # windowMain.show()
    window.show()
    sys.exit(app.exec_())