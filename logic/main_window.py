from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from UI import main_window

class MainWindow(QtWidgets.QMainWindow, main_window.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.openClassifierWindow)
        self.pushButton_2.clicked.connect(self.openRecordsWindow)
        self.pushButton_4.clicked.connect(MainWindow.close)
    def openClassifierWindow(self):
        pass
    def openRecordsWindow(self):
        pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())