from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from UI import main_window


class MainWindow(QtWidgets.QMainWindow, main_window.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.openClassifierWindow)
        self.pushButton_2.clicked.connect(self.openRecordsWindow)
        self.pushButton_3.clicked.connect(self.openStatisticWindow)
        self.pushButton_4.clicked.connect(self.exit)

    def setui(self, window):
        self.setupUi(window)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.openClassifierWindow)
        self.pushButton_2.clicked.connect(self.openRecordsWindow)
        self.pushButton_3.clicked.connect(self.openStatisticWindow)
        self.pushButton_4.clicked.connect(self.exit)

    def openClassifierWindow(self):
        pass

    def openRecordsWindow(self):
        pass

    def openStatisticWindow(self):
        pass

    def exit(self):
        pass

# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     app.setStyle("Fusion")
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())
