from PyQt5 import QtCore, QtGui, QtWidgets
from UI import statistic_figure

class MainWindow(QtWidgets.QMainWindow, statistic_figure.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton_4.clicked.connect(self.back)
    def setui(self, window):
        self.setupUi(window)
        self.pushButton_4.clicked.connect(self.back)
    def back(self):
        pass

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())