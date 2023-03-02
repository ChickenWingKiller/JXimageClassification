import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from UI import show_records
import pickle

class MainWindow(QtWidgets.QMainWindow, show_records.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        with open("X:\金相研磨图像识别项目\pythonProject\\results_record\\results.pickle", 'rb') as file:
            records_list = list()
            while True:
                try:
                    one_record = pickle.load(file)
                    records_list.append(one_record)
                except:
                    break
        self.tableWidget.setRowCount(len(records_list))
        self.tableWidget.verticalHeader().setHidden(True)
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setHorizontalHeaderLabels(['序号','时间', '图片名', '结果'])
        for i in range(len(records_list)):
            for j in range(4):
                if (j==0):
                    num = QtWidgets.QTableWidgetItem(str(i+1))
                    num.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                    self.tableWidget.setItem(i,j,num)
                    continue
                data = QtWidgets.QTableWidgetItem(str(records_list[i][j-1]))
                if (j==3):
                    data.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                self.tableWidget.setItem(i, j, data)
        self.tableWidget.resizeColumnToContents(0)  # 使列宽跟随内容改变
        self.tableWidget.resizeColumnToContents(1)  # 使列宽跟随内容改变
        self.tableWidget.setAlternatingRowColors(True)  # 使表格颜色交错显示
        self.tableWidget.setColumnWidth(0, 80)
        self.tableWidget.setColumnWidth(1, 210)
        self.tableWidget.setColumnWidth(2, 300)
        self.tableWidget.setColumnWidth(3, 160)
        self.pushButton.clicked.connect(self.back)

    def setui(self,window):
        self.setupUi(window)
        with open("X:\金相研磨图像识别项目\pythonProject\\results_record\\results.pickle", 'rb') as file:
            records_list = list()
            while True:
                try:
                    one_record = pickle.load(file)
                    records_list.append(one_record)
                except:
                    break
        self.tableWidget.setRowCount(len(records_list))
        self.tableWidget.verticalHeader().setHidden(True)
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setHorizontalHeaderLabels(['序号', '时间', '图片名', '结果'])
        for i in range(len(records_list)):
            for j in range(4):
                if (j == 0):
                    num = QtWidgets.QTableWidgetItem(str(i + 1))
                    num.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                    self.tableWidget.setItem(i, j, num)
                    continue
                data = QtWidgets.QTableWidgetItem(str(records_list[i][j - 1]))
                if (j == 3):
                    data.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                self.tableWidget.setItem(i, j, data)
        self.tableWidget.resizeColumnToContents(0)  # 使列宽跟随内容改变
        self.tableWidget.resizeColumnToContents(1)  # 使列宽跟随内容改变
        self.tableWidget.setAlternatingRowColors(True)  # 使表格颜色交错显示
        self.tableWidget.setColumnWidth(0, 80)
        self.tableWidget.setColumnWidth(1, 210)
        self.tableWidget.setColumnWidth(2, 300)
        self.tableWidget.setColumnWidth(3, 160)
        self.pushButton.clicked.connect(self.back)

    def back(self):
        pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())