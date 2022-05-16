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
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setHorizontalHeaderLabels(['时间', '图片名', '结果'])
        for i in range(len(records_list)):
            for j in range(3):
                data = QtWidgets.QTableWidgetItem(str(records_list[i][j]))
                self.tableWidget.setItem(i, j, data)
        self.tableWidget.resizeColumnToContents(0)  # 使列宽跟随内容改变
        self.tableWidget.resizeColumnToContents(1)  # 使列宽跟随内容改变
        self.tableWidget.setAlternatingRowColors(True)  # 使表格颜色交错显示
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
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setHorizontalHeaderLabels(['时间', '图片名', '结果'])
        for i in range(len(records_list)):
            for j in range(3):
                data = QtWidgets.QTableWidgetItem(str(records_list[i][j]))
                self.tableWidget.setItem(i, j, data)
        self.tableWidget.resizeColumnToContents(0)  # 使列宽跟随内容改变
        self.tableWidget.resizeColumnToContents(1)  # 使列宽跟随内容改变
        self.tableWidget.setAlternatingRowColors(True)  # 使表格颜色交错显示
        self.pushButton.clicked.connect(self.back)

    def back(self):
        pass

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())