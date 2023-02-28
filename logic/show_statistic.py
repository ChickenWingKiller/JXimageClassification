import pickle

from PyQt5 import QtCore, QtGui, QtWidgets, QtChart
from UI import statistic_figure
from logic import classify_image


class MainWindow(QtWidgets.QMainWindow, statistic_figure.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self, self.check_num())
        # self.create_piechart(self)
        self.pushButton_4.clicked.connect(self.back)


    def setui(self, window):
        self.setupUi(window, self.check_num())
        # self.create_piechart(window)
        self.pushButton_4.clicked.connect(self.back)

    def back(self):
        pass

    def check_num(self):
        pass_num = 0
        not_pass_num = 0
        with open("X:\金相研磨图像识别项目\pythonProject\\results_record\\results.pickle", 'rb') as file:
            while True:
                try:
                    one_record = pickle.load(file)
                    if (one_record[2] == "合格"):
                        pass_num += 1
                    else:
                        not_pass_num += 1
                except:
                    break
        return pass_num, not_pass_num

    def create_piechart(self, window):
        pass_num = classify_image.MainWindow.RESULT_STATISTIC['合格']
        notpass_num = classify_image.MainWindow.RESULT_STATISTIC['不合格']
        series = QtChart.QPieSeries()
        series.append('合格', pass_num)
        series.append('不合格', notpass_num)

        slice_pass = series.slices()[0]
        print(slice_pass)
        # slice_pass.setExploded(True)
        slice_pass.setLabelVisible(True)
        slice_pass.setPen(QtGui.QPen(QtCore.Qt.blue, 2))
        slice_pass.setBrush(QtCore.Qt.blue)

        slice_notpass = series.slices()[1]
        print(slice_notpass)
        slice_notpass.setLabelVisible(True)
        slice_notpass.setPen(QtGui.QPen(QtCore.Qt.red, 2))
        slice_notpass.setBrush(QtCore.Qt.red)

        chart = QtChart.QChart()
        chart.legend().hide()
        chart.addSeries(series)
        chart.createDefaultAxes()

        chart.setAnimationOptions(QtChart.QChart.SeriesAnimations)

        # chart.setTitle("饼图")

        chart.legend().setVisible(True)
        chart.legend().setAlignment(QtCore.Qt.AlignBottom)

        chartview = QtChart.QChartView(chart)
        chartview.setRenderHint(QtGui.QPainter.Antialiasing)

        window.setCentralWidget(chartview)
