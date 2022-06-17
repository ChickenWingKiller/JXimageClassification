import requests
import socket

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# host = socket.gethostname()  # 获取本地主机名
# ip = socket.gethostbyname(host)
# port = 9999  # 设置端口号
# s.connect((ip, port))
# while True:
#     message = s.recv(1024)
#     # if message != '':
#     # print(message.decode('utf-8'))
#     print(message)
#     s.close()
# import sys
#
# from PyQt5.QtWidgets import QApplication, QMainWindow
# from PyQt5.QtChart import QChart, QChartView, QPieSeries, QPieSlice
# from PyQt5.QtGui import QPainter, QPen
# from PyQt5.QtCore import Qt
#
#
# class Window(QMainWindow):
#     def __init__(self):
#         super().__init__()
#
#         self.setWindowTitle("PyQt5饼图")
#
#         # 显示位置
#         self.setGeometry(100, 100, 800, 600)
#         self.create_piechart()
#         self.show()
#
#     def create_piechart(self):
#         # 创建QPieSeries对象，它用来存放饼图的数据
#         series = QPieSeries()
#
#         # append方法中的数字，代表的是权重，完成可以改成其它，如80,70,60等等
#         series.append("Python", 8)
#         series.append("Java", 7)
#         series.append("C", 6)
#         series.append("C++", 5)
#         series.append("PHP", 4)
#         series.append("Swift", 3)
#
#         # 单独处理某个扇区
#         slice = QPieSlice()
#
#         # 这里要处理的是python项，是依据前面append的顺序，如果是处理C++项的话，那索引就是3
#         slice = series.slices()[0]
#
#         # 突出显示，设置颜色
#         slice.setExploded(True)
#         slice.setLabelVisible(True)
#         slice.setPen(QPen(Qt.red, 2))
#         slice.setBrush(Qt.red)
#
#         # 创建QChart实例，它是PyQt5中的类
#         chart = QChart()
#         # QLegend类是显示图表的图例，先隐藏掉
#         chart.legend().hide()
#         chart.addSeries(series)
#         chart.createDefaultAxes()
#
#         # 设置动画效果
#         chart.setAnimationOptions(QChart.SeriesAnimations)
#
#         # 设置标题
#         chart.setTitle("饼图示例")
#
#         chart.legend().setVisible(True)
#
#         # 对齐方式
#         chart.legend().setAlignment(Qt.AlignBottom)
#
#         # 创建ChartView，它是显示图表的控件
#         chartview = QChartView(chart)
#         chartview.setRenderHint(QPainter.Antialiasing)
#
#         self.setCentralWidget(chartview)
#
#
# App = QApplication(sys.argv)
# window = Window()
# sys.exit(App.exec_())
import sys
import os
import cv2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import QCameraInfo


class Video(QMainWindow):
    def __init__(self):
        super().__init__()

        self.camid = -1  # 当前摄像头编号
        self.initUi()
        self.get_cam()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.play)

    def initUi(self):
        self.centralwidget = QWidget()
        self.setCentralWidget(self.centralwidget)
        self.startButton = QPushButton("开启")
        self.closeButton = QPushButton("关闭")
        self.camButton = QPushButton("拍照")
        self.promptLabel = QLabel('请选择摄像头：', self)
        self.combo = QComboBox(self)  # 摄像头列表
        hbox = QHBoxLayout()
        hbox.addWidget(self.promptLabel)
        hbox.addWidget(self.combo)
        hbox.addStretch(1)
        hbox.addWidget(self.startButton)
        hbox.addWidget(self.closeButton)
        hbox.addWidget(self.camButton)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        self.vF = QLabel()
        vbox.addWidget(self.vF)
        self.centralwidget.setLayout(vbox)

        self.setGeometry(300, 300, 500, 500)
        self.setWindowTitle('简单拍照')

        self.startButton.clicked.connect(self.onStartVideo)
        self.closeButton.clicked.connect(self.onCloseVideo)
        self.camButton.clicked.connect(self.onCamera)
        self.combo.currentIndexChanged.connect(self.selectionchange)

        self.startButton.setEnabled(False)
        self.closeButton.setEnabled(False)
        self.camButton.setEnabled(False)

    def onStartVideo(self):

        if self.camid < 0:
            return

        # 初始化传入的摄像头句柄为实例变量,并得到摄像头宽度和高度
        # cv2.CAP_DSHOW
        self.cam = cv2.VideoCapture(self.camid, cv2.CAP_DSHOW)
        self.w = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # 设置GUI窗口的位置和尺寸
        self.setGeometry(300, 200, self.w + 20, self.h + 100)
        # 开启定时器
        self._timer.start(50)

        self.startButton.setEnabled(False)
        self.closeButton.setEnabled(True)
        self.camButton.setEnabled(True)
        self.combo.setEnabled(False)

    def onCloseVideo(self):
        self._timer.stop()

        if self.camid >= 0:
            self.startButton.setEnabled(True)
            self.closeButton.setEnabled(False)
            self.camButton.setEnabled(False)
            self.combo.setEnabled(True)

    def selectionchange(self):
        self.camid = self.combo.currentIndex()
        self.startButton.setEnabled(True)

    def play(self):
        """
        从摄像头得到图像 先转换为RGB格式 再生成QImage对象
        再用此QImage刷新vF实例变量 以刷新视频画面
        """
        r, f = self.cam.read()
        if r:
            self.vF.setPixmap(QPixmap.fromImage(QImage(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), self.w, self.h, 13)))
            # cv2.imwrite("c:\\temp\\aaa.jpg",f, [int(cv2.IMWRITE_JPEG_QUALITY),95])

    def get_cam(self):  # 使用QCameraInfo得到摄像头列表
        camlist = QCameraInfo.availableCameras()
        self.cnt = 0
        for cam in camlist:
            self.combo.addItem(str(self.cnt) + ' - ' + cam.description())
            self.cnt = self.cnt + 1
        if self.cnt == 0:
            ret = QMessageBox.information(self, "Attention", "没有找到摄像头", QMessageBox.Ok)

    def onCamera(self):  # 拍照并保存
        r, f = self.cam.read()
        if r:
            # self.vF.setPixmap(QPixmap.fromImage(QImage(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), self.w,self.h,13)))
            # cv2.imwrite("c:\\temp\\aaa.jpg",f, [int(cv2.IMWRITE_JPEG_QUALITY),95])
            newfileName, ok = QFileDialog.getSaveFileName(self, "文件另存为", "", "*.jpg")
            if newfileName:
                cv2.imwrite(newfileName, f, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                os.system(newfileName)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 初始化GUI窗口 并传入摄像头句柄
    win = Video()
    win.show()
    sys.exit(app.exec_())