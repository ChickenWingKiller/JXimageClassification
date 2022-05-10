from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageTk


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.reshape(-1, 7 * 7 * 512)
        x = self.dense1(x)
        return x


class Ui_MainWindow(object):
    def __init__(self):
        self.model = Net()
        self.modelPath = "../Trained Model/net89.pth"
        self.model.load_state_dict(torch.load(self.modelPath, map_location=torch.device('cpu')))
        self.image_path = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStyleSheet("background-color: rgb(252, 223, 86);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 801, 81))
        self.label.setStyleSheet("background-color: rgb(95, 125, 164);\n"
                                 "color: rgb(255, 255, 255);")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(0, 80, 531, 431))
        self.label_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(570, 140, 200, 50))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("background-color: rgb(95, 125, 164);\n"
                                      "color: rgb(255, 255, 255);")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(570, 280, 200, 50))
        self.pushButton_2.setMouseTracking(True)
        self.pushButton_2.setStyleSheet("background-color: rgb(95, 125, 164);\n"
                                        "font: 12pt \"Agency FB\";\n"
                                        "color: rgb(255, 255, 255);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(570, 420, 200, 50))
        self.pushButton_3.setMouseTracking(True)
        self.pushButton_3.setStyleSheet("background-color: rgb(95, 125, 164);\n"
                                        "font: 12pt \"Agency FB\";\n"
                                        "color: rgb(255, 255, 255);")
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(290, 520, 171, 51))
        self.label_3.setStyleSheet("background-color: rgb(95, 125, 164);")
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.pushButton.clicked.connect(self.open_image)
        # self.pushButton_2.clicked.connect(self.test_fun)
        self.pushButton_2.clicked.connect(self.classify_image)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def open_image(self):
        # dir = QtWidgets.QFileDialog() #创建文件对话框
        # dir.setDirectory('X:/') #设置初始路径为X盘
        # print(dir)
        from PyQt5.QtWidgets import QFileDialog
        dir = QFileDialog()  # 创建文件对话框
        dir.setFileMode(QFileDialog.ExistingFiles)  # 设置多选
        # dir.setDirectory('X:\\')  # 设置初始路径为C盘
        dir.setDirectory('C:/Users/lyf19/Desktop/Temp/金相研磨图像识别项目/image/train/Pass')  # 设置初始路径为C盘
        dir.setNameFilter('图片文件(*.jpg *.png *.bmp *.ico *.gif)')# 设置只显示图片文件
        if dir.exec_(): #判断是否选择了文件
            # print(dir.selectedFiles())
            self.image_path = dir.selectedFiles()[0]
            image = QtGui.QPixmap(self.image_path)
            self.label_2.setPixmap(image)
            self.label_2.setScaledContents(True) #自适应图片大小

    def classify_image(self):
        image = Image.open(self.image_path)
        resized_image = image.resize((224,224),Image.ANTIALIAS)
        image_numpy = np.array(resized_image)
        image_torch = torch.from_numpy(image_numpy)
        image_torch = image_torch[None, :]
        image_torch = image_torch.float()
        image_torch = image_torch.permute(0,3,1,2)
        output = self.model(image_torch)
        _, result = torch.max(output,1)
        if result == 0:
            self.label_3.setText("合格")
        else:
            self.label_3.setText("不合格")

    def test_fun(self):
        image_path = "../images/test-image.BMP"
        image_path = "../images/test-image-notpass.BMP"
        image = Image.open(image_path)
        resized_image = image.resize((224, 224), Image.ANTIALIAS)
        image = np.array(resized_image)
        image = torch.from_numpy(image)
        image = image[None, :]
        image = image.float()
        image = image.permute(0, 3, 1, 2)
        output = self.model(image)
        _, prediction = torch.max(output, 1)
        if prediction == 0:
            print('合格')
        else:
            print('不合格')

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow",
                                      "<html><head/><body><p align=\"center\"><span style=\" font-size:22pt;\">金相研磨图像识别器</span></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "上传图片"))
        self.pushButton_2.setText(_translate("MainWindow", "识别图片"))
        self.pushButton_3.setText(_translate("MainWindow", "保存此条记录"))
        self.label_3.setText(_translate("MainWindow", "合格/不合格"))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
