from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import pickle
import datetime
import cv2
import time
from UI import classifier_window
from camera_operation import camera_thread


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


class MainWindow(QtWidgets.QMainWindow, classifier_window.Ui_MainWindow):
    RESULT_STATISTIC = {'合格': 1, '不合格': 1}
    LABEL_2_SIZE = [531, 431]

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.model = Net()
        self.modelPath = "../Trained Model/net89.pth"
        self.modelPath = "X:\金相研磨图像识别项目\\pythonProject\\Trained Model\\net89.pth"
        self.model.load_state_dict(torch.load(self.modelPath, map_location=torch.device('cpu')))
        self.image_path = None
        self.result = None
        # self.camera = cv2.VideoCapture(0)
        self.camera_timer = QtCore.QTimer(self)
        self.camera_timer.timeout.connect(self.camera_view)
        # self.camera_thread = camera_thread.Camera_thread()

        self.pushButton.clicked.connect(self.open_image)
        self.pushButton_2.clicked.connect(self.classify_image)
        self.pushButton_3.clicked.connect(self.save_record)
        self.pushButton_4.clicked.connect(self.back)
        self.pushButton_5.clicked.connect(self.open_camera)
        self.pushButton_6.clicked.connect(self.take_picture)

    def setui(self, window):
        self.setupUi(window)
        self.model = Net()
        self.modelPath = "../Trained Model/net89.pth"
        self.modelPath = "X:\金相研磨图像识别项目\\pythonProject\\Trained Model\\net89.pth"
        self.model.load_state_dict(torch.load(self.modelPath, map_location=torch.device('cpu')))
        self.image_path = None
        self.result = None
        self.camera_thread = camera_thread.Camera_thread()
        self.pushButton.clicked.connect(self.open_image)
        self.pushButton_2.clicked.connect(self.classify_image)
        self.pushButton_3.clicked.connect(self.save_record)
        self.pushButton_4.clicked.connect(self.back)
        self.pushButton_5.clicked.connect(self.open_camera)
        self.pushButton_6.clicked.connect(self.take_picture)

    def open_image(self):
        from PyQt5.QtWidgets import QFileDialog
        dir = QFileDialog()  # 创建文件对话框
        dir.setFileMode(QFileDialog.ExistingFiles)  # 设置多选
        dir.setDirectory('C:/Users/lyf19/Desktop/Temp/金相研磨图像识别项目/image/train/Pass')  # 设置初始路径为C盘
        dir.setNameFilter('图片文件(*.jpg *.png *.bmp *.ico *.gif)')  # 设置只显示图片文件
        if dir.exec_():  # 判断是否选择了文件
            print(dir.selectedFiles())
            self.image_path = dir.selectedFiles()[0]
            image = QtGui.QPixmap(self.image_path)
            self.label_2.setPixmap(image)
            self.label_2.setScaledContents(True)  # 自适应图片大小

    def classify_image(self):
        if self.image_path == None:
            QtWidgets.QMessageBox.warning(None, '警告', '尚未上传图片', QtWidgets.QMessageBox.Ok)
        else:
            image = Image.open(self.image_path)
            resized_image = image.resize((224, 224), Image.ANTIALIAS)
            image_numpy = np.array(resized_image)
            image_torch = torch.from_numpy(image_numpy)
            image_torch = image_torch[None, :]
            image_torch = image_torch.float()
            image_torch = image_torch.permute(0, 3, 1, 2)
            output = self.model(image_torch)
            _, result = torch.max(output, 1)
            if result == 0:
                self.result = "合格"
                self.label_3.setText("合格")
                self.label_3.setText(
                    "<html><head/><body><p><span style=\" font-size:16pt;\">合格</span></p></body></html>")
            else:
                self.result = "不合格"
                self.label_3.setText("不合格")
                self.label_3.setText(
                    "<html><head/><body><p><span style=\" font-size:16pt;\">不合格</span></p></body></html>")

    def save_record(self):
        if self.image_path != None and self.result != None:
            if self.result == '合格':
                MainWindow.RESULT_STATISTIC['合格'] += 1
            else:
                MainWindow.RESULT_STATISTIC['不合格'] += 1
            path_list = self.image_path.split('/')
            image_name = path_list[-3] + '/' + path_list[-2] + '/' + path_list[-1]
            one_img_info = [datetime.datetime.now().strftime('%Y-%m-%d% H:%M:%S'), image_name, self.result]
            with open('X:\金相研磨图像识别项目\pythonProject\\results_record\\results.pickle', 'ab') as file:
                pickle.dump(one_img_info, file, 1)
        elif self.image_path != None and self.result == None:
            QtWidgets.QMessageBox.warning(None, '警告', '尚未对此图片进行识别', QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.warning(None, '警告', '尚未上传图片', QtWidgets.QMessageBox.Ok)

    def open_camera(self):
        # self.camera_thread.changePixmap.connect(self.setCameraImage)
        # self.camera_thread.start()
        if not self.camera_timer.isActive(): # 如果相机关着，打开相机
            self.cap = cv2.VideoCapture(0)
            self.camera_timer.start(20)
        else: # 如果相机开着，关闭相机
            self.camera_timer.stop()
            self.cap.release()

    def camera_view(self):
        ret, frame = self.cap.read()
        rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Qimage = QtGui.QImage(rgbImage[:], rgbImage.shape[1], rgbImage.shape[0], rgbImage.shape[1] * 3,
                                 QtGui.QImage.Format_RGB888)  # pyqt5转换成自己能放的图片格式
        resized_image = QtGui.QPixmap(Qimage.scaled(self.label_2.size().width(), self.label_2.size().height(), QtCore.Qt.KeepAspectRatio))
        self.label_2.setPixmap(resized_image)
        # self.cap.release()
        # resized_image = QtGui.QPixmap(Qimage).scaled(self.label_2.geometry()[0],
        #                                                 classify_image.MainWindow.LABEL_2_SIZE[1])  # 设置图片大小
        #     # resized_image = QtGui.QPixmap(image).scaled(431, 331, QtCore.Qt.KeepAspectRatio)  # 设置图片大小
        #     self.changePixmap.emit(resized_image)

    def setCameraImage(self, pixImage):
        self.label_2.setPixmap(pixImage)

    def take_picture(self):
        if self.camera_timer.isActive():
            ret, frame = self.cap.read();
            cv2.waitKey(0)
            # image_name = QtCore.QDateTime().currentDateTime().toString(QtCore.Qt.DefaultLocaleLongDate)
            image_name = time.time()
            cv2.imwrite('../images/camera_takes/' + str(int(image_name)) + '.jpg', frame)
        else:
            QtWidgets.QMessageBox.warning(None, '警告', '相机已关闭', QtWidgets.QMessageBox.Ok)

    def back(self):
        pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
