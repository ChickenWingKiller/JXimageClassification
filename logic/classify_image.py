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
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=6, eps=1e-05, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size=5, stride=5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(12, eps=1e-5, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size=5, stride=5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24, eps=1e-5, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(24 * 12 * 12, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(-1, 24 * 12 * 12)
        x = self.dense1(x)
        return x


class MainWindow(QtWidgets.QMainWindow, classifier_window.Ui_MainWindow):
    LABEL_2_SIZE = [531, 431]

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.model = Net()
        self.modelPath = "X:\金相研磨图像识别项目\\pythonProject\\Trained Model\\V1.0 model.pth"
        self.model.load_state_dict(torch.load(self.modelPath, map_location=torch.device('cpu')))
        self.image_path = None
        self.result = None
        self.camera = cv2.VideoCapture(0)
        self.camera_timer = QtCore.QTimer(self)
        self.camera_timer.timeout.connect(self.camera_view)
        self.photo_camera_takes = None
        self.camera_mode = False

        self.pushButton.clicked.connect(self.open_image)
        self.pushButton_2.clicked.connect(self.classify_image)
        self.pushButton_3.clicked.connect(self.save_record)
        self.pushButton_4.clicked.connect(self.back)
        self.pushButton_5.clicked.connect(self.open_camera)
        self.pushButton_6.clicked.connect(self.take_picture)

    def setui(self, window):
        self.setupUi(window)
        self.model = Net()
        self.modelPath = "X:\金相研磨图像识别项目\\pythonProject\\Trained Model\\V1.0 model.pth"
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
        if self.camera_mode:
            QtWidgets.QMessageBox.warning(None, '警告', '相机开启的状态下，无法上传本地图片', QtWidgets.QMessageBox.Ok)
        else:
            from PyQt5.QtWidgets import QFileDialog
            dir = QFileDialog()  # 创建文件对话框
            dir.setFileMode(QFileDialog.ExistingFiles)  # 设置多选
            dir.setDirectory('C:/Users/lyf19/Desktop/Temp/金相研磨图像识别项目/image/train/Pass')  # 设置初始路径为C盘
            dir.setNameFilter('图片文件(*.jpg *.png *.bmp *.ico *.gif)')  # 设置只显示图片文件
            if dir.exec_():  # 判断是否选择了文件
                # print(dir.selectedFiles())
                self.image_path = dir.selectedFiles()[0]
                image = QtGui.QPixmap(self.image_path)
                self.label_2.setPixmap(image)
                self.label_2.setScaledContents(True)  # 自适应图片大小

    def classify_image(self):
        if self.camera_mode:
            image_gray = cv2.cvtColor(self.photo_camera_takes, cv2.COLOR_BGR2GRAY)
            image_gray_canny = cv2.Canny(image_gray, 150, 250)
            row_min = np.min(np.nonzero(image_gray_canny)[0])
            row_max = np.max(np.nonzero(image_gray_canny)[0])
            col_min = np.min(np.nonzero(image_gray_canny)[1])
            col_max = np.max(np.nonzero(image_gray_canny)[1])
            cropped_image = self.photo_camera_takes[row_min:row_max, col_min:col_max]
            cropped_image = cv2.resize(cropped_image, (600, 600))
            cropped_image = torch.from_numpy(cropped_image)
            cropped_image = cropped_image[None, :]
            cropped_image = cropped_image.float()
            cropped_image = cropped_image.permute(0, 3, 1, 2)
            output = self.model(cropped_image)
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
        else:
            if self.image_path == None:
                QtWidgets.QMessageBox.warning(None, '警告', '尚未上传图片', QtWidgets.QMessageBox.Ok)
            else:
                image = cv2.imread(self.image_path)
                image_gray = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
                image_gray_canny = cv2.Canny(image_gray, 150, 250)
                row_min = np.min(np.nonzero(image_gray_canny)[0])
                row_max = np.max(np.nonzero(image_gray_canny)[0])
                col_min = np.min(np.nonzero(image_gray_canny)[1])
                col_max = np.max(np.nonzero(image_gray_canny)[1])
                cropped_image = image[row_min:row_max, col_min:col_max]
                cropped_image = cv2.resize(cropped_image, (600, 600))
                cropped_image = torch.from_numpy(cropped_image)
                cropped_image = cropped_image[None, :]
                cropped_image = cropped_image.float()
                cropped_image = cropped_image.permute(0, 3, 1, 2)
                output = self.model(cropped_image)
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
            path_list = self.image_path.split('/')
            image_name = path_list[-1]
            one_img_info = [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), image_name, self.result]
            with open('X:\金相研磨图像识别项目\pythonProject\\results_record\\results.pickle', 'ab') as file:
                pickle.dump(one_img_info, file, 1)
        elif self.image_path != None and self.result == None:
            QtWidgets.QMessageBox.warning(None, '警告', '尚未对此图片进行识别', QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.warning(None, '警告', '尚未上传图片', QtWidgets.QMessageBox.Ok)

    def open_camera(self):
        if not self.camera_timer.isActive():  # 如果相机关着，打开相机
            self.camera_mode = True
            self.camera_timer.start(20)
            self.label_4.setText(
                "<html><head/><body><p><span style=\" font-size:10pt;\">相机状态:已开启</span></p></body></html>")
            self.label_4.textFormat()
        else:  # 如果相机开着，关闭相机
            self.camera_mode = False
            self.camera_timer.stop()
            self.label_4.setText(
                "<html><head/><body><p><span style=\" font-size:10pt;\">相机状态:已关闭</span></p></body></html>")
            self.label_2.setPixmap(QtGui.QPixmap(""))

    def camera_view(self):
        ret, frame = self.camera.read()
        rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Qimage = QtGui.QImage(rgbImage[:], rgbImage.shape[1], rgbImage.shape[0], rgbImage.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)  # pyqt5转换成自己能放的图片格式
        resized_image = QtGui.QPixmap(
            Qimage.scaled(self.label_2.size().width(), self.label_2.size().height(), QtCore.Qt.KeepAspectRatio))
        resized_image = QtGui.QPixmap(Qimage.scaled(self.label_2.size().width(), self.label_2.size().height()))
        self.label_2.setPixmap(resized_image)

    def setCameraImage(self, pixImage):
        self.label_2.setPixmap(pixImage)

    def take_picture(self):
        if self.camera_timer.isActive():
            ret, frame = self.camera.read()
            self.camera_timer.stop()
            self.label_4.setText(
                "<html><head/><body><p><span style=\" font-size:10pt;\">相机状态:已关闭</span></p></body></html>")
            self.photo_camera_takes = frame
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
