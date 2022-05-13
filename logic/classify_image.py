from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import pickle
import datetime
from UI import classifier_window

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
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.model = Net()
        self.modelPath = "../Trained Model/net89.pth"
        self.modelPath = "X:\金相研磨图像识别项目\\pythonProject\\Trained Model\\net89.pth"
        self.model.load_state_dict(torch.load(self.modelPath, map_location=torch.device('cpu')))
        self.image_path = None
        self.result = None

        self.pushButton.clicked.connect(self.open_image)
        self.pushButton_2.clicked.connect(self.classify_image)
        self.pushButton_3.clicked.connect(self.save_record)
        self.pushButton_4.clicked.connect(self.back)
        # self.setui(self)

    # def setui(self, window):
    #     # self.setupUi(window)
    #     pass
    def open_image(self):
        # dir = QtWidgets.QFileDialog() #创建文件对话框
        # dir.setDirectory('X:/') #设置初始路径为X盘
        # print(dir)
        print(1)
        from PyQt5.QtWidgets import QFileDialog
        dir = QFileDialog()  # 创建文件对话框
        dir.setFileMode(QFileDialog.ExistingFiles)  # 设置多选
        # dir.setDirectory('X:\\')  # 设置初始路径为C盘
        dir.setDirectory('C:/Users/lyf19/Desktop/Temp/金相研磨图像识别项目/image/train/Pass')  # 设置初始路径为C盘
        dir.setNameFilter('图片文件(*.jpg *.png *.bmp *.ico *.gif)')  # 设置只显示图片文件
        if dir.exec_():  # 判断是否选择了文件
            # print(dir.selectedFiles())
            self.image_path = dir.selectedFiles()[0]
            image = QtGui.QPixmap(self.image_path)
            self.label_2.setPixmap(image)
            self.label_2.setScaledContents(True)  # 自适应图片大小

    def classify_image(self):
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
        else:
            self.result = "不合格"
            self.label_3.setText("不合格")

    def save_record(self):
        path_list = self.image_path.split('/')
        image_name = path_list[-3] + '/' + path_list[-2] + '/' + path_list[-1]
        one_img_info = [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), image_name, self.result]
        with open('../results_record/results.pickle', 'ab') as file:
            pickle.dump(one_img_info, file, 1)

    def back(self):
        pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())