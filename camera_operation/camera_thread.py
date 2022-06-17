import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from logic import classify_image


class Camera_thread(QtCore.QThread):
    changePixmap = QtCore.pyqtSignal(QtGui.QPixmap)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QtGui.QImage(rgbImage[:], rgbImage.shape[1], rgbImage.shape[0], rgbImage.shape[1] * 3,
                                     QtGui.QImage.Format_RGB888)  # pyqt5转换成自己能放的图片格式
                # resized_image = image.scaled(640, 480, QtCore.Qt.KeepAspectRatio)
                resized_image = QtGui.QPixmap(image).scaled(classify_image.MainWindow.LABEL_2_SIZE[0],
                                                            classify_image.MainWindow.LABEL_2_SIZE[1])  # 设置图片大小
                # resized_image = QtGui.QPixmap(image).scaled(431, 331, QtCore.Qt.KeepAspectRatio)  # 设置图片大小
                self.changePixmap.emit(resized_image)
