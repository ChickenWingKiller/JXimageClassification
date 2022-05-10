import sys
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QLabel
from PyQt5.QtGui import QPalette, QBrush, QPixmap


class Example1(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()  # 界面绘制交给InitUi方法

    def initUI(self):
        self.button = QPushButton(self)
        self.button.setGeometry(100,100,300,100)
        self.button.clicked.connect(self.goto)

        # 设置窗口的位置和大小
        self.setGeometry(300, 300, 800, 800)
        # 设置窗口的标题
        self.setWindowTitle('Example')

    def goto(self):
        pass

if __name__ == '__main__':
    # 创建应用程序和对象
    app = QApplication(sys.argv)
    ex = Example1()
    ex.show()
    sys.exit(app.exec_())