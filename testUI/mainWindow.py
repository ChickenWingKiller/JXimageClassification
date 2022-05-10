import sys
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QLabel
from PyQt5.QtGui import QPalette, QBrush, QPixmap

import change_window
import load_image

class new_change_window(change_window.Example1):
    def goto(self):
        # print(1)
        ex.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = load_image.Example()
    # ex1 = change_window.Example1()
    ex1 = new_change_window()
    ex1.show()
    sys.exit(app.exec_())