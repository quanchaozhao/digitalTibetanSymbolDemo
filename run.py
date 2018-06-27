# coding:utf-8
# 此文件用于启动该应用程序
# 非特殊情况请勿更改
#

import sys

from PyQt5.QtWidgets import *
from ui.main_window import Main_Window


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Main_Window()
    main_window.show()
    app.exec_()