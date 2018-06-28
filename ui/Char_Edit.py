
from __future__ import print_function,division
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ui.auto_ui.char_edit import Ui_Char_Edit_Dialog


class Char_Edit(QDialog,Ui_Char_Edit_Dialog):
    def __init__(self,parent=None,show_img=None,text=None):
        super(QDialog, self).__init__(parent)
        self.setFixedSize(638,529)
        self.setupUi(self)
        self.buttonBox.button(QDialogButtonBox.Ok).setText(u"修改")
        self.buttonBox.button(QDialogButtonBox.Cancel).setText(u"取消")
        self.textEdit.setFontPointSize(36)
        self.textEdit.setLineWrapMode(QTextEdit.NoWrap)
        self.textEdit.setFontFamily("IPAPANNEW")
        # self.parent().leacode保存的是ipapannew不能显示的Unicode编码
        # print(self.parent().leacode)
        if text in self.parent().leacode:
            self.textEdit.setFontFamily("IeaUnicode")
        self.textEdit.setText(text)
        self.textEdit.setAlignment(Qt.AlignCenter)
        if show_img is not None:
            self.show_img = show_img
            figure = plt.figure()
            self.ax = figure.add_axes([0.05, 0.05, 0.9, 0.9])
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.imshow(show_img,cmap='gray')
            self.canvas = FigureCanvas(figure)
            self.canvas.draw()
            self.verticalLayout.addWidget(self.canvas)
            plt.close()