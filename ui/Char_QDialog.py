# coding:utf8
from __future__ import print_function,division
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ui.auto_ui.char_dialog import Ui_Char_Dialog



class Char_QDialog(QDialog,Ui_Char_Dialog):

    # 用于展示显示的图像
    show_img = None
    # 记录切分的直线
    line = None
    # 记录切分的横坐标x
    cut_position = None

    def onclick(self,event):
        if event.button == 1:
            if event.xdata is not None and event.ydata is not None:
                x = event.xdata
                rows,cols = self.show_img.shape
                line = Rectangle((x,0),1,rows,color='red')
                if self.line is not None:
                    self.line.remove()
                self.line = line
                self.ax.add_patch(line)
                self.canvas.draw()
                self.cut_position = x
                plt.close()

    def __init__(self,parent=None,show_img=None):
        # region 初始化窗体中部件和布局
        super(Char_QDialog, self).__init__(parent)
        self.setFixedSize(633,555)
        self.setupUi(self)
        self.buttonBox.button(QDialogButtonBox.Ok).setText(u"切分")
        self.buttonBox.button(QDialogButtonBox.Cancel).setText(u"取消")
        if show_img is not None:
            self.show_img = show_img
            figure = plt.figure()
            self.ax = figure.add_axes([0.05, 0.05, 0.9, 0.9])
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.imshow(show_img,cmap='gray')
            self.canvas = FigureCanvas(figure)
            self.canvas.mpl_connect('button_press_event',self.onclick)
            self.canvas.draw()
            self.verticalLayout.addWidget(self.canvas)
            plt.close()
