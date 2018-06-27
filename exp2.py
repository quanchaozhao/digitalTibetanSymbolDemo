# coding:utf-8
# 引用文件
from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import *
from ui.auto_ui.untitledeeee import Ui_Dialog
import sys
# 定义类 继承 两个父类 一个是窗体的类 一个是UI类
class MyDialog(QDialog):
    def __init__(self,parent=None):
        super(MyDialog, self).__init__(parent)
        #使用 Ui_Dialog 中的 setupUi方法 生成UI样式
        layout1 = QVBoxLayout()
        # layout1.setSizeConstraint(QLayout.SetNoConstraint)
        for i in range(5):
            qframe = QFrame(self)
            qframe.setFixedSize(500,100)
            for j in range(5):
                label = QLabel(qframe)
                label.setText("%d,%d" % (i,j))
                label.setGeometry(QRect(j*20,7,20,50))



            btn = QPushButton(str(i))
            layout1.addWidget(btn)
            layout1.addWidget(qframe)

        self.setLayout(layout1)



if __name__=='__main__':
    app = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    app.exec_()