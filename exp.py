# coding:utf-8
# 引用文件



from PyQt5.QtWidgets import *
from ui.auto_ui.untitledeeee import Ui_Dialog
import sys
# 定义类 继承 两个父类 一个是窗体的类 一个是UI类
class TestDialog(QDialog,Ui_Dialog):
    def __init__(self,parent=None):
        super(TestDialog, self).__init__(parent)
        #使用 Ui_Dialog 中的 setupUi方法 生成UI样式
        self.setupUi(self)
if __name__=='__main__':
    app = QApplication(sys.argv)
    dialog = TestDialog()
    dialog.show()
    app.exec_()