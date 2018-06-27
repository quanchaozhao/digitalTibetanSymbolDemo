# coding-utf-8

from PyQt5.QtWidgets import *

from ui.Char_Edit import Char_Edit


class QLabelExt(QLabel):

    def __init__(self,img,parent=None):
        self.img = img
        super(QLabelExt, self).__init__(parent)

    def mousePressEvent(self,event):
        char_edit_dialog = Char_Edit(self, self.img, str(self.text()))
        ret_message = char_edit_dialog.exec_()
        if ret_message == 1:
            modified_str = char_edit_dialog.textEdit.toPlainText()
            if len(modified_str) > 0:
                self.setText(modified_str)
        pass