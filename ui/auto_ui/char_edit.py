# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'char_edit.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Char_Edit_Dialog(object):
    def setupUi(self, Char_Edit_Dialog):
        Char_Edit_Dialog.setObjectName("Char_Edit_Dialog")
        Char_Edit_Dialog.setEnabled(True)
        Char_Edit_Dialog.resize(638, 529)
        Char_Edit_Dialog.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.buttonBox = QtWidgets.QDialogButtonBox(Char_Edit_Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(20, 480, 591, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayoutWidget = QtWidgets.QWidget(Char_Edit_Dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 20, 601, 371))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.textEdit = QtWidgets.QTextEdit(Char_Edit_Dialog)
        self.textEdit.setGeometry(QtCore.QRect(20, 400, 601, 61))
        self.textEdit.setObjectName("textEdit")

        self.retranslateUi(Char_Edit_Dialog)
        self.buttonBox.accepted.connect(Char_Edit_Dialog.accept)
        self.buttonBox.rejected.connect(Char_Edit_Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Char_Edit_Dialog)

    def retranslateUi(self, Char_Edit_Dialog):
        _translate = QtCore.QCoreApplication.translate
        Char_Edit_Dialog.setWindowTitle(_translate("Char_Edit_Dialog", "Dialog"))

