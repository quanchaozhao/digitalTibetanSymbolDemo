# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'char_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Char_Dialog(object):
    def setupUi(self, Char_Dialog):
        Char_Dialog.setObjectName("Char_Dialog")
        Char_Dialog.setEnabled(True)
        Char_Dialog.resize(633, 555)
        Char_Dialog.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.buttonBox = QtWidgets.QDialogButtonBox(Char_Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(20, 510, 591, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayoutWidget = QtWidgets.QWidget(Char_Dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 20, 591, 481))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.retranslateUi(Char_Dialog)
        self.buttonBox.accepted.connect(Char_Dialog.accept)
        self.buttonBox.rejected.connect(Char_Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Char_Dialog)

    def retranslateUi(self, Char_Dialog):
        _translate = QtCore.QCoreApplication.translate
        Char_Dialog.setWindowTitle(_translate("Char_Dialog", "Dialog"))

