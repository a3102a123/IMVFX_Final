# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI/MainGUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainGUI(object):
    def setupUi(self, MainGUI):
        MainGUI.setObjectName("MainGUI")
        MainGUI.resize(1280, 720)
        self.centralwidget = QtWidgets.QWidget(MainGUI)
        self.centralwidget.setObjectName("centralwidget")
        self.ImageDisplayer = QtWidgets.QLabel(self.centralwidget)
        self.ImageDisplayer.setGeometry(QtCore.QRect(100, 20, 500, 480))
        self.ImageDisplayer.setStyleSheet("background-color:white;")
        self.ImageDisplayer.setText("")
        self.ImageDisplayer.setScaledContents(False)
        self.ImageDisplayer.setObjectName("ImageDisplayer")
        self.ImageDisplayer_2 = QtWidgets.QLabel(self.centralwidget)
        self.ImageDisplayer_2.setGeometry(QtCore.QRect(700, 20, 500, 480))
        self.ImageDisplayer_2.setStyleSheet("background-color:white;")
        self.ImageDisplayer_2.setText("")
        self.ImageDisplayer_2.setObjectName("ImageDisplayer_2")
        self.cutButton = QtWidgets.QPushButton(self.centralwidget)
        self.cutButton.setGeometry(QtCore.QRect(920, 530, 101, 31))
        self.cutButton.setObjectName("cutButton")
        MainGUI.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainGUI)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 21))
        self.menubar.setObjectName("menubar")
        MainGUI.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainGUI)
        self.statusbar.setObjectName("statusbar")
        MainGUI.setStatusBar(self.statusbar)

        self.retranslateUi(MainGUI)
        QtCore.QMetaObject.connectSlotsByName(MainGUI)

    def retranslateUi(self, MainGUI):
        _translate = QtCore.QCoreApplication.translate
        MainGUI.setWindowTitle(_translate("MainGUI", "MainWindow"))
        self.cutButton.setText(_translate("MainGUI", "Cut"))

