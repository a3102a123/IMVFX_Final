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
        self.AlphaBlending = QtWidgets.QGroupBox(self.centralwidget)
        self.AlphaBlending.setGeometry(QtCore.QRect(850, 580, 221, 71))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.AlphaBlending.setFont(font)
        self.AlphaBlending.setStyleSheet("background-color:white;")
        self.AlphaBlending.setObjectName("AlphaBlending")
        self.Alpha = QtWidgets.QSlider(self.AlphaBlending)
        self.Alpha.setGeometry(QtCore.QRect(10, 40, 160, 22))
        self.Alpha.setMaximum(10)
        self.Alpha.setOrientation(QtCore.Qt.Horizontal)
        self.Alpha.setObjectName("Alpha")
        self.Alpha_text = QtWidgets.QTextEdit(self.AlphaBlending)
        self.Alpha_text.setGeometry(QtCore.QRect(180, 30, 31, 31))
        self.Alpha_text.setObjectName("Alpha_text")
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
        self.AlphaBlending.setTitle(_translate("MainGUI", "Alpha value (blending)"))
        self.Alpha_text.setHtml(_translate("MainGUI", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>"))

