# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'icd_list.ui'
##
## Created by: Qt User Interface Compiler version 5.14.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient)
from PySide2.QtWidgets import *


class Ui_icd_list_gui(object):
    def setupUi(self, icd_list_gui):
        if not icd_list_gui.objectName():
            icd_list_gui.setObjectName(u"icd_list_gui")
        icd_list_gui.resize(450, 652)
        self.icd_list = QListWidget(icd_list_gui)
        self.icd_list.setObjectName(u"icd_list")
        self.icd_list.setGeometry(QRect(-5, 50, 451, 601))
        self.label = QLabel(icd_list_gui)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(0, 0, 441, 51))
        font = QFont()
        font.setPointSize(21)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignCenter)

        self.retranslateUi(icd_list_gui)

        QMetaObject.connectSlotsByName(icd_list_gui)
    # setupUi

    def retranslateUi(self, icd_list_gui):
        icd_list_gui.setWindowTitle(QCoreApplication.translate("icd_list_gui", u"Form", None))
        self.label.setText(QCoreApplication.translate("icd_list_gui", u"ICD codes", None))
    # retranslateUi

