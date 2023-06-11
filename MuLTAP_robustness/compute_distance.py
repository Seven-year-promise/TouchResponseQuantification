from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import QDir, Qt, QUrl, QRect, QCoreApplication, QMetaObject
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QAction

import sys
import cv2
import numpy as np
import os

class Ui_mainWindow(QMainWindow):

    def __init__(self):

        super().__init__()
        #mainWindow.resize(800, 800)
        self.video_path = "select the video path"
        self.video_frames = []
        self.frame_number = 15000

        self.distance_file = '_'
        self.position_flag1 = False
        self.position_flag2 = False
        self.negative_flag = False
        self.position_value1 = 0
        self.position_value2 = 0
        self.position_value3 = 0
        self.video_size = 720
        #self.openFile()
        #self.load_video()


        #self.centralWidget = QWidget(mainWindow)
        #self.centralWidget.setObjectName("centralWidget")
        self.keyframe = QLabel(self)
        self.keyframe.setObjectName("key frame")
        self.keyframe.setText("0")
        self.keyframe.setGeometry(QRect(self.video_size +30, 500, 50, 50))

        self.position1 = QLabel(self)
        self.position1.setObjectName("position1")
        self.position1.setText("0")
        self.position1.setGeometry(QRect(self.video_size +30, 60, 100, 50))

        self.position2 = QLabel(self)
        self.position2.setObjectName("position2")
        self.position2.setText("0")
        self.position2.setGeometry(QRect(self.video_size +30, 110, 100, 50))

        self.position3 = QLabel(self)
        self.position3.setObjectName("position3")
        self.position3.setText("0")
        self.position3.setGeometry(QRect(self.video_size +30, 160, 100, 50))

        self.percentage = QLabel(self)
        self.percentage.setObjectName("percentage")
        self.percentage.setText("0")
        self.percentage.setGeometry(QRect(self.video_size +30, 210, 100, 50))

        self.video = QLabel(self)
        self.video.setGeometry(QRect(10, 10, self.video_size + 10, self.video_size + 10))
        self.video.setText("一颗数据小白菜")
        self.video.setObjectName("label")
        self.video.setPixmap(QPixmap("open.png"))
        self.video.mousePressEvent = self.getPos

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(1)
        self.slider.setMaximum(self.frame_number)
        self.slider.setGeometry(10, self.video_size + 30, self.video_size, 30)
        self.slider.valueChanged[int].connect(self.changeValue)

        self.percentage_file_button = QPushButton('load percentage file', self)
        self.percentage_file_button.setGeometry(self.video_size +30, 10, 150, 50)
        self.percentage_file_button.clicked.connect(self.percentage_file_button_click)

        self.load_button = QPushButton('load data', self)
        self.load_button.setGeometry(self.video_size +200, 10, 150, 50)
        self.load_button.clicked.connect(self.load_button_click)

        self.exit_button = QPushButton('EXIT', self)
        self.exit_button.setGeometry(self.video_size + 400, 10, 150, 50)
        self.exit_button.clicked.connect(self.exit_button_click)

        self.pos_button = QPushButton('Positive', self)
        self.pos_button.setGeometry(self.video_size + 200, 60, 150, 50)
        self.pos_button.clicked.connect(self.pos_button_click)

        self.neg_button = QPushButton('Negative', self)
        self.neg_button.setGeometry(self.video_size + 400, 60, 150, 50)
        self.neg_button.clicked.connect(self.neg_button_click)

        self.setMouseTracking(True)
        #self.video_player()

        self.setGeometry(50,50,1200,1200)
        self.setWindowTitle("Compute Distance")
        self.show()

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie",
                        QDir.currentPath()+'/dataset/body')

        if fileName[-3:] == 'avi':
            self.video_path = fileName
            self.load_video()

    def openPercentageFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie",
                        QDir.currentPath())

        if fileName[-3:] == 'txt':
            self.percentage_file = open(fileName, 'w')

    def load_video(self):
        cap = cv2.VideoCapture(self.video_path)
        success, frame = cap.read()
        video_frames = []
        frame_cnt = 0
        while success:
            frame = cv2.resize(frame, (self.video_size,self.video_size))
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap01 = QPixmap.fromImage(qImg)
            video_frames.append(pixmap01)
            frame_cnt += 1
            success, frame = cap.read()
        self.video_frames = video_frames
        self.frame_number = frame_cnt

    def video_player(self):
        self.video.setPixmap(self.video_frames[0])

    def changeValue(self):
        slider_value = self.slider.value()
        txt = str(slider_value)
        self.keyframe.setText(txt)
        if len(self.video_frames) >1 :
            self.video.setPixmap(self.video_frames[slider_value-1])

    def load_button_click(self):
        self.openFile()
        self.video_player()
        self.slider.setMaximum(self.frame_number)

    def percentage_file_button_click(self):
        self.openPercentageFile()

    def exit_button_click(self):
        self.percentage_file.close()
        self.close()

    def pos_button_click(self):
        self.negative_flag = False

    def neg_button_click(self):
        self.negative_flag = True

    def getPos(self, event):
        if event.buttons() == Qt.LeftButton:
            x = event.pos().x()
            y = event.pos().y()
            x_txt = str(x)
            y_txt = str(y)
            if not self.position_flag1:
                self.position_value1 = [x, y]
                self.position1.setText(x_txt + ',' + y_txt)
                self.position_flag1 = not self.position_flag1
            elif not self.position_flag2:
                self.position_value2 = [x, y]
                self.position2.setText(x_txt + ',' + y_txt)
                self.position_flag2 = not self.position_flag2
            else:
                self.position_value3 = [x, y]
                self.position3.setText(x_txt + ',' + y_txt)
                self.position_flag1 = not self.position_flag1
                self.position_flag2 = not self.position_flag2

                x_1, y_1 = self.position_value1
                x_2, y_2 = self.position_value2
                x_3, y_3 = self.position_value3
                distance1 = np.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
                if self.negative_flag:
                    distance1 = -1*distance1
                distance2 = np.sqrt((x_1 - x_3)**2 + (y_1 - y_3)**2)
                percentage = distance1/ distance2
                percentage_txt = str(percentage)
                self.percentage.setText(percentage_txt)
                self.percentage_file.write(percentage_txt+'\n')
                src = self.video_path
                dst = self.video_path[:-4] + 'used.avi'

                # rename() function will
                # rename all the files
                os.rename(src, dst)


    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()

    ui = Ui_mainWindow()

    #ui.setupUi(mainWindow)
    #mainWindow.show()
    sys.exit(app.exec_())
