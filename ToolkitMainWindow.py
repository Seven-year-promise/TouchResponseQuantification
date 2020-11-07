from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import pyqtgraph as pg

from QtFunctions.Lines import QHLine, QVLine
from ImageProcessing import ImageProcessor

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
        self.video_cropped_frames = []
        self.video_sift_frames =[]
        self.video_binary_frames = []
        self.frame_number = 10000

        self.distance_file = '_'
        self.position_flag1 = False
        self.position_flag2 = False
        self.negative_flag = False
        self.cropped_flag = False
        self.binary_flag = False
        self.sift_flag = False

        self.position_value1 = 0
        self.position_value2 = 0
        self.position_value3 = 0
        self.video_size = 480
        self.well_infos = []

        self.file_paths = []
        self.current_path = ""

        self.bin_threshold = 0

        self.im_processor = ImageProcessor()

        self.msgbox = QMessageBox()
        self.msgbox.setIcon(QMessageBox.Information)

        #self.resize(1280, 720)
        #self.openFile()
        #self.load_video()

        self.main_widget = QWidget()  # set main controls
        self.main_layout = QGridLayout()  # set main layout
        self.main_widget.setLayout(self.main_layout)
        self.main_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)



        self.display_widget = QWidget()  # set main controls
        self.display_layout = QGridLayout()  # set main layout
        self.display_widget.setLayout(self.display_layout)
        self.display_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.display_widget, 0, 0)

        self.quantification_widget = QWidget()  # set main controls
        self.quantification_layout = QGridLayout()  # set main layout
        self.quantification_widget.setLayout(self.quantification_layout)
        self.quantification_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.quantification_widget, 1, 0)

        self.init_display_block()
        self.init_quantification_block()

        self.setCentralWidget(self.main_widget)

        self.setMouseTracking(True)
        #self.video_player()
        self.setGeometry(50, 50, 1280, 960)
        self.setWindowTitle("Touch Response Quantification")

        self.show()

    def init_display_block(self):
        self.video = QLabel()
        self.video.setText("一颗数据小白菜")
        self.video.setObjectName("label")
        self.video.setPixmap(QPixmap("0.jpg"))
        self.video.mousePressEvent = self.getPos
        #self.video.setStyleSheet('border:0px solid #cccccc;')
        #self.video.resize(self.video_size, self.video_size)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(self.frame_number)
        #self.slider.setStyleSheet('border:0px solid #cccccc;')
        self.slider.valueChanged[int].connect(self.sliderchangeValue)
        self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.keyframe = QLabel("0")
        self.keyframe.setObjectName("key frame")
        self.keyframe.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.display_layout.addWidget(QHLine(), 0, 0, 1, 6)
        self.display_layout.addWidget(QVLine(), 0, 0, 4, 1)
        self.display_layout.addWidget(self.video, 1, 1, 1, 1)
        self.display_layout.addWidget(self.keyframe, 2, 2, 1, 1)
        self.display_layout.addWidget(self.slider, 2, 1, 1, 1)

        self.im_processing_widget = QWidget()  # set main controls
        self.im_processing_layout = QGridLayout()  # set main layout
        self.im_processing_widget.setLayout(self.im_processing_layout)
        self.im_processing_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_layout.addWidget(self.im_processing_widget, 1, 3)
        self.init_im_processing_block()

        self.file_widget = QWidget()  # set main controls
        self.file_layout = QGridLayout()  # set main layout
        self.file_widget.setLayout(self.file_layout)
        self.file_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_layout.addWidget(self.file_widget, 1, 4)
        self.init_file_block()

        self.display_layout.addWidget(QHLine(), 3, 0, 1, 6)
        self.display_layout.addWidget(QVLine(), 0, 5, 4, 1)

    def init_im_processing_block(self):
        self.bin_threshold_label = QLabel("Binary Threshold")
        self.bin_threshold_label.setObjectName("bin_threshold_label")
        self.bin_threshold_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.bin_threshold_text = QLineEdit("0")
        self.bin_threshold_text.setObjectName("bin_threshold_text")
        self.bin_threshold_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.bin_threshold_text.textChanged.connect(self.bin_threshold_text_changed)

        self.position2 = QLabel("0")
        self.position2.setObjectName("position2")
        self.position2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.position3 = QLabel("0")
        self.position3.setObjectName("position3")
        self.position3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.percentage = QLabel("0")
        self.percentage.setObjectName("percentage")
        self.percentage.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.load_button = QPushButton('Load Files')
        self.load_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.load_button.clicked.connect(self.load_button_click)

        self.binarization_button = QPushButton('binarization')
        self.binarization_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.binarization_button.clicked.connect(self.binarization_button_click)

        self.crop_button = QPushButton('Crop Image')
        self.crop_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.crop_button.clicked.connect(self.crop_button_click)

        self.sift_button = QPushButton('SIFT Extraction')
        self.sift_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sift_button.clicked.connect(self.sift_button_click)

        self.exit_button = QPushButton('EXIT')
        self.exit_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.exit_button.clicked.connect(self.exit_button_click)

        self.pos_button = QPushButton('Positive')
        self.pos_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pos_button.clicked.connect(self.pos_button_click)

        self.neg_button = QPushButton('Negative')
        self.neg_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.neg_button.clicked.connect(self.neg_button_click)

        self.im_processing_layout.addWidget(self.load_button, 0, 0)
        self.im_processing_layout.addWidget(self.bin_threshold_label, 1, 0)
        self.im_processing_layout.addWidget(self.bin_threshold_text, 1, 1)
        self.im_processing_layout.addWidget(self.position2, 2, 1)
        self.im_processing_layout.addWidget(self.position3, 3, 1)
        self.im_processing_layout.addWidget(self.percentage, 4, 1)

        self.im_processing_layout.addWidget(self.crop_button, 5, 1)
        self.im_processing_layout.addWidget(self.binarization_button, 6, 1)
        self.im_processing_layout.addWidget(self.sift_button, 7, 1)
        self.im_processing_layout.addWidget(self.exit_button, 8, 1)
        self.im_processing_layout.addWidget(self.pos_button, 9, 1)
        self.im_processing_layout.addWidget(self.neg_button, 10, 1)

    def init_file_block(self):
        self.file_paths_label = QLabel("File Paths")
        self.file_paths_label.setObjectName("file_paths_label")

        self.files_list = QListWidget(self)
        self.files_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.files_list.itemDoubleClicked.connect(self.fileOnClicked)

        self.file_layout.addWidget(self.file_paths_label, 0, 0, 1, 1)
        self.file_layout.addWidget(self.files_list, 1, 0, 11, 1)

    def init_quantification_block(self):
        self.quantification_layout.addWidget(QHLine(), 0, 0, 1, 9)
        self.quantification_layout.addWidget(QVLine(), 0, 0, 2, 1)
        # =============================widget for latency time========================
        self.latency_time_widget = QWidget()  # set main controls
        self.latency_time_layout = QGridLayout()  # set main layout
        self.latency_time_widget.setLayout(self.latency_time_layout)
        self.latency_time_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.quantification_layout.addWidget(self.latency_time_widget, 1, 1)
        self.quantification_layout.addWidget(QVLine(), 0, 2, 2, 1)

        self.latency_time_label = QLabel("Latency Time: Figure")
        self.latency_time_label.setObjectName("latency_time_label")
        self.latency_time_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.latency_time_graph = pg.PlotWidget()
        self.latency_time_graph.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.latency_time_list = QListWidget(self)
        self.latency_time_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.latency_time_layout.addWidget(self.latency_time_label, 0, 0)
        self.latency_time_layout.addWidget(self.latency_time_graph, 1, 0)
        self.latency_time_layout.addWidget(self.latency_time_list, 1, 1)

        # =============================widget for cshape radius========================
        self.cshape_radius_widget = QWidget()  # set main controls
        self.cshape_radius_layout = QGridLayout()  # set main layout
        self.cshape_radius_widget.setLayout(self.cshape_radius_layout)
        self.cshape_radius_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.quantification_layout.addWidget(self.cshape_radius_widget, 1, 3)
        self.quantification_layout.addWidget(QVLine(), 0, 4, 2, 1)

        self.cshape_radius_label = QLabel("CShape Radius: Figure")
        self.cshape_radius_label.setObjectName("cshape_radius_label")
        self.cshape_radius_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cshape_radius_graph = pg.PlotWidget()
        self.cshape_radius_graph.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cshape_radius_list = QListWidget(self)
        self.cshape_radius_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cshape_radius_layout.addWidget(self.cshape_radius_label, 0, 0)
        self.cshape_radius_layout.addWidget(self.cshape_radius_graph, 1, 0)
        self.cshape_radius_layout.addWidget(self.cshape_radius_list, 1, 1)

        # =============================widget for response time========================
        self.response_time_widget = QWidget()  # set main controls
        self.response_time_layout = QGridLayout()  # set main layout
        self.response_time_widget.setLayout(self.response_time_layout)
        self.response_time_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.quantification_layout.addWidget(self.response_time_widget, 1, 5)
        self.quantification_layout.addWidget(QVLine(), 0, 6, 2, 1)

        self.response_time_label = QLabel("Response Time: Figure")
        self.response_time_label.setObjectName("response_time_label")
        self.response_time_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.response_time_graph = pg.PlotWidget()
        self.response_time_graph.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.response_time_list = QListWidget(self)
        self.response_time_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.response_time_layout.addWidget(self.response_time_label, 0, 0)
        self.response_time_layout.addWidget(self.response_time_graph, 1, 0)
        self.response_time_layout.addWidget(self.response_time_list, 1, 1)

        # =============================widget for moving distance========================
        self.moving_distance_widget = QWidget()  # set main controls
        self.moving_distance_layout = QGridLayout()  # set main layout
        self.moving_distance_widget.setLayout(self.moving_distance_layout)
        self.moving_distance_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.quantification_layout.addWidget(self.moving_distance_widget, 1, 7)
        self.quantification_layout.addWidget(QVLine(), 0, 8, 2, 1)

        self.moving_distance_label = QLabel("Moving Distance: Figure")
        self.moving_distance_label.setObjectName("moving_distance_label")
        self.moving_distance_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.moving_distance_graph = pg.PlotWidget()
        self.moving_distance_graph.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.moving_distance_list = QListWidget(self)
        self.moving_distance_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.moving_distance_layout.addWidget(self.moving_distance_label, 0, 0)
        self.moving_distance_layout.addWidget(self.moving_distance_graph, 1, 0)
        self.moving_distance_layout.addWidget(self.moving_distance_list, 1, 1)

        self.quantification_layout.addWidget(QHLine(), 2, 0, 1, 9)

    def loadfiles(self):
        self.current_path = QFileDialog.getExistingDirectory(self, "Open Movie", QDir.currentPath())

        self.path_files = os.listdir(self.current_path)
        print(self.path_files)

        self.files_list.clear()
        self.file_paths_label.setText("File Paths: " + self.current_path)

        self.files_list.addItems(self.path_files)

    def openfiles(self):

        fileName = self.current_path + "/" + self.files_list.currentItem().text()
        print(fileName)
        if fileName[-3:] == 'avi':
            self.video_path = fileName
            self.load_video()

    def fileOnClicked(self):
        self.openfiles()
        self.video_player()
        self.slider.setMaximum(self.frame_number)

    def openPercentageFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie",
                        QDir.currentPath())

        if fileName[-3:] == 'txt':
            self.percentage_file = open(fileName, 'w')

    def im2qImg(self, im):
        height, width, channel = im.shape
        bytesPerLine = 3 * width
        qImg = QImage(im.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap01 = QPixmap.fromImage(qImg)

        return pixmap01

    def load_video(self):
        cap = cv2.VideoCapture(self.video_path)
        success, frame = cap.read()
        video_frames = []
        frame_cnt = 0
        while success:
            frame = cv2.resize(frame, (self.video_size,self.video_size))

            video_frames.append(frame)
            frame_cnt += 1
            success, frame = cap.read()
        self.video_frames = video_frames
        self.frame_number = frame_cnt

    def video_player(self):
        self.video.setPixmap(self.im2qImg(self.video_frames[0]))

    def sliderchangeValue(self):
        slider_value = self.slider.value()
        txt = str(slider_value)
        self.keyframe.setText(txt)
        if len(self.video_frames) >1 :
            if self.cropped_flag:
                frame = self.video_cropped_frames[slider_value - 1]
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                self.video.setPixmap(self.im2qImg(frame_color))
            elif self.sift_flag:
                frame = self.video_sift_frames[slider_value - 1]
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                self.video.setPixmap(self.im2qImg(frame_color))
            elif self.binary_flag:
                frame = self.video_binary_frames[slider_value - 1]
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                self.video.setPixmap(self.im2qImg(frame_color))
            else:
                self.video.setPixmap(self.im2qImg(self.video_frames[slider_value-1]))

    def load_button_click(self):
        self.loadfiles()

        self.msgbox.setText("load files finished")
        #self.msgbox.setInformativeText("This is additional information")
        #self.msgbox.setWindowTitle("MessageBox demo")
        #self.msgbox.setDetailedText("The details are as follows:")

        self.msgbox.exec()

    def binarization_button_click(self):
        # TO DO
        i = 0
        self.video_binary_frames.clear()
        for frame in self.video_cropped_frames:
            frame_feature, _, _ = self.im_processor.feature_extraction(frame, method = "LRB", well_infos=self.well_infos)
            self.video_binary_frames.append(frame_feature)
            i += 1
            if i>5:
                break
        self.cropped_flag = False
        self.sift_flag = False
        self.binary_flag = True
        self.msgbox.setText("binarization finished")
        # self.msgbox.setInformativeText("This is additional information")
        # self.msgbox.setWindowTitle("MessageBox demo")
        # self.msgbox.setDetailedText("The details are as follows:")

        self.msgbox.exec()

    def crop_button_click(self):
        self.video_cropped_frames.clear()
        first_frame = self.video_frames[0]
        first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        success, (well_centerx, well_centery, well_radius) = self.im_processor.well_detection(first_frame_gray)
        self.well_infos = (well_centerx, well_centery, well_radius)
        mask = np.zeros(first_frame_gray.shape[:2], dtype="uint8")
        cv2.circle(mask, (well_centerx, well_centery), well_radius, 255, -1)
        mask2 = np.ones(first_frame_gray.shape[:2], dtype="uint8") * 255
        cv2.circle(mask2, (well_centerx, well_centery), well_radius, 0, -1)

        for frame in self.video_frames:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_masked = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)
            gray_masked += mask2
            self.video_cropped_frames.append(gray_masked)

        self.cropped_flag = True
        self.msgbox.setText("image crop finished")
        # self.msgbox.setInformativeText("This is additional information")
        # self.msgbox.setWindowTitle("MessageBox demo")
        # self.msgbox.setDetailedText("The details are as follows:")

        self.msgbox.exec()

    def sift_button_click(self):
        self.video_sift_frames.clear()
        i = 0
        for frame in self.video_cropped_frames:
            if i < 2:
                frame_feature, _, _ = self.im_processor.feature_extraction(frame)
                cv2.imshow("this", frame_feature)
                cv2.waitKey(0)
                self.video_sift_frames.append(frame_feature)
            else:
                self.video_sift_frames.append(frame)
            i+=1

        self.cropped_flag = False
        self.sift_flag = True
        self.msgbox.setText("SIFT detection finished")
        # self.msgbox.setInformativeText("This is additional information")
        # self.msgbox.setWindowTitle("MessageBox demo")
        # self.msgbox.setDetailedText("The details are as follows:")

        self.msgbox.exec()

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

    def bin_threshold_text_changed(self):
        self.bin_threshold = int(self.bin_threshold_text.text())