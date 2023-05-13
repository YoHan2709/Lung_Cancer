from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
import numpy as np
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class QPaintLabel3(QLabel):

    mpsignal = pyqtSignal(str)

    def __init__(self, parent):
        super(QLabel, self).__init__(parent)

        self.setMinimumSize(1, 1)
        self.setMouseTracking(False)
        self.image = None
        self.processedImage = None
        self.imgr, self.imgc = None, None
        self.imgpos_x, self.imgpos_y = None, None
        self.pos_x = 20
        self.pos_y = 20
        self.imgr, self.imgc = None, None
        # Stop when you encounter a list, 
        # the white on the picture is just a cover
        self.pos_xy = []
        
        # The center point of the cross! Each QLabel specifies 
        # a different center point, so that the same paintevent function can be used
        self.crosscenter = [0, 0]
        self.mouseclicked = None
        self.sliceclick = False
        
        # Decide which type of paintEvent to use, general represents general
        self.type = 'general'
        self.slice_loc = [0, 0, 0]
        self.slice_loc_restore = [0, 0, 0]
        self.mousein = False
        
        
        self.setMouseTracking(True)
        self.start_pos = QPoint()
        self.end_pos = QPoint()
        self.line_length = 0.0

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.end_pos = event.pos()
            self.update()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.end_pos = event.pos()
            self.line_length = self.calculate_length()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.end_pos = event.pos()
            self.line_length = self.calculate_length()
            self.update()
            
    def calculate_length(self):
        if self.start_pos.isNull() or self.end_pos.isNull():
            return 0.0
        return (((self.end_pos.x() - self.start_pos.x())*self.width_factor) ** 2 + ((self.end_pos.y() - self.start_pos.y()) * self.height_factor) ** 2) ** 0.5 

    # def leaveEvent(self, event):
    #     self.mousein = False
    #     self.slice_loc = self.slice_loc_restore
    #     self.update()

    def display_image(self, window=1, has_text = True):
        self.imgr, self.imgc = self.processedImage.shape[0:2]
        self.has_text = has_text
        qformat = QImage.Format_Indexed8
        if len(self.processedImage.shape) == 3:  # rows[0], cols[1], channels[2]
            if (self.processedImage.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.processedImage, self.processedImage.shape[1], self.processedImage.shape[0],
                     self.processedImage.strides[0], qformat)
        img = img.rgbSwapped()
        w, h = self.width(), self.height()
        img_height = self.processedImage.shape[0]
        img_width = self.processedImage.shape[1]
        
        self.height_factor = img_height / h
        self.width_factor = img_width / w
        
        if window == 1:
            self.setScaledContents(True)
            backlash = self.lineWidth() * 2
            self.setPixmap(QPixmap.fromImage(img).scaled(w - backlash, h - backlash, Qt.IgnoreAspectRatio))
            self.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        # Use a QFont to set the format of drawText
        loc = QFont()
        loc.setPixelSize(10)
        loc.setBold(True)
        loc.setItalic(True)
        loc.setPointSize(15)
        painter = QPainter(self)
        
        if self.pixmap():
            pixmap = self.pixmap()
            painter.drawPixmap(self.rect(), pixmap)

            if (self.has_text):
                painter.setPen(QPen(Qt.magenta, 10))
                painter.setFont(loc)
                painter.drawText(5, self.height() - 5, 'layer = %3d'
                                % (self.slice_loc[2]))

            if self.type == 'axial':
                # Draw a straight line
                painter.setPen(QPen(Qt.red, 3))
                painter.drawLine(self.crosscenter[0], 0, self.crosscenter[0], self.height())
                # Draw horizontal bars
                painter.setPen(QPen(Qt.cyan, 3))
                painter.drawLine(0, self.crosscenter[1], self.width(), self.crosscenter[1])
                # painting center
                painter.setPen(QPen(Qt.yellow, 3))
                painter.drawPoint(self.crosscenter[0], self.crosscenter[1])

            elif self.type == 'sagittal':
                # 畫直條
                painter.setPen(QPen(Qt.cyan, 3))
                painter.drawLine(self.crosscenter[0], 0, self.crosscenter[0], self.height())
                # 畫橫條
                painter.setPen(QPen(Qt.yellow, 3))
                painter.drawLine(0, self.crosscenter[1], self.width(), self.crosscenter[1])
                # 畫中心
                painter.setPen(QPen(Qt.red, 3))
                painter.drawPoint(self.crosscenter[0], self.crosscenter[1])
                
        if not self.start_pos.isNull() and not self.end_pos.isNull():
            pen = QPen(QColor("red"), 1)
            painter.setPen(pen)
            painter.drawLine(self.start_pos, self.end_pos)
            font = painter.font()
            font.setPointSize(12)
            painter.setFont(font)
            fm = QFontMetrics(font)
            text = "{:.2f} mm".format(self.line_length)
            if (self.line_length != 0):
                painter.drawText(self.end_pos, text)


def linear_convert(img):
    convert_scale = 255.0 / (np.max(img) - np.min(img))
    converted_img = convert_scale*img-(convert_scale*np.min(img))
    return converted_img
