from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
import math
import numpy as np
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class QPaintViewImage(QLabel):

    mpsignal = pyqtSignal(str)

    def __init__(self, parent):
        super(QLabel, self).__init__(parent)

        self.setMinimumSize(1, 1)
        self.image_src = None
        
        self.crosscenter = [0, 0]
        
        # Line
        self.setMouseTracking(True)
        self.pen = QPen(QColor(255, 0, 0), 1)
        self.draw_type = None
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.shape_rect = QRectF()
        
        # scale factor
        self.x_factor = 1
        self.y_factor = 1
        
    def rotate_pixmap(self):
        pixmap = self.pixmap()
        transform = QTransform().rotate(90)
        rotated_pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)
        self.setPixmap(rotated_pixmap)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = event.pos()
            self.update()
    
    def mouseMoveEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.end_point = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.end_point = event.pos()
            self.update()
            
    def calculate_length(self):
        if self.start_point.isNull() or self.end_point.isNull():
            return 0.0
        return (((self.end_point.x() - self.start_point.x())*self.width_factor) ** 2 + ((self.end_point.y() - self.start_point.y()) * self.height_factor) ** 2) ** 0.5 
    
    def calculate_box_area(self):
        if self.start_point.isNull() or self.end_point.isNull():
            return 0.0
        return abs(self.end_point.x() - self.start_point.x()) * abs(self.end_point.y() - self.start_point.y()) * self.height_factor * self.width_factor
    
    def calculate_box_h_w(self):
        if self.start_point.isNull() or self.end_point.isNull():
            return 0.0, 0.0
        return abs(self.end_point.x() - self.start_point.x()) * self.width_factor, abs(self.end_point.y() - self.start_point.y()) * self.height_factor
        
    def display_image(self, window=1):
        qformat = QImage.Format_Indexed8
        if len(self.image_src.shape) == 3:  # rows[0], cols[1], channels[2]
            if (self.image_src.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image_src, self.image_src.shape[1], self.image_src.shape[0],
                     self.image_src.strides[0], qformat)
        img = img.rgbSwapped()
        w, h = (639, 599) # size image const
        img_height = self.image_src.shape[0]
        img_width = self.image_src.shape[1]
        
        self.height_factor = img_height / h
        self.width_factor = img_width / w
        
        if window == 1:
            # self.setScaledContents(True)
            self.setPixmap(QPixmap.fromImage(img))#.scaled(w - backlash, h - backlash, Qt.IgnoreAspectRatio))
            self.setAlignment(QtCore.Qt.AlignCenter)
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
            
            
        if self.draw_type is not None and self.end_point is not None:
            painter.setPen(self.pen)
            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)
            # calc 
            line_length = self.calculate_length()
            self.shape_rect = QRectF(self.start_point, self.end_point)
            area = self.calculate_box_area()
            width, height = self.calculate_box_h_w()
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
            
            if self.draw_type == "line":
                if (line_length != 0):
                    painter.drawLine(self.start_point, self.end_point)
                    text = "{:.2f} mm".format(line_length)
                    painter.drawText(self.end_point, text)
            elif self.draw_type == 'box':
                if (area != 0):
                    painter.drawRect(self.shape_rect)
                    text = "Area: {:.2f} mm^2".format(area)
                    text_h = "Height: {:.2f}".format(height)
                    text_w = "Width: {:.2f}".format(width)
                    painter.drawText(self.end_point, text)
                    painter.drawText(x2, y2 + 20, text_h)
                    painter.drawText(x2, y2 + 40, text_w)
            elif self.draw_type == "circle":
                if (area != 0):
                    painter.drawEllipse(self.shape_rect)
                    text = "Area: {:.2f} mm^2".format(area * math.pi / 4)
                    text_h = "Height: {:.2f}".format(height / 2)
                    text_w = "Width: {:.2f}".format(width / 2)
                    painter.drawText(self.end_point, text)
                    painter.drawText(x2, y2 + 20, text_h)
                    painter.drawText(x2, y2 + 40, text_w)
            elif self.draw_type == "angle":
                if (angle != 0):
                    angle_str = str(round(angle, 2)) + "°"
                    painter.drawLine(self.start_point, self.end_point)
                    painter.setFont(QFont("Arial", 10))
                    painter.drawText(x2, y2, angle_str)