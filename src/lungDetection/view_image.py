import sys
from PyQt5.QtWidgets import QDialog, QGraphicsScene, QApplication, QButtonGroup, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon
from .matinqt import CFigureCanvas
from .preprocess import resample, segment_lung_mask
from skimage import measure
import cv2

class Image_View(QDialog):

    def __init__(self):
        super().__init__()
        loadUi('view_image.ui', self)
        self.setWindowTitle('Image View')
        self.setWindowIcon(QIcon('./resources/Logo.png'))
        self.setMouseTracking(False)
        
        # In this way, the "activated" Item can be 
        # converted into str and passed to the function of connect
        # (you can also use int and the like, which will be enum)
        self.colormapBox.activated[str].connect(self.colormap_choice)
        self.colormapDict = {'GRAY': None,
                             'AUTUMN': cv2.COLORMAP_AUTUMN,
                             'BONE': cv2.COLORMAP_BONE,
                             'COOL': cv2.COLORMAP_COOL,
                             'HOT': cv2.COLORMAP_HOT,
                             'HSV': cv2.COLORMAP_HSV,
                             'JET': cv2.COLORMAP_JET,
                             'OCEAN': cv2.COLORMAP_OCEAN,
                             'PINK': cv2.COLORMAP_PINK,
                             'RAINBOW': cv2.COLORMAP_RAINBOW,
                             'SPRING': cv2.COLORMAP_SPRING,
                             'SUMMER': cv2.COLORMAP_SUMMER,
                             'WINTER': cv2.COLORMAP_WINTER
                             }
        
        # button
        self.buttonGroup = QButtonGroup()
        self.buttonGroup.addButton(self.length_button)
        self.buttonGroup.addButton(self.box_button)
        self.buttonGroup.addButton(self.circle_button)
        self.buttonGroup.addButton(self.corner_button)

        # Connect button group to function
        self.buttonGroup.buttonClicked.connect(self.set_draw_type)
        
        self.rotate_button.clicked.connect(self.set_rotate_pixmap)
        
        self.save_button.clicked.connect(self.save_pixmap)
        
        self.cancel_button.clicked.connect(self.reject)
        
    def save_pixmap(self):
        # Prompt the user to select a file name and location
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self,"Save Pixmap", "","Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            # Save the pixmap to the selected file location
            self.image_label.pixmap().save(file_name)
            
    def set_rotate_pixmap(self):
        self.image_label.rotate_pixmap()
    
    def set_draw_type(self, radioButton):
        # Get text of selected radio button
        selection = radioButton.text()
        if selection == 'Length':
            self.image_label.draw_type = 'line'
        elif selection == 'Box':
            self.image_label.draw_type = 'box'
        elif selection == 'Circle':
            self.image_label.draw_type = 'circle'
        elif selection == 'Corner':
            self.image_label.draw_type = 'angle'
    
    def set_image(self, image):
        self.image_label.image_src = image
        self.image_label.display_image(1)
        

    def colormap_choice(self, text):
        self.colormap = self.colormapDict[text]
        self.set_image(cv2.applyColorMap(self.image_label.image_src, self.colormap))
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Image_View()
    icon = QIcon('./resources/Logo.png')
    ex.setWindowIcon(icon)
    ex.show()
    sys.exit(app.exec_())
