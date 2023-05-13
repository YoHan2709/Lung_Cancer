import sys
from PyQt5.QtWidgets import QDialog, QGraphicsScene, QApplication, QButtonGroup
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon
from .matinqt import CFigureCanvas
from .preprocess import resample, segment_lung_mask
from skimage import measure

class C3dView(QDialog):

    def __init__(self):
        super().__init__()
        loadUi('vol_view_module.ui', self)
        self.setWindowTitle('3D View')
        self.setWindowIcon(QIcon('./resources/Logo.png'))
        self.image = None
        self.setMouseTracking(False)
        self.imgs = None
        self.threshold = 0
        self.step = 3
        self.alpha = 0.3
        self.thresholdEdit.setText(str(self.threshold))
        self.stepEdit.setText(str(self.step))
        self.alphaEdit.setText(str(self.alpha))
        self.refreshButton.clicked.connect(self.refresh_clicked)
        self.graphicsView.setMinimumSize(2, 2)
        self.draw_type = 'Lung Core'
        
        self.buttonGroup = QButtonGroup()
        self.buttonGroup.addButton(self.bone_button)
        self.buttonGroup.addButton(self.lungfull_button)
        self.buttonGroup.addButton(self.lungonly_button)
        self.buttonGroup.addButton(self.pulse_button)
        # Connect button group to function
        self.buttonGroup.buttonClicked.connect(self.set_mode_chose)
        
    def set_mode_chose(self, radioButton):
        selection = radioButton.text()
        if selection == 'View Lung Only':
            self.draw_type = 'Lung Only'
        elif selection == 'View Bone':
            self.draw_type = 'Bone'
        elif selection == 'View Lung Full':
            self.draw_type = 'Lung Core'
        elif selection == 'View Pulse':
            self.draw_type = 'Pulse'
        self.vol_show(self.draw_type)

    def refresh_clicked(self):
        self.threshold = int(self.thresholdEdit.text())
        self.step = int(self.stepEdit.text())
        self.alpha = float(self.alphaEdit.text())
        self.vol_show(self.draw_type)

    def vol_show(self, type = 'Lung Core'):
        print("Shape before resampling\t", self.imgs.shape)
        imgs_after_resamp, spacing = self.imgs, [1, 1, 1]
        print("Shape after resampling\t", imgs_after_resamp.shape)
        
        if type == 'Lung Core':
            self.threshold = 0
            segmented_lungs = segment_lung_mask(self.imgs, False)
            imgs_after_resamp = segmented_lungs
        elif type == 'Lung Only':
            self.threshold = 0
            segmented_lungs_fill = segment_lung_mask(self.imgs, True)
            imgs_after_resamp = segmented_lungs_fill
        elif type == 'Pulse':
            self.threshold = 0
            segmented_lungs = segment_lung_mask(self.imgs, False)
            segmented_lungs_fill = segment_lung_mask(self.imgs, True)
            imgs_after_resamp = segmented_lungs_fill - segmented_lungs
        elif type == 'Bone':
            self.threshold = 400
        
        v, f = self.make_mesh(imgs_after_resamp, threshold=self.threshold, step_size=self.step)

        dr = CFigureCanvas()
        dr.plt_3d(v, f, alpha=self.alpha)
        graphicscene = QGraphicsScene()
        graphicscene.addWidget(dr)
        self.graphicsView.setScene(graphicscene)
        self.graphicsView.show()
        
    def make_mesh(self,img, threshold, step_size):
        p = img.transpose(2,1,0)
        verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size)
        return verts, faces


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = C3dView()
    icon = QIcon('./resources/Logo.png')
    ex.setWindowIcon(icon)
    ex.show()
    sys.exit(app.exec_())
