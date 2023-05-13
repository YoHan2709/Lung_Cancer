import sys
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon
import os
import cv2
import numpy as np
from .preprocess import load_scan, get_pixels_hu, load_dcm_info
from .preprocess import savenpy, resample
from .run_model import run_classifier, run_detector, take_position_nodule_proposal, take_info_proposal_nodule, take_pos_bbox
from .vol_view_module import C3dView
from .view_image import Image_View

class Lung_Detection_Module(QDialog):
    def __init__(self):
        super().__init__()
        path = os.getcwd()
        os.chdir(path + '/lungDetection')
        self.setWindowIcon(QIcon('./resources/Logo.png'))
        self.directory = os.getcwd()
        loadUi('lung_detection_module.ui', self)
        self.setWindowTitle('Lung Detection')
        
        self.image = None
        self.voxel = None # original image
        self.processedvoxel = None
        self.pre_image = None # preprocess image
        self.voxel_pre = None 
        self.nodule_image = None
        self.nodule_prob = None
        self.prob_patient = None
        
        self.dicomButton.clicked.connect(self.dicom_clicked)
        
        self.original_vSlider.valueChanged.connect(self.updateimg_original)
        
        self.preprocess_vSlider.valueChanged.connect(self.updateimg_preprocessing)
        
        self.volButton.clicked.connect(self.open_3dview)
        
        h = QSpacerItem(10, 10, QSizePolicy.Fixed, QSizePolicy.Fixed)
        v = QSpacerItem(10, 10, QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.axial_vBox.setSpacing(0) # original
        self.axial_vBox.insertSpacerItem(0, v)
        self.axial_vBox.insertSpacerItem(2, v)
        self.axial_hBox.setSpacing(0)
        self.axial_hBox.insertSpacerItem(0, h)
        self.axial_hBox.insertSpacerItem(2, h)
        self.saggital_vBox.setSpacing(0) # preprocess
        self.saggital_vBox.insertSpacerItem(0, v)
        self.saggital_vBox.insertSpacerItem(2, v)
        self.saggital_hBox.setSpacing(0)
        self.saggital_hBox.insertSpacerItem(0, h)
        self.saggital_hBox.insertSpacerItem(2, h)
        
        self.colormap = None
        
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
        self.colormap_hBox.insertStretch(2)
        self.colormap_hBox.insertSpacerItem(0, QSpacerItem(30, 0, QSizePolicy.Fixed,  QSizePolicy.Fixed))
        self.dcmInfo = None
        
        self.viewOriginalButton.clicked.connect(lambda: self.open_view_image(self.viewOriginalButton))
        self.viewProcessingButton.clicked.connect(lambda: self.open_view_image(self.viewProcessingButton))
        
        # nodule button
        self.view_nodule_1.clicked.connect(lambda: self.open_view_image(self.view_nodule_1))
        self.view_nodule_2.clicked.connect(lambda: self.open_view_image(self.view_nodule_2))
        self.view_nodule_3.clicked.connect(lambda: self.open_view_image(self.view_nodule_3))
        self.view_nodule_4.clicked.connect(lambda: self.open_view_image(self.view_nodule_4))
        self.view_nodule_5.clicked.connect(lambda: self.open_view_image(self.view_nodule_5))
        
        
        # self.imgLabel_1.mpsignal.connect(self.cross_center_mouse)
        # self.imgLabel_2.mpsignal.connect(self.cross_center_mouse)
        
    # Open view image
    def open_view_image(self, button):
        self.view_image = Image_View()
        text = button.text()
        if text == "View Original Slice":
            a_loc = self.original_vSlider.value()
            original_slice = (self.processedvoxel[a_loc, :, :]).astype(np.uint8).copy()
            self.view_image.set_image(original_slice)
        elif text == "View Preprocess Slice":
            a_loc = self.preprocess_vSlider.value()
            preprocess = (self.pre_image[a_loc, :, :]).astype(np.uint8).copy()
            self.view_image.set_image(preprocess)
        else:
            if text == "View Nodule 1":
                layer, x_start, y_start, x_end, y_end = take_pos_bbox(self.info_pbb[0])
                preprocess = (self.pre_image[layer, :, :]).astype(np.uint8).copy()
            elif text == "View Nodule 2":
                layer, x_start, y_start, x_end, y_end = take_pos_bbox(self.info_pbb[1])
                preprocess = (self.pre_image[layer, :, :]).astype(np.uint8).copy()
            elif text == "View Nodule 3":
                layer, x_start, y_start, x_end, y_end = take_pos_bbox(self.info_pbb[2])
                preprocess = (self.pre_image[layer, :, :]).astype(np.uint8).copy()
            elif text == "View Nodule 4":
                layer, x_start, y_start, x_end, y_end = take_pos_bbox(self.info_pbb[3])
                preprocess = (self.pre_image[layer, :, :]).astype(np.uint8).copy()
            elif text == "View Nodule 5":
                layer, x_start, y_start, x_end, y_end = take_pos_bbox(self.info_pbb[4])
                preprocess = (self.pre_image[layer, :, :]).astype(np.uint8).copy()
            
            # Draw horizontal lines
            preprocess[y_start, x_start:x_end+1] = 255
            preprocess[y_end, x_start:x_end+1] = 255
            # Draw vertical lines
            preprocess[y_start:y_end+1, x_start] = 255
            preprocess[y_start:y_end+1, x_end] = 255
            self.view_image.set_image(preprocess)
        
        print(button.text())
        self.view_image.show()
        
    def colormap_choice(self, text):
        self.colormap = self.colormapDict[text]
        self.updateimg_original()
        self.updateimg_preprocessing()
        
    def set_directory(self):
        os.chdir(self.directory)
        
    def dicom_clicked(self):
        dname = QFileDialog.getExistingDirectory(self, 'choose dicom directory')
        print(dname)
        self.load_dicomfile(dname)
        
    def open_3dview(self):
        self.volWindow.setWindowTitle('3D View')
        self.volWindow.vol_show()
        self.volWindow.show()
                
    def load_dicomfile(self, dname):
        self.dcmList.clear()
        
        # img for original
        patient = load_scan(dname)
        imgs, spacing = get_pixels_hu(patient)
        imgs_resample, space = resample(imgs, spacing, [1, 1, 1])
        self.imgLabel_1.setMouseTracking(False)
        self.imgLabel_2.setMouseTracking(False)
        self.voxel = self.linear_convert(imgs_resample)
        self.processedvoxel = self.voxel.copy()
        self.update_shape_original()
        self.updateimg_original()
        # self.imgLabel_1.setMaximumHeight(imgs_resample.shape[0])
        # self.imgLabel_1.setMaximumWidth(imgs_resample.shape[1])
        # self.imgLabel_1.setMinimumHeight(imgs_resample.shape[0])
        # self.imgLabel_1.setMinimumHeight(imgs_resample.shape[1])
        
        # 3d view
        self.volWindow = C3dView()
        self.volWindow.imgs = np.asarray(imgs_resample)
        
        # img for preprocessing
        name_dcm = dname.split('/')[-1]
        path_load = './work/' + name_dcm + '/' + name_dcm + '_clean.npy'
        if (not os.path.exists(path_load)):
            os.mkdir('./work/' + name_dcm)
            savenpy(name_dcm, dname, './work/' + name_dcm)
        img_pre = np.load(path_load)[0]
        self.pre_image = img_pre 
        self.update_shape_preprocessing()
        self.updateimg_preprocessing()
        # self.imgLabel_2.setMaximumHeight(img_pre.shape[0])
        # self.imgLabel_2.setMaximumWidth(img_pre.shape[1])
        # self.imgLabel_2.setMinimumHeight(img_pre.shape[0])
        # self.imgLabel_2.setMinimumHeight(img_pre.shape[1])
        
        # img for detection and classifier
        path_load = './work/bbox_result/' + name_dcm + '/' + name_dcm + '_pbb.npy'
        if (not os.path.exists(path_load)):
            run_detector(dname)
        prob, image_crops, prob_list = run_classifier(dname)
        info = take_position_nodule_proposal(name_dcm)
        self.info_pbb = info
        self.nodule_image = image_crops
        self.nodule_prob = prob_list[0]
        self.prob_patient = prob
        self.label_prob.setText("The probability of this patient's cancer: " + str(round(prob*100,2)) + "%")
        self.updateimg_nodule_pred()
        
        self.set_directory()
        self.dcmInfo = load_dcm_info(dname, self.privatecheckBox.isChecked())
        self.updatelist()
        
    def update_shape_original(self):
        self.v1, self.v2, self.v3 = self.processedvoxel.shape
        self.original_vSlider.setMaximum(self.v1-1)
        
    def update_shape_preprocessing(self):
        self.pre_v1, self.pre_v2, self.pre_v3 = self.pre_image.shape
        self.preprocess_vSlider.setMaximum(self.pre_v1-1)
        
    def updateimg_original(self):
        a_loc = self.original_vSlider.value()
        c_loc = self.v1 // 2
        s_loc = self.v2 // 2
        axial = (self.processedvoxel[a_loc, :, :]).astype(np.uint8).copy()
        self.imgLabel_1.slice_loc = [s_loc, c_loc, a_loc]

        if self.colormap is None:
            self.imgLabel_1.processedImage = axial
        else:
            self.imgLabel_1.processedImage = cv2.applyColorMap(axial, self.colormap)
        self.imgLabel_1.display_image(1)
        
    def updateimg_preprocessing(self):
        a_loc = self.preprocess_vSlider.value()
        c_loc = self.pre_v1 // 2
        s_loc = self.pre_v2 // 2
        axial = (self.pre_image[a_loc, :, :]).astype(np.uint8).copy()
        self.imgLabel_2.slice_loc = [s_loc, c_loc, a_loc]
        
        if self.colormap is None:
            self.imgLabel_2.processedImage = axial
        else:
            self.imgLabel_2.processedImage = cv2.applyColorMap(axial, self.colormap)
        self.imgLabel_2.display_image(1)
        
    def updateimg_nodule_pred(self):
        # self.info_pbb info nodule position
        # self.nodule_image image nodule_96x96
        if len(self.info_pbb) >= 1:
            self.nodule_1.processedImage = self.nodule_image[0, 0, 0, :, :, 48].astype(np.uint8).copy()
            self.nodule_info_1.setText(take_info_proposal_nodule(self.info_pbb[0]) + '\nMalignancy: ' + str(round(self.nodule_prob[0] * 100, 2)) + '%')
            self.nodule_1.display_image(1, False)
        
        if len(self.info_pbb) >= 2:
            self.nodule_2.processedImage = self.nodule_image[0, 1, 0, :, :, 48].astype(np.uint8).copy()
            self.nodule_info_2.setText(take_info_proposal_nodule(self.info_pbb[1])+ '\nMalignancy: ' + str(round(self.nodule_prob[1] * 100, 2)) + '%')
            self.nodule_2.display_image(1, False)
            
            
        if len(self.info_pbb) >= 3:
            self.nodule_3.processedImage = self.nodule_image[0, 2, 0, :, :, 48].astype(np.uint8).copy()
            self.nodule_info_3.setText(take_info_proposal_nodule(self.info_pbb[2])+ '\nMalignancy: ' + str(round(self.nodule_prob[2] * 100, 2)) + '%')
            self.nodule_3.display_image(1, False)
            
            
        if len(self.info_pbb) >= 4:
            self.nodule_4.processedImage = self.nodule_image[0, 3, 0, :, :, 48].astype(np.uint8).copy()
            self.nodule_info_4.setText(take_info_proposal_nodule(self.info_pbb[3])+ '\nMalignancy: ' + str(round(self.nodule_prob[3] * 100, 2)) + '%')
            self.nodule_4.display_image(1, False)
            
            
        if len(self.info_pbb) == 5:
            self.nodule_5.processedImage = self.nodule_image[0, 4, 0, :, :, 48].astype(np.uint8).copy()
            self.nodule_info_5.setText(take_info_proposal_nodule(self.info_pbb[4])+ '\nMalignancy: ' + str(round(self.nodule_prob[4] * 100, 2)) + '%')
            self.nodule_5.display_image(1, False)
            
        
    def updatelist(self):
        for item in self.dcmInfo:
            # For simple strings, it doesn’t matter if you don’t need QListWidgetItem packaging
            self.dcmList.addItem(QListWidgetItem('%-20s\t:  %s' % (item[0], item[1])))
        
    @staticmethod
    def linear_convert(img):
        convert_scale = 255.0 / (np.max(img) - np.min(img))
        converted_img = convert_scale * img - (convert_scale * np.min(img))
        return converted_img

        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Lung_Detection_Module()
    ex.show()
    sys.exit(app.exec_())
