# -*- coding: utf-8 -*-
"""
# Load the existing model and the corresponding run test data set for prediction
# The management of pytorch memory and GPU memory is relatively scattered. There may be multiple parameters involved in a project. If you are proficient, it is easy to use finely.
#pytorch part of the comment:
1) pin_memory: When the computer has sufficient memory, you can set pin_memory=True. When the system is stuck, or the swap memory is used too much, set pin_memory=False
2) num_workers: Indicates the number of threads to read samples, that is, the number of processes that use multi-process loading training and test data, 0 means not using multi-process, the more the number, the larger the memory usage
3) collate_fn: splicing multiple sample data into a batch (how to splice?), generally using the default splicing method
4) DataBowl3Detector: It is a subclass that inherits torch and is used to define our own datasets (loading npy, cropping, sampling, augmentation, splicing, label label changes, cutting and merging in the test phase, etc.)
5) DataLoader: After defining our own dataset in DataBowl3Detector, we can load data through torch's DataLoader
6) SplitComb.py: Only in the test phase preprocessing step2 appears, used for imgs data cutting and patch merging
"""

from .config_submit import config as config_submit

import os
import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from .model import acc
from .model import DataBowl3Detector, collate
from .model import DataBowl3Classifier
# from data_classifier import DataBowl3Classifier

from .utils import *
from .model import nms
from .model import SplitComb
from .model import test_detect
from .model import test_casenet
from .model import get_model_detector
from .model import CaseNet, config_classifier_net
from .preprocess import savenpy
from importlib import import_module
import pandas
import warnings

warnings.filterwarnings("ignore")

def run_detector(path_folder):
    # check OS
    if "\\" in path_folder:
    # This is a Windows path, split by '\'
        path_components = path_folder.split("\\")
    else:
        # This is a Unix-style path, split by '/'
        path_components = path_folder.split("/")
    name = path_components[-1]
        
    if torch.cuda.is_available():
        print("Have GPU")
    else:
        print("None GPU")

    if not os.path.exists('./work/' + name + '/' + name + '_label.npy') or not os.path.exists(
        './work/' + name + '/' + name + '_clean.npy'):
        if not os.path.exists('./work/' + name):
            os.mkdir('./work/' + name + '/')
        savenpy(name,path_folder,'./work/' + name)
    else:
        print("->> Preprocess for " + name + " is done !!! <<-")
    
    
    if os.path.exists('./work/bbox_result/' + name + "/" + name + '_pbb.npy') and os.path.exists(
            './work/bbox_result/' + name + '/' + name +'_lbb.npy'):
        if not os.path.exists('./work/bbox_result/' + name):
            os.mkdir('./work/bbox_result/' + name + '/')
        print("->> Detector for " + name + " is done !!! <<-")
        return 0
    else:
        print("Start detector ...")
    
    # The address of the prediction dataset
    datapath = config_submit['datapath'] + name
    # The final prediction result address of the prediction data set
    prep_result_path = config_submit['preprocess_result_path'] + name
    skip_prep = config_submit['skip_preprocessing']
    skip_detect = config_submit['skip_detect']

    testsplit = [f.split('_')[0] for f in os.listdir(datapath) if f.endswith('_clean.npy')]

    # N-NET Model (nodule detection) detector
    ## detector_model.py script object
    #nodmodel = import_module(name = config_submit['detector_model'].split('.py')[0])

    ## model Initialize and return some other parameters 
    # (eg: config1 is the anchor, some hyper parameters related to training)
    config1, nod_net, loss, get_pbb = get_model_detector()

    ## Load the trained weights. Note that it is not directly loaded by model.
    # load like keras. The torch.load() method (torch.save() method is similar)
    checkpoint = torch.load(config_submit['detector_param'])

    ## state_dict：pytorch-specific state dictionary, 
    # {layer: params} mapping relationship, 
    # and only save trainable_layer and corresponding parameters

    ## Then merge the corresponding network structure and state_dict state dictionary parameters
    nod_net.load_state_dict(checkpoint['state_dict'])

    # cuda、cudnn and other multi-GPU configuration related
    # torch.cuda.set_device(0)
    # nod_net = nod_net.cuda()
    # cudnn.benchmark = True
    # nod_net = DataParallel(nod_net)

    # bbox-cube The storage address of the prediction result,
    # because it is a detection, so the relevant coordinates will be saved
    bbox_result_path = './work/bbox_result/' 
    if not os.path.exists(bbox_result_path):
        os.mkdir(bbox_result_path)
    #testsplit = [f.split('_clean')[0] for f in os.listdir(prep_result_path) if '_clean' in f]

    if not os.path.exists(bbox_result_path + name):
        os.mkdir(bbox_result_path + name)

    if not skip_detect:
        # The target detection prediction results of the first stage model
        margin = 32
        sidelen = 144
        config1['datadir'] = prep_result_path
        
        # A batch is one original image with batch_size of 1, but SplitComb extracts 
        # 12 patches based on each image, 
        # and the demo_test is predicted at the same time, causing GPU memory overflow
        split_comber = SplitComb(sidelen,
                                config1['max_stride'],
                                config1['stride'],
                                margin,
                                pad_value=config1['pad_value'])
        
        # DataBowl3Detector is used to process the preprocessed npy dataset, extract 3D patches, 
        # and fix the crop dimension to (128x128x128x1) to do difficult negative sample mining
        dataset = DataBowl3Detector(testsplit,
                                    config1,
                                    phase='test',
                                    split_comber=split_comber)
        
        # DataLoader Function to convert a data sample into a pytorch-specific format
        test_loader = DataLoader(dataset,
            batch_size = 1,
            shuffle = False,
            pin_memory=False,
            collate_fn=collate) 
        # Here, adjust num_workers to a small size to avoid the pytorch data_loader method taking up too much memory
        
        test_detect(test_loader, nod_net, get_pbb,
                    bbox_result_path,config1,
                    n_gpu=config_submit['n_gpu'])
        print()
        return 1
    
    
def run_classifier(path_folder):
    # check OS
    if "\\" in path_folder:
    # This is a Windows path, split by '\'
        path_components = path_folder.split("\\")
    else:
        # This is a Unix-style path, split by '/'
        path_components = path_folder.split("/")
    name = path_components[-1]
        
    if torch.cuda.is_available():
        print("Have GPU")
    else:
        print("None GPU")
    
    # The address of the prediction dataset
    datapath = config_submit['datapath'] + name
    testsplit = [f.split('_')[0] for f in os.listdir(datapath) if f.endswith('_clean.npy')]
    
    # C-NET model (nodule cancer classification) classifier
    ## classifier_model.py script object
    #casemodel = import_module(config_submit['classifier_model'].split('.py')[0])
    
    ## The model is initialized, and the boxes-cube prediction with probability top=5 
    # in the first stage is taken as the input of the next stage
    casenet = CaseNet(topk=5)
    
    prep_result_path = config_submit['preprocess_result_path'] + name
    bbox_result_path = './work/bbox_result/' +  name
    
    ## There are also hyper parameters
    config2 = config_classifier_net
    ## Load the trained weights. 
    # Note that it is not directly loaded by model.load like keras. 
    # The torch.load() method (torch.save() method is similar)
    
    # with open(config_submit['classifier_param'], 'rb') as f:
    #     checkpoint = torch.load(f)
    checkpoint = torch.load(config_submit['classifier_param'], encoding='latin1')
    
    ## Then merge the corresponding network structure and state_dict state dictionary parameters
    casenet.load_state_dict(checkpoint['state_dict'])

    # torch.cuda.set_device(0)
    # casenet = casenet.cuda()
    # cudnn.benchmark = True
    # casenet = DataParallel(casenet)

    filename = config_submit['outputfile']

    # The address of the boxes-cube output in the first stage
    config2['bboxpath'] = bbox_result_path
    
    # The output address of the sigmoid probability and coordinate 
    # conversion coefficient corresponding to the cube in the second stage
    config2['datadir'] = prep_result_path



    dataset = DataBowl3Classifier(testsplit, config2, phase = 'test')
    # test_casenet function prediction
    predlist, x, out = test_casenet(casenet,dataset)
    
    # save result
    df = pandas.DataFrame({'id':testsplit, 'cancer':predlist})
    
    print(predlist[0], x.shape, out.shape)
    return predlist[0], x.detach().numpy(), out.detach().numpy()
    
def take_preprocess_img(name):
    path_load = './work/' + name + '/' + name + '_clean.npy'
    img = np.load(path_load)
    return img

def take_position_nodule_proposal(name):
    pbb = np.load('./work/bbox_result/' + name + '/' + name + '_pbb.npy')
    pbb = pbb[pbb[:,0]>-1]
    pbb = nms(pbb,0.05)
    
    size = pbb.shape[0]
    info_pbb = []
    for i in range(size):
        info_pbb.append([int(elm) for elm in list(pbb[i][1:])])  
    return info_pbb
    
def take_info_proposal_nodule(pbb_elm):
    x_pos = pbb_elm[2] - pbb_elm[3]
    y_pos = pbb_elm[1] - pbb_elm[3]
    height = pbb_elm[3] * 2
    width = pbb_elm[3] * 2
    return "Layer: %d \nPosition: (%d, %d) \nHeight Box: %d \nWidth Box: %d" % (pbb_elm[0],
                                                                                       x_pos,
                                                                                       y_pos,
                                                                                       height,
                                                                                       width)
    
def take_pos_bbox(pbb_elm):
    x_pos = pbb_elm[2] - pbb_elm[3]
    y_pos = pbb_elm[1] - pbb_elm[3]
    x_end = x_pos + pbb_elm[3] * 2
    y_end = y_pos + pbb_elm[3] * 2
    return pbb_elm[0], x_pos, y_pos, x_end, y_end
# INPUT_PATH = "D:/Baitap/NCKH/DICOM/Data/PAT002DSB"
# run_detector(INPUT_PATH)
# run_classifier(INPUT_PATH)
# print(take_position_nodule_proposal('PAT002DSB'))