# -*- coding: utf-8 -*-
import argparse
import os
import time
import numpy as np
from importlib import import_module
import shutil
# from utils import *
import sys

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

import matplotlib.pyplot as plt


def test_casenet(model,testset):
    """
    The cancer classification prediction process of the second stage model, return predlist
    :param model: 
    :param testset: The dimension is (96, 96, 96, 1), which is obtained from top5 cube proposals and cropping
    :return: 
    """
    data_loader = DataLoader(
        testset,
        batch_size = 1,
        shuffle = False,
        pin_memory=True)
    #model = model.cuda()
    model.eval()
    predlist = []
    
    #   weight = torch.from_numpy(np.ones_like(y).float().cuda()
    for i,(x,coord) in enumerate(data_loader):
        if torch.cuda.is_available():
            coord = Variable(coord).cuda()
            x = Variable(x).cuda()
        coord = Variable(coord)
        x = Variable(x)
        # print("Size x : ", x.shape)
        
        ## nodulePred：Prediction results of the 4 coordinate conversion 
        # coefficients of pulmonary nodules (3 coordinates x, y, z and radius of the origin)
        
        ## casePred：The sigmoid probability value of lung nodules predicted as cancer
        nodulePred,casePred,out = model(x,coord)
        predlist.append(casePred.data.cpu().numpy())
        # print(out)
        # fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
        # for i in range(5):
        #     axes[i].imshow(x[0, i, 0, :, :, 48], cmap='gray')  # show the middle slice of the z-axis
        #     axes[i].set_title('Nodule ' + str(i) + '\nMalignancy: ' + str(round(out.detach().numpy()[0][i]*100, 2)) + '%')
        #     axes[i].set_xticks([])
        #     axes[i].set_yticks([])
        # plt.show()
        #print([i,data_loader.dataset.split[i,1],casePred.data.cpu().numpy()])
    predlist = np.concatenate(predlist)
    return predlist, x, out