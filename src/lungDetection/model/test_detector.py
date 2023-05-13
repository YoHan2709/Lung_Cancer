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

def test_detect(data_loader, net, get_pbb, save_dir, config, n_gpu):
    start_time = time.time()
    net.eval()
    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        # target:Nodule position information of the current index batch
        # coord:All nodule position information of the current index batch
        # nzhw:
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('-')[0].split('/')[-1]
        shortname = name.split('_clean')[0]
        data = data[0][0]
        coord = coord[0][0]
        isfeat = False # isfeat=False unchanging
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        # n_per_run = n_gpu
        # data = data[0:1] # Only one patch is taken from all 12 patches of each picture for prediction,
        # and there will be problems with merging in the future
        
        # (12,1,208,208,208)，The batch_size of the original image is 1, 
        # because of patch clipping and sampling, 
        # 12 patches are loaded. It is equivalent to predicting 
        # 12 pictures at the same time, causing GPU memory overflow
        print(data.size()) 
        splitlist = range(0,len(data)+1,n_gpu)
        if splitlist[-1]!=len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist)-1):
            if torch.cuda.is_available():
                input = Variable(data[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
                inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
                
            # place cuda here
            input = Variable(data[splitlist[i]:splitlist[i+1]], volatile = True)
            inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]], volatile = True)
            
            if isfeat:
                output,feature = net(input,inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                # RuntimeError: DataLoader worker (pid 22623) is killed by signal: Killed.
                # (The problem of memory overflow can be solved by reducing the number of num_workers and sample pictures)
                
                # RuntimeError: CUDA out of memory. Tried to allocate xxx.xx MiB.
                # (GPU memory overflow problem, reduce the number of patches for prediction at the same time, or add del output)
                output = net(input,inputcoord) 
            outputlist.append(output.data.cpu().numpy())
            del output # Avoid GPU memory overflow plus this sentence
        output = np.concatenate(outputlist,0)
        # The combine operation combines all patches to calculate the final prediction result
        output = split_comber.combine(output,nzhw=nzhw) 
        
        # not run here
        if isfeat:
            feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])[:,:,:,:,:,np.newaxis]
            # feature = split_comber.combine(feature, sidelen)[...,0]
            
        thresh = -3
        pbb,mask = get_pbb(output,thresh,ismask=True)
        
        # not run here
        if isfeat:
            feature_selected = feature[mask[0],mask[1],mask[2]]
            np.save(os.path.join(save_dir, shortname+'_feature.npy'), feature_selected)
            
        #tp,fp,fn,_ = acc(pbb,lbb,0,0.1,0.1)
        #print([len(tp),len(fp),len(fn)])
        
        print([i_name,shortname])
        e = time.time()
        
        np.save(os.path.join(save_dir, shortname+'_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, shortname+'_lbb.npy'), lbb)
    end_time = time.time()


    print('elapsed time is %3.2f seconds' % (end_time - start_time))
    print
    print
    