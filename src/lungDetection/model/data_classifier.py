import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import collections
import random
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate
from .layers import nms,iou
import pandas

class DataBowl3Classifier(Dataset):
    def __init__(self, split, config, phase = 'test'):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        
        self.random_sample = config['random_sample']
        self.T = config['T']
        self.topk = config['topk']
        self.crop_size = config['crop_size']
        self.stride = config['stride']
        self.augtype  = config['augtype']
        self.filling_value = config['filling_value']
        
        #self.labels = np.array(pandas.read_csv(config['labelfile']))
        
        datadir = config['datadir']
        bboxpath  = config['bboxpath']
        self.phase = phase
        self.candidate_box = []
        self.pbb_label = []
        
        idcs = split
        self.filenames = [os.path.join(datadir, '%s_clean.npy' % idx.split('-')[0]) for idx in idcs]
        
        if self.phase!='test':
            self.yset = 1-np.array([f.split('-')[1][2] for f in idcs]).astype('int')
 
        
        for idx in idcs:
            pbb = np.load(os.path.join(bboxpath,idx+'_pbb.npy'))
            pbb = pbb[pbb[:,0]>config['conf_th']]
            pbb = nms(pbb, config['nms_th'])
            
            lbb = np.load(os.path.join(bboxpath,idx+'_lbb.npy'))
            pbb_label = []
            
            for p in pbb:
                isnod = False
                for l in lbb:
                    score = iou(p[1:5], l)
                    if score > config['detect_th']:
                        isnod = True
                        break
                pbb_label.append(isnod)

            self.candidate_box.append(pbb)
            self.pbb_label.append(np.array(pbb_label))
        self.crop = simpleCrop(config,phase)
        

    def __getitem__(self, idx,split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))#seed according to time

        pbb = self.candidate_box[idx]
        pbb_label = self.pbb_label[idx]
        conf_list = pbb[:,0]
        T = self.T
        topk = self.topk
        img = np.load(self.filenames[idx])
    
        chosenid = conf_list.argsort()[::-1][:topk]
        croplist = np.zeros([topk,1,self.crop_size[0],self.crop_size[1],self.crop_size[2]]).astype('float32')
        coordlist = np.zeros([topk,3,int(self.crop_size[0]/self.stride),int(self.crop_size[1]/self.stride),int(self.crop_size[2]/self.stride)]).astype('float32')
        padmask = np.concatenate([np.ones(len(chosenid)),np.zeros(self.topk-len(chosenid))])
        isnodlist = np.zeros([topk])

        
        for i,id in enumerate(chosenid):
            target = pbb[id,1:]
            isnod = pbb_label[id]
            crop,coord = self.crop(img,target)
            crop = crop.astype(np.float32)
            croplist[i] = crop
            coordlist[i] = coord
            isnodlist[i] = isnod
       
        return torch.from_numpy(croplist).float(), torch.from_numpy(coordlist).float()
        
    def __len__(self):
        return len(self.candidate_box)
        

        
class simpleCrop():
    def __init__(self,config,phase):
        self.crop_size = config['crop_size']
        self.scaleLim = config['scaleLim']
        self.radiusLim = config['radiusLim']
        self.jitter_range = config['jitter_range']
        self.isScale = config['augtype']['scale'] and phase=='train'
        self.stride = config['stride']
        self.filling_value = config['filling_value']
        self.phase = phase
        
    def __call__(self,imgs,target):
        if self.isScale:
            radiusLim = self.radiusLim
            scaleLim = self.scaleLim
            scaleRange = [np.min([np.max([(radiusLim[0]/target[3]),scaleLim[0]]),1])
                         ,np.max([np.min([(radiusLim[1]/target[3]),scaleLim[1]]),1])]
            scale = np.random.rand()*(scaleRange[1]-scaleRange[0])+scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float')/scale).astype('int')
        else:
            crop_size = np.array(self.crop_size).astype('int')
        if self.phase=='train':
            jitter_range = target[3]*self.jitter_range
            jitter = (np.random.rand(3)-0.5)*jitter_range
        else:
            jitter = 0
        start = (target[:3]- crop_size/2 + jitter).astype('int')
        pad = [[0,0]]
        for i in range(3):
            if start[i]<0:
                leftpad = -start[i]
                start[i] = 0
            else:
                leftpad = 0
            if start[i]+crop_size[i]>imgs.shape[i+1]:
                rightpad = start[i]+crop_size[i]-imgs.shape[i+1]
            else:
                rightpad = 0
            pad.append([leftpad,rightpad])
        imgs = np.pad(imgs,pad,'constant',constant_values =self.filling_value)
        crop = imgs[:,start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
        
        normstart = np.array(start).astype('float32')/np.array(imgs.shape[1:])-0.5
        normsize = np.array(crop_size).astype('float32')/np.array(imgs.shape[1:])
        xx,yy,zz = np.meshgrid(np.linspace(normstart[0],normstart[0]+normsize[0],int(self.crop_size[0]/self.stride)),
                           np.linspace(normstart[1],normstart[1]+normsize[1],int(self.crop_size[1]/self.stride)),
                           np.linspace(normstart[2],normstart[2]+normsize[2],int(self.crop_size[2]/self.stride)),indexing ='ij')
        coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')

        if self.isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop,[1,scale,scale,scale],order=1)
            newpad = self.crop_size[0]-crop.shape[1:][0]
            if newpad<0:
                crop = crop[:,:-newpad,:-newpad,:-newpad]
            elif newpad>0:
                pad2 = [[0,0],[0,newpad],[0,newpad],[0,newpad]]
                crop = np.pad(crop,pad2,'constant',constant_values =self.filling_value)

        return crop,coord
