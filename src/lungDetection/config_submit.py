# -*- coding: utf-8 -*-
config = {'datapath':'./work/',
          'preprocess_result_path':'./work/',
          'outputfile':'prediction.csv',
          
          'detector_model':'net_detector',
         'detector_param':'./model/pretrain/detector.ckpt',
         'classifier_model':'net_classifier',
         'classifier_param':'./model/pretrain/classifier.ckpt',
         'n_gpu':1,
         'n_worker_preprocessing':None,
         'use_exsiting_preprocessing':False,
         'skip_preprocessing':True, # Skip the preprocessing process, 
         # here because all the data has been 
         # preprocessed before, so it is set to True
         
         'skip_detect':False} # Skip stage 1 detection prediction process