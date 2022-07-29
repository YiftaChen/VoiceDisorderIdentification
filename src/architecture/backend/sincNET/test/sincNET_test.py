from click import option
from architecture.backend.sincNET.model import SincNet
from datasets.SvdExDataset import create_datasets
from torch.utils.data import DataLoader
from architecture.classifier.basicConvClassification import SincNETClassifier
import torch

import core
import socket

import core
import socket


directory = core.params.dataset_locations[socket.gethostname()]
dataset,_,_ = create_datasets(directory,split=(0.8,0.1,0.1),hp={"filter_gender":None},filter_gender=None)
dl = DataLoader(dataset,batch_size=50)

data = iter(dl).next()
data = data['data']
data = data.squeeze(1)

classifier = SincNETClassifier()

classifier.eval()

res = classifier(data)

a=5





# SincNetOptions = {'input_dim': 50000,
#           'fs': 50000, # sample rate
#           'cnn_N_filt': [80,60,60], # channels (first channel is sinc_conv others are just 1d_conv)
#           'cnn_len_filt': [1024,5,5], # filter length
#           'cnn_max_pool_len': [3,3,3],
#           'cnn_use_laynorm_inp': True,
#           'cnn_use_batchnorm_inp': False,
#           'cnn_use_laynorm':[True,True,True],
#           'cnn_use_batchnorm':[False,False,False],
#           'cnn_act': ["relu","relu","relu"],
#           'cnn_drop':[0.0,0.0,0.0]
#           }

# model = SincNet(SincNetOptions)
# print(model)

# data = data.squeeze(1)

# res = model(data)

# model_output_size = model(torch.zeros(torch.Size((1,50000)))).shape[1]

# a=5










# def read_conf_inp(cfg_file):
 
#     # parser=OptionParser()
#     # (options,args)=parser.parse_args()
    
#     options = {}

#     Config = ConfigParser.ConfigParser()
#     Config.read(cfg_file)

#     # #[data]
#     # options.tr_lst=Config.get('data', 'tr_lst')
#     # options.te_lst=Config.get('data', 'te_lst')
#     # options.lab_dict=Config.get('data', 'lab_dict')
#     # options.data_folder=Config.get('data', 'data_folder')
#     # options.output_folder=Config.get('data', 'output_folder')
#     # options.pt_file=Config.get('data', 'pt_file')

#     #[windowing]
#     options['fs']=Config.get('windowing', 'fs')
#     options['cw_len']=Config.get('windowing', 'cw_len')
#     options['cw_shift']=Config.get('windowing', 'cw_shift')

#     #[cnn]
#     options['cnn_N_filt']=Config.get('cnn', 'cnn_N_filt')
#     options['cnn_len_filt']=Config.get('cnn', 'cnn_len_filt')
#     options['cnn_max_pool_len']=Config.get('cnn', 'cnn_max_pool_len')
#     options['cnn_use_laynorm_inp']=Config.get('cnn', 'cnn_use_laynorm_inp')
#     options['cnn_use_batchnorm_inp']=Config.get('cnn', 'cnn_use_batchnorm_inp')
#     options['cnn_use_laynorm']=Config.get('cnn', 'cnn_use_laynorm')
#     options['cnn_use_batchnorm']=Config.get('cnn', 'cnn_use_batchnorm')
#     options['cnn_act']=Config.get('cnn', 'cnn_act')
#     options['cnn_drop']=Config.get('cnn', 'cnn_drop')


#     #[dnn]
#     options['fc_lay']=Config.get('dnn', 'fc_lay')
#     options['fc_drop']=Config.get('dnn', 'fc_drop')
#     options['fc_use_laynorm_inp']=Config.get('dnn', 'fc_use_laynorm_inp')
#     options['fc_use_batchnorm_inp']=Config.get('dnn', 'fc_use_batchnorm_inp')
#     options['fc_use_batchnorm']=Config.get('dnn', 'fc_use_batchnorm')
#     options['fc_use_laynorm']=Config.get('dnn', 'fc_use_laynorm')
#     options['fc_act']=Config.get('dnn', 'fc_act')

#     #[class]
#     options['class_lay']=Config.get('class', 'class_lay')
#     options['class_drop']=Config.get('class', 'class_drop')
#     options['class_use_laynorm_inp']=Config.get('class', 'class_use_laynorm_inp')
#     options['class_use_batchnorm_inp']=Config.get('class', 'class_use_batchnorm_inp')
#     options['class_use_batchnorm']=Config.get('class', 'class_use_batchnorm')
#     options['class_use_laynorm']=Config.get('class', 'class_use_laynorm')
#     options['class_act']=Config.get('class', 'class_act')


#     #[optimization]']
#     options['lr']=Config.get('optimization', 'lr')
#     options['batch_size']=Config.get('optimization', 'batch_size')
#     options['N_epochs']=Config.get('optimization', 'N_epochs')
#     options['N_batches']=Config.get('optimization', 'N_batches')
#     options['N_eval_epoch']=Config.get('optimization', 'N_eval_epoch')
#     options['seed']=Config.get('optimization', 'seed')
    
#     return options
