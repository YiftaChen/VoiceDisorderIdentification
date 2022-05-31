import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from transformations.transform import WaveformToInput as TorchTransform
from architecture.backend.yamnet.params import YAMNetParams
from architecture.backend.yamnet.model import yamnet
from architecture.backend.yamnet.model import yamnet_category_metadata

from architecture.classifier.classification import Classifier

from torchaudio.transforms import Spectrogram,Resample,MelSpectrogram
from transformations.transform import ToTensor,CutAndResize,PadWhiteNoise,Truncate,FinalTrans

from datasets.ESC50Dataset import ESC50Dataset
import trainer.train_svd as svd_trainer
from trainer.MulticlassTrainer import MulticlassTrainer
import torch.nn as nn
import torch.optim 
from ray import tune

data_trans = nn.Sequential(ToTensor(),PadWhiteNoise(70000),Truncate(70000),Resample(orig_freq=50000,new_freq=15000),MelSpectrogram(sample_rate=15000,n_fft=500,n_mels=64),CutAndResize(-1,(96,64)),FinalTrans())

# dataset = ESC50Dataset('/home/chenka@staff.technion.ac.il/ESC-50-master/audio')
dataset = ESC50Dataset('/home/chenka@staff.technion.ac.il/Desktop/SVD',data_trans=data_trans)

def train_model(config):
    print(f'test config: {config}')
    model = Classifier(config["mlp_layers"],out_dim=2,activation=nn.LeakyReLU(negative_slope=0.01),freeze_backend_grad=False)    
    loss = nn.CrossEntropyLoss()
   
    opt = torch.optim.Adam(
        [
            dict(params=model.parameters()),            
            # dict(params=model.backend.layer14.parameters(),lr=config['lr']*0.01),
            # dict(params=model.backend.layer13.parameters(),lr=config['lr']*0.01),       
            # dict(params=model.backend.layer12.parameters(),lr=config['lr']*0.01)
            # dict(params=model.backend.layer11.parameters(),lr=config['lr']*0.01),
            # dict(params=model.backend.layer10.parameters(),lr=config['lr']*0.01),
            # dict(params=model.backend.layer9.parameters(),lr=config['lr']*0.01),
            # dict(params=model.backend.layer8.parameters(),lr=config['lr']*0.01),
            # dict(params=model.backend.layer7.parameters(),lr=config['lr']*0.01),
            # dict(params=model.backend.layer6.parameters(),lr=config['lr']*0.01),
            # dict(params=model.backend.layer5.parameters(),lr=config['lr']*0.01),
            # dict(params=model.backend.layer4.parameters(),lr=config['lr']*0.01),
            # dict(params=model.backend.layer3.parameters(),lr=config['lr']*0.01),
            # dict(params=model.backend.layer2.parameters(),lr=config['lr']*0.01),  
            # dict(params=model.backend.layer1.parameters(),lr=config['lr']*0.01)   
        ]
        ,lr=config["lr"],weight_decay=config['wd'])
    hyper_params = {
        'train_batch_size':128,
        'vald_batch_size':128,
        'test_batch_size':128,
        'num_workers':2,
        'epochs':200
    }
    trainer = MulticlassTrainer(dataset=dataset,model=model,optimizer=opt,hyper_params=hyper_params,verbose=False)
    model = trainer.train()


# config={
#     'lr':0.001,
#     'mlp_layers':[4096,1024,64],
#     'wd':0.001
# }
# train_model(config)

config={
    'lr':tune.grid_search([0.01,0.001]),
    'mlp_layers':[4096,1024,64],
    'wd':tune.grid_search([0.001,0.0001])
}
analysis = tune.run(train_model,config=config,resources_per_trial={'gpu':1},verbose=False)