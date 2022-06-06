import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from transformations.transform import Derivative, WaveformToInput as TorchTransform
from architecture.backend.yamnet.params import YAMNetParams
from architecture.backend.yamnet.model import yamnet
from architecture.backend.yamnet.model import yamnet_category_metadata

from architecture.classifier.classification import Classifier

from torchaudio.transforms import Spectrogram,Resample,MelSpectrogram
from transformations.transform import ToTensor,PadWhiteNoise,Truncate,SetRange,WaveformToInput,AddDerivatives
from torchvision.transforms import Resize,Normalize

from datasets.SVDDataset import SVDDataset
from datasets.SVDBinaryDataset import SVDBinaryDataset
import trainer.train_svd as svd_trainer
from trainer.MulticlassTrainer import MulticlassTrainer
import torch.nn as nn
import torch.optim 
from ray import tune

dataset = SVDBinaryDataset('/home/chenka@staff.technion.ac.il/Desktop/SVD')

def train_model(config):
    print(f'dataset len: {len(dataset)}')
    print(f'test config: {config}')
    model = Classifier(config["mlp_layers"],out_dim=2,activation=nn.LeakyReLU(negative_slope=0.01),freeze_backend_grad=False)        
   
    opt = torch.optim.Adam(
        [
            dict(params=model.classifier.parameters()), 
            dict(params=model.backend.parameters(),lr=0.001*config['lr'])             
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


config={
    'lr':0.0001,
    'mlp_layers':[512],
    'wd':0
}
train_model(config)

# config={
#     'lr':tune.grid_search([0.01,0.001]),
#     'mlp_layers':[4096,1024,64],
#     'wd':tune.grid_search([0.001,0.0001])
# }
# analysis = tune.run(train_model,config=config,resources_per_trial={'gpu':1},verbose=False)