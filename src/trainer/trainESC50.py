import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from transformations.transform import WaveformToInput as TorchTransform
from architecture.backend.yamnet.params import YAMNetParams
from architecture.backend.yamnet.model import yamnet
from architecture.backend.yamnet.model import yamnet_category_metadata

from architecture.classifier.classification import Classifier

from datasets.ESC50Dataset import ESC50Dataset
import trainer.train_svd as svd_trainer
from trainer.MulticlassTrainer import MulticlassTrainer
import torch.nn as nn
import torch.optim 


dataset = ESC50Dataset('/home/chenka@staff.technion.ac.il/ESC-50-master/audio')

def train_model(config):
    print(f'test config: {config}')
    model = Classifier(config["mlp_layers"],out_dim=50,activation=nn.LeakyReLU(negative_slope=0.01),freeze_backend_grad=False)    
    loss = nn.CrossEntropyLoss()
   
    opt = torch.optim.Adam(
        [
            dict(params=model.classifier.parameters()),
            dict(params=model.backend.layer14.parameters(),lr=config['lr']*0.01),
         
        ]
        ,lr=config["lr"])
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
    'lr':0.01,
    'mlp_layers':[512]
}
train_model(config)