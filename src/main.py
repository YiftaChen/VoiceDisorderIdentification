import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from transformations.transform import WaveformToInput as TorchTransform
from architecture.backend.yamnet.params import YAMNetParams
from architecture.backend.yamnet.model import yamnet
from architecture.backend.yamnet.model import yamnet_category_metadata

from architecture.classifier.classification import Wav2Vec2Classifier,HubertClassifier

from datasets.SvdExDataset import SvdExtendedVoiceDataset
import trainer.train_svd as svd_trainer

import torch.nn as nn
import torch.optim 


def train_model(config):
    print(f'test config: {config}')
    dataset = SvdExtendedVoiceDataset(r"/home/yiftach.ede@staff.technion.ac.il/Desktop/SVD",hp = config,classification_binary=True)

    model = HubertClassifier(config["mlp_layers"])    
    loss = nn.BCEWithLogitsLoss()
    # params_non_frozen = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.Adam(
        [
            dict(params=model.classifier.parameters()),
            # dict(params=model.backend.layer13.parameters(),lr=config['lr']*0.001),
            # dict(params=model.backend.layer12.parameters(),lr=config['lr']*0.01),
            # dict(params=model.backend.layer11.parameters(),lr=config['lr']*0.01),
        ]
        ,lr=config["lr"])
    hyper_params = {
        'train_batch_size':128,
        'vald_batch_size':128,
        'test_batch_size':128,
        'num_workers':2,
        'epochs':200
    }
    trainer = svd_trainer.Trainer(dataset=dataset,model=model,optimizers=opt,critereon=loss,hyper_params=hyper_params,verbose=False)
    model = trainer.train()
    
config={
    'lr':1e-3,
    'mlp_layers':[128],
    'augmentations':["TimeInversion"]
    # 'mlp_layers':[512]
    # 'activation':nn.LeakyReLU(negative_slope=0.01)
    }

train_model(config)