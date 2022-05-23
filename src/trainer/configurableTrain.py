import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from transformations.transform import WaveformToInput as TorchTransform
from architecture.backend.yamnet.params import YAMNetParams
from architecture.backend.yamnet.model import yamnet
from architecture.backend.yamnet.model import yamnet_category_metadata

from architecture.classifier.classification import Classifier

from datasets.SvdExDataset import SvdCutOffShort
import trainer.train_svd as svd_trainer
import torch.nn as nn
import torch.optim 
from ray import tune
import pickle

dataset = SvdCutOffShort(r"/home/chenka@staff.technion.ac.il/Desktop/SVD",classification_binary=True,overfit_test = True)

def train_model(config):
    print(f'test config: {config}')
    model = Classifier(config["mlp_layers"],activation=nn.ReLU())    
    loss = nn.BCEWithLogitsLoss()
    params_non_frozen = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.Adam(params_non_frozen,lr=config["lr"])
    hyper_params = {
        'train_batch_size':128,
        'vald_batch_size':128,
        'test_batch_size':128,
        'num_workers':2,
        'epochs':200
    }
    trainer = svd_trainer.Trainer(dataset=dataset,model=model,optimizers=opt,critereon=loss,hyper_params=hyper_params)
    model = trainer.train()
    


config={
    'lr':tune.grid_search([0.1,0.01,0.001]),
    'mlp_layers':[tune.grid_search([1024, 512,256]),tune.grid_search([1024,512,256])]
    }
analysis = tune.run(train_model,config=config,resources_per_trial={'gpu':1},verbose=True)


# config={
#     'lr':0.01,
#     'mlp_layers':[1024,1024,256]
# }
# train_model(config)

# with open('analysisFile','wb') as a_file:
#     pickle.dump(analysis,a_file)