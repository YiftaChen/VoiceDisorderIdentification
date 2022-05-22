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

dataset = SvdCutOffShort(r"/home/chenka@staff.technion.ac.il/Desktop/SVD",classification_binary=True,overfit_test = False)

def train_model(config):
    model = Classifier([512])
    loss = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(),lr=config["lr"])
    hyper_params = {
        'train_batch_size':128,
        'vald_batch_size':128,
        'test_batch_size':128,
        'num_workers':2,
        'epochs':80
    }
    trainer = svd_trainer.Trainer(dataset=dataset,model=model,optimizers=opt,critereon=loss,hyper_params=hyper_params)
    model = trainer.train()
    

# train_model({'lr':0.1})


analysis = tune.run(train_model,config={'lr':tune.grid_search([0.1,0.01,0.001,0.0001])},metric="acc",resources_per_trial={'gpu':1},verbose=True)

# with open('analysisFile','wb') as a_file:
#     pickle.dump(analysis,a_file)