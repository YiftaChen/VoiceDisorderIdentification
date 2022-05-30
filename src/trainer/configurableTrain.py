import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from transformations.transform import WaveformToInput as TorchTransform
from architecture.backend.yamnet.params import YAMNetParams
from architecture.backend.yamnet.model import yamnet
from architecture.backend.yamnet.model import yamnet_category_metadata

from architecture.classifier.classification import Classifier

from datasets.SvdExDataset import SvdExtendedVoiceDataset
import trainer.train_svd as svd_trainer
import torch.nn as nn
import torch.optim 
from ray import tune
import pickle


def train_model(config):
    print(f'test config: {config}')
    dataset = SvdExtendedVoiceDataset(r"/home/yiftach.ede@staff.technion.ac.il/Desktop/SVD",hp = config,classification_binary=True)
    torch.multiprocessing.set_start_method('spawn')

    model = Classifier(config["mlp_layers"],activation=nn.LeakyReLU(negative_slope=0.01),freeze_backend_grad=False)    
    loss = nn.BCEWithLogitsLoss()
    # params_non_frozen = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.Adam(
        [
            dict(params=model.classifier.parameters()),
            dict(params=model.backend.layer14.parameters(),lr=config['yamnet_l14_lr']),
            dict(params=model.backend.layer13.parameters(),lr=config['yamnet_l13_lr']),
            # dict(params=model.backend.layer12.parameters(),lr=config['lr']*0.01),
            # dict(params=model.backend.layer11.parameters(),lr=config['lr']*0.01),
        ]
        ,lr=config["lr"],weight_decay = config['l2_reg'])
    hyper_params = {
        'train_batch_size':128,
        'vald_batch_size':128,
        'test_batch_size':128,
        'num_workers':2,
        'epochs':200
    }
    trainer = svd_trainer.Trainer(dataset=dataset,model=model,optimizers=opt,critereon=loss,early_stop=50,hyper_params=hyper_params,verbose=False)
    model = trainer.train()
    


config={
    'lr':tune.grid_search([1e-2,1e-3]),
    'mlp_layers':[tune.grid_search([512,256,128])],
    'augmentations':tune.grid_search([
        ["TimeInversion"],
        []
    ]),
    'l2_reg':tune.grid_search([1e-2,1e-3]),
    'yamnet_l14_lr':tune.grid_search([1e-3,1e-4]),
    'yamnet_l13_lr':tune.grid_search([1e-3,1e-4,1e-5,0])
    # 'mlp_layers':[512]
    # 'activation':nn.LeakyReLU(negative_slope=0.01)
    }
analysis = tune.run(train_model,config=config,resources_per_trial={'gpu':1},verbose=False)


# config={
#     'lr':0.01,
#     'mlp_layers':[512,128]
# }
# train_model(config)

# with open('analysisFile','wb') as a_file:
#     pickle.dump(analysis,a_file)