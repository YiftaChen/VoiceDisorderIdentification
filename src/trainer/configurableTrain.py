import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from transformations.transform import WaveformToInput as TorchTransform
from architecture.backend.yamnet.params import YAMNetParams
from architecture.backend.yamnet.model import yamnet
from architecture.backend.yamnet.model import yamnet_category_metadata

from architecture.classifier.classification import YamnetClassifier,Wav2Vec2Classifier,DistilHUBERTClassifier,HubertClassifier
from datasets.SvdExDataset import SvdExtendedVoiceDataset,SvdFilterA
import trainer.train_svd as svd_trainer
import torch.nn as nn
import torch.optim 
from ray import tune
import pickle
from ray.tune.integration.wandb import (
    WandbLoggerCallback,
    WandbTrainableMixin,
    wandb_mixin,
)
import wandb

count = 0
@wandb_mixin
def train_model(config):
    wandb.run.name = f"HubertAblationStudy_{wandb.run.name.split('_')[-1]}"
    wandb.run.save()
    dataset = SvdFilterA(r"/home/yiftach.ede@staff.technion.ac.il/Desktop/SVD",hp = config,classification_binary=True)
    # torch.multiprocessing.set_start_method('spawn')

    model = HubertClassifier(config["mlp_layers"],configuration = config['configuration']).to(device="cuda:0")  
    loss = nn.BCEWithLogitsLoss()
    # params_non_frozen = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.Adam(
        [
            dict(params=model.classifier.parameters()),
            dict(params=model.model.encoder.parameters(),lr=config['backend_encoder_lr']),
            # dict(params=model.backend.layer13.parameters(),lr=config['yamnet_l13_lr']),
            # dict(params=model.backend.layer12.parameters(),lr=config['lr']*0.01),
            # dict(params=model.backend.layer11.parameters(),lr=config['lr']*0.01),
        ]
        ,lr=config["lr"],weight_decay = config['l2_reg'])
    hyper_params = {
        'train_batch_size':32,
        'vald_batch_size':32,
        'test_batch_size':32,
        'num_workers':2,
        'epochs':50
    }
    trainer = svd_trainer.Trainer(dataset=dataset,model=model,optimizers=opt,critereon=loss,early_stop=30,hyper_params=hyper_params,verbose=False)
    model = trainer.train()
    


config={
    'lr':tune.grid_search([1e-2,1e-1,5e-2,5e-3]),
    'backend_encoder_lr':tune.grid_search([1e-4,1e-5]),

    'mlp_layers':[],
    'augmentations':tune.grid_search([
        [],["TimeInversion"]
    ]),
    'configuration':tune.grid_search(["base","large"]),
    'filter_letter':tune.grid_search(['a','i','u',None]),
    'filter_gender':tune.grid_search(['female','male',None]),
    'l2_reg':tune.grid_search([0.001,0.01,0]),
    # 'mlp_layers':[512]
    # 'activation':nn.LeakyReLU(negative_slope=0.01)
    "wandb": {"api_key": "19e347e092a58ca11a380ad43bd1fd5103f4d14a", "project": "VoiceDisorder","group":"HubertAblationStudy"},

    }
analysis = tune.run(train_model,config=config,resources_per_trial={'gpu':1},verbose=False,name="HubertAblationStudy",
callbacks=[WandbLoggerCallback(
        project="VoiceDisorder",
        api_key="19e347e092a58ca11a380ad43bd1fd5103f4d14a",
        log_config=True)]),
        



# config={
#     'lr':0.01,
#     'mlp_layers':[512,128]
# }
# train_model(config)

# with open('analysisFile','wb') as a_file:
#     pickle.dump(analysis,a_file)