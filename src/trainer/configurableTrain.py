import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from transformations.transform import WaveformToInput as TorchTransform
from architecture.backend.yamnet.params import YAMNetParams
from architecture.backend.yamnet.model import yamnet
from architecture.backend.yamnet.model import yamnet_category_metadata

from architecture.classifier.classification import YamnetClassifier,Wav2Vec2Classifier,DistilHUBERTClassifier,HubertClassifier
from datasets.SvdExDataset import create_datasets
import trainer.train_svd as svd_trainer
import torch.nn as nn
import torch.optim 
from ray import tune
import ray
import pickle
import socket
import core

from ray.tune.integration.wandb import (
    WandbLoggerCallback,
    WandbTrainableMixin,
    wandb_mixin,
)
import wandb
from itertools import chain, combinations



count = 0
@wandb_mixin
def train_model(config):
    run_name =  f"ConvolutionalClassificationHead{wandb.run.name.split('_')[-1]}"
    wandb.run.name = run_name
    wandb.run.save()
    torch.autograd.set_detect_anomaly(True)
    directory = core.params.dataset_locations[socket.gethostname()]

    datasets = create_datasets(directory,split=(0.8,0.1,0.1),hp=config,filter_gender=config['filter_gender'])

    # torch.multiprocessing.set_start_method('spawn')

    model = HubertClassifier(config["mlp_layers"]).to(device="cuda:0")  
    print(model)
    loss = nn.BCEWithLogitsLoss()
    # params_non_frozen = filter(lambda p: p.requires_grad, model.parameters())
    # assert False, f"model params {model}"

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
        'train_batch_size':200,
        'vald_batch_size':200,
        'test_batch_size':200,
        'num_workers':2,
        'epochs':50,
        'checkpoints':config['checkpoints'],
        'name': run_name
    }
    trainer = svd_trainer.Trainer(datasets=datasets,model=model,optimizers=opt,critereon=loss,early_stop=200,hyper_params=hyper_params,verbose=False)
    model = trainer.train()
    
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))


config={
    'lr':tune.grid_search([1e-2,1e-3,2.45e-2,2.45e-3]),
    'backend_encoder_lr':tune.grid_search([1e-4,1e-5]),
        'augmentations':None,
    'mlp_layers':[],
    # 'delta':tune.grid_search([10]),
    'configuration':tune.grid_search(["base","large"]),
    'filter_letter':None,
    'filter_pitch':None,
    # 'filter_letter':tune.grid_search(list(powerset(["a","i","u"]))),
    # 'filter_pitch':tune.grid_search(list(powerset(["l","n","lhl","h"]))),
    'filter_gender':tune.grid_search(['female','male',None]),
    'l2_reg':tune.grid_search([0.001,0.01,0]),
    # 'mlp_layers':[512]
    # 'activation':nn.LeakyReLU(negative_slope=0.01)
    "wandb": {"api_key": "19e347e092a58ca11a380ad43bd1fd5103f4d14a", "project": "VoiceDisorder","group":"ConvolutionalClassificationHead"},
    "checkpoints": core.params.project_dir[socket.gethostname()] + "/checkpoints"
    }
            

chosen_config={
    'lr':1e-3,
    'backend_encoder_lr':1e-5,

    'mlp_layers':[],    
    'augmentations':[],
    'configuration':'base',
    'filter_letter':'a',
    'filter_pitch':["n","lhl"],
    'filter_gender':tune.grid_search(["male","female"]),
    'l2_reg':0.001,
    # 'mlp_layers':[512]
    # 'activation':nn.LeakyReLU(negative_slope=0.01)
    "wandb": {"api_key": "19e347e092a58ca11a380ad43bd1fd5103f4d14a", "project": "VoiceDisorder","group":"ChosenConfigTest"},

    }
# ray.init(address="132.68.58.49:6123")

# simple_run_config={
#     'lr':1e-3,
#     'backend_encoder_lr':1e-5,

#     'mlp_layers':[],    
#     'augmentations':[],
#     'configuration':'base',
#     'filter_letter':'a',
#     'filter_pitch':["n","lhl"],
#     'filter_gender':tune.grid_search(["male","female"]),
#     'l2_reg':0.001,
#     # 'mlp_layers':[512]
#     # 'activation':nn.LeakyReLU(negative_slope=0.01)
#     "wandb": {"api_key": "19e347e092a58ca11a380ad43bd1fd5103f4d14a", "project": "VoiceDisorder","group":"SimpleRunNoGridSearch"}
# }

analysis = tune.run(train_model,config=config,resources_per_trial={'gpu':1},verbose=False,name="ConvolutionalClassificationHead",
callbacks=[WandbLoggerCallback(
        project="VoiceDisorder",
        api_key="19e347e092a58ca11a380ad43bd1fd5103f4d14a",
        log_config=True)])
        



# config={
#     'lr':0.01,
#     'mlp_layers':[512,128]
# }
# train_model(config)

# with open('analysisFile','wb') as a_file:
#     pickle.dump(analysis,a_file)