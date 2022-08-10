import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from transformations.transform import WaveformToInput as TorchTransform
from architecture.backend.yamnet.params import YAMNetParams
from architecture.backend.yamnet.model import yamnet
from architecture.backend.yamnet.model import yamnet_category_metadata

from architecture.classifier.classification import YamnetClassifier,Wav2Vec2Classifier,HubertMulticlassClassifier,HubertClassifier
from datasets.SvdExDataset import create_datasets
import trainer.InclusiveMulticlassTrainer as svd_trainer
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
    run_name =  f"ConvMulticlassClassificationHead{wandb.run.name.split('_')[-1]}"
    wandb.run.name = run_name
    wandb.run.save()
    torch.autograd.set_detect_anomaly(True)
    directory = core.params.dataset_locations[socket.gethostname()]
    
    datasets = create_datasets(directory,split=(0.8,0.1,0.1),hp=config,filter_gender=config['filter_gender'],classification_binary=config['binary_classification'])

    model = HubertMulticlassClassifier(config).to(device="cuda:0")  
    # print(model)
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
        'epochs':100,
        'checkpoints':config['checkpoints'],
        'name': run_name,
    }
    trainer = svd_trainer.MulticlassTrainer(datasets=datasets,model=model,optimizer=opt,early_stop=200,hyper_params=hyper_params,verbose=False)
    model = trainer.train()
    
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))

    #    hp['classification_head_kernels'] hp['classification_head_strides'])
config={
    'binary_classification':False,
    'lr':tune.grid_search([1e-2,1e-3,2.45e-2,2.45e-3]),
    'backend_encoder_lr':tune.grid_search([1e-4,1e-5]),
        'augmentations':None,
    'mlp_layers':[],
    # 'delta':tune.grid_search([10]),
    'configuration':tune.grid_search(["base"]),
    'filter_letter':None,
    'filter_pitch':None,
    # 'classification_inner_dim':tune.grid_search([1,64]),
    # 'classification_head_strides':tune.grid_search([(1,2)]),
    # 'classification_head_kernels':tune.grid_search([[(7,7)]*5,[(7,7)]*3,[(7,7)]*4,[(5,5)]*5,[(5,5)]*4,[(5,5)]*3,[(3,3)]*5,[(3,3)]*4,[(3,3)]*3]),
    'lstm_layer_count':tune.grid_search([1]),
    'hidden_dim':tune.grid_search([1]),
    'bidirectional':tune.grid_search([False]),
    'dropout':tune.grid_search([0]),
    'filter_gender':tune.grid_search([None]),
    'l2_reg':tune.grid_search([0]),

    "wandb": {"api_key": "19e347e092a58ca11a380ad43bd1fd5103f4d14a", "project": "VoiceDisorder","group":"ConvMulticlassClassificationHead"},
    "checkpoints":r"/home/yiftach.ede/VoiceDisorderIdentification/checkpoints"
    }
            


# ray.init(address="132.68.58.49:6123")

analysis = tune.run(train_model,config=config,resources_per_trial={'gpu':1},verbose=False,name="ConvolutionalClassificationHeadGridSearch",
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