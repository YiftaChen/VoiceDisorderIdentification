import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from transformations.transform import WaveformToInput as TorchTransform
from architecture.backend.yamnet.params import YAMNetParams
from architecture.backend.yamnet.model import yamnet
from architecture.backend.yamnet.model import yamnet_category_metadata

from architecture.classifier.classification import YamnetClassifier,Wav2Vec2Classifier,HubertMulticlassClassifier,HubertClassifier
from datasets.SvdExDataset import create_datasets_split_by_subjects
import trainer.InclusiveMulticlassTrainer as svd_trainer
import torch.nn as nn
import torch.optim 
from ray import tune
import ray
import pickle
import socket
import core
from tqdm import tqdm

from ray.tune.integration.wandb import (
    WandbLoggerCallback,
    WandbTrainableMixin,
    wandb_mixin,
)
import wandb
from itertools import chain, combinations

def get_metadata_of_dataset(dataset):
    bins = torch.zeros(11)
    size = 0
    with tqdm(dataset) as pbar:                        
        for sample in pbar:
            bins += sample['classification']
            size += 1

    return bins,size


count = 0
@wandb_mixin
def train_model(config):
    run_name =  f"ConvMulticlassClassificationHeadNull"    
    torch.autograd.set_detect_anomaly(True)
    directory = core.params.dataset_location
    
    datasets = create_datasets_split_by_subjects(directory,split=(0.8,0.1,0.1),hp=config,filter_gender=config['filter_gender'],classification_binary=config['binary_classification'])

    train_dataset = datasets[0]    
    train_data = get_metadata_of_dataset(train_dataset)
    print('train_data')
    print(train_data)

    valid_dataset = datasets[1]
    valid_data = get_metadata_of_dataset(valid_dataset)
    print('valid_data')
    print(valid_data)

    # bins = get_weight_of_classifications_in_dataset(train_dataset)
    # print(bins)

    model = HubertMulticlassClassifier(config).to(device="cuda:0")  
    # print(model)

    # loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1000]*11))
    loss = None   

    opt = torch.optim.Adam(
        [
            dict(params=model.classifier.parameters()),
            dict(params=model.model.encoder.parameters(),lr=config['backend_encoder_lr']),          
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
    trainer = svd_trainer.MulticlassTrainer(datasets=datasets,model=model,optimizer=opt,early_stop=200,hyper_params=hyper_params,verbose=False,criterion=loss)
    model = trainer.train()
    
config={
    'binary_classification':False,
    'lr':1e-3,
    'backend_encoder_lr':1e-4,
    'augmentations':None,
    'mlp_layers':[],
    # 'delta':tune.grid_search([10]),
    'configuration':"base",
    'filter_letter':None,
    'filter_pitch':None,
    # 'classification_inner_dim':tune.grid_search([1,64]),
    # 'classification_head_strides':tune.grid_search([(1,2)]),
    # 'classification_head_kernels':tune.grid_search([[(7,7)]*5,[(7,7)]*3,[(7,7)]*4,[(5,5)]*5,[(5,5)]*4,[(5,5)]*3,[(3,3)]*5,[(3,3)]*4,[(3,3)]*3]),
    'filter_gender':None,
    'l2_reg':0,

    # "wandb": {"api_key": "19e347e092a58ca11a380ad43bd1fd5103f4d14a", "project": "VoiceDisorder","group":"ConvMulticlassClassificationHead"},
    "checkpoints": core.params.checkpoints_dir
    }
     

train_model(config)