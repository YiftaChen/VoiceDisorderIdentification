import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from transformations.transform import WaveformToInput as TorchTransform
from architecture.backend.yamnet.params import YAMNetParams
from architecture.backend.yamnet.model import yamnet
from architecture.backend.yamnet.model import yamnet_category_metadata

from architecture.classifier.classification import YamnetClassifier,Wav2Vec2Classifier,HubertMulticlassClassifier,HubertMulticlassSingleLabelClassifier,HubertClassifier
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

def get_metadata_of_dataset(dataset,is_data_oneHot=True):
    bins = torch.zeros(11)
    size = 0
    with tqdm(dataset) as pbar:                        
        for sample in pbar:
            if is_data_oneHot:
                bins += sample['classification']
            else:
                bins[sample['classification']] += 1
            size += 1
    
    return bins,size

def get_pos_weights(pos_bins_in_dataset, amount_of_samples_in_dataset):
    neg_bins_in_dataset = amount_of_samples_in_dataset - pos_bins_in_dataset
    pos_weights = neg_bins_in_dataset / (pos_bins_in_dataset + 1e-7)
    return pos_weights

def train_singleLabel_model(config):
    run_name =  f"ConvMulticlassClassificationHeadNull"    
    torch.autograd.set_detect_anomaly(True)
    directory = core.params.dataset_location
    
    datasets = create_datasets_split_by_subjects(directory,split=(0.8,0.1,0.1),hp=config,\
        filter_gender=config['filter_gender'],classification_binary=config['binary_classification'],only_single_pathology=True,oneHot=False)
 
    train_dataset = datasets[0]    
    train_data_bins,train_data_size = get_metadata_of_dataset(train_dataset,is_data_oneHot=False)
    print('train_data_bins')
    print(train_data_bins)
    print('train_data_size')
    print(train_data_size)

    valid_dataset = datasets[1]
    valid_data_bins,valid_data_size = get_metadata_of_dataset(valid_dataset,is_data_oneHot=False)
    print('valid_data_bins')
    print(valid_data_bins)
    print('valid_data_size')
    print(valid_data_size)

    model = HubertMulticlassSingleLabelClassifier(config, freeze_backend_grad=False,num_classes=11).to(device="cuda:0")         

    train_weights = (train_data_size/(train_data_bins + 1e-7)).to(device="cuda:0")         

    print('train_weights')
    print(train_weights)

    loss = nn.CrossEntropyLoss(weight=train_weights)   

    opt = torch.optim.Adam(
        [
            dict(params=model.classifier.parameters()),
            dict(params=model.model.encoder.parameters(),lr=config['backend_encoder_lr']),          
        ]
        ,lr=config["lr"],weight_decay = config['l2_reg'])
    hyper_params = {
        'train_batch_size':128,
        'vald_batch_size':128,
        'test_batch_size':128,
        'num_workers':2,
        'epochs':100,
        'checkpoints':config['checkpoints'],
        'name': run_name,
    }
    trainer =\
     svd_trainer.MultiClassSingleLabelTrainer(datasets=datasets,model=model,optimizer=opt,\
        early_stop=200,hyper_params=hyper_params,verbose=True,criterion=loss,confusion_mat_every=1,classes_amount=10)    
    model = trainer.train()

def train_multiLabel_model(config):
    run_name =  f"ConvMulticlassClassificationHeadNull"    
    torch.autograd.set_detect_anomaly(True)
    directory = core.params.dataset_location
    
    datasets = create_datasets_split_by_subjects(directory,split=(0.8,0.1,0.1),hp=config,\
        filter_gender=config['filter_gender'],classification_binary=config['binary_classification'],only_single_pathology=False)
    

    train_dataset = datasets[0]    
    train_data_bins,train_data_size = get_metadata_of_dataset(train_dataset)
    print('train_data_bins')
    print(train_data_bins)
    print('train_data_size')
    print(train_data_size)

    valid_dataset = datasets[1]
    valid_data_bins,valid_data_size = get_metadata_of_dataset(valid_dataset)
    print('valid_data_bins')
    print(valid_data_bins)
    print('valid_data_size')
    print(valid_data_size)

    # bins = get_weight_of_classifications_in_dataset(train_dataset)
    # print(bins)

    model = HubertMulticlassClassifier(config, freeze_backend_grad=False).to(device="cuda:0")  
    # print(model)

    pos_weights = get_pos_weights(train_data_bins,train_data_size).to(device="cuda:0")
    print('pos_weights')
    print(pos_weights)

    pos_weights_valid = get_pos_weights(valid_data_bins,valid_data_size).to(device="cuda:0")
    print('pos_weights_valid')
    print(pos_weights_valid)
    

    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    # loss = nn.BCEWithLogitsLoss()   

    opt = torch.optim.Adam(
        [
            dict(params=model.classifier.parameters()),
            dict(params=model.model.encoder.parameters(),lr=config['backend_encoder_lr']),          
        ]
        ,lr=config["lr"],weight_decay = config['l2_reg'])
    hyper_params = {
        'train_batch_size':128,
        'vald_batch_size':128,
        'test_batch_size':128,
        'num_workers':2,
        'epochs':100,
        'checkpoints':config['checkpoints'],
        'name': run_name,
    }
    trainer =\
     svd_trainer.MulticlassTrainer(datasets=datasets,model=model,optimizer=opt,\
        early_stop=200,hyper_params=hyper_params,verbose=True,criterion=loss,confusion_mat_every=1)    
    model = trainer.train()


count = 0
@wandb_mixin
def train_model(config):
    # train_multiLabel_model(config)
    train_singleLabel_model(config)

    
config={
    'binary_classification':False,
    'lr':1e-3,
    'backend_encoder_lr':1e-4,
    'augmentations':None,
    'mlp_layers':[],    
    'configuration':"base",
    'filter_letter':None,
    'filter_pitch':None,   
    'filter_gender': None,
    'l2_reg':0,

    # "wandb": {"api_key": "19e347e092a58ca11a380ad43bd1fd5103f4d14a", "project": "VoiceDisorder","group":"ConvMulticlassClassificationHead"},
    "checkpoints": core.params.checkpoints_dir
    }
     

train_model(config)