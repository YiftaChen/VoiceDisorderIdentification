import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from transformations.transform import WaveformToInput as TorchTransform
from architecture.backend.yamnet.params import YAMNetParams
from architecture.backend.yamnet.model import yamnet
from architecture.backend.yamnet.model import yamnet_category_metadata

from architecture.classifier.classification import Wav2Vec2Classifier,HubertClassifier

from datasets.SvdExDataset import create_datasets
import trainer.train_svd as svd_trainer

import torch.nn as nn
import torch.optim 
import socket
import core


def test_model(model_id):
    model_folder = f"/home/yiftach.ede@staff.technion.ac.il/VoiceDisorderIdentification/checkpoints/{model_id}"
    checkpoint = torch.load(f"{model_folder}/accuracy_based_model.pt")
    model = HubertClassifier([])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device="cuda:0")
    directory = core.params.dataset_locations[socket.gethostname()]
    config = {'filter_gender':None, 'lr':0}
    datasets = create_datasets(directory,split=(0.8,0.1,0.1),hp=config,filter_gender=config['filter_gender'],seed=checkpoint['seed'])
    opt = torch.optim.Adam(params=model.parameters(),lr=config["lr"])
    hyper_params = {
            'train_batch_size':128,
            'vald_batch_size':128,
            'test_batch_size':128,
            'num_workers':2,
            'epochs':200,
            'checkpoints':"/home/yiftach.ede@staff.technion.ac.il/VoiceDisorderIdentification/checkpoints",
            'name': model_id

    }

    # }    # dataset = SvdExtendedVoiceDataset(r"/home/yiftach.ede@staff.technion.ac.il/Desktop/SVD",hp = config,classification_binary=True)
    loss = nn.BCEWithLogitsLoss()
    trainer = svd_trainer.Trainer(datasets=datasets,model=model,optimizers=opt,critereon=loss,hyper_params=hyper_params,verbose=False)
    trainer.test(model)
    # loss = nn.BCEWithLogitsLoss()
    # # params_non_frozen = filter(lambda p: p.requires_grad, model.parameters())
    # opt = torch.optim.Adam(
    #     [
    #         dict(params=model.classifier.parameters()),
    #         # dict(params=model.backend.layer13.parameters(),lr=config['lr']*0.001),
    #         # dict(params=model.backend.layer12.parameters(),lr=config['lr']*0.01),
    #         # dict(params=model.backend.layer11.parameters(),lr=config['lr']*0.01),
    #     ]
    #     ,lr=config["lr"])
    # hyper_params = {
    #     'train_batch_size':128,
    #     'vald_batch_size':128,
    #     'test_batch_size':128,
    #     'num_workers':2,
    #     'epochs':200
    # }
    # trainer = svd_trainer.Trainer(dataset=dataset,model=model,optimizers=opt,critereon=loss,hyper_params=hyper_params,verbose=False)
    # model = trainer.train()
    

test_model("ConvolutionalClassificationHead00000")