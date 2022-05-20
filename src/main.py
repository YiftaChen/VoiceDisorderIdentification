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
import trainer.train_svd 
import torch.nn as nn
import torch.optim 

dataset = SvdCutOffShort(r"/Users/yiftachedelstain/Development/VoiceDisorderIdentification/data/SVD - Extended",classification_binary=True,overfit_test = False)
model = Classifier([512])
loss = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters())
hyper_params = {
    'train_batch_size':128,
    'vald_batch_size':128,
    'test_batch_size':128,
    'num_workers':2,
    'epochs':6
}
trainer = trainer.train_svd.Trainer(dataset=dataset,model=model,optimizers=opt,critereon=loss,hyper_params=hyper_params)
model = trainer.train()
trainer.test(model)