from __future__ import print_function, division
from logging import root
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
import scipy.io
from tqdm import tqdm
from torchaudio.transforms import ToTensor
# from transform import ToTensor

class SvdExtendedVoiceDataset(Dataset):
    """Saarbruken blah blah"""

    def __init__(self, root_dir, data_transform=None,label_transform=None, class_definitions:dict=None):
        self.root_dir = root_dir
        self.data_transform = data_transform
        self.label_transform = label_transform
        self.class_definitions=class_definitions if class_definitions!= None else {'Leukoplakie':1,'Kehlkopftumor':2,'Stimmlippenkarzinom':3,'GERS':4,'Kontaktpachydermie':5} # Placeholder for actual definitions
        self.files = []
        for root, dirs, files in os.walk(root_dir):
            self.files += [os.path.join(root,f) for f in files if f.endswith('.wav')]
            # assert len(files) == 0 or (len(files) != 0 and 
        assert len(self.files) > 0,"Directory should not be empty"
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()
        if isinstance(index,list):
            index = index[0]
        samplerate, data = wavfile.read(self.files[index])
        classification = self.class_definitions[self.files[index].split('/')[-2]]
        
        if self.data_transform != None:
            data = self.data_transform(data)
        if self.label_transform != None:
            label = self.label_transform(classification)
        return {'data':data, 'sampling_rate':samplerate,'classification':classification}

# def forward(self,sample):
#     data,sample_rate,label = sample...

if __name__ == "__main__":
    dataset = SvdExtendedVoiceDataset(r"/Users/yiftachedelstain/Development/Data",data_transform=ToTensor())
    for idx,item in enumerate(tqdm(dataset)):
        # print(item)
        print(item)