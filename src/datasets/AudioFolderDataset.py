from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset
from scipy.io import wavfile
import torch.nn as nn


class AudioFolderDataset(Dataset):
    def __init__(self, root_dir, data_transform = None, label_transform = None):
        self.root_dir = root_dir               
        self.files = []
        self.data_transform = data_transform
        self.label_transform = label_transform
        for root, dirs, files in os.walk(root_dir):
            self.files += [os.path.join(root,f) for f in files if not f.startswith('.') and  f.endswith('.wav')]
       
        assert len(self.files) > 0,"Directory should not be empty"
    def __len__(self):
        return len(self.files)

    def getClassificationByFileName(self,fileName):
        raise Exception("logic for getting classification by file name is required")

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()
        if isinstance(index,list):
            index = index[0]
        samplerate, data = wavfile.read(self.files[index])
        label = self.getClassificationByFileName(self.files[index])     

        if self.data_transform != None:
            data = self.data_transform(data)
        if self.label_transform != None:
            label = self.label_transform(label)

        return {'data':data, 'sampling_rate':samplerate,'classification':label}
