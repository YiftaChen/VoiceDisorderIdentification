from datasets.AudioFolderDataset import AudioFolderDataset
import os
import torch.nn as nn
from transformations import ToTensor,Truncate,ToOneHot,WaveformToInput,PadWhiteNoise


default_data_trans = nn.Sequential(ToTensor(),PadWhiteNoise(70000),Truncate(70000),WaveformToInput())

class SVDBinaryDataset(AudioFolderDataset):
    def __init__(self, root_dir,data_trans=default_data_trans):
        super().__init__(root_dir,data_trans)        

    def getClassificationByFileName(self, fileName: str) -> int:       
        return int('Healthy' not in fileName) # 0 - healthy, 1 - pathological

    def fileNameFilter(self, fileName):
        return '-a_l.' in fileName or '-a_h.' in fileName or '-a_n.' in fileName


