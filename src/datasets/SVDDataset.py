from datasets.AudioFolderDataset import AudioFolderDataset
import os
import torch.nn as nn
from transformations import ToTensor,Truncate,ToOneHot,WaveformToInput,PadWhiteNoise
from core.params import PathologiesToIndex


default_data_trans = nn.Sequential(ToTensor(),PadWhiteNoise(70000),Truncate(70000),WaveformToInput())

class SVDDataset(AudioFolderDataset):
    def __init__(self, root_dir,data_trans=default_data_trans):
        super().__init__(root_dir,data_trans)        

    def getClassificationByFileName(self, fileName: str) -> int:           
        classification_str = fileName.split('/')[-3]
        return PathologiesToIndex[classification_str]

    def fileNameFilter(self, fileName):
        return '-a_l.' in fileName or '-a_h.' in fileName or '-a_n.' in fileName


