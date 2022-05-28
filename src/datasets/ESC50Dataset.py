from datasets.AudioFolderDataset import AudioFolderDataset
import os
import torch.nn as nn
from transformations import ToTensor,Truncate,ToOneHot,WaveformToInput,PadWhiteNoise


# data_trans = nn.Sequential(ToTensor(),PadWhiteNoise(70000),Truncate(70000),WaveformToInput())
default_data_trans = nn.Sequential(ToTensor(),PadWhiteNoise(70000),Truncate(70000),WaveformToInput())


class ESC50Dataset(AudioFolderDataset):
    def __init__(self, root_dir,data_trans=default_data_trans):
        super().__init__(root_dir,data_trans)

    def getClassificationByFileName(self, fileName: str) -> int:
        return int('Healthy' in fileName)
        str = os.path.basename(fileName).split('.')[0].split('-')[3]
        classification=int(str)
        return classification


