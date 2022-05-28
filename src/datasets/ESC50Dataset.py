from datasets.AudioFolderDataset import AudioFolderDataset
import os
import torch.nn as nn
from transformations import ToTensor,Truncate,ToOneHot,WaveformToInput


data_trans = nn.Sequential(ToTensor(),Truncate(44100),WaveformToInput())
#label_trans = nn.Sequential(ToOneHot(num_classes=50))

class ESC50Dataset(AudioFolderDataset):
    def __init__(self, root_dir):
        super().__init__(root_dir,data_trans)

    def getClassificationByFileName(self, fileName: str) -> int:
        str = os.path.basename(fileName).split('.')[0].split('-')[3]
        classification=int(str)
        return classification


