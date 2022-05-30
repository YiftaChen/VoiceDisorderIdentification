from __future__ import print_function, division
import os
from random import sample
import torch
from torch.utils.data import Dataset
from scipy.io import wavfile
import torch.nn as nn
from architecture.backend.yamnet.model import Identity
from transformations import PadWhiteNoise,ToTensor,Truncate,ToOneHot,WaveformToInput,Inflate,Deflate
import librosa
from core.params import CommonParams as cfg,PathologiesToIndex
from torch_audiomentations import Compose, PitchShift,TimeInversion,AddBackgroundNoise





def create_transformations(augmentations):
    print(augmentations)
    name_to_aug = {
        "PitchShift":PitchShift(
            p=0.5,
            p_mode="per_example",
            mode="per_example",
            sample_rate=50000,
            min_transpose_semitones = -4.0,
            max_transpose_semitones = 4.0,
        ),
        "TimeInversion":
        TimeInversion(
            p=0.5, 
            p_mode="per_example",
            mode="per_example"
        ),
    }
    transforms = [name_to_aug[augmentation] for augmentation in augmentations]
    return Compose(transforms)

# WaveformToInput())
default_label_transforms = nn.Sequential(ToOneHot())

class SvdExtendedVoiceDataset(Dataset):
    """Saarbruken blah blah"""

    def __init__(self, root_dir, hp,label_transform=default_label_transforms, class_definitions=None,classification_binary=True):        
        self.root_dir = root_dir
        audiomentations = create_transformations(hp['augmentations'])
        data_transform = nn.Sequential(ToTensor(),Inflate(),audiomentations,Deflate(),PadWhiteNoise(70000),Truncate(70000))

        self.data_transform = data_transform
        self.label_transform = label_transform
        self.classification_binary = classification_binary
        self.class_definitions=class_definitions if class_definitions!= None else PathologiesToIndex# Placeholder for actual definitions
        self.files = []
        for root, dirs, files in os.walk(root_dir):
            self.files += [os.path.join(root,f) for f in files if not f.startswith('.') and  f.endswith('.wav')]
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
        classification = self.class_definitions[self.files[index].split('/')[-3]]
        
        if self.data_transform != None:
            data = self.data_transform(data)
        if self.label_transform != None and not self.classification_binary:
            label = self.label_transform(classification)
        if self.classification_binary:
            label = classification!=0
        return {'data':data, 'sampling_rate':samplerate,'classification':label}

class SvdCutOffShort(SvdExtendedVoiceDataset):
    """Saarbruken blah blah, cut off samples smaller than 0.96"""
    def __init__(self, root_dir, hp,label_transform=default_label_transforms, class_definitions=None,classification_binary=True,overfit_test = False):
        super().__init__(root_dir,hp,label_transform,class_definitions,classification_binary)
        import random
        self.files = [file for file in self.files if librosa.get_duration(filename=file)>=cfg.VOICE_SAMPLE_MIN_LENGTH]
        random.shuffle(self.files)
        if overfit_test:
            self.files = self.files[:40]


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    hp = {}
    hp["augmentations"]=["PitchShift","TimeInversion"]
    dataset = SvdExtendedVoiceDataset(r"/Users/yiftachedelstain/Development/VoiceDisorderIdentification/data/SVD - Extended/",hp,classification_binary=True)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

    for idx,item in enumerate(tqdm(loader)):
        print(item['data'].shape) 
