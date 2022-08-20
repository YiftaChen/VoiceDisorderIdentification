from __future__ import print_function, division
import os
from random import sample
# from cv2 import transform
import torch
from torch.utils.data import Dataset
from scipy.io import wavfile
import torch.nn as nn
from architecture.backend.yamnet.model import Identity
from transformations import PadWhiteNoise,ToTensor,Truncate,ToOneHot,WaveformToInput,Inflate,Deflate,CFloat
import librosa
import random, sys
from core.params import CommonParams as cfg,PathologiesToIndex
# from torch_audiomentations import Compose, PitchShift,TimeInversion,AddBackgroundNoise
from  torchaudio.transforms import Spectrogram,TimeStretch, TimeMasking, FrequencyMasking, InverseSpectrogram,GriffinLim
import wandb
import numpy as np

def create_transformations(augmentations):
    print(augmentations)
    name_to_aug = {
        "TimeStretch":TimeStretch(fixed_rate=0.8),
        "FrequencyMasking":
        FrequencyMasking(
            freq_mask_param=80
        ),
        "TimeMasking":
            TimeMasking(time_mask_param=80),
    }

    transforms = [name_to_aug[augmentation] for augmentation in augmentations]
    transforms = [Spectrogram()]+ transforms + [CFloat(),InverseSpectrogram()]
    return nn.Sequential(*transforms)

default_label_transforms = nn.Sequential(ToOneHot())

def create_datasets(root_dir,split:tuple,hp,filter_gender=None,seed=None,**kwargs)->list():
    assert sum(split)==1, f"Splits fraction array should sum up to 1"
    split = np.cumsum(split)
    files_array = []
    if hp["filter_gender"] != None:
        root_dir=os.path.join(root_dir,hp["filter_gender"])
    for root, dirs, files in os.walk(root_dir):

        if kwargs['classification_binary']!=True:
            files_array += [os.path. join(root,f) for f in files if not f.startswith('.') and  f.endswith('.wav') and 'healthy' not in f.lower()]
        else:
            files_array += [os.path. join(root,f) for f in files if not f.startswith('.') and  f.endswith('.wav')]
    if kwargs['classification_binary']!=True:
        data = {}
        subjects = [(path,path.split('/')[-3],path.split('/')[-1].split('-')[0]) for path in files_array]
        for path,pathology,subject in subjects:
            if pathology == "Healthy":
                continue
            if subject not in data:
                data[subject]={}
            if pathology not in data[subject].keys():
                data[subject][pathology]=[path]
            else:
                data[subject][pathology]+=[path]
        files_array = [(list(sublist[1].items())[0][1],list(sublist[1].keys())) for sublist in data.items()]
        files_array =  [(item,tup[1]) for tup in files_array for item in tup[0]]


    if seed == None:    
        seed = random.randrange(sys.maxsize)
    random.Random(seed).shuffle(files_array)

    split = [int(s * len(data.keys()))for s in split][:-1]

    files_split = np.split(files_array, split)

    hp['seed']=seed
    assert False, f"files split {files_split}"
    splits = [SvdExtendedVoiceDataset(sp,hp,**kwargs) for sp in files_split]

    return splits

# removes duplicate recordings from the split.
# for example, if files_array_splits looks something like this (notice first 3 are the same recording):
# [('some/path/pathology-A/sub-pathology-AA/123-a_n.wav',['Pathology-A','Pathology-B']),
#  ('some/path/pathology-A/sub-pathology-AAA/123-a_n.wav',['Pathology-A','Pathology-B']),
#  ('some/path/pathology-B/sub-pathology-BB/123-a_n.wav',['Pathology-A','Pathology-B']),
#  ('some/path/pathology-C/sub-pathology-CC/777-a_n.wav',['Pathology-C'])]
#
# will result something like:
# [('some/path/pathology-A/sub-pathology-AA/123-a_n.wav',['Pathology-A','Pathology-B']),
#  ('some/path/pathology-C/sub-pathology-CC/777-a_n.wav',['Pathology-C])]
def remove_duplicated_from_files_array_splits(files_array_splits):    
    files_array_splits_with_names = [(os.path.split(data[0])[1],data) for data in files_array_splits[0]]
    file_names = []
    filtered_list = []
    for (file_name,data) in files_array_splits_with_names:
        if (file_name not in file_names):
            file_names.append(file_name)
            filtered_list.append(data)

    return filtered_list

def get_single_pathology_data(data):
    data_single_pathologies = {}    
    for subject in data:
        if len(data[subject].keys()) == 1:
            data_single_pathologies[subject] = data[subject]       
    
    return data_single_pathologies


def create_datasets_split_by_subjects(root_dir,split:tuple,hp,filter_gender=None,seed=None,only_single_pathology=False,**kwargs)->list():
    assert sum(split)==1, f"Splits fraction array should sum up to 1"    
    files_array = []
    if hp["filter_gender"] != None:
        root_dir=os.path.join(root_dir,hp["filter_gender"])
    for root, dirs, files in os.walk(root_dir):
        files_array += [os.path. join(root,f) for f in files if not f.startswith('.') and  f.endswith('.wav')]    

    if seed == None:    
        seed = random.randrange(sys.maxsize)
    random.Random(seed).shuffle(files_array)

    if kwargs['classification_binary']!=True:
        data = {}
        subjects = [(path,path.split('/')[-3],path.split('/')[-1].split('-')[0]) for path in files_array]
        for path,pathology,subject in subjects:
            if pathology == "Healthy":
                continue
            if subject not in data:
                data[subject]={}
            if pathology not in data[subject].keys():
                data[subject][pathology]=[path]
            else:
                data[subject][pathology]+=[path]    

    if only_single_pathology:
        data = get_single_pathology_data(data)    
    
    split = np.cumsum(split)
    split = [int(s * len(data.keys()))for s in split][:-1]
    subjects_split = np.split(list(data.keys()),split)
    files_array_splits = []
    for sp in subjects_split:
        files_split = [[(f,list(data[key].keys())) for key in sp for d in data[key] for f in data[key][d]]]

        # Im not sure if this is intended, but after this, files_array_splits will contain duplicates of the same recording if it has more than one pathology.
        # Also will contain duplicates if its the same pathology but in different sub-pathologies that were merged by Ariel.
        # So, for the time being I added this "fix"

        split_filtered = remove_duplicated_from_files_array_splits(files_split)
        files_array_splits += [split_filtered]

    hp['seed']=seed
    splits = [SvdExtendedVoiceDataset(sp,hp,**kwargs) for sp in files_array_splits]
    return splits

# TODO: make this inherit AudioFolderDataset
class SvdExtendedVoiceDataset(Dataset):
    """Saarbruken blah blah"""

    def __init__(self, files, hp,label_transform=default_label_transforms, class_definitions=None,classification_binary=True):        
    # audiomentations = create_transformations(hp['augmentations'])
        data_transform = nn.Sequential(ToTensor(),PadWhiteNoise(50000),Truncate(50000))
        
        self.data_transform = data_transform
        self.label_transform = label_transform
        self.classification_binary = classification_binary
        self.class_definitions=class_definitions if class_definitions!= None else PathologiesToIndex# Placeholder for actual definitions
        self.seed = hp['seed']
        self.files = files
            # assert len(files) == 0 or (len(files) != 0 and 
        assert len(self.files) > 0,f"Directory should not be empty, it is {self.files}"
    
    def _load_wav(self,wav_file):
        return wavfile.read(wav_file)
    def _get_class(self,wav_file_path):
        return self.class_definitions[wav_file_path.split('/')[-3]],  wav_file_path.split('/')[-3]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()
        if isinstance(index,list):
            index = index[0]
        # assert False, self.files[index]
        if self.classification_binary:
            samplerate, data = self._load_wav(self.files[index])
            classification_index,classification = self._get_class(self.files[index])
        else:
            samplerate, data = self._load_wav(self.files[index][0])
            classification = self.files[index][1]
            # assert False,f"classification {classification}"
            # if "Healthy" in classification:
            #     assert False, f"fuck, classification {classification}"
            classification_index = [self.class_definitions[index] for index in self.files[index][1]]
        if self.data_transform != None:
            data = self.data_transform(data)
        if self.label_transform != None and not self.classification_binary:
            label = self.label_transform(classification_index)
        if self.classification_binary:
            label = classification_index!=0
        return {'data':data, 'sampling_rate':samplerate,'classification':label}

class SvdCutOffShort(SvdExtendedVoiceDataset):
    """Saarbruken blah blah, cut off samples smaller than 0.96"""
    def __init__(self, files, hp,label_transform=default_label_transforms, class_definitions=None,classification_binary=True,overfit_test = False):
        super().__init__(files,hp,label_transform,class_definitions,classification_binary)
        import random
        self.files = [file for file in self.files if librosa.get_duration(filename=file)>=cfg.VOICE_SAMPLE_MIN_LENGTH]
        random.shuffle(self.files)
        if overfit_test:
            self.files = self.files[:40]


class SvdWindowedDataset(SvdExtendedVoiceDataset):
    """Saarbruken blah blah, cut off samples smaller than 0.96"""
    def __init__(self, files, hp,label_transform=default_label_transforms, class_definitions=None,classification_binary=True,overfit_test = False,delta=1):
        super().__init__(files,hp,label_transform,class_definitions,classification_binary)
        import random
        def _filter_pitch(filename):
            if hp["filter_pitch"] != None:
                return filename.split("_")[1].split(".")[0] in hp["filter_pitch"]
            return True
        def _filter_sound(filename):
            if hp["filter_letter"] != None:
                # assert False, f"filename split {filename.split('_')}"
                return filename.split("_")[0].split("-")[1] in hp["filter_letter"]
            return True
        self.delta=delta
        self.files = [file for file in self.files if _filter_sound(file) and _filter_pitch(file)]
        self.files = self._inflate_sound_files(self.files)
        
        if overfit_test:
            random.shuffle(self.files)
            self.files = self.files[:40]
    
    def _load_wav(self,wav_file):
        window_index = wav_file["window_index"]
        file_path = wav_file["path"]
        sample_rate,data = wavfile.read(file_path)
        start_index = int(self.delta*window_index*cfg.SVD_SAMPLE_RATE)
        end_index = int(self.delta*(window_index+1)*cfg.SVD_SAMPLE_RATE)
        return sample_rate,data[start_index:end_index]
    def _get_class(self,wav_file):
        wav_file_path = wav_file["path"]
        return self.class_definitions[wav_file_path.split('/')[-3]], wav_file_path.split('/')[-3]

    def _inflate_sound_files(self,files):
        def get_window_unt(f):
            length = librosa.get_duration(filename=f)*cfg.SVD_SAMPLE_RATE
            length = 0 if length-cfg.SVD_SAMPLE_RATE<0 else length-cfg.SVD_SAMPLE_RATE
            return int(length/(self.delta*cfg.SVD_SAMPLE_RATE))+1
        return [{'path':file,'window_index':i} for file in files for i in range(get_window_count(file))]
        
if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    hp = {}
    hp["augmentations"] = None
    hp["filter_pitch"] = None
    hp["filter_letter"] = None
    hp["filter_gender"] = None
    sets = create_datasets_split_by_subjects(r"/home/yiftach.ede@staff.technion.ac.il/data/SVD",split=(0.8,0.1,0.1),hp=hp,filter_gender=None,classification_binary=False)
    loader =  DataLoader(
            sets[0],
            batch_size=20,
            shuffle=True,
            num_workers=1
        )    
    for item in loader:
        print(item)
        pass
    # print(sets[0][0])
