import argparse
from importlib.resources import path
import os 
import json 
import collections
from isort import file
# from pyrsistent import T
from scipy.io import wavfile
import scipy.io
import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf 

def sf_load_from_int16(fname):
    x, sr = sf.read(fname, dtype='int16', always_2d=False)
    x = x / 2 ** 15
    x = x.T.astype(np.float32)
    return x, sr



def iterate_files(path):
    children = []
    for root, _ , files in tqdm(os.walk(path)):
        files = list([f for f in files])
        files = [os.path.join(root,f) for f in files if f.endswith('wav') and not f.endswith('reversed.wav')]
        waveforms_files = [(f,sf_load_from_int16(f)) for f in files]
        for f,(waveform,sr) in waveforms_files:
            filename = f.split('.')[0]
            sf.write(f"{filename}_reversed.wav",waveform[::-1],sr)
        # children+=[os.path.join(root,f) for f in files if f.endswith('wav')]
    # return children

def consolidate_files(children):
    children = {child.split('/')[-1]: child for child in children}
    return list(children.values())

if __name__=="__main__":
    print(iterate_files("/Users/yiftachedelstain/Development/VoiceDisorderIdentification/data/SVD - Extended"))