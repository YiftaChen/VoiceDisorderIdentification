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

def iterate_filesize_pathology_structure(path):
    try:
        path = str(path, 'utf-8')
    except:
        pass
    if str(path).endswith('wav'):
        wav=librosa.load(path)
        return (path.split('/')[-1].split('-')[0],path.split('/')[-1].split('-')[1].split('.')[0],librosa.get_duration(y=wav[0]))
    children = [os.path.join(path,child) for child in os.listdir(path) if child!='.DS_Store']
    children = [iterate_filesize_pathology_structure(child) for child in children]

    if len(children)== 0:
        return children
    if isinstance(children[0],tuple):
        children = list(set(children))
        pathology = path.split('/')[-1]
        return [(child,pathology) for child in children]
    children = [child for sublist in children for child in sublist]
    return children


def iterate_pathology_structure(path):
    try:
        path = str(path, 'utf-8')
    except:
        pass

    if str(path).endswith('wav'):
        return path.split('/')[-1].split('-')[0]
    children = [os.path.join(path,child) for child in os.listdir(path) if child!='.DS_Store']
    children = [iterate_pathology_structure(child) for child in children]
    if len(children)== 0:
        return children
    if isinstance(children[0],str):
        children = list(set(children))
        pathology = path.split('/')[-1]
        return [(child,pathology) for child in children]
    children = [child for sublist in children for child in sublist]
    return children

def iterate_pathology_to_subpathology_structure(path):
    try:
        path = str(path, 'utf-8')
    except:
        pass
    if len(os.listdir(path))== 0:
        return None
    if str(os.listdir(path)[0]).endswith('wav'):
        return path.split('/')[-1]
    children = [os.path.join(path,child) for child in os.listdir(path) if child!='.DS_Store']
    children = [iterate_pathology_to_subpathology_structure(child) for child in children]
    children = [child for child in children if child!=None]
    if isinstance(children[0],str):
        children = list(set(children))
        pathology = path.split('/')[-1]
        return [(child,pathology) for child in children]
    children = [child for sublist in children for child in sublist]
    return children


def consolidate_subjects(subjects):
    subj_dict = {}
    pathology_dict = {}
    for (subject,pathology) in subjects:
        # print(subject)
        if subject[0] not in subj_dict:
            subj_dict[subject]=[]
        if pathology not in pathology_dict:
            pathology_dict[pathology]=[]
        subj_dict[subject] += [pathology]
        pathology_dict[pathology]+= [subject]
    
    return subj_dict,pathology_dict

def consolidate_pathologies(pathologies,pathology_to_subjects):
    pathology_dict = {}
    for (sub_pathology,pathology) in pathologies:
        if pathology not in pathology_dict:
            pathology_dict[pathology]={}
        
        pathology_dict[pathology][sub_pathology] = len(pathology_to_subjects[sub_pathology])
    # for (pathology,_) in pathology_dict.items():
    #     pathology_dict[pathology] = list(sorted(pathology_dict[pathology]))
    return dict(sorted(pathology_dict.items()))


def iterate_files(path):
    children = []
    for root, _ , files in os.walk(path):
        files = list([str(f,'utf-8') for f in files])
        children+=[os.path.join(str(root,'utf-8'),f) for f in files if f.endswith('wav')]
    return children

def consolidate_files(children):
    children = {child.split('/')[-1]: child for child in children}
    return list(children.values())

def parse_file(children,pathology_subpathology):
    for child in children:
        path = child.split('/')
        pathology = path[-2]
        subject = path[-1]
        idx = subject.split('-')[0]
        t = subject.split('-')[1].split('.')[0]
        yield {'pathology':pathology_subpathology[pathology],
                'id':idx,
                'recording_type':t,
                'length':librosa.get_duration(filename=child)} 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str,
                    help='path to the dataset')
    parser.add_argument('--json_outpath', type=str,
                    help='path to output the json to')
    
    args = parser.parse_args()
    directory = os.fsencode(args.path)
    subjects = iterate_files(directory)
    subjects = consolidate_files(subjects)
    # for subject in tqdm(subjects):
        # print(subject)
    data = []
    for d in  tqdm(parse_file(subjects,dict(iterate_pathology_to_subpathology_structure(args.path)))):
        data+=[d]
    # print(data)
    with open(args.json_outpath,'w') as f:
        f.write(json.dumps(data,indent=4))
    # print()
    # count = len([(subj,patho) for (subj,patho) in subj_dict.items() if len(patho)>1])
    # path_dict_counts = dict([(pathology,len(subjects)) for (pathology,subjects) in path_dict.items()])
    # # print(path_dict_counts)
    # # print(count)
    # res = iterate_pathology_to_subpathology_structure(directory)
    # print(len(res))
    # res = [f"{pathology}@{sub_path}" for (pathology,sub_path) in res]
    # res = list(set(res))
    # res = [tuple(ill.split('@')) for ill in res]
    
    # with open(args.json_outpath,'w') as f:
    #     f.write(json.dumps(consolidate_pathologies(res,path_dict),indent=4))