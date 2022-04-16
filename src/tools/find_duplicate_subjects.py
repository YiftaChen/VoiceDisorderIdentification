import argparse
from importlib.resources import path
import os 
import json 
import collections

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
        if subject not in subj_dict:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str,
                    help='path to the dataset')
    parser.add_argument('--json_outpath', type=str,
                    help='path to output the json to')
    
    args = parser.parse_args()
    directory = os.fsencode(args.path)
    subjects = iterate_pathology_structure(directory)
    subj_dict,path_dict = consolidate_subjects(subjects)
    count = len([(subj,patho) for (subj,patho) in subj_dict.items() if len(patho)>1])
    path_dict_counts = dict([(pathology,len(subjects)) for (pathology,subjects) in path_dict.items()])
    # print(path_dict_counts)
    # print(count)
    res = iterate_pathology_to_subpathology_structure(directory)
    print(len(res))
    res = [f"{pathology}@{sub_path}" for (pathology,sub_path) in res]
    res = list(set(res))
    res = [tuple(ill.split('@')) for ill in res]
    
    with open(args.json_outpath,'w') as f:
        f.write(json.dumps(consolidate_pathologies(res,path_dict),indent=4))