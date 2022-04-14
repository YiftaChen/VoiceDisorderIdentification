import argparse
import os 

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

def consolidate_subjects(subjects):
    subj_dict = {}
    for (subject,pathology) in subjects:
        if subject not in subj_dict:
            subj_dict[subject]=[]
        subj_dict[subject] += [pathology]
    return subj_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str,
                    help='path to the dataset')
    
    args = parser.parse_args()
    directory = os.fsencode(args.path)
    subjects = iterate_pathology_structure(directory)
    subj_dict = consolidate_subjects(subjects)
    count = len([(subj,patho) for (subj,patho) in subj_dict.items() if len(patho)>1])
    print(len(subj_dict.items()))
    print(count)