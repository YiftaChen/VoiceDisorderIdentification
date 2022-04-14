import argparse
import os 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str,
                    help='path to the dataset')
    
    args = parser.parse_args()
    directory = os.fsencode(args.path)
    
    for file in os.listdir(directory):
        assert file in [b"male",b"female"], f"structure should follow with genders"
        path = os.path.join(directory,file)
        for pathology in os.listdir(path):
            sub_path = os.path.join(path,pathology)
            print(f"{pathology}")
            for sub_pathology in os.listdir(sub_path):
                print(f"    {sub_pathology}")
                
