import os
from core.params import results_dir

def get_specific_line_of_file(file_path, line):
    with open(file_path) as file:
        lines = file.readlines()
        return lines[line]
        

def get_csv_from_results_dir(results_dir, epoch):
    lines = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            line = get_specific_line_of_file(root+"/"+file, epoch)
            line = file.split('_')[0] + "," + line
            lines.append(line)
    
    with open(results_dir+"/csv_summary.csv",'+w') as csv_file:
        headers = 'Pathology,Precision,Recall,F1-Score\n'
        csv_file.write(headers)   
        csv_file.writelines(lines)   


get_csv_from_results_dir(results_dir + "/precision_recall_fscores",8)



