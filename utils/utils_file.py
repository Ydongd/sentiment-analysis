import json
import os


def read_json_to_txt(input,output):
    return

def creat_label(path):
    labels = []
    with open(os.path.join(path,"train.txt"),"r",encoding="utf-8") as f:
        for l in f.readlines():
            if not l.startswith("\n"):
                label = l.strip().split('\t')[0]
                if label not in labels:
                    labels.append(label)
    labels = ['0', '1']
    with open(os.path.join(path,"labels.txt"),"w") as f:
        for label in labels:
            f.write(label + "\n")

def save_result_json(input_json_path,output_json_path,test_json_path):
    return

def get_score(ground_truth_path, output_path):
    return

if __name__ == '__main__':
    pass
