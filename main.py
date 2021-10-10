import os
from utils.utils_file import read_json_to_txt,save_result_json,creat_label

input_path = './datasets/'
output_path = './output'

train_set = "train.txt"
dev_set = "dev.txt"
test_set = "test.txt"

test_predictions = "test_predictions.txt"

def main():
    # json to txt
    creat_label(input_path)

    # train/predict
    os.system("sh main.sh")

if __name__ == '__main__':
    main()
