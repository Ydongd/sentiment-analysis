## A easy sentiment analysis baseline, can also be used for text classification

## Usage

I created it for my own use. The model is trained on `ChnSentiCorp` dataset. Given a text, the model will give a score between 0 and 1 which represent the sentiment of the text. I use sliding window to handle long text which bert need to truncate.  More model will be tried in the future.

## Prerequisites

|   Library    | Version |
| :----------: | :-----: |
|    Python    |  3.8.5  |
|    torch     |  1.9.0  |
|   sklearn    |   0.0   |
|  tb-nightly  |  2.7.0  |
| transformers |  4.9.2  |

## Structure

1. Easy to understand from the file name.
2. Remember to put your `datasets` and `download bert` under the root file.
3. Modified from the general Bert template.

## How to execute

1. To run the code, just `python main.py`.
2. You can modify `main.py` and `main.sh` as you like. 

## Model
1. Now the model is simply bert + linear + sigmoid.

## Datasets
1. Put the txt or json datasets under `./datasets`. Now there isn't a preprocess routine for json datasets.
2. Data format: `'label' + '\t' + 'sentence'`.

## Main APP

You should only use the `run.py` file with the following arguments ( The table contains only a few major parameters, more parameters can be seen in `run.py` ) :

|       Argument       |                       Description                       |
| :------------------: | :-----------------------------------------------------: |
|      --data-dir      |  The input data dir. Should contain the training files  |
|     --model_type     |                   Select which model                    |
| --model_name_or_path |                Path to pre_trained model                |
|     --output_dir     |        Output directory of model and predictions        |
|       --label        |          Path to a file containing all labels           |
|   --max_seq_length   |         The maximum total input sentence length         |
|      --do_train      |                 Whether to run training                 |
|      --do_eval       |           Whether to run eval on the dev set            |
|     --do_predict     |       Whether to run predictions on the test set        |
|        --text        |           Predict sentiment for a single text           |
|    --predict_file    |        Whether to predict sentiment from a file         |
|    --sliding_len     | Length of sliding window for long text(over max length) |

## See the code for more details