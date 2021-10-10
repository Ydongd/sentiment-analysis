#!/bin/bash

DATA_DIR='./datasets/'
MODEL_TYPE='bert'
MODEL_NAME_OR_PATH='./roberta'

OUTPUT_DIR='./output'
LABEL='./datasets/labels.txt'

CUDA_VISIBLE_DEVICES='0' python run.py \
--data_dir $DATA_DIR \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL_NAME_OR_PATH \
--output_dir $OUTPUT_DIR \
--labels $LABEL \
--do_train \
--adv_training fgm \
--num_train_epochs 5 \
--max_seq_length 512 \
--logging_steps -1 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 1 \
--learning_rate 3e-4 \
--bert_lr 5e-5 \
--classifier_lr  3e-4 \
--overwrite_cache \
--overwrite_output_dir \
