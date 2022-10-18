#!/bin/bash

IDENTY="RobData"
MODEL="bert-ept-apt"
GPU=0
LR=2e-5
EPOCH=30
MAXL=128
BS=8
GRAD_ACC=4


ALPHA=0.1
BETA=1
AP_LAYER=5

export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=8

TASK='aug_ori_pawsx'
SAVE_DIR="/Code/EAR_Train/${TASK}-${IDENTY}-${MODEL}/LR${LR}-epoch${EPOCH}-alpha${ALPHA}-beta${BETA}-layer${AP_LAYER}/"
mkdir -p $SAVE_DIR

DATA_DIR="/Data/AugWithOrign/pawsx"

python3 /apdcephfs/share_1157269/karlding/EAR/run_classify.py \
--model_type $MODEL  \
--model_name_or_path "/Data/modle_card/bert-base-multilingual-cased"  \
--train_language 'en' \
--task_name $TASK  \
--do_train  \
--do_eval  \
--do_predict  \
--data_dir $DATA_DIR  \
--per_gpu_train_batch_size $BS  \
--learning_rate  $LR \
--num_train_epochs $EPOCH  \
--max_seq_length $MAXL  \
--output_dir  $SAVE_DIR  \
--eval_all_checkpoints  \
--overwrite_output_dir  \
--save_steps 500  \
--log_file "train.log"  \
--predict_languages "en,de,es,fr,ja,ko,zh"  \
--save_only_best_checkpoint  \
--seed 7 \
--eval_test_set \
--per_gpu_eval_batch_size 256 \
--gradient_accumulation_steps $GRAD_ACC \
--bi_stream \
--save_standard_dev \
--alpha $ALPHA \
--beta $BETA \
--ap_layer $AP_LAYER

