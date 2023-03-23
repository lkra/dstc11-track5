#!/bin/bash

# GODEL_MODEL="microsoft/GODEL-v1_1-base-seq2seq"
python train.py \
	--model_name_or_path 'microsoft/GODEL-v1_1-base-seq2seq'  \
	--dataset_name ../dstc11/dstc11_dataset.py   \
	--output_dir ../dstc11/ckpt   \
	--per_device_train_batch_size=16  \
	--per_device_eval_batch_size=16  \
	--max_target_length 128  \
	--max_length 512  \
	--num_train_epochs 50  \
	--save_steps 10000  \
	--num_beams 5  \
	--exp_name wow-test \
	--preprocessing_num_workers 24 \
	--save_every_checkpoint