#!/usr/bin bash

DATA_NAME=/home/lkrause/data/volume_2/dstc/dstc11-track5/CLTeamL/approaches/GODEL/dstc11/dstc11_val.jsonl
OUTPUT_DIR=/home/lkrause/data/volume_2/dstc/dstc11-track5/CLTeamL/approaches/GODEL/dstc11/pred/
MODEL_PATH=/home/lkrause/data/volume_2/dstc/dstc11-track5/CLTeamL/approaches/GODEL/dstc11/ckpt/checkpoint-epoch-1764

python generate.py --model_name_or_path ${MODEL_PATH}  \
	--output_dir ${OUTPUT_DIR}  \
	--per_device_eval_batch_size=16  \
	--max_target_length 128 \
	--max_length 512  \
	--preprocessing_num_workers 24  \
	--num_beams 5
