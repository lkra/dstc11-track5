# verify and evaluate the output
eval_dataset=val
model_name_exp=godel
rg_output_file=pred/${eval_dataset}/e2e.${model_name_exp}.json
rg_output_score_file=pred/${eval_dataset}/e2e.${model_name_exp}.score.json
python -m scripts.check_results --dataset ${eval_dataset} --dataroot data --outfile ${rg_output_file}
python -m scripts.scores --dataset ${eval_dataset} --dataroot data --outfile ${rg_output_file} --scorefile ${rg_output_score_file}
