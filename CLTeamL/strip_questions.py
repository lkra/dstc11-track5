import json

import spacy
from tqdm import tqdm

from utils.helpers import read_data, split_summary_question

DATASET = "val"  # val or train

# define classifiers and other processing  tools
nlp = spacy.load("en_core_web_sm")

# loop over references and predictions
data, predictions = read_data(dataset_to_read=DATASET, dataroot="./../data/")
for (log, reference), prediction in tqdm(zip(data, predictions)):
    # skip if instance without response
    if not prediction['target']:
        continue

    # find questions
    prediction_response = prediction['response']
    summary, optional_question = split_summary_question(prediction_response, nlp)

    # assign summary only
    prediction['response'] = summary

with open(f'./../pred/val/baseline.rg.stripquestions.json', "w") as jsonfile:
    json.dump(predictions, jsonfile, indent=2)


# python -m scripts.scores --dataset val --dataroot data/ --outfile pred/val/baseline.rg.stripquestions.json --scorefile pred/val/baseline.rg.stripquestions.score.json