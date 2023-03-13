import json

from scripts.dataset_walker import DatasetWalker
from scripts.scores import Metric


def read_data():
    data = DatasetWalker(dataset="val", dataroot="./../../data/", labels=True, incl_knowledge=True)

    with open("./../../pred/val/baseline.rg.bart-base.json", 'r') as f:
        predictions = json.load(f)

    return data, predictions


data, predictions = read_data()
metric = Metric()
count_cases = {"no_knowledge": 0, "knowledge": 0, "question": 0, "no_question": 0}

for (item, ground_truth), prediction in zip(data, predictions):
    # separate items that do need knowledge
    if not ground_truth["target"]:
        count_cases["no_knowledge"] += 1

    else:
        count_cases["knowledge"] += 1

        last_utterance = item[-1]["text"]
        response = ground_truth["response"]
        predicted_response = prediction["response"]

        # Count items that do not end with a question mark
        if last_utterance[-1] == '?':
            count_cases["question"] += 1
        else:
            count_cases["no_question"] += 1

            print(f"\n{last_utterance}")
            print(f"\tGROUND_TRUTH: {response}")
            print(f"\tPREDICTED:    {predicted_response}")

            # metric.update(ref, pred)

print(count_cases)

# TODO dialogue acts
# MIDAS

# TODO check for entity name in text (function for this?!)

#

