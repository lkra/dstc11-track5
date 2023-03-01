import json

from scripts.dataset_walker import DatasetWalker


def read_data():
    data = DatasetWalker(dataset="val", dataroot="./../../data/", labels=True, incl_knowledge=True)

    with open("./../../pred/val/baseline.rg.bart-base.json", 'r') as f:
        predictions = json.load(f)

    return data, predictions


def dummy_question(text):
    return text[-1] == '?'
