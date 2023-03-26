import json
import statistics

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.dataset_walker import DatasetWalker
from utils.nlp_helpers import get_sentiment


def read_predictions(dataset_to_read="val", dataroot="./../../pred/", prediction_file="baseline.rg.bart-base.json"):
    with open(f"{dataroot}{dataset_to_read}/{prediction_file}", 'r') as f:
        predictions = json.load(f)

    return predictions


def write_predictions(predictions, prediction_file, dataset_to_read="val", dataroot="./../../pred/"):
    with open(f"{dataroot}{dataset_to_read}/{prediction_file}", "w") as jsonfile:
        json.dump(predictions, jsonfile, indent=2)


def read_preprocessed_data_and_predictions(dataset_to_read="val", dataroot="./../../data/",
                                           prediction_file="baseline.rg.bart-base.json"):
    df = pd.read_csv(f'{dataroot}../CLTeamL/data_analysis/output/analysis_{dataset_to_read}.csv')

    if dataset_to_read == "val":
        pred_data = DatasetWalker(dataset=dataset_to_read, dataroot=dataroot, labels=True,
                                  labels_file=f"{dataroot}../pred/{dataset_to_read}/{prediction_file}",
                                  incl_knowledge=True)
        predictions = [el[1] for el in pred_data]
    else:
        predictions = [None] * len(df)

    return df, predictions


def group_metrics_by(df, column):
    groups = df.groupby(column)[column,
                                'bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL'].agg(avg_bleu=('bleu', 'mean'),
                                                                                    avg_meteor=('meteor', 'mean'),
                                                                                    avg_rouge1=('rouge1', 'mean'),
                                                                                    avg_rouge2=('rouge2', 'mean'),
                                                                                    avg_rougeL=('rougeL', 'mean'),
                                                                                    num_samples=(column, 'count'))
    return groups


def process_knowledge(item, nlp):
    if not item['knowledge']:  # in case knowledge is empty
        item_knowledge, item_know_sentiment, item_know_avg_sentiment = [], [], 0
        item_domain, item_doc_type = 'empty', 'empty'
    else:
        item_knowledge = [el['sent'] if 'sent' in el.keys() else el['question'] + el['answer']
                          for el in item['knowledge']]
        item_know_sentiment = [get_sentiment(el, nlp) for el in item_knowledge]
        item_know_avg_sentiment = statistics.mean(item_know_sentiment) if item_know_sentiment else None
        item_domain = item['knowledge'][0]['domain']
        item_doc_type = item['knowledge'][0]['doc_type']

    return item_knowledge, item_know_sentiment, item_know_avg_sentiment, item_domain, item_doc_type


def score_predictions(reference_response, prediction_response, bleu_metric, meteor_metric, rouge_scorer):
    if not reference_response or not prediction_response:
        bleu, meteor, rouge1, rouge2, rougeL = None, None, None, None, None

    else:

        try:
            # calculate metrics
            bleu = bleu_metric.evaluate_example(prediction_response, reference_response)['bleu'] / 100.0
            meteor = meteor_metric.evaluate_example(prediction_response, reference_response)['meteor']
            scores = rouge_scorer.score(reference_response, prediction_response)
            rouge1 = scores['rouge1'].fmeasure
            rouge2 = scores['rouge2'].fmeasure
            rougeL = scores['rougeL'].fmeasure

        except:
            print(f"Error on {reference_response}, {prediction_response}")
            bleu, meteor, rouge1, rouge2, rougeL = None, None, None, None, None

    return bleu, meteor, rouge1, rouge2, rougeL


def hardest_examples(n_samples):
    dataset = pd.read_csv(f'./../data_analysis/output/analysis_train.csv')
    dataset.sort_values(by="ref_know_nr", ascending=False, inplace=True)

    print(dataset['ref_know_nr'].describe())

    sns.kdeplot(data=dataset, x="ref_know_nr")
    plt.show()

    dataset = dataset[dataset['ref_know_nr'] < dataset['ref_know_nr'].mean() + dataset['ref_know_nr'].std()]
    dataset = dataset[dataset['ref_know_nr'] > dataset['ref_know_nr'].mean()]

    return dataset[:n_samples].index
