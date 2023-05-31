import json
import statistics

import pandas as pd

from scripts.dataset_walker import DatasetWalker
from utils.nlp_helpers import get_sentiment, calculate_entropy


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
                                'ref_know_avg_sentiment','ref_response_length', 'ref_response_sent_nr',
                                'bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL'].agg(avg_bleu=('bleu', 'mean'),
                                                                                    avg_meteor=('meteor', 'mean'),
                                                                                    avg_rouge1=('rouge1', 'mean'),
                                                                                    avg_rouge2=('rouge2', 'mean'),
                                                                                    avg_rougeL=('rougeL', 'mean'),
                                                                                    num_samples=(column, 'count'),
                                                                                    avg_know_sentiment=(
                                                                                    'ref_know_avg_sentiment', 'mean'),
                                                                                    avg_len=(
                                                                                    'ref_response_length', 'mean'),
                                                                                    avg_sentences=(
                                                                                    'ref_response_sent_nr', 'mean')
                                                                                    )
    return groups


def process_knowledge(item, nlp):
    if not item['knowledge']:  # in case knowledge is empty
        item_faqs, item_reviews = None, None
        item_knowledge, item_know_nr = [], None
        item_know_sentiment, item_know_avg_sentiment = [], None
        item_know_std_sentiment, item_know_entropy_sentiment = None, None
        item_domain, item_doc_type = 'empty', 'empty'
    else:
        # Sort and separate
        item['knowledge'] = sorted(item['knowledge'], key=lambda d: d['doc_type'])
        item_faqs = [el['question'] + el['answer'] for el in item['knowledge'] if el['doc_type'] == 'faq']
        item_reviews = [el['sent'] for el in item['knowledge'] if el['doc_type'] == 'review']

        # overall knowledge
        item_know_nr = len(item['knowledge'])
        item_knowledge = item_faqs + item_reviews
        item_know_sentiment = [get_sentiment(el, nlp) for el in item_reviews]
        item_know_avg_sentiment = statistics.mean(item_know_sentiment) if item_know_sentiment else None
        item_know_std_sentiment = statistics.stdev(item_know_sentiment) \
            if item_know_sentiment and len(item_know_sentiment) > 2 else None
        item_know_entropy_sentiment = calculate_entropy(item_know_sentiment, item_know_avg_sentiment) \
            if item_know_sentiment else None
        item_domain = item['knowledge'][0]['domain']
        item_doc_type = item['knowledge'][0]['doc_type']

    return item_know_nr, item_knowledge, item_faqs, item_reviews, \
           item_know_sentiment, item_know_avg_sentiment, item_know_std_sentiment, item_know_entropy_sentiment, \
           item_domain, item_doc_type


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
