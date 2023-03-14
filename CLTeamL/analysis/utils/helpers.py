import json
from collections import Counter
from itertools import chain

import pandas as pd

from scripts.dataset_walker import DatasetWalker
from utils.constants import NGRAM_SIZE


def read_data(dataset_to_read="val", dataroot="./../../data/"):
    data = DatasetWalker(dataset=dataset_to_read, dataroot=dataroot, labels=True, incl_knowledge=True)
    if dataset_to_read == "val":
        with open(f"{dataroot}../pred/{dataset_to_read}/baseline.rg.bart-base.json", 'r') as f:
            predictions = json.load(f)
    else:
        predictions = [None] * len(data)

    return data, predictions


def save_data(df, filename):
    df.to_csv(f'{filename}.csv')
    df.to_pickle(f'{filename}.pkl')


def dummy_question(text):
    return text[-1] == '?'


def split_summary_question(text, nlp):
    doc = nlp(text)
    sents = [x.text for x in doc.sents]
    # for sent in doc.sents:
    #     print(sent.text)

    if dummy_question(sents[-1]):
        question = sents[-1]
        summary = ". ".join(sents[:-1])
    else:
        question = None
        summary = ". ".join(sents)

    return summary, question


def get_sentiment(summary, nlp):
    doc = nlp(summary)
    sentiment = doc._.blob.polarity

    return sentiment


def group_metrics_by(df, column):
    groups = df.groupby(column)[column,
                                'bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL'].agg(avg_bleu=('bleu', 'mean'),
                                                                                    avg_meteor=('meteor', 'mean'),
                                                                                    avg_rouge1=('rouge1', 'mean'),
                                                                                    avg_rouge2=('rouge2', 'mean'),
                                                                                    avg_rougeL=('rougeL', 'mean'),
                                                                                    num_samples=(column, 'count'))
    return groups


def find_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))


def get_ngram_count(df, column, ngram_size=NGRAM_SIZE):
    df['ngrams'] = df[column].map(lambda x: find_ngrams(x.split(" "), ngram_size) if x else x)
    ngrams = [x for x in df['ngrams'].tolist() if x]
    ngrams = list(chain(*ngrams))
    ngram_counts = Counter(ngrams)
    ngram_counts = pd.DataFrame.from_dict(ngram_counts, orient='index', columns=['count'])

    return ngram_counts
