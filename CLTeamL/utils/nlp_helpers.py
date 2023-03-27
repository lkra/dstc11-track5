from collections import Counter
from itertools import chain

import pandas as pd

from utils.constants import NGRAM_SIZE


def get_num_sentences(text, nlp):
    doc = nlp(text)
    return len(list(doc.sents))


def get_sentiment(summary, nlp):
    doc = nlp(summary)
    return doc._.blob.polarity


def dummy_question(text):
    return text[-1] == '?'


def split_summary_question(text, nlp):
    # split text into sentences
    doc = nlp(text)
    sents = [x.text for x in doc.sents]
    if len(sents) == 1:
        sents[-1] = sents[-1].rstrip(",")
        sents = sents[-1].split(',')

    # separate questions from summaries
    if sents and dummy_question(sents[-1]):
        question = sents[-1]
        summary = ". ".join(sents[:-1])
    else:
        question = None
        summary = ". ".join(sents)

    return summary, question


def find_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))


def get_ngram_count(df, column, ngram_size=NGRAM_SIZE):
    df['ngrams'] = df[column].map(lambda x: find_ngrams(x.split(" "), ngram_size) if type(x) == str else x)
    ngrams = [x for x in df['ngrams'].tolist() if type(x) == list]
    ngrams = list(chain(*ngrams))
    ngram_counts = Counter(ngrams)
    ngram_counts = pd.DataFrame.from_dict(ngram_counts, orient='index', columns=['count'])

    return ngram_counts
