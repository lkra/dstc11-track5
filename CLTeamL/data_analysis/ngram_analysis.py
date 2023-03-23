import pandas as pd

from utils.nlp_helpers import get_ngram_count

DATASET = "val"  # val or train

# read in data frame
df = pd.read_csv(f'output/analysis_{DATASET}.csv')
df = df[df['target']]

# find ngrams
bigrams = get_ngram_count(df, 'ref_response_question', ngram_size=2)
bigrams.to_csv(f"output/bigrams_{DATASET}.csv")

trigrams = get_ngram_count(df, 'ref_response_question', ngram_size=3)
trigrams.to_csv(f"output/trigrams_{DATASET}.csv")
