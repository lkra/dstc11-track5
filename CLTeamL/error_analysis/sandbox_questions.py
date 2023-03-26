import pandas as pd

from utils.beam_search import train_question_beam_search, beam_search_question
from utils.tfidf import calculate_tfidf

DATASET = "train"  # val or train
NGRAM_SIZE = 10
TOP_FREQUENT_NGRAMS = 0.05

# Create vocabulary and examples
full_questions = pd.read_csv(f'data_analysis/output/scores_per_reference_question_{DATASET}.csv')
model, index = train_question_beam_search(full_questions, ngram_size=NGRAM_SIZE)
prediction = beam_search_question(model, index, phrase_to_complete="would you like to know more")

# read in data frame
trigrams = pd.read_csv(f'data_analysis/output/trigrams_{DATASET}.csv', names=["trigram", "count"], header=0)
tfidf = calculate_tfidf(trigrams, keep_top=TOP_FREQUENT_NGRAMS)

print("done")
