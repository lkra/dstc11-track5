import pandas as pd

from analysis.utils.beam_search import train_question_beam_search, beam_search_question
from analysis.utils.tfidf import calculate_tfidf

DATASET = "train"  # val or train
NGRAM_SIZE = 10
TOP_FREQUENT_NGRAMS = 0.05

# Create vocabulary and examples
full_questions = pd.read_csv(f'./analysis/output/scores_per_reference_question_{DATASET}.csv')
model, index = train_question_beam_search(full_questions, ngram_size=NGRAM_SIZE)
prediction = beam_search_question(model, index, phrase_to_complete="would you like to know more")

# read in data frame
trigrams = pd.read_csv(f'./analysis/output/trigrams_{DATASET}.csv', names=["trigram", "count"], header=0)
tfidf = calculate_tfidf(trigrams, keep_top=TOP_FREQUENT_NGRAMS)

# python -m scripts.scores --dataset val --dataroot data/ --outfile pred/val/baseline.rg.appendquestions.json --scorefile pred/val/baseline.rg.appendquestions.score.json

print("done")
