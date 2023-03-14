import pandas as pd

from analysis.utils.beam_search import beam_search_question
from analysis.utils.tfidf import calculate_tfidf

DATASET = "train"  # val or train
NGRAM_SIZE = 7
TOP_FREQUENT_NGRAMS = 0.05

# Create vocabulary and examples
full_questions = pd.read_csv(f'./analysis/output/scores_per_reference_question_{DATASET}.csv')
prediction = beam_search_question(full_questions, ngram_size=NGRAM_SIZE)

# read in data frame
trigrams = pd.read_csv(f'./analysis/output/trigrams_{DATASET}.csv', names=["trigram", "count"], header=0)
tfidf = calculate_tfidf(trigrams, keep_top=TOP_FREQUENT_NGRAMS)

# with open(f'./../pred/val/baseline.rg.appendquestions.json', "w") as jsonfile:
#     json.dump(predictions, jsonfile, indent=2)

# python -m scripts.scores --dataset val --dataroot data/ --outfile pred/val/baseline.rg.appendquestions.json --scorefile pred/val/baseline.rg.appendquestions.score.json

print("done")
