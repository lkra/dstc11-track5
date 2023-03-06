import pandas as pd

from utils.helpers import group_metrics_by, get_ngram_count, save_data

DATASET = "train"  # val or train

# read in data frame
df = pd.read_pickle(f'output/analysis_{DATASET}.pkl')

# group by USER UTTERANCE dialogue act
groups = group_metrics_by(df, 'question_dialogue_act')
save_data(groups, f"output/scores_per_act_{DATASET}")

# group by REFERENCE RESPONSE optional questions
df['reference_response_question'] = df['reference_response_question'].str.lower()
groups = group_metrics_by(df, 'reference_response_question')
save_data(groups, f"output/scores_per_reference_question_{DATASET}")

# group by REFERENCE RESPONSE optional questions (start)
df['reference_response_question_start'] = df['reference_response_question'].apply(
    lambda x: " ".join(x.split()[:5]) if x else x)
groups = group_metrics_by(df, 'reference_response_question_start')
save_data(groups, f"output/scores_per_reference_question_start_{DATASET}")

# group by REFERENCE RESPONSE optional questions (end)
df['reference_response_question_end'] = df['reference_response_question'].apply(
    lambda x: " ".join(x.split()[-5:]) if x else x)
groups = group_metrics_by(df, "reference_response_question_end")
save_data(groups, f"output/scores_per_reference_question_end_{DATASET}")

# find ngrams
bigrams = get_ngram_count(df, 'reference_response_question', ngram_size=2)
save_data(bigrams, f"output/bigrams_{DATASET}")

trigrams = get_ngram_count(df, 'reference_response_question', ngram_size=3)
save_data(trigrams, f"output/trigrams_{DATASET}")

print("done!")
