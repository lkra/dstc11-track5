import pandas as pd


def extract_last_act(history):
    history = list(history)
    return history[-1]


# read in data frame
df = pd.read_pickle("output/analysis.pkl")

# group by
df['last_act'] = df['dialogue_act_history'].apply(lambda x: extract_last_act(x))
groups = df.groupby('last_act')['question', 'bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL'].agg(
    {'bleu': 'mean', 'meteor': 'mean', 'rouge1': 'mean', 'rouge2': 'mean', 'rougeL': 'mean', 'question': 'count'})

groups.to_pickle("scores_per_act.pkl")
print("done!")

# prev dialogue, knowledge, ref response
