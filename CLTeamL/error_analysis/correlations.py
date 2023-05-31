import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(font_scale=0.6)

APPROACH = 'baseline.rg.bart-base'

COLUMNS_TO_PLOT = [
    'turn_nr', 'dialogue_act_history',
    'user_utterance', 'user_utterance_dialogue_act',
    'ref_domain', 'ref_doc_type',
    'ref_knowledge', 'ref_know_nr',  # 'ref_faqs', 'ref_reviews',
    'ref_know_sentiment',
    'ref_know_avg_sentiment', 'ref_know_std_sentiment', 'ref_know_entropy_sentiment',
    'ref_response', 'ref_response_length', 'ref_response_sent_nr',
    'ref_response_summary', 'ref_response_summary_sentiment', 'ref_response_question',
    'pred_domain', 'pred_doc_type',
    'pred_knowledge', 'pred_know_nr',  # 'pred_faqs', 'pred_reviews',
    'pred_know_sentiment',
    'pred_know_avg_sentiment', 'pred_know_std_sentiment', 'pred_know_entropy_sentiment',
    'pred_response', 'pred_response_length', 'pred_response_sent_nr',
    'pred_response_summary', 'pred_response_summary_sentiment', 'pred_response_question',
    'bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL'
]

# Only input (features)
DATA_COLUMNS = [
    'turn_nr', 'dialogue_act_history',
    'user_utterance', 'user_utterance_dialogue_act',
    'ref_domain', 'ref_doc_type',
    'ref_knowledge', 'ref_know_nr',  # 'ref_faqs', 'ref_reviews',
    'ref_know_sentiment',
    'ref_know_avg_sentiment', 'ref_know_std_sentiment', 'ref_know_entropy_sentiment',
    'ref_response', 'ref_response_length', 'ref_response_sent_nr',
    'ref_response_summary', 'ref_response_summary_sentiment', 'ref_response_question',
]

# Input and output (features and predictions)
PREDICTIONS_COLUMNS = [
    'turn_nr', 'dialogue_act_history',
    # 'user_utterance', 'user_utterance_dialogue_act',
    'ref_domain', 'ref_doc_type',
    'ref_knowledge', 'ref_know_nr',  # 'ref_faqs', 'ref_reviews',
    # 'ref_know_sentiment', 'ref_know_avg_sentiment', 'ref_know_std_sentiment', 'ref_know_entropy_sentiment',
    'ref_response', 'ref_response_length', 'ref_response_sent_nr',
    # 'ref_response_summary', 'ref_response_summary_sentiment', 'ref_response_question',
    'pred_domain', 'pred_doc_type',
    'pred_knowledge', 'pred_know_nr',  # 'pred_faqs', 'pred_reviews',
    # 'pred_know_sentiment', 'pred_know_avg_sentiment',  # 'pred_know_std_sentiment', 'pred_know_entropy_sentiment',
    'pred_response', 'pred_response_length', 'pred_response_sent_nr',
    # 'pred_response_summary', 'pred_response_summary_sentiment', 'pred_response_question',
    'bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL'
]

# read in data frame
df = pd.read_csv(f'./output/error_analysis_{APPROACH}.csv')

# change data types from object to int
df = df.infer_objects()
df = df.astype({'user_utterance_dialogue_act': "category"})

# Filter
df = df[df['target']]
df = df[PREDICTIONS_COLUMNS]

# calculate correlations
matrix = df.corr().round(2)

# plot heatmap
m = sns.heatmap(matrix, vmax=1, vmin=-1, center=0, cmap="Blues", annot=True, annot_kws={"fontsize": 6})
# m.set_xticklabels(m.get_xticklabels(), rotation=55, va='center_baseline')
plt.subplots_adjust(bottom=0.4, left=0.3)
plt.savefig(f'./output/correlations_{APPROACH}.png', bbox_inches='tight')
plt.show()
