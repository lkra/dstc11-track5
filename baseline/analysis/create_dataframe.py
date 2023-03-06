import json

import pandas as pd
import seaborn as sns
from rouge_score import rouge_scorer
from summ_eval.bleu_metric import BleuMetric
from summ_eval.meteor_metric import MeteorMetric
from tqdm import tqdm

from baseline.analysis.utils.dialogue_act_classifier import MidasDialogTagger
from baseline.analysis.utils.helpers import dummy_question

sns.set()

# load reference JSON
with open('/home/lkrause/data/volume_2/dstc/dstc11-track5/data/val/labels.json', 'r') as f:
    references = json.load(f)

# load prediction JSON
with open('/home/lkrause/data/volume_2/dstc/dstc11-track5/pred/val/baseline.rg.bart-base.json', 'r') as f:
    predictions = json.load(f)

# load logs
with open('/home/lkrause/data/volume_2/dstc/dstc11-track5/data/val/logs.json', 'r') as f:
    logs = json.load(f)

# create dataframe
df = pd.DataFrame(
    columns=['question', 'turn_nr', 'reference_response', 'reference_length', 'ref_know_nr', 'reference_domain',
             'reference_doc_type',
             'prediction_response', 'prediction_length', 'pred_know_nr', 'prediction_domain', 'prediction_doc_type',
             'bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL'])

# define metrics
bleu_metric = BleuMetric()
meteor_metric = MeteorMetric()
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# define classifiers
dialogue_act_tagger = MidasDialogTagger(model_path="./models/midas_classifier.pt")

# loop over references and predictions
for reference, prediction, log in tqdm(zip(references, predictions, logs)):
    # skip if instance without response
    if not reference['target']:
        continue

    print(reference, prediction)

    # get reference data
    # print(reference)
    reference_response = reference['response']
    reference_length = len(reference['response'])
    ref_know_nr = len(reference['knowledge'])
    reference_domain = reference['knowledge'][0]['domain']
    reference_doc_type = reference['knowledge'][0]['doc_type']

    # get prediction data
    # print(prediction)
    if not prediction['target']:  # in case response is empty
        pass
    else:
        prediction_response = prediction['response']
        prediction_length = len(prediction['response'])
        pred_know_nr = len(prediction['knowledge'])
        if not prediction['knowledge']:  # in case knowledge is empty
            prediction_domain = 'empty'
            prediction_doc_type = 'empty'
        else:
            prediction_domain = prediction['knowledge'][0]['domain']
            prediction_doc_type = prediction['knowledge'][0]['doc_type']

    # calculate metrics
    bleu = bleu_metric.evaluate_example(prediction_response, reference_response)['bleu'] / 100.0
    # print(f"bleu: {bleu}")

    meteor = meteor_metric.evaluate_example(prediction_response, reference_response)['meteor']
    # print(f"meteor: {meteor}")

    scores = scorer.score(reference_response, prediction_response)
    rouge1 = scores['rouge1'].fmeasure
    # print(f"rouge1: {rouge1}")
    rouge2 = scores['rouge2'].fmeasure
    # print(f"rouge2: {rouge2}")
    rougeL = scores['rougeL'].fmeasure
    # print(f"rougeL: {rougeL}")

    # classify!
    dialogue_act_history = []
    for item in log:
        dialogue_act = dialogue_act_tagger.extract_dialogue_act(item['text'])
        dialogue_act_history.append(dialogue_act[0].value)

    # add question
    question = log[-1]["text"]
    print(question)

    # number of turns
    turn_nr = len(log)

    # find questions
    if dummy_question(prediction_response):
        # TODO: @lea: I found a question!
        pass

    # append data to dataframe
    df = df.append({
        'question': question,
        'turn_nr': turn_nr,
        'reference_response': reference_response,
        'reference_length': reference_length,
        'ref_know_nr': ref_know_nr,
        'reference_domain': reference_domain,
        'reference_doc_type': reference_doc_type,
        'dialogue_act_history': dialogue_act_history,
        'prediction_response': prediction_response,
        'prediction_length': prediction_length,
        'pred_know_nr': pred_know_nr,
        'prediction_domain': prediction_domain,
        'prediction_doc_type': prediction_doc_type,
        'bleu': bleu,
        'meteor': meteor,
        'rouge1': rouge1,
        'rouge2': rouge2,
        'rougeL': rougeL
    }, ignore_index=True)

# remove duplicate domain values
# df['reference_domain'] = df['reference_domain'].apply(lambda x: x[0])
# df['prediction_domain'] = df['prediction_domain'].apply(lambda x: x[0])

# # visualisationsss
# print('create visualsss')
# sns.pairplot(df, hue='prediction_domain', size=2.5)
# print('save visualsss')
# plt.savefig('prediction_domain.png')


df.to_pickle("output/analysis.pkl")
