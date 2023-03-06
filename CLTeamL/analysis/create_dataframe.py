import pandas as pd
import spacy
from rouge_score import rouge_scorer
from summ_eval.bleu_metric import BleuMetric
from summ_eval.meteor_metric import MeteorMetric
from tqdm import tqdm

from utils.dialogue_act_classifier import MidasDialogTagger
from utils.helpers import read_data, save_data, split_summary_question

DATASET = "train"  # val or train

# create dataframe
df = pd.DataFrame(
    columns=['turn_nr', 'dialogue_act_history',
             'question', 'question_dialogue_act',
             'reference_length', 'reference_response', 'reference_response_summary', 'reference_response_question',
             'ref_knowledge', 'ref_know_nr', 'reference_domain', 'reference_doc_type',
             'prediction_length', 'prediction_response', 'pred_know_nr', 'prediction_domain', 'prediction_doc_type',
             'bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL'])

# define metrics
bleu_metric = BleuMetric()
meteor_metric = MeteorMetric()
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# define classifiers and other processing  tools
dialogue_act_tagger = MidasDialogTagger(model_path="./models/midas_classifier.pt")
nlp = spacy.load("en_core_web_sm")

# loop over references and predictions
data, predictions = read_data(dataset_to_read=DATASET)
for (log, reference), prediction in tqdm(zip(data, predictions)):
    # skip if instance without response
    if not reference['target']:
        continue

    ####################### Process dialogue context #######################
    # number of turns
    turn_nr = len(log)

    # classify!
    dialogue_act_history = []
    for item in log:
        dialogue_act = dialogue_act_tagger.extract_dialogue_act(item['text'])
        dialogue_act_history.append(dialogue_act[0].value)

    ####################### Process user utterance #######################
    # add question
    question = log[-1]["text"]

    ####################### Process reference response #######################
    # get reference data
    reference_response = reference['response']
    reference_length = len(reference['response'])
    ref_know_nr = len(reference['knowledge'])
    if not reference['knowledge']:  # in case knowledge is empty
        ref_knowledge = []
        reference_domain = 'empty'
        reference_doc_type = 'empty'
    else:
        ref_knowledge = [el['sent'] for el in reference['knowledge'] if 'sent' in el.keys()]
        reference_domain = reference['knowledge'][0]['domain']
        reference_doc_type = reference['knowledge'][0]['doc_type']

    # find questions
    summary, optional_question = split_summary_question(reference_response, nlp)

    ####################### Process prediction response #######################
    # get prediction data
    if not prediction or not prediction['target']:  # in case response is empty
        prediction_response = 'empty'
        prediction_length = 'empty'
        pred_know_nr = 'empty'
        prediction_domain = 'empty'
        prediction_doc_type = 'empty'
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

    ####################### Process evaluation #######################
    # calculate metrics
    bleu = bleu_metric.evaluate_example(prediction_response, reference_response)['bleu'] / 100.0
    meteor = meteor_metric.evaluate_example(prediction_response, reference_response)['meteor']
    scores = scorer.score(reference_response, prediction_response)
    rouge1 = scores['rouge1'].fmeasure
    rouge2 = scores['rouge2'].fmeasure
    rougeL = scores['rougeL'].fmeasure

    # append data to dataframe
    print(f"\nQUESTION: {question}"
          f"\n\tREFERENCE RESPONSE: {reference_response}\n\tPREDICTED RESPONSE: {prediction_response}")
    df = pd.concat([df, pd.DataFrame.from_records([{
        # Related to dialogue context
        'turn_nr': turn_nr,
        'dialogue_act_history': dialogue_act_history,

        # Related to current user utterance
        'question': question,
        'question_dialogue_act': dialogue_act_history[-1],

        # Related to reference
        'reference_response': reference_response,
        'reference_length': reference_length,
        'ref_knowledge': ref_knowledge,
        'ref_know_nr': ref_know_nr,
        'reference_domain': reference_domain,
        'reference_doc_type': reference_doc_type,
        'reference_response_summary': summary,
        'reference_response_question': optional_question,

        # Related to prediction
        'prediction_response': prediction_response,
        'prediction_length': prediction_length,
        'pred_know_nr': pred_know_nr,
        'prediction_domain': prediction_domain,
        'prediction_doc_type': prediction_doc_type,

        # Related to evaluation
        'bleu': bleu,
        'meteor': meteor,
        'rouge1': rouge1,
        'rouge2': rouge2,
        'rougeL': rougeL
    }])])

save_data(df, f'output/analysis_{DATASET}')
