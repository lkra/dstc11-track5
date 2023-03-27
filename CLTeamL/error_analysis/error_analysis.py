import argparse

from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
import spacy
from rouge_score import rouge_scorer
from summ_eval.bleu_metric import BleuMetric
from summ_eval.meteor_metric import MeteorMetric
from tqdm import tqdm

from utils.helpers import read_preprocessed_data_and_predictions, process_knowledge, score_predictions
from utils.nlp_helpers import get_sentiment, split_summary_question, get_num_sentences


def main(args):
    approach = args.prediction_file.rsplit(".", 1)[0]

    # create dataframe
    pred_df = pd.DataFrame(
        columns=['target',
                 'pred_domain', 'pred_doc_type',
                 'pred_knowledge', 'pred_know_nr', 'pred_know_sentiment', 'pred_know_avg_sentiment',
                 'pred_response', 'pred_response_length', 'pred_response_sent_nr',
                 'pred_response_summary', 'pred_response_summary_sentiment', 'pred_response_question',
                 'bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL',
                 'appropriateness', 'relevance'])

    # define classifiers and other processing  tools
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')

    # loop over references and predictions
    ref_df, predictions = read_preprocessed_data_and_predictions(dataset_to_read=args.dataset,
                                                                 prediction_file=args.prediction_file)
    for prediction in tqdm(predictions):
        if not prediction or not prediction['target']:  # in case response is empty
            pred_df = pd.concat([pred_df, pd.DataFrame.from_records([{'target': False}])])
            continue

        ####################### Process prediction response #######################
        # get prediction data
        pred_response = prediction['response']
        pred_length = len(prediction['response'])
        pred_sent_length = get_num_sentences(prediction['response'], nlp)
        pred_know_nr = len(prediction['knowledge'])
        pred_knowledge, pred_know_sentiment, pred_know_avg_sentiment, pred_domain, pred_doc_type = process_knowledge(
            prediction, nlp)

        # separate summary and find questions
        pred_summary, pred_optional_question = split_summary_question(pred_response, nlp)

        # analyse summary sentiment
        pred_summary_sentiment = get_sentiment(pred_summary, nlp)

        pred_df = pd.concat([pred_df, pd.DataFrame.from_records([{
            'target': True,
            # Related to prediction
            'pred_response': pred_response,
            'pred_response_length': pred_length,
            'pred_response_sent_nr': pred_sent_length,
            'pred_knowledge': pred_knowledge,
            'pred_know_nr': pred_know_nr,
            'pred_know_sentiment': pred_know_sentiment,
            'pred_know_avg_sentiment': pred_know_avg_sentiment,
            'pred_domain': pred_domain,
            'pred_doc_type': pred_doc_type,
            'pred_response_summary': pred_summary,
            'pred_response_summary_sentiment': pred_summary_sentiment,
            'pred_response_question': pred_optional_question,

        }])])

    ####################### Process evaluation #######################
    # Merge two dfs
    pred_df.reset_index(inplace=True, drop=True)
    df = pd.concat([ref_df, pred_df], axis=1)

    # define metrics
    bleu_metric = BleuMetric()
    meteor_metric = MeteorMetric()
    rouge_metric = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # calculate metrics
    df[['bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL']] = df.apply(
        lambda x: score_predictions(x['ref_response'], x['pred_response'], bleu_metric, meteor_metric, rouge_metric),
        axis=1, result_type='expand')

    df.to_csv(f'./output/error_analysis_{approach}.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='val', type=str,
                        help="Which dataset to process ")
    parser.add_argument('--prediction_file', default='baseline.rg.bart-base.json', type=str,
                        help="Which file with predictions to compare to the ground truth")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
