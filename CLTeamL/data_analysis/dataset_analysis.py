import argparse

from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
import spacy
from tqdm import tqdm

from scripts.dataset_walker import DatasetWalker
from utils.dialogue_act_classifier import MidasDialogTagger
from utils.helpers import process_knowledge
from utils.nlp_helpers import get_sentiment, split_summary_question, get_num_sentences


def main(args):
    # create dataframe
    df = pd.DataFrame(
        columns=['target',
                 'turn_nr', 'dialogue_act_history',
                 'user_utterance', 'user_utterance_dialogue_act',
                 'ref_domain', 'ref_doc_type',
                 'ref_knowledge', 'ref_know_nr', 'ref_know_sentiment', 'ref_know_avg_sentiment',
                 'ref_response', 'ref_response_length', 'ref_response_sent_nr',
                 'ref_response_summary', 'ref_response_summary_sentiment', 'ref_response_question'])

    # define classifiers and other processing  tools
    dialogue_act_tagger = MidasDialogTagger(model_path='./models/midas_classifier.pt')
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')

    # loop over references and predictions
    data = DatasetWalker(dataset=args.dataset, dataroot="./../../data/", labels=True, incl_knowledge=True)
    for (log, reference) in tqdm(data):
        # skip if instance without response
        if not reference['target']:
            df = pd.concat([df, pd.DataFrame.from_records([{'target': False}])])
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
        user_utterance = log[-1]['text']

        ####################### Process reference response #######################
        # get reference data
        ref_response = reference['response']
        ref_length = len(reference['response'])
        ref_sent_length = get_num_sentences(reference['response'], nlp)
        ref_know_nr = len(reference['knowledge'])
        ref_knowledge, ref_know_sentiment, ref_know_avg_sentiment, ref_domain, ref_doc_type = process_knowledge(
            reference, nlp)

        # separate and find questions
        ref_summary, ref_optional_question = split_summary_question(ref_response, nlp)

        # analyse summary sentiment
        ref_summary_sentiment = get_sentiment(ref_summary, nlp)

        # append data to dataframe
        print(f'\nUSER UTTERANCE: {user_utterance}\n\tREFERENCE RESPONSE: {ref_response}')
        df = pd.concat([df, pd.DataFrame.from_records([{
            'target': True,

            # Related to dialogue context
            'turn_nr': turn_nr,
            'dialogue_act_history': dialogue_act_history,

            # Related to current user utterance
            'user_utterance': user_utterance,
            'user_utterance_dialogue_act': dialogue_act_history[-1],

            # Related to reference
            'ref_response': ref_response,
            'ref_response_length': ref_length,
            'ref_response_sent_nr': ref_sent_length,
            'ref_knowledge': ref_knowledge,
            'ref_know_nr': ref_know_nr,
            'ref_know_sentiment': ref_know_sentiment,
            'ref_know_avg_sentiment': ref_know_avg_sentiment,
            'ref_domain': ref_domain,
            'ref_doc_type': ref_doc_type,
            'ref_response_summary': ref_summary,
            'ref_response_summary_sentiment': ref_summary_sentiment,
            'ref_response_question': ref_optional_question
        }])])

    df.to_csv(f'./output/analysis_{args.dataset}.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='val', type=str,
                        help="Which dataset to process ")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
