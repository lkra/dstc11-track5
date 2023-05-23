import argparse

import spacy
from tqdm import tqdm

from utils.helpers import read_predictions, write_predictions
from utils.nlp_helpers import split_summary_question
from utils.questions_helpers import most_frequent_question


def main(args):
    # define classifiers and other processing  tools
    approach = args.prediction_file.rsplit(".", 1)[0]
    nlp = spacy.load("en_core_web_sm")
    question_to_append = most_frequent_question() if args.question_selection == 1 else ''

    # loop over references and predictions
    dataset_to_read = "test" if args.test_set else "val"
    predictions = read_predictions(dataset_to_read=dataset_to_read, prediction_file=args.prediction_file)

    for prediction in tqdm(predictions):
        # skip if instance without response
        if prediction['target']:
            # find questions
            prediction_response = prediction['response']
            summary, optional_question = split_summary_question(prediction_response, nlp)

            # assign
            if args.question_style == 0:
                # remove all questions
                prediction['response'] = summary

            if args.question_style == 1:
                # append questions where there is none
                if not optional_question:
                    if args.question_selection == 1:
                        prediction['response'] = prediction_response + " " + question_to_append
                        print(f'Appended a question: {question_to_append}')

    write_predictions(predictions, f'{approach}_question-style{args.question_style}.json',
                      dataset_to_read=dataset_to_read)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--question_style', default=1, type=int, choices=[0, 1],
                        help="Which action to take: "
                             "0 always remove questions from the end; "
                             "1 always append question at the end; ")
    parser.add_argument('--question_selection', default=0, type=int, choices=[0, 1],
                        help="Mechanism to select the question to append: "
                             "1 patch with the most frequent question; ")
    parser.add_argument('--test_set', default=False, action='store_true',
                        help="Run on the test set instead of the validation set")
    parser.add_argument('--prediction_file', default='baseline.rg.bart-base.json',
                        help="Which file with predictions to post process")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)

# python -m scripts.scores
# --dataset val
# --dataroot data/
# --outfile pred/val/baseline.rg.question-style0.json
# --scorefile pred/val/baseline.rg.question-style0.score.json
