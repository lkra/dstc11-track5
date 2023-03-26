import argparse

import spacy
from tqdm import tqdm

from utils.beam_search import most_frequent_question
from utils.helpers import read_predictions, write_predictions
from utils.nlp_helpers import split_summary_question


def main(args):
    # define classifiers and other processing  tools
    nlp = spacy.load("en_core_web_sm")
    if args.question_selection == 0:
        question_to_append = most_frequent_question()

    # loop over references and predictions
    dataset_to_read = "test" if args.test_set else "val"
    predictions = read_predictions(dataset_to_read=dataset_to_read, prediction_file=args.input_file)

    for prediction in tqdm(predictions):
        # skip if instance without response
        if not prediction['target']:
            continue

        # find questions
        prediction_response = prediction['response']
        summary, optional_question = split_summary_question(prediction_response, nlp)

        # assign
        if args.question_style == 0:
            prediction['response'] = summary

        if args.question_style == 1:
            if not optional_question:
                if args.question_selection == 0:
                    prediction['response'] = prediction_response + " " + question_to_append

    output_file = args.input_file.split(".")
    output_file[-2] = output_file[-2] + f"_question-style{args.question_style}"
    output_file = ".".join(output_file)

    write_predictions(predictions, output_file, dataset_to_read=dataset_to_read)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--question_style', default=1, type=int, choices=[0, 1],
                        help="Which action to take: "
                             "0 always remove questions from the end; "
                             "1 always append question at the end; ")
    parser.add_argument('--question_selection', default=0, type=int, choices=[0, 1],
                        help="Mechanism to select the question to append: "
                             "0 patch with the most frequent question; ")
    parser.add_argument('--test_set', default=False, action='store_true',
                        help="Run on the test set instead of the validation set")
    parser.add_argument('--input_file', default='baseline.rg.bart-base.json',
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
