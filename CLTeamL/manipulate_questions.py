import argparse
import json

import spacy
from tqdm import tqdm

from utils.beam_search import most_frequent_question
from utils.helpers import split_summary_question


def main(args):
    # define classifiers and other processing  tools
    nlp = spacy.load("en_core_web_sm")
    if args.question_selection == 0:
        question_to_append = most_frequent_question()

    # loop over references and predictions
    dataset_to_read = "test" if args.test_set else "val"
    with open(f"./../pred/{dataset_to_read}/{args.input_file}", 'r') as f:
        predictions = json.load(f)

    for prediction in tqdm(predictions):
        # skip if instance without response
        if not prediction['target']:
            continue

        # find questions
        prediction_response = prediction['response']
        summary, optional_question = split_summary_question(prediction_response, nlp)

        # assign
        if args.action == 0:
            prediction['response'] = summary

        if args.action == 1:
            if not optional_question:
                if args.question_selection == 0:
                    prediction['response'] = prediction_response + " " + question_to_append

    output_file = args.input_file.split(".")
    output_file[-2] = output_file[-2] + f"_question-style{args.action}"
    output_file = ".".join(output_file)
    with open(f'./../pred/{dataset_to_read}/{output_file}', "w") as jsonfile:
        json.dump(predictions, jsonfile, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default=1, type=int, choices=[0, 1],
                        help="Which action to take: "
                             "0 always remove questions from the end; "
                             "1 always append question at the end; ")
    parser.add_argument('--question_selection', default=0, type=int, choices=[0, 1],
                        help="Mechanism to select the question to append: "
                             "0 patch with the most frequent question; "
                             "1 beam_search; ")
    parser.add_argument('--test_set', default=False, action='store_true',
                        help="Run on the test set instead of the validation set")
    parser.add_argument('--input_file', default='baseline.rg.prompt-style1.json',
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
