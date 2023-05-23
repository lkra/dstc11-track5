import argparse

from utils.helpers import read_predictions, write_predictions
from utils.questions_helpers import most_frequent_question


def clean_question_step(response):
    question_step = response.split("\n\n(3) response:", 1)[0]

    # Remove None questions
    if "None" in question_step:
        question_step = ''

    return question_step


def parse_api_response(prediction: dict, prompt_style: int = 0, clean_style: int = 0):
    response = prediction['response']

    if prompt_style in [0, 1]:
        return response
    elif prompt_style in [2, 3]:
        try:
            if clean_style == 0:
                response = response.split("\n\n(3) response:", 1)[1]
                # Remove None questions
                if ". None" in response:
                    response = response[:-4]

                return response.lstrip()
            elif clean_style == 1:
                if prompt_style in [2]:
                    [summary_step, response] = response.split("\n\n(2) follow-up:", 1)
                    question_step = clean_question_step(response)
                    return f"{summary_step} {question_step.lstrip()}"
                elif prompt_style in [3]:
                    prompt = prediction['prompt'][0]['content']
                    # take the part between summary and followup
                    summary_step = prompt.split("(1) summary:", 1)[1].split("\n\n(2) follow-up:", 1)[0]
                    question_step = clean_question_step(response)
                    return f"{summary_step.strip()} {question_step.lstrip()}"

        except Exception:
            return response


def main(args):
    approach = args.prediction_file.rsplit(".", 1)[0]
    dataset_to_read = "test" if args.test_set else "val"
    data = read_predictions(dataset_to_read, prediction_file=args.prediction_file)
    mfq = most_frequent_question()

    for index, prediction in enumerate(data):
        if prediction['target']:
            if 'response' in prediction.keys() and not prediction['response'].isupper():
                prediction['response'] = parse_api_response(prediction, args.prompt_style, args.clean_style)
            else:
                if 'response' in prediction.keys():
                    print(prediction['response'])
                prediction['response'] = mfq

    # Write final results to file
    write_predictions(data, f'{approach}_clean-style{args.clean_style}.json', dataset_to_read=dataset_to_read)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_style', default=2, type=int, choices=[2, 3],
                        help="Which prompt style to use: "
                             "2 ChatGPT 3steps style only using one user message; ")
    parser.add_argument('--clean_style', default=0, type=int, choices=[0, 1],
                        help="Which cleaning strategy to use: "
                             "0 take the explicit response part; "
                             "1 take the step 1 and 2 hack; ")
    parser.add_argument('--test_set', default=False, action='store_true',
                        help="Run on the test set instead of the validation set")
    parser.add_argument('--prediction_file', default='baseline.rg.bart-base.json', type=str,
                        help="Which file with predictions to compare to the ground truth")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
