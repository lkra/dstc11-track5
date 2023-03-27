import argparse

from utils.helpers import read_predictions, write_predictions


def parse_api_response(response: str, prompt_style: int = 0, clean_style: int = 0):
    if prompt_style in [0, 1]:
        return response
    elif prompt_style == 2:
        try:
            if clean_style == 0:
                response = response.split("\n\n(3) response:", 1)[1]
                return response.lstrip()
            elif clean_style == 1:
                [step1, response] = response.split("\n\n(2) follow-up:", 1)
                step2 = response.split("\n\n(3) response:", 1)[0]
                return f"{step1} {step2.lstrip()}"
        except:
            return response


def main(args):
    approach = args.prediction_file.rsplit(".", 1)[0]
    dataset_to_read = "test" if args.test_set else "val"
    data = read_predictions(dataset_to_read, prediction_file=args.prediction_file)

    for prediction in data:
        if prediction['target']:
            prediction['response'] = parse_api_response(prediction['response'], args.prompt_style, args.clean_style)

    # Write final results to file
    write_predictions(data, f'{approach}_clean-style{args.clean_style}.json', dataset_to_read=dataset_to_read)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_style', default=2, type=int, choices=[2],
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
