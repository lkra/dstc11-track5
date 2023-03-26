import argparse
import os
import time
from typing import Union

import numpy as np
import openai
import tiktoken
from dotenv import load_dotenv
from tqdm import tqdm

from scripts.dataset_walker import DatasetWalker
from utils.helpers import read_predictions, write_predictions


def parse_api_response(response: str, prompt_style: int = 0, clean_style: int = 0):
    if prompt_style in [0, 1]:
        return response
    elif prompt_style == 2:
        try:
            if clean_style == 0:
                return response.split("\n\n(3) response:", 1)[1]
            elif clean_style == 1:
                [step1, response] = response.split("\n\n(2) follow-up:", 1)
                step2 = response.split("\n\n(3) response:", 1)[0]
                return f"{step1} {step2.lstrip()}"
        except:
            return response


def main(args):
    dataset_to_read = "test" if args.test_set else "val"
    data = read_predictions(dataset_to_read, prediction_file=f'baseline.rg.prompt-style{args.prompt_style}.json')

    for prediction in data:
        if prediction['target']:
            prediction['response'] = parse_api_response(prediction['response'], args.prompt_style, args.clean_style)

    # Write final results to file
    write_predictions(data, f'baseline.rg.prompt-style{args.prompt_style}_clean-style{args.clean_style}.json',
                      dataset_to_read=dataset_to_read)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_style', default=2, type=int, choices=[0, 1, 2],
                        help="Which prompt style to use: "
                             "0 ChatGPT style using user messages for dialogue history; "
                             "1 GPT3 style; "
                             "2 ChatGPT style only using one user messages with max_tokens=50; ")
    parser.add_argument('--clean_style', default=0, type=int, choices=[0, 1],
                        help="Which cleaning strategy to use: "
                             "0 take the explicit response part; "
                             "1 take the step 1 and 2 hack; ")
    parser.add_argument('--test_set', default=False, action='store_true',
                        help="Run on the test set instead of the validation set")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
