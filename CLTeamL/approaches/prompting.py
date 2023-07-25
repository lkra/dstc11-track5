import os
import sys
from pathlib import Path

SCRIPTS_PATH = "./../../"
sys.path.append(str(Path(SCRIPTS_PATH).resolve()))
CLTEAML_PATH = "./../"
sys.path.append(str(Path(CLTEAML_PATH).resolve()))
print(sys.path)

import argparse
import time

import openai
import spacy
from dotenv import load_dotenv
from tqdm import tqdm

from scripts.dataset_walker import DatasetWalker
from utils.helpers import read_predictions, write_predictions
from utils.nlp_helpers import split_summary_question
from utils.prompting_helpers import EXAMPLE_SEPARATOR, gpt3, chatgpt, format_knowledge, resolve_speaker, \
    num_tokens_from_messages, append_prompt_examples, hardest_examples, random_examples

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_examples(n: int, prompt_style: int = 0) -> str:
    """
    Get n example prompts from example_data dataset, using `prompt_style` and `goal`
    """
    nlp = spacy.load('en_core_web_sm')

    if prompt_style in [2]:
        # Choose examples by performance
        examples = hardest_examples(n)
    else:
        # Choose samples by index
        examples = random_examples(n)

    # Retrieve them from the dataset and format their prompts
    example_prompts = []
    for log, reference in examples:
        # Format example in the same way
        prompt_sample = get_prompt_text(log, reference, prompt_style=prompt_style)

        summary, question = split_summary_question(reference['response'], nlp)
        prompt_end = f"""{summary}\n\n(2) follow-up:\n         {question}\n\n(3) response:\n         {reference['response']}"""

        example_prompts.append(prompt_sample[0]['content'] + prompt_end)

    example_prompts = [el for el in example_prompts if el != ""]
    return EXAMPLE_SEPARATOR.join(example_prompts)


def get_prompt_text(dialogue_context: list,
                    reference: dict,
                    prompt_style: float = 0) -> (str, str):
    """ Function to create a prompt """
    # skip if instance does not need to be verbalized
    if not reference['target']:
        prompt = ""

    # Baseline for ChatGPT
    elif prompt_style == 0:
        dialogue_context = [{"role": f"{resolve_speaker(el['speaker'])}", "content": f"{el['text']}"} for el in
                            dialogue_context]
        knowledge = format_knowledge(reference, prompt_style)
        knowledge = {"role": "system",
                     "content": f"You are a helpful assistant with access to the following:\n\t{knowledge}"}
        prompt = [knowledge] + dialogue_context

    # Baseline for GPT3
    elif prompt_style == 1:
        dialogue_context = '\n\t'.join([f"{el['speaker']}: {el['text']}" for el in dialogue_context])
        knowledge = format_knowledge(reference, prompt_style)
        prompt = f"""-------------------
DIALOGUE:\n\t{dialogue_context}
\nKNOWLEDGE:\n\t{knowledge}
\nRESPONSE: """

    # Step by step, instructions for ChatGPT
    elif prompt_style == 2:
        dialogue_context = '\n\t'.join(
            [f"{resolve_speaker(el['speaker'], prompt_style)}: {el['text']}" for el in dialogue_context])
        knowledge = format_knowledge(reference, prompt_style)
        prompt = f"""You are assisting a user. Create a response for the user, using the following procedure:
(1) First, summarise the available knowledge into a couple sentences.
(2) Then, create a short follow-up question given the dialogue history.
(3) Create the final response to the user as <summary><follow-up>

Knowledge:
    {knowledge}

Dialogue history:
    {dialogue_context}


Solution:
(1) summary:
        """
        prompt = [{"role": "user", "content": prompt}]

    # GPT3 for summarization only
    elif prompt_style == 3.1:
        knowledge = format_knowledge(reference, prompt_style)
        prompt = f"""Summarize the following into one or two sentences max:\n\n{knowledge}"""

    # Step by step, instructions for ChatGPT (emphasize brief)
    elif prompt_style == 3:
        dialogue_context = '\n\t'.join(
            [f"{resolve_speaker(el['speaker'], prompt_style)}: {el['text']}" for el in dialogue_context])
        knowledge = format_knowledge(reference, prompt_style)
        prompt = f"""You are assisting a user. Create a response for the user, using the following procedure:
(1) First, summarise the available knowledge into a couple sentences.
(2) Then, create a short follow-up question given the dialogue history.
(3) Create a final brief response to the user as <summary><follow-up>

Knowledge:
    {knowledge}

Dialogue history:
    {dialogue_context}


Solution:
(1) summary:
        """
        prompt = [{"role": "user", "content": prompt}]

    else:
        raise ValueError(f"Unknown prompt style: {prompt_style}")

    return prompt


def main(args):
    dataset_to_read = "test" if args.test_set else "val"
    dev_data = DatasetWalker(dataset=dataset_to_read, dataroot="./../../data/", labels=dataset_to_read == 'val',
                             incl_knowledge=True)
    # Hack to use the baseline knowledge
    prediction_file = "baseline.ks.deberta-v3-base.json" if args.test_set else "baseline.rg.bart-base.json"
    dev_data.labels = read_predictions(dataset_to_read=dataset_to_read, prediction_file=prediction_file)

    example_prompt = ''
    if args.n_shot > 0:
        # Get examples
        example_prompt = get_examples(args.n_shot, args.prompt_style)

    # Prepare storage of labels
    total_tokens = 0
    for i, (log, reference) in tqdm(enumerate(dev_data)):
        # skip if instance without response
        if reference['target']:
            # Get the prompt for asking about the sample
            prompt = get_prompt_text(log, reference, prompt_style=args.prompt_style)
            prompt = append_prompt_examples(prompt, example_prompt, args.prompt_style) if args.n_shot > 0 else prompt
            prompt_tokens = num_tokens_from_messages(prompt,
                                                     model='text-davinci-002' if args.prompt_style == 1
                                                     else "gpt-3.5-turbo")

            if args.dry_run:
                prediction = {"text": "DUMMY RUN"}

            else:
                # Pass through model
                try:
                    # Ugly hack to adhere to rate limit (1 / s)
                    time.sleep(0.4)
                    if args.prompt_style in [0, 2]:
                        # ChatGPT call
                        prediction = chatgpt(prompt)
                        reference['finish_reason'] = prediction['finish_reason']
                    elif args.prompt_style in [1]:
                        # GPT3 call
                        [prediction] = gpt3([prompt])
                    elif args.prompt_style in [3]:
                        # Create summary with GPT3
                        partial_prompt = get_prompt_text(log, reference, prompt_style=3.1)
                        [summary] = gpt3([partial_prompt])
                        # Insert summary in ChatGPT prompt
                        prompt_end = f"""{summary['text']}\n\n(2) follow-up:\n         """
                        prompt[0]['content'] = prompt[0]['content'] + prompt_end
                        # ChatGPT call
                        prediction = chatgpt(prompt)
                    else:
                        raise ValueError(f"Unknown prompt style: {args.prompt_style}")

                except Exception:
                    prediction = {"text": "FAILED RUN"}

            # Count tokens to calculate price
            total_tokens += prompt_tokens
            print(f"\t{total_tokens} tokens, {total_tokens * (0.002 / 1000)} dollars")

            # Parse responses
            reference['prompt'] = prompt
            reference['prompt_tokens'] = prompt_tokens
            reference['response'] = prediction['text']

            # Ugly logging
            print(f"INDEX: {i}, RESPONSE: {reference['response']}")
            if i % 500 == 0:
                # Write results to file as backup
                print(f"Saving {i + 1} records to file as backup")
                write_predictions(dev_data.labels, f'baseline.rg.prompt-style{args.prompt_style}_n{args.n_shot}.json',
                                  dataset_to_read=dataset_to_read)

    # Write final results to file
    write_predictions(dev_data.labels, f'baseline.rg.prompt-style{args.prompt_style}_n{args.n_shot}.json',
                      dataset_to_read=dataset_to_read, dataroot=args.write_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_style', default=0, type=int, choices=[0, 1, 2, 3],
                        help="Which prompt style to use: "
                             "0 ChatGPT style using user messages for dialogue history; "
                             "1 GPT3 style; "
                             "2 ChatGPT 3steps style only using one user message; "
                             "3 GPT3 for summarization, ChatGPT 3steps style; ")
    parser.add_argument('--n_shot', default=0, type=int,
                        help="How many examples to give in the prompt")
    parser.add_argument('--test_set', default=False, action='store_true',
                        help="Run on the test set instead of the validation set")
    parser.add_argument('--prediction_file', default='baseline.ks.deberta-v3-base.json', type=str,
                        help="Which file with predictions to compare to the ground truth")
    parser.add_argument('--write_folder', default='./../../pred/', type=str,
                        help="Which folder to write predictions to")
    parser.add_argument('--dry_run', default=False, action='store_true',
                        help="Create the prompts but do not send them to the API")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
