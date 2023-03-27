import argparse
import os
import time
from typing import Union

import numpy as np
import openai
import spacy
import tiktoken
from dotenv import load_dotenv
from tqdm import tqdm

from scripts.dataset_walker import DatasetWalker
from utils.helpers import read_predictions, write_predictions, hardest_examples
from utils.nlp_helpers import split_summary_question

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def gpt3(prompts: list, model: str = "text-davinci-002"):
    """ functions to call GPT3 predictions, works on batches """
    response = openai.Completion.create(
        model=model,
        prompt=prompts,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=1
    )
    return [
        {
            'prompt': prompt,
            'text': response['text'],
            'tokens': response['logprobs']['tokens'],
            'logprob': response['logprobs']['token_logprobs']
        }
        for response, prompt in zip(response.choices, prompts)
    ]


def chatgpt(utterances: list, model: str = "gpt-3.5-turbo"):
    """ functions to call ChatGPT predictions """
    response = openai.ChatCompletion.create(
        model=model,
        messages=utterances,
        temperature=0,
        top_p=1,
        max_tokens=100,  # Manually change to 60 with prompt style 1
        frequency_penalty=0,
        presence_penalty=0
    )
    return {
        'prompt': utterances,
        'text': response['choices'][0]['message']['content'],
        'finish_reason': response['choices'][0]['finish_reason']
    }


def format_knowledge(reference: dict):
    if "knowledge" not in reference.keys():
        return ""

    k = []
    for el in reference['knowledge']:
        if 'sent' in el.keys():
            k.append(f"{el['doc_type'].upper()}: {el['sent']}")
        else:
            k.append(f"{el['doc_type'].upper()}: Q: {el['question']} A: {el['answer']}")

    k = '\n\t'.join(k)
    return k


def resolve_speaker(speaker: str, prompt_style: int = 0):
    if speaker == "U":
        return "user" if prompt_style == 0 else "USER"
    elif speaker == "S":
        return "assistant" if prompt_style == 0 else "YOU"


def num_tokens_from_messages(messages: Union[list, str], model: str = "gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    encoding = tiktoken.encoding_for_model(model)
    if model == "gpt-3.5-turbo":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    if model == "text-davinci-002":
        num_tokens = len(encoding.encode(messages))
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")


def get_examples(n: int, example_data, prompt_style: int = 0) -> str:
    """
    Get n example prompts from example_data dataset, using `prompt_style` and `goal`
    """
    temp = hardest_examples(n)
    nlp = spacy.load('en_core_web_sm')

    # Choose samples by index
    shown_example_idx = np.random.randint(0, len(example_data), size=n)

    # Retrieve them from the dataset and format their prompts
    example_prompts = []
    for idx in shown_example_idx:
        # Format example in the same way
        log, reference = example_data[idx]
        prompt_sample, prompt_tokens = get_prompt_text(log, reference, response=True, prompt_style=prompt_style)

        summary, question = split_summary_question(reference['response'], nlp)
        prompt_end = f"""{summary}\n\n(2) follow-up:\n         {question}"""

        example_prompts.append(prompt_sample + prompt_end)

    example_prompts = [el for el in example_prompts if el != ""]
    return "\n".join(example_prompts)


def get_prompt_text(dialogue_context: list,
                    reference: dict,
                    response: bool = False,
                    prompt_style: int = 0) -> (str, str):
    """ Function to create a prompt """
    # skip if instance does not need to be verbalized
    if not reference['target']:
        return "", 0

    # Baseline for ChatGPT
    if prompt_style == 0:
        dialogue_context = [{"role": f"{resolve_speaker(el['speaker'])}", "content": f"{el['text']}"} for el in
                            dialogue_context]
        knowledge = format_knowledge(reference)
        knowledge = {"role": "system",
                     "content": f"You are a helpful assistant with access to the following:\n\t{knowledge}"}

        prompt = [knowledge] + dialogue_context
        return prompt, num_tokens_from_messages(prompt)

    # Baseline for GPT3
    elif prompt_style == 1:
        dialogue_context = '\n\t'.join([f"{el['speaker']}: {el['text']}" for el in dialogue_context])
        knowledge = format_knowledge(reference)
        if response:
            response = reference['response']
        else:
            response = ''

        prompt = f"""-------------------
DIALOGUE:\n\t{dialogue_context}
\nKNOWLEDGE:\n\t{knowledge}
\nRESPONSE: {response}"""

        return prompt, num_tokens_from_messages(prompt, model='text-davinci-002')

    # Step by step, instructions first ChatGPT
    elif prompt_style == 2:
        dialogue_context = '\n\t'.join(
            [f"{resolve_speaker(el['speaker'], prompt_style)}: {el['text']}" for el in dialogue_context])
        knowledge = format_knowledge(reference)

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

        return prompt, num_tokens_from_messages(prompt)


    else:
        raise ValueError(f"Unknown prompt style: {prompt_style}")


def main(args):
    dataset_to_read = "test" if args.test_set else "val"
    dev_data = DatasetWalker(dataset=dataset_to_read, dataroot="./../../data/", labels=True, incl_knowledge=True)
    # Hack to use the baseline knowledge
    dev_data.labels = read_predictions()

    example_prompt = ''
    if args.n_shot > 0:
        # Get examples
        train_data = DatasetWalker(dataset="train", dataroot="./../../data/", labels=True, incl_knowledge=True)
        train_data.filter_knowledge_only()
        example_prompt = get_examples(args.n_shot, train_data, args.prompt_style)

    # Prepare storage of labels
    total_tokens = 0
    for i, (log, reference) in tqdm(enumerate(dev_data)):
        # skip if instance without response
        if reference['target']:
            # Get the prompt for asking about the sample
            prompt, prompt_tokens = get_prompt_text(log, reference, prompt_style=args.prompt_style)

            if args.dry_run:
                prediction = {"text": "DUMMY RUN"}

            else:
                # Pass through model
                try:
                    # Ugly hack to adhere to rate limit (1 / s)
                    time.sleep(0.4)
                    if args.prompt_style in [0, 2]:
                        # ChatGPT call
                        prompt = [example_prompt] + prompt if example_prompt != '' else prompt
                        prediction = chatgpt(prompt)
                        reference['finish_reason'] = prediction['finish_reason']
                    elif args.prompt_style in [1]:
                        # GPT3 call
                        prompt = f"{example_prompt}\n{prompt}"
                        [prediction] = gpt3([prompt])
                    elif args.prompt_style in [3]:
                        # Create summary with GPT3
                        partial_prompt = ''
                        [summary] = gpt3([partial_prompt])
                        # Insert summary in ChatGPT prompt
                        prompt = prompt.format(summary=summary)
                        # ChatGPT call
                        prompt = [example_prompt] + prompt if example_prompt != '' else prompt
                        prediction = chatgpt(prompt)
                        reference['finish_reason'] = prediction['finish_reason']
                    else:
                        raise ValueError(f"Unknown prompt style: {args.prompt_style}")
                except:
                    prediction = {"text": "DUMMY RUN"}

            # Count tokens to calculate price
            total_tokens += prompt_tokens
            print(f"\t{total_tokens} tokens, {total_tokens * (0.002 / 1000)} dollars")

            # Parse responses
            reference['prompt'] = prompt
            reference['prompt_tokens'] = prompt_tokens
            reference['response'] = prediction['text']
            print(f"INDEX: {i}, RESPONSE: {reference['response']}")

            if i % 500 == 0:
                # Write results to file as backup
                print(f"Saving {i + 1} records to file as backup")
                write_predictions(dev_data.labels, f'baseline.rg.prompt-style{args.prompt_style}_n{args.n_shot}.json',
                                  dataset_to_read=dataset_to_read)

    # Write final results to file
    write_predictions(dev_data.labels, f'baseline.rg.prompt-style{args.prompt_style}_n{args.n_shot}.json',
                      dataset_to_read=dataset_to_read)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_style', default=0, type=int, choices=[0, 1, 2],
                        help="Which prompt style to use: "
                             "0 ChatGPT style using user messages for dialogue history; "
                             "1 GPT3 style; "
                             "2 ChatGPT 3steps style only using one user message; ")
    parser.add_argument('--n_shot', default=0, type=int,
                        help="How many examples to give in the prompt")
    parser.add_argument('--test_set', default=False, action='store_true',
                        help="Run on the test set instead of the validation set")
    parser.add_argument('--dry_run', default=False, action='store_true',
                        help="Create the prompts but do not send them to the API")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
