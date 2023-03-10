import argparse
import json
import os
import time

import numpy as np
import openai
import tiktoken
from dotenv import load_dotenv
from tqdm import tqdm

from scripts.dataset_walker import DatasetWalker

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def gpt3(prompts, model="text-davinci-002"):
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


def chatgpt(utterances, model="gpt-3.5-turbo"):
    """ functions to call ChatGPT predictions """
    response = openai.ChatCompletion.create(
        model=model,
        messages=utterances,
        temperature=0,
        top_p=1,
        max_tokens=100,
        frequency_penalty=0,
        presence_penalty=0
    )
    return {
        'prompt': utterances,
        'text': response['choices'][0]['message']['content'],
        'finish_reason': response['choices'][0]['finish_reason']
    }


def format_knowledge(reference):
    k = '\n\t'.join([f"{el['doc_type'].upper()}: {el['sent']}" for el in reference['knowledge'] if 'sent' in el.keys()])
    return k


def resolve_speaker(speaker):
    if speaker == "U":
        return "user"
    elif speaker == "S":
        return "assistant"


def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
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
    # Choose samples by index
    shown_example_idx = np.random.randint(0, len(example_data), size=n)

    # Retrieve them from the dataset and format their prompts
    example_prompts = []
    for idx in shown_example_idx:
        # Format example in the same way
        log, reference = example_data[idx]
        prompt_sample = get_prompt_text(log, reference, response=True, prompt_style=prompt_style)
        example_prompts.append(prompt_sample)

    example_prompts = [el for el in example_prompts if el != ""]
    return "\n".join(example_prompts)


def get_prompt_text(dialogue_context: list,
                    reference: dict,
                    response: bool = False,
                    prompt_style: int = 0) -> (str, str):
    """ Function to create a prompt """
    # skip if instance does not need to be verbalized
    if not reference['target']:
        return ""

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

    else:
        raise ValueError(f"Unknown prompt style: {prompt_style}")


def main(args):
    train_data = DatasetWalker(dataset="train", dataroot="./../data/", labels=True, incl_knowledge=True)
    dataset_to_read = "test" if args.test_set_predictions else "val"
    dev_data = DatasetWalker(dataset=dataset_to_read, dataroot="./../data/", labels=True, incl_knowledge=True)
    # Hack to use the baseline knowledge
    with open(f"./../pred/{dataset_to_read}/baseline.rg.bart-base.json", 'r') as f:
        dev_data.labels = json.load(f)

    # Prepare storage of labels
    total_tokens = 0
    for log, reference in tqdm(dev_data):
        # skip if instance without response
        if reference['target']:
            # Get the prompt for asking about the sample
            prompt, prompt_tokens = get_prompt_text(log, reference, prompt_style=args.prompt_style)

            # Pass through model
            if args.prompt_style in [0]:
                # ChatGPT style
                prediction = chatgpt(prompt)
                reference['finish_reason'] = prediction['finish_reason']

            elif args.prompt_style in [1]:
                # Get examples
                if args.n_shot > 0:
                    example_prompt = get_examples(args.n_shot, train_data)
                    prompt = f"{example_prompt}\n{prompt}"

                # GPT3 style
                [prediction] = gpt3([prompt])

            else:
                raise ValueError(f"Unknown prompt style: {args.prompt_style}")

            # Ugly hack to adhere to rate limit (1 / s)
            time.sleep(1)

            # Count tokens to calculate price
            total_tokens += prompt_tokens
            print(f"{total_tokens} tokens, {total_tokens * (0.02 / 1000)} dollars")

            # Parse responses
            reference['prompt'] = prompt
            reference['prompt_tokens'] = prompt_tokens
            reference['response'] = prediction['text']

    # Write results to file as backup
    with open(f'./../pred/{dataset_to_read}/baseline.rg.prompt-style{args.prompt_style}.json', 'w') as f:
        json.dump(dev_data.labels, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_style', default=1, type=int, choices=[0, 1, 2, 3, 4, 5, 6],
                        help="Which prompt style to use: "
                             "0 ChatGPT style; "
                             "1 GPT3 style; ")
    parser.add_argument('--n_shot', default=0, type=int,
                        help="How many examples to give in the prompt")
    parser.add_argument('--test_set_predictions', default=False, action='store_true',
                        help="Run on the test set instead of the validation set")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)

# python -m scripts.scores --dataset val --dataroot data/ --outfile pred/val/prompting.rg.prompt-style0.json --scorefile pred/val/prompting.rg.prompt-style0.score.json
