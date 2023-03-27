import argparse
import json
import os

import openai
from dotenv import load_dotenv

from scripts.knowledge_reader import KnowledgeReader

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


def main(args):
    augmented_knowledge = KnowledgeReader("../data/aug/")
    entities = augmented_knowledge.get_entity_list(args.domain)

    new_knowledge = {args.domain: {}}

    for entity in entities:
        entity_id = entity["id"]
        entity_name = entity["name"]
        entity_obj = augmented_knowledge.knowledge[args.domain][str(entity_id)]
        faqs = {
            doc_id: {
                "question": doc_obj["title"],
                "answer": doc_obj["body"]
            }
            for doc_id, doc_obj in entity_obj["docs"].items()
        }

        new_entity = {
           entity_id : {
                "name": entity_name,
                "faqs": faqs,
           }
        }

        prompt_text = f"Given this example: {new_entity}, can you generate three more reviews, not more than 2 sentences, as: travele type: review?"
        prompt = [{
            "role": "system",
            "content": prompt_text,
        }]

        output = chatgpt(prompt)
        response = output["text"]
        entity_reviews = response.split("\n\n")

        reviews = {}
        for i, review in enumerate(entity_reviews):
            split_review = review.split(":")
            traveler_type = split_review[0]
            traveler_review = split_review[1]
            # traveler_review = {str(j): sentence if sentence[-1] == "." else f"{sentence}." for j, sentence in enumerate(traveler_review.split("."))}

            traveler_review = {}
            for j, sentence in enumerate(split_review[1].split(".")):
                sentence = sentence.strip().replace('"', "")
                if sentence:
                    if sentence[-1] != ".":
                        sentence = f"{sentence}."
                    traveler_review[str(j)] = sentence

            reviews[str(i)] = {
                "traveler_type": traveler_type,
                "sentences": traveler_review,
            }

        new_knowledge[args.domain][str(entity_id)] = {
            "name": entity_name,
            "reviews": reviews,
            "faqs": faqs,
        }

    with open(args.output_file, "w") as f:
        json.dump(new_knowledge, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_style', default=1, type=int, choices=[0, 1],
                        help="Which prompt style to use: "
                             "0 ChatGPT style; "
                             "1 GPT3 style; ")
    parser.add_argument('--domain', default="taxi", type=str,
                        help="TBA")
    parser.add_argument('--output_file', type=str,
                        help="TBA")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
