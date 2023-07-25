import json
import os
import time
import traceback

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# process entity reviews
def process_entity(reviews_all, domain, entity_name):
    # Prompt
    prompt = f'Please provide new reviews for the {entity_name} but for five different traveler_type. ' \
             f'Continue the counting of the reviews and make sure the new reviews are in a dict format like this: ' \
             f'"<id>": {{"traveler_type": "<traveler_type>", "sentences": {{"<id>": "<review>", "<id>": "<review>", "<id>": "<review>", "<id>": "<review>"}}}}, ' \
             f'These are the existing reviews: {reviews_all}. ' \
             f'Take this start and continue. Use double quotes to comply with json format.' \
             f'"10": {{"traveler_type": ' \


    prompt = [{"role": "user", "content": prompt}]

    # print(f'**PROMPT** {prompt}')

    try:
        # Send request
        time.sleep(0.4)
        response = chatgpt(prompt)

        # print(f"**RESPONSE** {response['text']}")

        start = f'"10": {{"traveler_type": '
        reviews_new = "{" + start + response['text'] + "}"

        print(f'**MERGED** {reviews_new}')

        reviews_new_dict = json.loads(reviews_new)

        reviews_all.update(reviews_new_dict)


    except Exception:
        traceback.print_exc()
        print(f"Did nae work.")

    return reviews_all


def chatgpt(utterances, model="gpt-3.5-turbo"):
    """ functions to call ChatGPT predictions """
    response = openai.ChatCompletion.create(
        model=model,
        messages=utterances,
        temperature=0,
        top_p=1,
        max_tokens=1024,
        frequency_penalty=0,
        presence_penalty=0
    )
    return {
        'prompt': utterances,
        'text': response['choices'][0]['message']['content'],
        'finish_reason': response['choices'][0]['finish_reason']
    }


# Load knowledge from file
with open('/Users/lea/projects/dstc/dstc11-track5/data/knowledge.json') as f:
    knowledge = json.load(f)

    # Process each entity in knowledge
    for domain in knowledge:
        for entity, entity_info in knowledge[domain].items():
            entity_name = entity_info["name"]
            entity_reviews = entity_info['reviews']
            if len(entity_reviews) < 14:
                print(f'Entity: {entity_name}')
                entity_info['reviews'] = process_entity(entity_reviews, domain, entity_name)

    # Save extended knowledge to file
    with open('/data/knowledge_aug_reviews.json', 'w') as nf:
        json.dump(knowledge, nf)
