#!/usr/bin/env python
#  coding=utf-8
#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.

import json
import random
import copy
import fire
import jsonlines


def process(filepath):
    kbs = json.load(open(f'{filepath}/knowledge.json'))
    import pdb
    examples = []

    random.seed(2022)
    for folder in ['train', 'val']: # TODO exclude val for validation, include when test set is released
        logs = json.load(open(f'{filepath}/{folder}/logs.json'))
        labels = json.load(open(f'{filepath}/{folder}/labels.json'))
        for log, label in zip(logs, labels):

            if label['target']:
                history = [i['text'] for i in log]
                response = label['response']
                kb_str = []
                for kb in label['knowledge']:
                    domain, entity, doc_type, doc = kb['domain'], kb['entity_id'], kb['doc_type'], kb['doc_id']
                    if doc_type == "review":
                        sent = kb['sent_id']
                        kb = kbs[domain][str(entity)]['reviews'][str(doc)]['sentences'][str(sent)]
                        kb_str.append(f'R: {kb}')
                    else:

                        kb = kbs[domain][str(entity)]['faqs']['docs'][str(doc)]
                        question, answer = kb['question'], kb['answer']
                        kb_str.append(f'Q: {question} A: {answer}')
                kb_str = ' '.join(kb_str)
                history = ' EOS '.join(history)
                example = {}
                example['Context'] = history
                example['Knowledge'] = kb_str
                example['Response'] = response
                examples.append(copy.deepcopy(example))

        if folder == 'train':
            with jsonlines.open(f'dstc11_{folder}.jsonl', mode='w') as writer:
                for i in examples:
                    if random.random() < 0.025:
                        writer.write(i)
        else:
            with jsonlines.open(f'dstc11_{folder}.jsonl', mode='w') as writer:
                for i in examples:
                    writer.write(i)


def main():
    fire.Fire(process)


if __name__ == '__main__':
    main()
