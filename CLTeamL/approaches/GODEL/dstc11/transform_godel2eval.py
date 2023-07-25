import json

with open('baseline.rg.bart-base.json', 'r') as f:
    godel = json.load(f)

with open('input.json', 'r') as f:
    baseline = json.load(f)

# transform data into list of sentences
sentences = []
for sublist in godel:
    sentence = ' '.join(sublist)
    sentences.append(sentence)

# loop through the list of dictionaries
for d in baseline:
    if d.get('target') and 'knowledge' in d:
        # replace the response with the next sentence from the list
        index = 0
        for k in d['knowledge']:
            if k.get('doc_type') == 'review':
                d['response'] = sentences[index]
                index += 1
        # if all sentences have been used, break out of the loop
        if index == len(sentences):
            break

with open('transformed_godel_baseline.json', 'w') as f:
    json.dump(sentences, f)

# write the modified json to a file
with open('baseline.rg.godel_formatted.json', 'w') as f:
    json.dump(baseline, f, indent=2)
