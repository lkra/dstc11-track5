import json

with open('/Users/lea/projects/dstc/dstc11-track5/data/train/logs.json') as f:
    train = json.load(f)

with open('/Users/lea/projects/dstc/dstc11-track5/data/val/logs.json') as f:
    val = json.load(f)

train = train + val

with open('/Users/lea/projects/dstc/dstc11-track5/data/train/logs.json', 'w') as f:
    json.dump(train, f)
