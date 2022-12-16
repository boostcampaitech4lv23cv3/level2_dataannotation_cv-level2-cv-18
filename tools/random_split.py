# For UFO format Only
# Convert "annotation.json" to ["train.json", "val.json"]
# Usage : python random_split.py ~/~/input/data/ICDAR17_Korean/ufo

import os
import sys
import numpy as np
import json
import pandas
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


## File exists check
if len(sys.argv) != 2 or os.path.isdir(sys.argv[1]) == False:
    print("Usage : python random_split.py DATA_DIR")
    print("    Ex) python random_split.py ~/input/data/ICDAR17_Korean/ufo")
    exit()

ann_file = os.path.join(sys.argv[1], "annotation.json")
if os.path.isfile(ann_file) == False:
    print("Not found Annotation file")
    print(f" --> Required [{ann_file}]")
    exit()

with open(ann_file) as json_file:
    file_contents = json_file.read()
annotation_json = (json.loads(file_contents))['images']

shuffle_keys = list(annotation_json.keys())
random.shuffle(shuffle_keys)

train_size = max(1, int(len(shuffle_keys) * 0.8))
tes_size = len(shuffle_keys) - train_size

train = dict()
test = dict()

for i in range(train_size):
    train[shuffle_keys[i]] = annotation_json[shuffle_keys[i]]

for i in range(train_size, train_size + tes_size):
    test[shuffle_keys[i]] = annotation_json[shuffle_keys[i]]

print(f'Total: {len(shuffle_keys)}')
print(f'Train: {len(train)}')
print(f'Valid: {len(test)}')

with open(os.path.join(sys.argv[1], "train.json"), "w") as json_file:
    json.dump({"images": train}, json_file, indent=4)
with open(os.path.join(sys.argv[1], "val.json"), "w") as json_file:
    json.dump({"images": test}, json_file, indent=4)

print("Success!")
