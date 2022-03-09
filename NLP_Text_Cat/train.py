import numpy as np
from utils import *
import argparse
import json
from os.path import exists

parser = argparse.ArgumentParser(description='Train the Naive Bayes model')
parser.add_argument("training_file_path", help = "path to file with paths of\
    training corpus and labels")
parser.add_argument("word_data_file", help = "file to store word data from training")
args = parser.parse_args()
training_file_path = args.training_file_path

# estimate priors from current training data and add to overall priors data
d_priors = est_prior(training_file_path)

# dict of word occurences. For each word, counts for occurences of that word
# in each category of document are listed
d_word_occurs = {}

# form dict whose keys are the categories and values are 0
d_cats_default = zero_cat_dict(d_priors)

training_file_handle = open(training_file_path, "r")
training_lines = training_file_handle.readlines()

for line in training_lines:
    data_path = line.split()[0]
    true_cat = line.split()[1]

    data = open(data_path, "r")
    data_str = data.read()

    # prepare data into more concise form
    tokenized_data = tokenize(data_str)
    filtered_data = filter(tokenized_data)
    stemmed_data = stem(filtered_data)

    for idx, token in enumerate(stemmed_data):
        if token not in d_word_occurs:
            d_word_occurs[token] = deepcopy(d_cats_default)
            d_word_occurs[token][true_cat] = 1
        elif token not in stemmed_data[0:idx]:
            d_word_occurs[token][true_cat] = d_word_occurs[token].get(true_cat, 0) + 1

out_file = open(args.word_data_file, "w")
json.dump([d_priors, d_word_occurs], out_file)