import numpy as np
from utils import *      
import argparse
import json

parser = argparse.ArgumentParser(description='Test the Naive Bayes model on a corpus')
parser.add_argument("test_files_list_path")
args = parser.parse_args()
test_files_list_path = args.test_files_list_path

# read the word data from the json file generated during training
training_word_data = open("training_word_data.json", "r")
d_prior_counts, d_word_occurs = json.load(training_word_data)

file_list = open(test_files_list_path).read().splitlines()
out_file = open("corpus1_predictions.labels", "w")

for path in file_list:
    cur_file = open(path, "r")
    file_str = cur_file.read()
    cat_est = estimate_category(file_str, d_prior_counts, d_word_occurs)
    out_file.write(path + " " + cat_est + "\n")