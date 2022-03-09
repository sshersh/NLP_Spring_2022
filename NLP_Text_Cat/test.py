import numpy as np
from utils import *      
import argparse
import json

corpus_cats = {"corpus1": ["Str", "Pol", "Dis", "Cri", "Oth"], 
            "corpus2": ["I", "O"],
            "corpus3":  ["Wor", "USN", "Sci", "Fin", "Spo", "Ent"]}

def estimate_category(doc_string, d_priors, d_word_occurs, cat_options=None):
    """ Run the trained Bayes model on a document

    doc_string      :   text string
    d_priors        :   dict of priors
    d_word_occurs   :   dict of word occurences in each category
    cat_options     :   optional list of valid categories (based on corpus)
    """
    tokenized_doc = tokenize(doc_string)
    filtered_doc = filter(tokenized_doc)
    stemmed_doc = stem(filtered_doc)

    d_log_priors = get_log_priors(d_priors)
    d_post = deepcopy(d_log_priors)

    if cat_options is None:
        cat_options = list(d_priors.keys())

    for token in stemmed_doc:
        if token in d_word_occurs:
            for cat in cat_options:
                if cat in d_word_occurs[token]:
                    d_post[cat] += np.log((d_word_occurs[token][cat] + 1) /
                                    (d_priors[cat] + 1))
    # return the argmax
    return max(d_post, key=d_post.get)  

parser = argparse.ArgumentParser(description='Test the Naive Bayes model on a corpus')
parser.add_argument("test_files_list_path", help="path to file with test corpus paths")
parser.add_argument("word_data_file", help = "file with training word data")
parser.add_argument("output_file_name", help = "name of file with data and predicted labels")
args = parser.parse_args()
test_files_list_path = args.test_files_list_path

# read the word data from the json file generated during training
training_word_data = open(args.word_data_file, "r")
d_prior_counts, d_word_occurs = json.load(training_word_data)

file_list = open(test_files_list_path).read().splitlines()
out_file = open(args.output_file_name, "w")

# pass through 1st 10 lines to determine which corpus is being used
corpus_counts = {"corpus1":0,
                 "corpus2":0,
                "corpus3":0}
cur_corpus = ""
# for idx, line in enumerate(file_list):
#     if idx == 10:
#         break
#     path = line.split()[0]
#     cur_file = open(path, "r")
#     file_str = cur_file.read()
#     cat_est = estimate_category(file_str, d_prior_counts, d_word_occurs)

#     # increment the corpus whose category has been detected
#     for corpus in corpus_cats:
#         if cat_est in corpus_cats[corpus]:
#             corpus_counts[corpus] += 1
#             break
#     cur_corpus = max(corpus_counts, key=corpus_counts.get) 
    
# print("Corpus Detected: ", cur_corpus)

# run the Bayes model
for line in file_list:
    path = line.split()[0]
    cur_file = open(path, "r")
    file_str = cur_file.read()
    cat_est = estimate_category(file_str, d_prior_counts, d_word_occurs)
    out_file.write(path + " " + cat_est + "\n")