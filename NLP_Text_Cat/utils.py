from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from copy import deepcopy
import numpy as np

def tokenize(data_str):
    """Tokenization is the first step, so the file handle is input to this function.

    data_str    :   input text string
    """
    tokenized_data = word_tokenize(data_str)
    return tokenized_data

def filter(tokenized_arr):
    """Filter the input data by removing tokens that add little info.

    tokenized_arr    :   array of tokens
    """
    filtered = []
    for token in tokenized_arr:
        if len(token) > 2 or token in ['!']:
            filtered.append(token)
    return filtered
            

def stem(tokenized_arr):
    """ Stem the words 
    """
    stemmer = PorterStemmer()
    stemmed = []
    for token in tokenized_arr:
        stemmed.append(stemmer.stem(token))
    return stemmed

def est_prior(training_file_path):
    """ Estimate the Prior distribution of categories by counting number of occurences of each category

    training_file  :   path to file with labels

    returns :   dict with labels and counts (how many files correspond to each 
                category)
    """
    d_priors = {}

    training_file = open(training_file_path)
    lines = training_file.readlines()

    for line in lines:
        cur_label = line.split()[-1]

        if cur_label in d_priors:
            d_priors[cur_label] += 1
        else:
            d_priors[cur_label] = 1

    return d_priors

def zero_cat_dict(d_priors):
    d_cats_zero = deepcopy(d_priors)
    for cat in d_cats_zero:
        d_cats_zero[cat] = 0
    return d_cats_zero

def get_log_priors(d_priors):
    """ d_priors returned by est_prior just stores counts. Need log probabilies

    d_priors    :   dict of prior counts
    """
    total_training_docs = 0
    d_log_priors = {}
    for cat in d_priors:
        total_training_docs += d_priors[cat]
    for cat in d_priors:
        d_log_priors[cat] = np.log(d_priors[cat] / total_training_docs)
    
    return d_log_priors