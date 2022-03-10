# Project 1: Text Categorization. 

Text categorization of news articles and captions using bag-of-words Naive Bayes

## Development Environment
- Linux (WSL 2)
- Python >= 3.8
- NLTK >= 3.6

## Training
To train the model on a labelled corpus of text documents, cd into the TC_provided directory and run:

```python ../train.py <training_file_path> <word_data_file>```

Where <training_file_path> is the path to the training file containing document paths along with their labels, and <word_data_file> is the file that word occurences will be written to. 

## Running
To run the model on a dataset, run:

```python ../test.py <test_files_list_path> <word_data_file> <output_file_name>```

Where <test_files_list_path> is the dataset and <output_file_name> is the name of the file the labelled data will be written to

## Description

The same method was used for training all 3 corpuses. During both training and running the model, each document is passed through the NLTK tokenizer which separates it into a list of tokens. This list is passed through NLTK's Porter Stemmer which replaces the tokens with their stems. For example "having" is stemmed to "have". The stemmed tokens are then passed through a filter which eliminates tokens of length 2 characters or less. This filter improved overall performance by a few percentage points for each corpus since these tokens are usually particles and prepositions so they typically don't add information to the document. This can be thought of as a crude version of TF*IDF where the IDF metric measures the importance of each word.

Training the model involves adding counts of each word found in each category, stored in a large dictionary. The dictionary is written to word_data.json for storage.

For corpus 2 and 3, testing sets were not provided, so the training sets were divided into a "training subset" (90% of the original file) and a "tuning subset" (10% of the original file) using a bash script. For corpus3, the lines were presorted by their labels so the file had to be shuffled randomly so the training and tuning subsets were statistically similar. 

Laplace smoothing was used to correct for the sparsity of the data. The k value used was 0.25, found empirically using binary seach. 

The final ratios of correct predictions achieved were:

|Corpus   | Ratio |
|---------|-------|
|Corpus 1 | 89%   |
|Corpus 2 | 83%   |
|Corpus 3 | 90%   |