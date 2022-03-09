# Project 1: Text Categorization. 

Text categorization of news articles and captions using bag-of-words Naive Bayes

## Required Python and Libraries
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