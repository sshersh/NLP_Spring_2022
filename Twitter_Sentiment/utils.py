import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def prepare_data(df):
    df.drop(['selected_text', 'textID'], axis=1, inplace=True)
    print(df.head())

    df.dropna(inplace=True)

    # get rid of "neutral" values since we want to do binary classification
    df1 = df[df['sentiment'] != 'neutral']
    df1.to_csv('./Data/no_neutral_tweets')

    num_samples = df1.shape[0]
    print('\nThe dataset contains ' + str(num_samples) + ' samples')

    # separate text and labels
    full_text = df1['text']
    labels = df1['sentiment']

    labels_encoded = []

    class_names = ['negative', 'positive']

    for label in labels:
        labels_encoded.append(1 if (label == 'positive') else 0)

    # create the tokenizer
    tk = Tokenizer(lower=True, filters='')

    # find the frequency of each word (TF)
    tk.fit_on_texts(full_text)

    # Twitter already has a character limit. In the worst case each token is a single character.
    max_line_len = 120

    # convert the tweets to sequences of TF counts
    train_tokenized = tk.texts_to_sequences(full_text)

    # pad the sequences to the character limit
    data = pad_sequences(train_tokenized, maxlen=max_line_len)

    return (data, labels_encoded, num_samples)

def get_training_data(data, labels, num_samples, ratio):
    # separate into training and testing data. Split is 80/20
    num_train_samples = int(num_samples * ratio)
    train_samples = data[:num_train_samples]
    train_labels = labels[:num_train_samples]

    return (train_samples, train_labels)

def get_testing_data(data, labels, num_samples, ratio):
    # separate into training and testing data. Split is 80/20
    num_train_samples = int(num_samples * (1 - ratio))
    test_samples = data[:num_train_samples]
    test_labels = labels[:num_train_samples]

    return (test_samples, test_labels)