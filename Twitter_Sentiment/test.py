import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from utils import *

model = keras.models.load_model('./Models/mod')

df = pd.read_csv('./Data/Tweets.csv', header=0)

data_encoded, labels_encoded, num_samples = prepare_data(df)
test_samples, test_labels = get_testing_data(data_encoded, labels_encoded, num_samples, 0.8)

loss, accuracy = model.evaluate(test_samples, test_labels)
print("Loss: ", loss)
print("Accuracy: ", accuracy)