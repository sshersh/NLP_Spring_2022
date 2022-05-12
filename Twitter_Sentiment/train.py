import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

# was getting a lot of annoying warnings whenever importing tensorflow
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, Dense, LSTM, Softmax
    from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('Tweets.csv', header=0)
df.drop(['selected_text', 'textID'], axis=1, inplace=True)
print(df.head())

#simple data preparation code taken from 
# https://www.kaggle.com/code/yasserh/predicting-twitter-sentiments-top-ml-models
# and
# https://scorrea92.medium.com/nlp-twitter-sentiment-analysis-with-tensorflow-15e1b2594cfa

num_samples = df.shape[0]
print('\nThe dataset contains ' + str(num_samples) + ' samples')
df.dropna(inplace=True)

# get rid of "neutral" values since we want to do binary classification
df1 = df[df['sentiment'] != 'neutral']

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

# separate into training and testing data. Split is 80/20
num_train_samples = int(num_samples*0.8)
train_samples = data[:num_train_samples]
train_labels = labels_encoded[:num_train_samples]

test_samples = data[num_train_samples:]
test_labels = labels_encoded[num_train_samples:]

max_vocab = 50000
embedding_columns = 64

# build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim = max_vocab,
        output_dim = embedding_columns,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Softmax()
])

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(train_samples, train_labels, epochs=10, batch_size=256, validation_split=0.2)

history_dict = history.history
print(history_dict.keys())

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
fig = plt.figure()
ax1 = fig.add_subplot()
ax1.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
ax1.plot(epochs, val_loss, 'b', label='Validation loss')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

ax2 = fig.add_subplot()
ax2.plot(epochs, acc, 'ro', label='Training accuracy')
# b is for "solid blue line"
ax2.plot(epochs, val_acc, 'r', label='Validation accuracy')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.savefig("test")

loss, accuracy = model.evaluate(test_samples, test_labels)
print("Loss: ", loss)
print("Accuracy: ", accuracy)