import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from utils import *

# was getting a lot of annoying warnings whenever importing tensorflow
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, Dense, LSTM

df = pd.read_csv('./Data/Tweets.csv', header=0)
data, labels_encoded, num_samples = prepare_data(df)

train_samples, train_labels = get_training_data(data, labels_encoded, num_samples, 0.8)

max_vocab = 50000
embedding_columns = 64

# build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim = max_vocab,
        output_dim = embedding_columns,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1),
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

plt.savefig("test_loss_accuracy")

model.save('./Models/mod')