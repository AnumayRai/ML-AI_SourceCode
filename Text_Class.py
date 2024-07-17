import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the IMDB dataset
imdb, info = tf.keras.datasets.imdb.load_data(num_words=10000)

# Split the data into train and test sets
train_data = imdb[0]
test_data = imdb[1]

# Prepare the tokenizer
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_data[0])
word_index = tokenizer.word_index

# Convert the text to sequences
train_sequences = tokenizer.texts_to_sequences(train_data[0])
test_sequences = tokenizer.texts_to_sequences(test_data[0])

# Pad the sequences
maxlen = 200
train_data = pad_sequences(train_sequences, maxlen=maxlen)
test_data = pad_sequences(test_sequences, maxlen=maxlen)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=maxlen),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_data[1],
          validation_data=(test_data, test_data[1]),
          epochs=10,
          batch_size=32)
