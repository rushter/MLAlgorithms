from __future__ import print_function

import logging
import random

import numpy as np
import sys

from mla.datasets import load_nietzsche
from mla.neuralnet import NeuralNet
from mla.neuralnet.constraints import SmallNorm
from mla.neuralnet.layers import Activation, Dense
from mla.neuralnet.layers.recurrent import LSTM, RNN
from mla.neuralnet.optimizers import RMSprop

logging.basicConfig(level=logging.DEBUG)


# Example taken from: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


X, y, text, chars, char_indices, indices_char = load_nietzsche()
# Round the number of sequences for batch processing
items_count = X.shape[0] - (X.shape[0] % 64)
maxlen = X.shape[1]
X = X[0:items_count]
y = y[0:items_count]

print(X.shape, y.shape)
# LSTM OR RNN
# rnn_layer = RNN(128, return_sequences=False)
rnn_layer = LSTM(128, return_sequences=False, )

model = NeuralNet(
    layers=[
        rnn_layer,
        # Flatten(),
        # TimeStepSlicer(-1),
        Dense(X.shape[2]),
        Activation('softmax'),
    ],
    loss='categorical_crossentropy',
    optimizer=RMSprop(learning_rate=0.01),
    metric='accuracy',
    batch_size=64,
    max_epochs=1,
    shuffle=False,

)

for _ in range(25):
    model.fit(X, y)
    start_index = random.randint(0, len(text) - maxlen - 1)

    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)
    for i in range(100):
        x = np.zeros((64, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.
        preds = model.predict(x)[0]
        next_index = sample(preds, 0.5)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
