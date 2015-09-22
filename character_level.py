from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np
import random
import sys

import IPython


class CharacterLevelLSTM:
    """
    character level model LSTM
    """
    def __init__(self, input_len, hidden_len, output_len, return_sequence=True):
        self.input_len = input_len
        self.hidden_len = hidden_len
        self.output_len = output_len
        self.return_sequence = return_sequence
        self.model = Sequential()

    def build(self, dropout=0.2):
        """
        Stacked LSTM with specified dropout rate
        :param dropout: dropout rate
        :return: model built with softmax activation, cross entropy loss and rmsprop optimizer
        """
        # 2 layer LSTM with specified number of nodes in the hidden layer.
        self.model.add(LSTM(self.input_len, self.hidden_len, return_sequences=self.return_sequence))
        self.model.add(Dropout(dropout))
        self.model.add(LSTM(self.hidden_len, self.hidden_len, return_sequences=False))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(self.hidden_len, self.output_len))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    def sample(self, a, temperature=1.0):
        """
        softmax function for reinforcement learning
        :param a: prob array
        :param temperature: affects diversity
        :return:
        """
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))


def get_data():
    """
    retrieves data from a plain txt file and formats it
    using 1-of-k encoding
    :return: relevant data extracted
    """
    # should be plain txt file
    text = open('input.txt', 'r').read().lower()

    # vocab
    chars = set(text)
    print("total chars: ", len(chars))
    char_to_indices = dict((char, idx) for idx, char in enumerate(chars))
    indices_to_chars = dict((idx, char) for idx, char in enumerate(chars))

    # separate into array of sentences (max 20 chars)
    max_len = 20
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - max_len, step):
        sentences.append(text[i: i + max_len])
        next_chars.append(text[i + max_len])
    print("total # of sentences: ", len(sentences))

    # 1-of-k encoding (all zeros except for a single one at
    # the index of the character in the vocab)
    # all input sentences encoded
    x = np.zeros((len(sentences), max_len, len(chars)), dtype=np.bool)
    # expected outputs for each sentence
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            # mark the each corresponding character in a sentence as 1
            x[i, t, char_to_indices[char]] = 1
        # mark the corresponding character in expected output as 1
        y[i, char_to_indices[next_chars[i]]] = 1

    return text, max_len, len(chars), char_to_indices, indices_to_chars, x, y


def train():
    """
    Trains the network and outputs the generated text.
    Trains using batch size of 100, 60 epochs total.
    :return:
    """
    text, max_len, input_len, char_to_indices, indices_to_chars, x, y = get_data()
    # two layered LSTM 512 hidden nodes and a dropout rate of 0.5
    lstm = CharacterLevelLSTM(input_len, 100, input_len)
    print("Building Model...")
    # IPython.embed()
    lstm.build(dropout=0.5)

    # train model and output generated text
    for iteration in range(1, 60):
        print("==============================================================")
        print("Iteration: ", iteration)
        lstm.model.fit(x, y, batch_size=100, nb_epoch=1)

        start_index = random.randint(0, len(text) - max_len - 1)
        for T in [0.2, 0.5, 1.0, 1.2]:
            print("------------Temperature", T)
            generated = ''
            sentence = text[start_index:start_index + max_len]
            generated += sentence
            print("Generating with seed: " + sentence)
            sys.stdout.write(generated)

            # generate 400 chars
            for i in range(400):
                seed = np.zeros((1, max_len, input_len))
                # format input
                for t, char in enumerate(sentence):
                    seed[0, t, char_to_indices[char]] = 1

                # get predictions
                # verbose = 0, no logging
                predictions = lstm.model.predict(seed, verbose=0)[0]
                next_index = lstm.sample(predictions, T)
                next_char = indices_to_chars[next_index]
                # print next char
                sys.stdout.write(next_char)
                sys.stdout.flush()

                # use current output as input to predict the next character
                # in the sequence
                generated += next_char
                sentence = sentence[1:] + next_char
            print()

if __name__ == '__main__':
    train()