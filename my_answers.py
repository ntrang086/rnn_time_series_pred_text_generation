import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import string


def window_transform_series(series, window_size):
    """Transforms the input series and window-size into a set of 
    input/output pairs for use with an RNN model."""
    # containers for input/output pairs
    X = [series[i:i+window_size] for i in range(len(series)-window_size)]
    y = series[window_size:]
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

def build_part1_RNN(window_size):
    """Build an RNN to perform regression on time series input/output data."""
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model

def cleaned_text(text):
    """Return the text input with only ascii lowercase and the punctuation
    given below included."""
    punctuation = ['!', ',', '.', ':', ';', '?']
    valid_chars = set(string.ascii_letters + ''.join(punctuation) \
                      + string.whitespace) - {'\t', '\n', '\x0b', '\x0c', '\r'}
    text_unique_chars = list(set(text))
    for c in text_unique_chars:
    	if c not in valid_chars:
    		text = text.replace(c,' ')
    return text

def window_transform_text(text, window_size, step_size):
    """Transform the input text and window-size into a set of input/output 
    pairs for use with an RNN model."""
    # containers for input/output pairs
    inputs = []
    outputs = []
    for i in range(window_size, len(text), step_size):
    	inputs.append(text[i-window_size:i])
    	outputs.append(text[i])
    return inputs,outputs

def build_part2_RNN(window_size, num_chars):
    """Build an RNN model with a single LSTM hidden layer with softmax activation."""
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    return model
