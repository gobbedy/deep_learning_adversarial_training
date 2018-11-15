#!/usr/bin/env python

from keras.datasets import imdb
from keras.callbacks import EarlyStopping
import argparse
from keras import Sequential
from keras.layers import Embedding, SimpleRNN, Dense,  LSTM
from keras import backend as K
import tensorflow as tf
from tensorflow.python.client import timeline


run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

vocabulary_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
print(X_train[0])
print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))

from keras.preprocessing import sequence

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--state_dimensions", type=int, help="number of state dimensions", required=True)
parser.add_argument("-a", "--architecture", type=str, help="architecture (Vanilla or LSTM)", required=True)
parser.add_argument("-o", "--output_dir", type=str, help="output directory", required=True)
args = parser.parse_args()

dim = args.state_dimensions
#dim = 20
architecture = args.architecture
#architecture = "Vanilla"
output_dir = "."


patience = 5
# architectures = ["Vanilla", "LSTM"]
# architectures = ["LSTM"]
# architecture = "Vanilla"
early_stopping_callback = EarlyStopping(monitor='loss', min_delta=0, patience=patience, verbose=0, mode='auto',
                                        baseline=None, restore_best_weights=True)

callbacks = [early_stopping_callback]

dropout = 0.5

dimensions = [200]
# dimensions = [20, 50, 100, 200, 500]
# dimensions = [500]
# dimensions = [200, 500]
# arr = [1, 5, 10, 15]
# arr = [15]
# for architecture in architectures:
# for dim in dimensions:
embedding_size = 32
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))

if architecture == "Vanilla":
    model.add(SimpleRNN(dim, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform',
                        recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None,
                        recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                        kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=dropout,
                        recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False,
                        stateful=False, unroll=False))
else:
    model.add(LSTM(dim, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,
                   kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                   bias_initializer='zeros',
                   unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None,
                   bias_regularizer=None,
                   activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
                   bias_constraint=None, dropout=dropout, recurrent_dropout=0.0, implementation=1,
                   return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False))

model.add(Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 100
#num_epochs = 1000
num_epochs = 1

X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
model.fit(X_train2, y_train2, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks)

filename = args.output_dir + '/' + architecture + "_earlystop_patience" + str(patience) \
           + "_batchsize" + str(batch_size) + "_dropout05.txt"
#filename = output_dir + '/' + architecture + "_earlystop_patience" + str(patience) \
#           + "_batchsize" + str(batch_size) + "_dropout05.txt"
file = open(filename, "a")
scores = model.evaluate(X_test, y_test, verbose=0)
file.write(str(scores) + " " + str(dim) + " " + str(num_epochs) + " " + str(batch_size) + " " + str(
    early_stopping_callback.stopped_epoch) + "\n")
file.close()
print('Test accuracy:', scores[1])



