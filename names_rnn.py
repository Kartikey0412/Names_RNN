#This code is for predicting next letter in human names, by training a recurrent neural network on a list of 8000
#human names. In the next code (lstm_names.py) I have implemented a LSTM model for it. Here I have used a 'relu'
#activation and 64 number of hidden rnn units. The plot shows the decrease in the loss with training. In the end
# I have built a names generator function which makes human names from a seed character.



import tensorflow as tf
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os
import sys
sys.path.append("..")
import keras
from keras import backend as K
import download_utils
import tqdm_utils
import keras_utils
from keras_utils import reset_tf_session

#importing from google drive, list of names
from google.colab import drive
drive.mount('/content/gdrive')

start_token = " "
pad_token = "#"

with open('/content/gdrive/My Drive/ names_rnn.txt') as f:
    names = f.read()[:-1].split('\n')
    print(names[0:5])
    names = [start_token + name for name in names]

#identifying all tokens in names, i.e. the individual letters
tokens = set(''.join(names[:]))
print(tokens)
tokens = list(tokens)
n_tokens = len(tokens)
print ('n_tokens:', n_tokens)

assert 50 < n_tokens < 60

#dicitionary from token to id to be fed into rnn
token_to_id = {}
for i in range(n_tokens):
    token_to_id[tokens[i]] = i
print(token_to_id)

#names to numeric id matrix
def to_matrix(names, max_len=None, pad=0, dtype=np.int32):

    max_len = max_len or max(map(len, names))
    names_ix = np.zeros([len(names), max_len], dtype) + pad

    for i in range(len(names)):
        name_ix = list(map(token_to_id.get, names[i]))
        names_ix[i, :len(name_ix)] = name_ix

    return names_ix

#Start buidling model
s = keras_utils.reset_tf_session()

import keras
from keras.layers import concatenate, Dense, Embedding

rnn_num_units = 64  # size of hidden state
embedding_size = 16  # for characters

#layers of rnn
embed_x = Embedding(n_tokens, embedding_size)
get_h_next = Dense(rnn_num_units, activation = "relu")
get_probas = Dense(n_tokens, activation="softmax")


#using current input and previous state to get probabilities of output and next state
def rnn_one_step(x_t, h_t):
    # convert character id into embedding
    x_t_emb = embed_x(tf.reshape(x_t, [-1, 1]))[:, 0]

    # concatenate x_t embedding and previous h_t state

    x_and_h = tf.concat([x_t_emb, h_t], 1)

    h_next = get_h_next(x_and_h)

    output_probas = get_probas(h_next)

    return output_probas, h_next


input_sequence = tf.placeholder(tf.int32, (None, MAX_LENGTH))  # batch of token ids
batch_size = tf.shape(input_sequence)[0]

predicted_probas = []
h_prev = tf.zeros([batch_size, rnn_num_units])  # initial hidden state

for t in range(MAX_LENGTH):
    x_t = input_sequence[:, t]
    probas_next, h_next = rnn_one_step(x_t, h_prev)

    h_prev = h_next
    predicted_probas.append(probas_next)

# combine predicted_probas into [batch, time, n_tokens] tensor
predicted_probas = tf.transpose(tf.stack(predicted_probas), [1, 0, 2])

# next to last token prediction is not needed
predicted_probas = predicted_probas[:, :-1, :]

#flatten predictions to [batch*time, n_tokens]
predictions_matrix = tf.reshape(predicted_probas, [-1, n_tokens])

# flatten answers (next tokens) and one-hot encode them
answers_matrix = tf.one_hot(tf.reshape(input_sequence[:, 1:], [-1]), n_tokens)

#loss and optimizer
from keras.objectives import categorical_crossentropy
loss = tf.reduce_mean(categorical_crossentropy(answers_matrix, predictions_matrix))
optimize = tf.train.AdamOptimizer().minimize(loss)

#train
from random import sample

s.run(tf.global_variables_initializer())

batch_size = 32
history = []

for i in range(1000):
    batch = to_matrix(sample(names, batch_size), max_len=MAX_LENGTH)
    loss_i, _ = s.run([loss, optimize], {input_sequence: batch})

    history.append(loss_i)

    if (i + 1) % 100 == 0:
        clear_output(True)
        plt.plot(history, label='loss')
        plt.legend()
        plt.show()

assert np.mean(history[:10]) > np.mean(history[-10:]), "RNN didn't converge"

#given a seed phrase generating sample names from the trained rnn network
def generate_sample(seed_phrase=start_token, max_length=MAX_LENGTH):

    x_sequence = [token_to_id[token] for token in seed_phrase]
    s.run(tf.assign(h_t, h_t.initial_value))

    # feed the seed phrase, if any
    for ix in x_sequence[:-1]:
        s.run(tf.assign(h_t, next_h), {x_t: [ix]})

    # start generating
    for _ in range(max_length - len(seed_phrase)):
        x_probs, _ = s.run([next_probs, tf.assign(h_t, next_h)], {x_t: [x_sequence[-1]]})
        x_sequence.append(np.random.choice(n_tokens, p=x_probs[0]))

    return ''.join([tokens[ix] for ix in x_sequence if tokens[ix] != pad_token])

