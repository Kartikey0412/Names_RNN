#In this code I have implemented a LSTM model on the same 8000 names used for training of the RNN network. I have
#implemented the tensorflow MultiRNNCell with two basicLSTMCell.
#to_matrix function has been borrowed from the names_RNN.py to convert characters to numbers
#implemeted categorical_crossentropy loss with AdamOptimizer



s2 = keras_utils.reset_tf_session()

iterations = 300
batch_size = 32
n_input = 1
time_steps = MAX_LENGTH
rnn_num_units = 64

import keras
from keras.layers import concatenate, Dense, Embedding

rnn_num_units = 64  # size of hidden state
embedding_size = 16  # for characters

# an embedding layer that converts character ids into embeddings
embed_x = Embedding(n_tokens, embedding_size)

# get_probas = Dense(n_tokens, activation="softmax")

input_sequence2 = tf.placeholder(tf.int32, (None, time_steps))

inputs_embedded2 = embed_x(input_sequence2)

print(inputs_embedded2.shape)

weights = {
    'out': tf.Variable(tf.random_normal([rnn_num_units, n_tokens]))
}

biases = {
    'out': tf.Variable(tf.random_normal([n_tokens]))
}

input = tf.unstack(inputs_embedded2, time_steps, 1)
print(input)

rnn_cell = tf.contrib.rnn.MultiRNNCell(
    [tf.contrib.rnn.BasicLSTMCell(rnn_num_units), tf.contrib.rnn.BasicLSTMCell(rnn_num_units)])

outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, input, dtype=tf.float32)
# print(len(outputs))
print(outputs)
# outputs = tf.unstack(outputs, time_steps,1)
# print(outputs)
pred_list = []
for i in outputs:
    pred = tf.matmul(i, weights['out']) + biases['out']
    pred_list.append(pred)

# print(pred_list)

# combine predicted_probas into [batch, time, n_tokens] tensor
pred_list = tf.transpose(pred_list, [1, 0, 2])

pred_list = pred_list[:, :-1, :]
print(pred_list)
# next to last token prediction is not needed
# predicted_probas = predicted_probas[:, :-1, :]
predictions_matrix = tf.reshape(pred_list, [-1, n_tokens])
print(predictions_matrix.shape)
y = tf.one_hot(tf.reshape(input_sequence2[:, 1:], [-1]), n_tokens)
print(y.shape)
# pred = rnn(x, weights, biases)

# pred = tf.reshape(pred, [-1, n_tokens])

from keras.objectives import categorical_crossentropy

loss = tf.reduce_mean(categorical_crossentropy(y, predictions_matrix))
optimize = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()
from random import sample

s2.run(init)

history = []

for i in range(iterations):
    batch = to_matrix(sample(names, batch_size), max_len=MAX_LENGTH)

    loss_i, _ = s2.run([loss, optimize], {input_sequence2: batch})
    history.append(loss_i)

    if (i + 1) % 100 == 0:
        clear_output(True)
        plt.plot(history, label='loss')
        plt.legend()
        plt.show()