import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import random
import collections
import time
import Data_Process
import pdb
from random import sample
"""
metadata: three list (word2index, index2words, emoticons)
idx_input: tokenized lines
"""
metadata, idx_input = Data_Process.load_data(PATH='')

dictionary = metadata['w2idx']
reverse_dictionary = metadata['idx2w']
emoticons = metadata['emoticons']
training_data = idx_input


# classification_number = len(dictionary)
classification_number = 40

# Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 1
n_input = 10 
batch_size = 200

# number of units in RNN cell
n_hidden = 1024

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, classification_number])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, classification_number]))
}
biases = {
    'out': tf.Variable(tf.random_normal([classification_number]))
}

def rand_batch_gen(x, y, batch_size):
    sample_idx = sample(list(np.arange(len(x))), batch_size)
    new_x = []
    new_y = []
    for index in sample_idx:
        new_x.append(x[index])
        new_y.append(y[index])
    return np.array(new_x), np.array(new_y)


def RNN(x, weights, biases):

    # x.shape = [10, 200, 1](n_input, batch_size, num_of_token single time step)
    x = tf.unstack(x, n_input, 1)


    # 3-layer RNN each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out'] , outputs

pred, outputs = RNN(x, weights, biases)

# Loss and optimizer
softmax_result = tf.nn.softmax(logits=pred)
cost_in_one_batch = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0

    while step < training_iters:

        trainX, trainY = rand_batch_gen(training_data, emoticons, batch_size)
        symbols_in_keys = trainX
        symbols_in_keys = np.reshape(symbols_in_keys, [-1, n_input, 1])

        symbols_out_onehot = np.zeros([batch_size, classification_number], dtype=float)
        count = 0
        for instance in trainY:
        	symbols_out_onehot[count][instance-1] = 1.0
        	count += 1
        pdb.set_trace()
        _, acc, loss, correct_portion, outs = session.run([optimizer, accuracy, cost, correct_pred, outputs], \
                                                feed_dict={x: symbols_in_keys, y: symbols_out_onehot})

        print("outputs shape:", len(outs), outs[-1].shape)
        print("iteration:", step)
        print("Accuracy:", acc)
        step += 1
