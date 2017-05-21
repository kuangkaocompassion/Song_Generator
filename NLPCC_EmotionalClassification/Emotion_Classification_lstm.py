import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import random
import collections
import time
import Data_Process
import pdb
from random import sample
import csv
"""
metadata: three list (word2index, index2words, emoticons)
idx_input: tokenized lines
"""
metadata, idx_input = Data_Process.load_data(PATH='')

dictionary = metadata['w2idx']
reverse_dictionary = metadata['idx2w']
emoticons = metadata['emoticons']
exp_info = metadata['exp_info']
training_data = idx_input


# classification_number: number of emoticons
classification_number = 7

# Parameters
num_of_layer = 3
learning_rate = 0.01
training_epochs = 60
display_step = 1
n_input = exp_info['limit_length'] 
batch_size = 100
n_hidden = 128
keep_prob = 0 # DROPOUT ratio
exp_info['num_layers'] = num_of_layer
exp_info['learning_rate'] = learning_rate
exp_info['num_hidden_states'] = n_hidden
exp_info['best_acc'] = 0
exp_info['output_keep_prob'] = keep_prob

# Experiment Note
note = "delete two emotions: fear & surprise"

# Open Record: Experiment Information
csvin = open('Experiment_Record.csv', 'a')
csvin_writer = csv.writer(csvin)

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

def rand_batch_gen(x, y):
    """
    Goal: random sample out order of every sentence
    x: sentences
    y: emoticons
    """
    sample_idx = sample(list(np.arange(len(x))), len(x))
    new_x = []
    new_y = []
    for index in sample_idx:
        new_x.append(x[index])
        new_y.append(y[index])
    return np.array(new_x), np.array(new_y)


def RNN(x, weights, biases, layer):

    # x.shape = [10, 200, 1](n_input, batch_size, number_of_tokens in single time step)
    x = tf.unstack(x, n_input, 1)

    # 3-layer RNN each layer has n_hidden units.
    # cell = rnn.DropoutWrapper(rnn.BasicLSTMCell(n_hidden), output_keep_prob = keep_prob)
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden) for ly in range(layer)])

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out'] 


pred = RNN(x, weights, biases, layer=num_of_layer)

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
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    session.run(init)
    number_of_epoch = 0

    while number_of_epoch < training_epochs:
        acc_total = 0
        loss_total = 0
        trainX_all, trainY_all = rand_batch_gen(training_data, emoticons)
        num_of_sentences = float(len(trainX_all))
        total_num_of_epochs = int(num_of_sentences//batch_size) 

        for number_of_batch in range(total_num_of_epochs):
            symbols_in_keys = trainX_all[number_of_batch * batch_size : (number_of_batch+1) * (batch_size)]
            symbols_in_keys = np.reshape(symbols_in_keys, [-1, n_input, 1])

            symbols_out_onehot = np.zeros([batch_size, classification_number], dtype=float)
            count = 0
            for instance in trainY_all[number_of_batch * batch_size: (number_of_batch+1) * (batch_size)]:
                symbols_out_onehot[count][instance] = 1.0
                count += 1
            
            _, acc, loss, correct_portion = session.run([optimizer, accuracy, cost, correct_pred], \
                                                    feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
            acc_total += acc
            loss_total += loss
            # print("num_of_batch:",number_of_batch)
        print("=====PER EPOCH FINISHED=====")
        print("Number_of_Epoch:", number_of_epoch)
        print("AVGAccuracy_after_an_epoch:", float(acc_total)/float(total_num_of_epochs))
        print("AVGLoss_after_an_epoch", float(loss_total)/float(total_num_of_epochs))
        if float(acc_total)/float(total_num_of_epochs) > exp_info['best_acc']:
            exp_info['best_acc'] = float(acc_total)/float(total_num_of_epochs)
        number_of_epoch += 1
    csvin_writer.writerow([time.strftime("%Y.%m.%d"), 
                           "{}%".format(exp_info['tokens_left']),
                           exp_info['least_frequency'],
                           exp_info['number_of_lines'],
                           exp_info['num_layers'],
                           exp_info['learning_rate'],
                           exp_info['num_hidden_states'],
                           exp_info['output_keep_prob'],
                           exp_info['best_acc'],
                           note
                           ])
    csvin.close()
