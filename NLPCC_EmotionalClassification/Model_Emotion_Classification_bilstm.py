import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import random
import sys
import collections
import time
import Model_Data_Process
import pdb
import pickle
from random import sample
import csv

# MODEL USAGE: 'USE' or 'TRAIN'
# >'USE'  : USE model to classify texts
# >'TRAIN': TRAIN emotion classification model
# >'FINETUNE': FINE TUNE the model
flag = sys.argv[1]
restore_model = sys.argv[2]
training_epochs = 0

if flag=='USE':
    use_newFilename = sys.argv[3]
else:
    # EPOCH = sys.argv[3]
    training_epochs = int(sys.argv[3])

if flag=='TRAIN':
    save_model_path = 'CKPT/train_model'
elif flag=='FINETUNE':
    save_model_path = 'CKPT/finetune_model'

"""
===USE===
EMOTION_DIC: number <-> emotion category
use_sentences: tokenized sentences
use_idx_sentences: tokenized sentences
use_csvin : csv file to store classification result
"""
if flag == 'USE':
    EMOTION_DIC = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4:'like', 5:'sadness', 6:'surprise'}
    with open('USE_metadata.pkl', 'rb') as f:
            use_metadata = pickle.load(f)
    use_sentences = use_metadata['USE_sentences']   # Chinese Sentence
    use_idx_sentences = np.load('USE_idx_input.npy')   # to index

    # use_csvin = open('jay_lyrics_withEMO.csv', 'w')
    use_csvin = open(use_newFilename, 'w') 
    use_csvin_writer = csv.writer(use_csvin)

"""
===TRAIN FINETUNE===
metadata: three list (word2index, index2words, emoticons)
idx_input: tokenized sentences
"""
if flag=='TRAIN':
    metadata_filename = 'Model_metadata.pkl'
    idx_input_filename = 'Model_idx_input.npy'
elif flag=='FINETUNE':
    metadata_filename = 'Finetune_metadata.pkl'
    idx_input_filename = 'Finetune_idx_input.npy'


if flag == 'TRAIN' or flag =='FINETUNE':
    metadata, idx_input = Model_Data_Process.load_data(metadata_filename, idx_input_filename)
    dictionary = metadata['w2idx']
    reverse_dictionary = metadata['idx2w']
    exp_info = metadata['exp_info']
    num_toal_sentences = len(idx_input)
    emoticons_train = metadata['emoticons'][0:int(num_toal_sentences*0.8)]
    emoticons_test = metadata['emoticons'][int(num_toal_sentences*0.8):]
    training_data = idx_input[0:int(num_toal_sentences*0.8)]
    testing_data = idx_input[int(num_toal_sentences*0.8):]


# classification_number: number of emoticons
classification_number = 7

# Parameters
num_of_layer = 2
learning_rate = 0.001
# training_epochs = 5
n_input = 30
batch_size = 100
n_hidden = 256
keep_prob = 0 # DROPOUT ratio

# Experiment Record
# Open Experiment Record
if flag == 'TRAIN' or flag == 'FINETUNE':
    exp_info['epchs'] = training_epochs
    exp_info['num_layers'] = num_of_layer
    exp_info['learning_rate'] = learning_rate
    exp_info['num_hidden_states'] = n_hidden
    exp_info['best_acc_train'] = 0
    exp_info['best_acc_test'] = 0
    exp_info['output_keep_prob'] = keep_prob
    note = "bidirectional: learning_rate = 0.001; AdamOptimizer"

    csvin = open('MODEL_Record.csv', 'a')
    csvin_writer = csv.writer(csvin)

"""
===MODEL===
# rand_batch_gen: random sample out order of every sentence(with emotion tag)
# Bi_LSTM: Core Model Structure
"""
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, classification_number])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([2*n_hidden, classification_number]))
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


def Bi_LSTM(x, weights, biases, layer):

    # x.shape = [10, 200, 1](n_input, batch_size, number_of_tokens in single time step)
    x = tf.unstack(x, n_input, 1)

    # Forward direction cell
    lstm_fw_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden) for ly in range(layer)])
    # Backward direction cell
    lstm_bw_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden) for ly in range(layer)])

    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    # there are n_input outputs but we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = Bi_LSTM(x, weights, biases, layer=num_of_layer)

# Loss and optimizer
softmax_result = tf.nn.softmax(logits=pred)
cost_in_one_batch = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Launch the graph
if flag == 'TRAIN' or flag == 'FINETUNE':
    if flag == 'TRAIN':
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        session.run(init)
    if flag == 'FINETUNE':
        session = tf.Session()
        # saver.restore(session, 'CKPT/my-model-100')         # change
        saver.restore(session, 'CKPT/'+restore_model)
        print("MODEL RESTORE\n")   
    number_of_epoch = 0
    print("START TRAINING")
    while number_of_epoch < training_epochs:
        acc_total = 0
        loss_total = 0
        trainX_all, trainY_all = rand_batch_gen(training_data, emoticons_train)
        testX_all, testY_all = testing_data, emoticons_test
        num_of_sentences = float(len(trainX_all))
        total_num_of_epochs = int(num_of_sentences//batch_size)

        # train
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
        if float(acc_total)/float(total_num_of_epochs) > exp_info['best_acc_train']:
            exp_info['best_acc_train'] = float(acc_total)/float(total_num_of_epochs)
        number_of_epoch += 1

        # test
        print("-"*10 + "TEST" + "-"*10)
        symbols_in_keys_test = np.reshape(testX_all, [-1, n_input, 1])
        symbols_out_onehot_test = np.zeros([len(emoticons_test), classification_number], dtype=float)
        count = 0
        for instance in testY_all:
            symbols_out_onehot_test[count][instance] = 1.0
            count += 1
        acc_test, loss_test = session.run( [accuracy, cost], feed_dict={x: symbols_in_keys_test, y: symbols_out_onehot_test})
        if acc_test > exp_info['best_acc_test']:
            exp_info['best_acc_test'] = acc_test
        print("Accuracy:", acc_test)
        print("Loss:", loss_test)
        print("\n")
    saver.save(session, save_model_path, global_step=training_epochs)
    csvin_writer.writerow([time.strftime("%Y.%m.%d"),
                           exp_info['epchs'],
                           exp_info['limit_length'],
                           exp_info['number_of_lines'],
                           exp_info['num_layers'],
                           exp_info['learning_rate'],
                           exp_info['num_hidden_states'],
                           exp_info['output_keep_prob'],
                           exp_info['best_acc_train'],
                           exp_info['best_acc_test'],
                           note
                           ])
    csvin.close()
if flag == 'USE':
    saver = tf.train.Saver()
    with tf.Session() as session:
        # saver.restore(session, 'CKPT/my-model-100')
        saver.restore(session, 'CKPT/'+restore_model)
        print("MODEL RESTORE\n") 
        for num_line in range(len(use_sentences)):
            symbols_in_keys_use = np.reshape(use_idx_sentences[num_line], [-1, n_input, 1])
            softmax_result_use = session.run( [softmax_result], feed_dict={x: symbols_in_keys_use})
            # print(type(softmax_result_use[0][0]))

            use_csvin_writer.writerow([
                EMOTION_DIC[softmax_result_use[0][0].argmax()],
                ''.join(use_sentences[num_line])
                ])

    use_csvin.close()
