import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMStateTuple, _LSTMStateTuple
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib.legacy_seq2seq import rnn_decoder
import random
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
flag='TRAIN'

metadata_filename = 'USE_metadata.pkl'
idx_input_filename = 'USE_idx_input.npy'
metadata, idx_input = Model_Data_Process.load_data(metadata_filename, idx_input_filename)
exp_info = dict()

emotion_distribution = np.load('Jay_lyrics_notag_withEMO.npy')


"""
===TRAIN FINETUNE===
metadata: three list (word2index, index2words, emoticons)
idx_input: tokenized sentences
"""
if flag =='TRAIN':
    dictionary = metadata['w2idx']
    reverse_dictionary = metadata['idx2w']
    sentence_data = metadata['USE_sentences']
    num_toal_sentences = len(idx_input)
    # pdb.set_trace()
    training_data = idx_input[0:int(num_toal_sentences*0.8)]
    training_data_emo = emotion_distribution[0:int(num_toal_sentences*0.8)]
    testing_data = idx_input[int(num_toal_sentences*0.8):]
    testing_data_emo = emotion_distribution[int(num_toal_sentences*0.8):]
# pdb.set_trace()
# classification_number: number of emoticons
classification_number = 7

# Parameters
vocab_size = 7044
num_of_layer = 2
learning_rate = 0.01
training_epochs = 50
n_input = 30
batch_size = 100
n_hidden = 256
keep_prob = 0 # DROPOUT ratio
flag_test = 0

# Experiment Record
# Open Experiment Record
exp_info['epchs'] = training_epochs
exp_info['num_layers'] = num_of_layer
exp_info['learning_rate'] = learning_rate
exp_info['num_hidden_states'] = n_hidden
exp_info['best_acc_train'] = 0
exp_info['best_acc_test'] = 0
exp_info['output_keep_prob'] = keep_prob
note = "learning_rate = 0.01; AdamOptimizer"

csvin = open('Seq2Seq_MODEL_Record.csv', 'a')
csvin_writer = csv.writer(csvin)

"""
===MODEL===
# rand_batch_gen: random sample out order of every sentence(with emotion tag)
# Bi_LSTM: Core Model Structure
"""
x_inputs = tf.placeholder(tf.int32, [None, n_input])
y_inputs = tf.placeholder(tf.int32, [None, n_input])
x_inputs_emo_distribution = tf.placeholder(tf.float32, [None, 7])

target = tf.cast(y_inputs, tf.int32)

# RNN output node weights and biases
weights = {
    'decoder_out': tf.Variable(tf.random_normal([n_hidden+classification_number, vocab_size])),    
}
biases = {
    'decoder_out': tf.Variable(tf.random_normal([vocab_size]))
}
embedding = tf.get_variable("embedding", [vocab_size, n_hidden])



def rand_batch_gen(data, data_emo):  
    sample_idx = sample(list(np.arange(len(data)-1)), len(data)-1)
    new_x = []
    new_y = []
    count = 0
    for index in sample_idx:
        new_x.append(data[index])
        new_y.append(data[index + 1])
        if count == 0:
            new_emo = data_emo[index][np.newaxis, :]
        else:
            new_emo = np.concatenate((new_emo, data_emo[index][np.newaxis, :]), axis=0)
        count += 1
    return np.array(new_x), np.array(new_y), new_emo


def seq2seq_encoder(encoder_input, layer):
    encoder_input = tf.nn.embedding_lookup(embedding, encoder_input)
    encoder_input = tf.split(encoder_input, n_input, 1)
    encoder_input = [tf.squeeze(input_, [1]) for input_ in encoder_input]

    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden) for ly in range(layer)])
    outputs, states = rnn.static_rnn(rnn_cell, encoder_input, dtype=tf.float32)

    return outputs, states

def loop(prev, _):
    prev = tf.matmul(prev, weights['decoder_out']) + biases['decoder_out']
    prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
    prev_symbol = tf.cast(prev_symbol, tf.float32)
    return tf.nn.embedding_lookup(embedding, prev_symbol)

# emo_state.shape = [200, 7]
def modified_state(state, emo_state):
    final_tuple_state =()
    for ly_state in state:
        original_state = ly_state
        original_memory_state = original_state.c
        original_hidden_state = original_state.h
        
        new_h = tf.concat([original_hidden_state, emo_state], 1)
        new_c = tf.concat([original_memory_state, emo_state], 1)
        new_tuple_state = LSTMStateTuple(new_c, new_h)
        final_tuple_state += (new_tuple_state,)
    return final_tuple_state
"""
def modified_rnn_decoder(input, state, cell, loop_function=None, scope=None):
    with variable_scope.variable_scope(scope or "modified_rnn_decoder"):
        state = initial_state
        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
          if loop_function is not None and prev is not None:
            with variable_scope.variable_scope("loop_function", reuse=True):
              inp = loop_function(prev, i)
          if i > 0:
            variable_scope.get_variable_scope().reuse_variables()
          output, state = cell(inp, state)
          outputs.append(output)
          if loop_function is not None:
            prev = output
    return outputs, state
"""
def seq2seq_decoder(decoder_input, initial_state, layer, function=loop):
    input_go = tf.ones([batch_size, 1])
    input_go = tf.cast(input_go, tf.int32)
    decoder_input = tf.concat([input_go, decoder_input], 1)
    decoder_input = tf.nn.embedding_lookup(embedding, decoder_input)
    decoder_input = tf.split(decoder_input, n_input+1, 1)
    decoder_input = [tf.squeeze(input_, [1]) for input_ in decoder_input]

    decoder_rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden + classification_number) for ly in range(layer)])
    # outputs, states = modified_rnn_decoder(decoder_input, initial_state, decoder_rnn_cell, loop_function=function if flag_test else None)
    outputs, states = rnn_decoder(decoder_input, initial_state, decoder_rnn_cell, loop_function=function if flag_test else None)
    return outputs, states


encoder_outputs, encoder_states = seq2seq_encoder(x_inputs, layer=num_of_layer)
encoder_states_modified = modified_state(encoder_states, x_inputs_emo_distribution)
# pdb.set_trace()
decoder_outputs, decoder_states = seq2seq_decoder(y_inputs, encoder_states_modified, layer=num_of_layer)

# test
decoder_outputs_last = decoder_outputs

decoder_outputs = decoder_outputs[0:-1] # get first 30 words
decoder_outputs_reshape = tf.reshape(tf.concat(decoder_outputs, 1), [-1, n_hidden+classification_number])
decoder_logits = tf.matmul(decoder_outputs_reshape, weights['decoder_out']) + biases['decoder_out']

decoder_probs = tf.nn.softmax(decoder_logits)
decoder_word_index = tf.argmax(decoder_probs, 1)

loss = legacy_seq2seq.sequence_loss_by_example(
                [decoder_logits],
                [tf.reshape(target, [-1])],
                [tf.ones([batch_size*n_input])])

cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# tvars = tf.trainable_variables()
# grads = tf.gradients(cost, tvars)
# grads, _ = tf.clip_by_global_norm(grads, 5)
# train_op = optimizer.apply_gradients(zip(grads, tvars))
# self.merged_op = tf.summary.merge_all()


# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Launch the graph

if flag == 'TRAIN':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    session.run(init)
    number_of_epoch = 0
    print("START TRAINING")
    while number_of_epoch < training_epochs:
        flag_test = 0

        trainX_all, trainY_all, trainX_emo_all = rand_batch_gen(training_data, training_data_emo)
        num_of_sentences = float(len(trainX_all)) 
        total_num_of_batches = int(num_of_sentences//batch_size) 
        print("EPOCH:", number_of_epoch)

        # train
        for number_of_batch in range(total_num_of_batches):
            input_line = trainX_all[number_of_batch * batch_size : (number_of_batch+1) * (batch_size)]
            input_line_emo = trainX_emo_all[number_of_batch * batch_size : (number_of_batch+1) * (batch_size)]

            output_line = trainY_all[number_of_batch * batch_size : (number_of_batch+1) * (batch_size)]
            # y_line = np.reshape(y_line, [-1, n_input, 1])
            # feed_dict = {x_inputs: input_line, y_inputs:output_line}
            feed_dict = {x_inputs: input_line, y_inputs:output_line, x_inputs_emo_distribution:input_line_emo}
            # _, loss_val, cost_val = session.run([train_op, loss, cost], feed_dict)
            _, loss_val, cost_val = session.run([optimizer, loss, cost], feed_dict)
            # test
            # decoder_states_test = session.run([decoder_states], feed_dict)
            print("batch:",number_of_batch,"; cost:",cost_val)
            # pdb.set_trace()

            if number_of_batch%2 == 0 and number_of_batch>0:
                flag_test = 1
                testX_all, testY_all, testX_emo_all = rand_batch_gen(testing_data, testing_data_emo)
                testx = testX_all[0:200]
                testx_emo = testX_emo_all[0:200]
                testy = testY_all[1:201]
                test_outputs_text = session.run([decoder_word_index], feed_dict = {x_inputs: testx, y_inputs:testy, x_inputs_emo_distribution:testx_emo })
                test_outputs_text = [test_outputs_text[0][0:n_input].tolist(), test_outputs_text[0][n_input:n_input*2].tolist()]
                
                for line1, line2 in zip(testx, test_outputs_text):
                    q=''
                    a=''
                    for q_word, a_word in zip(line1,line2):
                        if q_word != 0 and a_word != 0 and q_word != 1 and a_word != 1:
                            q += reverse_dictionary[q_word]
                            a += reverse_dictionary[a_word]
                    print("first line:", q)
                    print("second line:", a)

                print("%" * 20)
                # pdb.set_trace()

        number_of_epoch += 1
