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

def set_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--purpose", help="purpose: TRAIN, USE", type=str)
    parser.add_argument("-n", "--name", help="name: input file", type=str)
    parser.add_argument("-m", "--method", help="process method: token, char", type=str)
    # parser.add_argument("-of", "--original_file", help="name: original training file", type=str)
    # parser.add_argument("-cmd", "--commandinput", help="input by command: 1 or 0", type=int)
    parser.add_argument("-e", "--epochs", help="training/finetune epochs: int", type=int)
    parser.add_argument("-nf", "--newfile", help="result filename of USE: csv", type=str)
    args = parser.parse_args()
    args = default_check(args)
    return args

class SongGenerator(object):
    def __init__(self, args):
        # global information
        self.purpose = args.purpose
        self.InputFile = 'TEST/' + args.name.replace('.csv', '_idx_input.npy')
        self.InputEmo = args.name.replace('.csv', '_withEMO.npy')
        self.flag_test =  0 if self.purpose == 'TRAIN' else 1
        
        self.metadata = 'TEST/' + args.name.replace('.csv', '_metadata.pkl') # w2idx, idx2w, sentence, IdxInput
        self.w2idx = self.metadata['w2idx']
        self.idx2w = self.metadata['idx2w']
        self.sentence = self.metadata['lines']
        self.NumSentence = len(self.sentence)
        # create Model for: TRAIN, TEST
        self.Model_var()
        self.x_input, y_input, x_emo_dist, target, weight, biase, embedding = self.Model_init()
        self.Model_Main()
        if self.purpose == 'TRAIN':
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.session.run(init)
    
    def LoadData(self, METADATA, INDEX_INPUT,):
        # read data control dictionaries
        with open(PATH + METADATA, 'rb') as f:
            metadata = pickle.load(f)
        # read numpy arrays
        idx_input = np.load(PATH + INDEX_INPUT)
        return metadata, idx_input

    
    def SplitData(self, ratio):

    """
    MODEL STRUCTURE
    """
    def Model_var(self):
        self.NumClass = 7
        self.SizeVocab = 7044
        self.NumLayer = 2
        self.lr = 0.01 #learning rate
        self.NumEpoch = 50
        self.NumInput = 30
        self.SizeBatch = 50
        self.NumHidden = 256
    
    # initialize: x, y x_emo_dist, target, weight, bias, embedding
    def Model_init(self):
        x_input = tf.placeholder(tf.int32, [None, self.NumInput]) # x
        y_input = tf.placeholder(tf.int32, [None, self.NumInput]) # y
        # emotion distribution of x_input
        x_emo_dist = tf.placeholder(tf.float32, [None, self.NumClass]) 
        target = tf.cast(y_input, tf.int32)
        # weights
        weight = {
        'EmoDecoderOut': tf.Variable(tf.random_normal([self.NumHidden + self.NumClass,\
            self.SizeVocab])),\
        'BaseDecoderOut': tf.Variable(tf.random_normal([self.NumHidden,\
            self.SizeVocab]))
                  }
        # biases
        bias = {
        'EmoDecoderOut': tf.Variable(tf.random_normal([self.SizeVocab])),
        'BaseDecoderOut': tf.Variable(tf.random_normal([self.SizeVocab]))
                 }
        embedding = tf.get_variable("embedding", [SizeVocab, NumHidden])

        return x_input, y_input, x_emo_dist, target, weight, bias, embedding
    
    def seq2seq_encoder(self, encoder_input, layer):
        encoder_input = tf.nn.embedding_lookup(self.embedding, encoder_input)
        encoder_input = tf.split(encoder_input, self.NumInput, 1)
        encoder_input = [tf.squeeze(input_, [1]) for input_ in encoder_input]

        rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.NumHidden) for ly in range(layer)])
        outputs, states = rnn.static_rnn(rnn_cell, encoder_input, dtype=tf.float32)

        return outputs, states
    
    def seq2seq_decoder(self, decoder_input, initial_state, layer, function=loop):
        input_go = tf.ones([self.SizeBatch, 1])
        input_go = tf.cast(input_go, tf.int32)
        decoder_input = tf.concat([input_go, decoder_input], 1)
        decoder_input = tf.nn.embedding_lookup(self.embedding, decoder_input)
        decoder_input = tf.split(decoder_input, self.NumInput+1, 1)
        decoder_input = [tf.squeeze(input_, [1]) for input_ in decoder_input]

        decoder_rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.NumHidden + self.NumClass) for ly in range(self.NumLayer)])
        outputs, states = rnn_decoder(decoder_input, initial_state, decoder_rnn_cell, loop_function=self.loop if self.flag_test else None)
        return outputs, states
    
    def loop(prev, _):
        prev = tf.matmul(prev, weights['EmoDecoderOut']) + biases['EmoDecoderOut']
        prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
        prev_symbol = tf.cast(prev_symbol, tf.float32)
        return tf.nn.embedding_lookup(self.embedding, prev_symbol)
    
    def modified_state(self, state, emo_state):
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
    
    def Model_Main(self):
        EncoOutput, EncoState = seq2seq_encoder(self.x_input, layer= self.NumLayer)
        EncoState_m = modified_state(EncoState, self.x_emo_dist) # modified states
        DecoOutput, DecoState = seq2seq_decoder(self.y_input, EncoState_m, layer= self.NumLayer)
         # get first 30 words
        DecoOutput = DecoOutput[0:-1]
        # reshape decoder output
        DecoOutput_r = tf.reshape(tf.concat(DecoOutput, 1), [-1, self.NumHidden+ self.NumClass])
        DecoLogit = tf.matmul(DecoOutput_r, self.weight['EmoDecoderOut']) + bias['EmoDecoderOut']

        DecoProb = tf.nn.softmax(DecoLogit)
        DecoWord_idx = tf.argmax(DecoProb, 1)

        loss = legacy_seq2seq.sequence_loss_by_example(
                        [DecoLogit],
                        [tf.reshape(self.target, [-1])],
                        [tf.ones([self.SizeBatch*self.NumInput])])

        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)

        # Initializing the variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

