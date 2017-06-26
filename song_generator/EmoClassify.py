import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import random
import sys
import collections
import time
# import dataset.process
import process
import pdb
import pickle
from random import sample
import csv

# MODEL USAGE: 'USE' or 'TRAIN'
# >'USE'  : USE model to classify texts
# >'TRAIN': TRAIN emotion classification model
# >'FINETUNE': FINE TUNE the model

# 062617: problem in training
class Emo_Classifier(object):
	def __init__(self, args):
		# pdb.set_trace()
		self.purpose = args.purpose
		self.filename = args.name
		self.method = args.method

		if self.purpose == 'USE':
			self.use_init(args)
		else:
			self.training_init(args)
			self.exp_init()

		self.EMOTION_DIC = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4:'like', 5:'sadness', 6:'surprise'}
		self.classification_number = 7

		self.num_of_layer = 2
		self.training_epochs= 100
		self.learning_rate = 0.001
		self.n_input = 30
		self.batch_size = 100
		self.n_hidden = 256
		self.keep_prob = 0 # DROPOUT ratio

		self.x, y, weights, biases, pred = self.model_init()
		
		###### TODO ########
		# Loss and optimizer
		self.softmax_result = tf.nn.softmax(logits=pred)
		cost_in_one_batch = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
		# Model evaluation
		correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		# Initializing the variables
		init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()

	def model_init(self):
		x = tf.placeholder("float", [None, self.n_input, 1])
		y = tf.placeholder("float", [None, self.classification_number])
		# RNN output node weights and biases
		weights = {
			'out': tf.Variable(tf.random_normal([2*self.n_hidden, self.classification_number]))
		}
		biases = {
			'out': tf.Variable(tf.random_normal([self.classification_number]))
		}    
		pred = self.Bi_LSTM(x, weights, biases, layer=self.num_of_layer)
		return x, y, weights, biases, pred

	def Bi_LSTM(self, x, weights, biases, layer):
		x = tf.unstack(x, self.n_input, 1)
		# Forward direction cell
		lstm_fw_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden) for ly in range(layer)])
		# Backward direction cell
		lstm_bw_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden) for ly in range(layer)])
		outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
		# there are n_input outputs but we only want the last output
		return tf.matmul(outputs[-1], weights['out']) + biases['out']

    # Experiment Record
	def exp_init(self):
		self.exp_info['epchs'] = self.training_epochs
		self.exp_info['num_layers'] = self.num_of_layer
		self.exp_info['learning_rate'] = self.learning_rate
		self.exp_info['num_hidden_states'] = self.n_hidden
		self.exp_info['best_acc_train'] = 0
		self.exp_info['best_acc_test'] = 0
		self.exp_info['output_keep_prob'] = self.keep_prob
		self.note = "bidirectional: learning_rate = 0.001; AdamOptimizer"
		self.csvin = open('MODEL_Record.csv', 'a')
		self.csvin_writer = csv.writer(self.csvin)
	
	# use information initialize
	def use_init(self, args):
		self.use_newFilename = args.newfile
		self.restore_matadata_name = 'TEST/'+self.filename.replace('.csv', '_metadata.pkl')
		self.idx_input_filename = 'TEST/'+self.filename.replace('.csv', '_idx_input.npy')
		# self.restore_model = 'original_classify_model-100'
		self.restore_model = 'finetune_model-80'										
		self.csvin = open(self.use_newFilename, 'w')
		self.csvin_writer = csv.writer(self.csvin)		
		self.csvin_writer.writerow(['emotion','lyrics','anger','disgust','fear','happiness','like','sadness','surprise'])

	# Training information initialize
	def training_init(self, args):
		if self.purpose == 'TRAIN':
			self.save_model_path = 'CKPT/train_model'
			# restore file
			self.metadata_filename = 'Model_metadata.pkl'
			self.idx_input_filename = 'Model_idx_input.npy'        
		if self.purpose == 'FINETUNE':
			self.origi_filename = args.original_file
			self.save_model_path = 'CKPT/finetune_model'
			self.metadata_filename = 'Finetune_metadata.pkl'
			self.idx_input_filename = 'Finetune_idx_input.npy'

	def load_metadata(self):    # read data control dictionaries
		with open(self.restore_matadata_name, 'rb') as f:
			metadata = pickle.load(f)
		return metadata 
  
	def update_emotion(self, emotion_distribution,num_line, softmax_result_use):
		if num_line == 0:
			new_emotion_distribution = softmax_result_use[0]
		else:
			new_emotion_distribution = np.concatenate((emotion_distribution,softmax_result_use[0]), axis=0)
		return new_emotion_distribution
	
	def write_emotion_output(self, num_line, use_sentences, softmax_result_use):
		self.csvin_writer.writerow([
			self.EMOTION_DIC[softmax_result_use[0][0].argmax()],
			''.join(use_sentences[num_line]),
			softmax_result_use[0][0][0],
			softmax_result_use[0][0][1],
			softmax_result_use[0][0][2],
			softmax_result_use[0][0][3],
			softmax_result_use[0][0][4],
			softmax_result_use[0][0][5],
			softmax_result_use[0][0][6],
		])

	def classify(self):
		use_metadata = self.load_metadata()
		use_sentences = use_metadata['lines']   				# Chinese Sentence
		use_idx_sentences = np.load(self.idx_input_filename)	# to index

		with tf.Session() as session:
			# model restore
			# pdb.set_trace()
			self.saver.restore(session, 'CKPT/'+self.restore_model)
			print("MODEL RESTORE\n") 
			
			emotion_distribution = []
			for num_line in range(len(use_idx_sentences)):
				symbols_in_keys_use = np.reshape(use_idx_sentences[num_line], [-1, self.n_input, 1])
				softmax_result_use = session.run( [self.softmax_result], feed_dict={self.x: symbols_in_keys_use})
				emotion_distribution = self.update_emotion(emotion_distribution,num_line, softmax_result_use)
				self.write_emotion_output(num_line, use_sentences, softmax_result_use)

			print("Finish Classifying !!\n")
			print("SAVE EMOTION ARRAY")
			np.save(self.use_newFilename.replace('.csv','.npy'), emotion_distribution)
			print("SAVE EMOTION NUMPY:", self.use_newFilename.replace('.csv','.npy'))

	def USE_emotion_classifier(self):
		print("PURPOSE:", self.purpose)
		print("-"*20)
		self.classify()
		self.csvin.close()