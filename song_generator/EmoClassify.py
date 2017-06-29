import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import random
import sys
import collections
import time
import argparse
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

# example:
# python3 EmoClassify.py -p TRAIN 
def set_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--purpose", help="purpose: TRAIN, FINETUNE", type=str)
    parser.add_argument("-n", "--name", help="name: process target csv file", type=str)    
    args = parser.parse_args()
    return args

class EC_DataGenerator(object):
    def __init__(self, args):
        self.seq_length = 30
        self.batch_size = 100
        self.classfication_number = 7
        self.purpose = args.purpose

        if self.purpose == 'TRAIN':
            self.data_separate_point = 3940
            # save model's checkpoint
            self.save_model_path = 'CKPT/emotion_classifier/emo_classify_original_model'
            # restore file
            self.metadata_path = 'DATASET/emotion_classifier/train/Weibo_data_train_metadata.pkl'
            self.idx_input_path = 'DATASET/emotion_classifier/train/Weibo_data_train_idx_input.npy'        
        if self.purpose == 'FINETUNE':
            self.data_separate_point = 200
            # self.origi_filename = args.original_file
            self.save_model_path = 'CKPT/emotion_classifier/emo_classify_finetune_model'
            self.metadata_path = 'DATASET/emotion_classifier/finetune/FineTune_Data_Jay-1_finetune_metadata.pkl'
            self.idx_input_path = 'DATASET/emotion_classifier/finetune/FineTune_Data_Jay-1_finetune_idx_input.npy'
       
        # load metadata, index_input
        self.metadata = self.LoadMetadata()
        self.emo_num = self.metadata['emoticons']
        self.train_data_emo = self.TransEmo(self.emo_num)
        self.train_data = np.load(self.idx_input_path)
        self.data_size = len(self.train_data)
        # pdb.set_trace()
        # pointer position to generate current batch
        self._pointer = 0
    
    def LoadMetadata(self):
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        return metadata

    def TransEmo(self, emo_list):
        new_array = np.zeros((len(emo_list), self.classfication_number), dtype=np.float32)
        for count, num in enumerate(emo_list):
            new_array[count][num] = 1
        return new_array

    def next_batch(self):
        batch_x = []
        batch_y = []
        for num in range(self.batch_size):
            # pdb.set_trace()
            rand_idx = np.random.randint(0, high=self.data_size-self.data_separate_point)
            batch_x.append(self.train_data[rand_idx])
            batch_y.append(self.train_data_emo[rand_idx])
        return np.array(batch_x), np.array(batch_y)

    def test_batch(self):
        batch_x = []
        batch_y = []
        for num in range(self.batch_size):
            rand_idx = np.random.randint(self.data_size-self.data_separate_point, high=self.data_size)
            batch_x.append(self.train_data[rand_idx])
            batch_y.append(self.train_data_emo[rand_idx])
        return np.array(batch_x), np.array(batch_y)        

class EmotionClassifier(object):
    def __init__(self, args):
        # pdb.set_trace()
        self.purpose = args.purpose

        if self.purpose == 'USE':
            self.use_init(args)
            self.filename = args.name

        self.EMOTION_DIC = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4:'like', 5:'sadness', 6:'surprise'}
        self.classification_number = 7

        self.num_of_layer = 2
        self.training_epochs= 100
        self.learning_rate = 0.001
        self.n_input = 30
        self.batch_size = 100
        self.n_hidden = 256
        self.keep_prob = 0 # DROPOUT ratio

        self.x, self.y, weights, biases, pred = self.model_init()
        
        ###### TODO ########
        # Loss and optimizer
        self.softmax_result = tf.nn.softmax(logits=pred)
        self.cost_in_one_batch = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        # Model evaluation
        self.correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        # Initializing the variables

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
        self.use_newFilename = 'TEST/'+ args.newfile
        self.restore_matadata_name = 'TEST/' + self.filename.replace('.csv', '_metadata.pkl')
        self.idx_input_filename = 'TEST/' + self.filename.replace('.csv','_idx_input.npy')
        # self.restore_model = 'original_classify_model-100'
        self.restore_model = 'finetune_model-80'                                        
        self.csvin = open(self.use_newFilename, 'w')
        self.csvin_writer = csv.writer(self.csvin)      
        self.csvin_writer.writerow(['emotion','lyrics','anger','disgust','fear','happiness','like','sadness','surprise'])

    def LoadMetadata(self):    # read data control dictionaries
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
        use_metadata = self.LoadMetadata()
        use_sentences = use_metadata['lines']                   # Chinese Sentence
        use_idx_sentences = np.load(self.idx_input_filename)    # to index

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

def EC_train(model, data):
    save_model_path = data.save_model_path
    epochs = 100
    number_of_epoch = 0
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=6)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    if model.purpose == 'TRAIN':
        session.run(init)
    else:
        ckpt = 'CKPT/emotion_classifier/emo_classify_original_model-75'
        print("restore:", ckpt)
        saver.restore(session, ckpt)        

    
    print("START TRAINING!!")
    while number_of_epoch < epochs:
        cost_total = 0
        num_of_batch_per_epoch = int(data.data_size//data.batch_size)

        #-------
        # train:
        print('EPOCH:{} start!'.format(number_of_epoch))
        for number_of_batch in range(num_of_batch_per_epoch):
            x_train, y_train = data.next_batch()
            x_train = np.reshape(x_train, [-1, model.n_input, 1])
            # pdb.set_trace()
            
            feed_dict = {  
                        model.x: x_train, 
                        model.y: y_train, 
                        }
            
            _, cost, acc = session.run([model.optimizer, model.cost, model.accuracy], feed_dict )
            print("cost:", cost, ";accuracy:", acc)
        # test:
        x_test, y_test = data.test_batch()
        x_test = np.reshape(x_test, [-1, model.n_input, 1])

        feed_dict = {  
                    model.x: x_test, 
                    model.y: y_test, 
                    }
        acc = session.run([model.accuracy], feed_dict )
        print('EPOCH:{}, testing_accuracy:{:4f}'.format(number_of_epoch, acc[0]))
        if number_of_epoch%15 == 0:
            saver.save(session, save_model_path, global_step=number_of_epoch)
        number_of_epoch += 1   

if __name__ == '__main__':
    args = set_argparse()
    TrainData = EC_DataGenerator(args)
    with tf.variable_scope('emotion_classifier'):
        Model = EmotionClassifier(args)
    EC_train(Model, TrainData)

    # pdb.set_trace()








