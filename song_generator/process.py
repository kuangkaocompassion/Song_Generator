# This Python file uses the following encoding: utf-8
import os, sys
import pdb
import csv
import jieba
import random
import sys
import nltk
import itertools
import pickle
import numpy as np
import time
import argparse
from collections import defaultdict

class Data_Process():

    def __init__(self, limit_length, args, vocab_size=None):
        self.purpose = args.purpose
        self.filename = args.name
        self.method = args.method
        
        if self.purpose == 'TRAIN':
            self.metadata = self.filename.replace('.csv', '_train_metadata.pkl')
        if self.purpose == 'USE':
            self.metadata = self.filename.replace('.csv', '_metadata.pkl')
            self.restore_matadata_name = 'DATASET/emotion_classifier/finetune/FineTune_Data_Jay-1_finetune_metadata.pkl'   # use Jay model
        elif self.purpose == 'FINETUNE':
            self.origi_filename = args.original_file
            self.metadata = self.filename.replace('.csv', '_finetune_metadata.pkl')
            self.restore_matadata_name = 'DATASET/emotion_classifier/train/'+ self.origi_filename + '_train_metadata.pkl'

        self.vocab_size = vocab_size
        # self.seq_length = seq_length
        self.limit_length = limit_length
        self.forbidden_symbol = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\'0123456789！＠＃＄％＾＆＊（）＿＋＝『』｜「」><`。：，'
        self.EMOTION_DIC = EMOTION_DIC = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'like':4, 'sadness':5, 'surprise':6}
        

    def pad_seq(self, seq, lookup, maxlen):
        indices = []
        for word in seq:
            if word in lookup:
                indices.append(lookup[word])
            else:
                indices.append(lookup['unk'])
        return indices + [0]*(maxlen - len(seq))

    def index_token(self, tokens , vocab_size, REUSE=False):
        # get frequency distribution
        freq_dist = nltk.FreqDist(tokens)
        # get vocabulary of 'vocab_size' most used words
        if REUSE:
            vocab = freq_dist.most_common(len(freq_dist))
            index2word = [token[0] for token in vocab ]
            index2word = self.original_idx2w + index2word 
            index2word = ['_'] + ['unk'] + ['<go>']+ list(set(index2word[3:])) 
            word2index = dict([(w,i) for i,w in enumerate(index2word)] )
        else:
            vocab = freq_dist.most_common(vocab_size)
            # index2word
            index2word = ['_'] + ['unk'] + ['<go>'] + [ token[0] for token in vocab ]
            # word2index
            word2index = dict([(w,i) for i,w in enumerate(index2word)] )
        return index2word, word2index, vocab, len(freq_dist)

    def zero_pad(self, qtokenized, w2idx, upperbound):
        # num of rows
        data_len = len(qtokenized)

        # numpy arrays to store indices
        idx_q = np.zeros([data_len, upperbound], dtype=np.int32) 

        for i in range(data_len):
            q_indices = self.pad_seq(qtokenized[i], w2idx, upperbound)

            #print(len(idx_q[i]), len(q_indices))
            #print(len(idx_a[i]), len(a_indices))
            idx_q[i] = np.array(q_indices)
        return idx_q


    def filter_lines(self, line_seq, emoticon_seq):
        new_line_seq = []
        new_emoticon_seq = []
        
        for line_num in range(len(line_seq)):
            new_line = ''.join([ ch for ch in line_seq[line_num] if ch not in self.forbidden_symbol])
            if len(new_line)>0 and new_line != " ":
                new_line.rstrip()
                if '\u3000' in str(new_line):
                    temp = new_line.split('\u3000')
                    new_line = ''.join(temp)
                new_line_seq.append(new_line)
                new_emoticon_seq.append(emoticon_seq[line_num])
        return new_line_seq, new_emoticon_seq

    def segmentation_to_token(self, sequence):
        new_sequence = []
        tokens = []

        for line in sequence:
            list_tokens = []
            words = jieba.cut(line, cut_all= False)
            temp = ",".join(words).strip()
            temp_list = temp.split(',')
            for token in temp_list:
                if len(token)>0 and token != " ":
                    list_tokens.append(token)
                    tokens.append(token)
            new_sequence.append(list_tokens)
        
        return new_sequence, tokens
    
    def segmentation_to_char(self, sequence):
        new_sequence = []
        total_char = ''
        for line in sequence:
            total_char += line
        characters = list(set(total_char))
        # pdb.set_trace()
        for line in sequence:
            line_char = [char for char in line]
            new_sequence.append(line_char)  
        
        return new_sequence, characters

    def read_lines(self):
        if self.purpose == 'TRAIN':
            filename = 'DATASET/emotion_classifier/train/' + self.filename
        elif self.purpose == 'FINETUNE':
            filename = 'DATASET/emotion_classifier/finetune/' + self.filename
        sentence_list = []
        emoticon_list = []
        with open(filename, 'r') as csvfile:
            csvfile = csv.reader(csvfile)
            sentence_list = []
            emoticon_list = []
            for line in csvfile:
                sentence_list.append(line[1])
                emoticon_list.append(self.EMOTION_DIC[line[0]])
        return sentence_list, emoticon_list

    def use_read_lines(self, filename):
        with open('TEST/'+filename, 'r') as csvfile:
            csvfile = csv.reader(csvfile)
            sentence_list = []
            for line in csvfile:
                if line!= []:
                    sentence_list.append(line[0])       # only sentence
        return sentence_list

    def reduce_size(self, TokenizedLines, emoticons, upperbound):
        new_input_tokenized = []
        new_emoticons = []
        for num in range(len(TokenizedLines)):
            if len(TokenizedLines[num]) < upperbound:
                new_input_tokenized.append(TokenizedLines[num])
                new_emoticons.append(emoticons[num])
        return new_input_tokenized, new_emoticons
    
    def use_reduce_size(self, TokenizedLines, upperbound):
        new_input_tokenized = []
        for num in range(len(TokenizedLines)):
            if len(TokenizedLines[num]) < upperbound:
                new_input_tokenized.append(TokenizedLines[num])
        return new_input_tokenized

    def ratio_of_unk(self, sentence_batch):
        length=self.limit_length
        num_below_half = 0
        num_above_half = 0
        num_total_sentence = len(sentence_batch)
        for sentence in sentence_batch:
            count_no_meaning = 0
            for index_num in sentence:
                if index_num == 1:
                    count_no_meaning += 1
            if count_no_meaning > 0 and count_no_meaning < float(length)/2.0:
                num_below_half += 1
            if count_no_meaning > float(length)/2.0:
                num_above_half += 1
        return float(num_below_half)*100/num_total_sentence, float(num_above_half)*100/num_total_sentence
    
    def load_metadata(self):
            # read data control dictionaries
        with open(self.restore_matadata_name, 'rb') as f:
            metadata = pickle.load(f)
        return metadata 
    
    def use_process_data(self):
        print("PURPOSE:", self.purpose)
        print("-"*20)
        use_metadata = self.load_metadata()
        use_w2idx = use_metadata['w2idx']

        self.lines = self.use_read_lines(filename=self.filename)
        # pdb.set_trace()
        self.input_tokenized,_ = self.segmentation_to_char(self.lines)
        self.input_tokenized = self.use_reduce_size(self.input_tokenized, self.limit_length)
        self.idx_input = self.zero_pad(self.input_tokenized, use_w2idx, upperbound=self.limit_length)

        # save the necessary dictionaries
        new_metadata = {'lines' : self.lines, 
                        'w2idx' : use_metadata['w2idx'],
                        'idx2w' : use_metadata['idx2w'],
                       }

        # write to disk : data control dictionaries
        with open('TEST/'+ self.metadata, 'wb') as f:
                pickle.dump(new_metadata, f)
        print("SAVE:", 'TEST/'+ self.filename.replace('.csv','_idx_input.npy'))
        print("SAVE:", 'TEST/'+ self.metadata)
        np.save('TEST/'+ self.filename.replace('.csv','_idx_input.npy'), self.idx_input)

    def train_process_data(self):
        print("PURPOSE:", self.purpose)
        print("-"*20)
        print('\n>> Read lines from file')
        self.lines, self.emoticons = self.read_lines()

        print('\n>> INFO about DataSet')
        self.freq_emotion = nltk.FreqDist(self.emoticons)
        for emotion in self.EMOTION_DIC.keys():
            print(emotion,":", self.freq_emotion[self.EMOTION_DIC[emotion]])
        
        print('\n>> Filter lines from lines')
        self.lines, self.emoticons = self.filter_lines(self.lines, self.emoticons)
        # pdb.set_trace()
        # self.lines, self.emoticons = filter_emotions(lines, emoticons, filter=FILTERED_EMO)
        print("=====info: filtered lines=====")
        print("sample lines:", self.lines[0:3])

        print('\n>> Tokenize every lines')
        if self.method == 'token':
            self.input_tokenized, self.input_tokens = self.segmentation_to_token(self.lines)
        elif self.method == 'char':
            self.input_tokenized, self.input_tokens = self.segmentation_to_char(self.lines)
        print('\n>> Filtered out lines with too many tokens')
        self.input_tokenized, self.emoticons = self.reduce_size(self.input_tokenized, self.emoticons, self.limit_length)
        # pdb.set_trace()

        print('\n >> Index2words AND Word2Index')
        if self.purpose == 'TRAIN':
            self.idx2w, self.w2idx, self.tokens_freq, self.origi_num_tokens = self.index_token( self.input_tokens, vocab_size= self.vocab_size)
            print("=====info: index2word, word2index=====")
            print("number of lines:", len(self.input_tokenized), "original:15690")
            print("number of reduced tokens:", len(self.tokens_freq))
            print("sample tokenized lines:", self.input_tokenized[120: 123])
            print("ratio of tokens left:", float(len(self.tokens_freq))*100/float(self.origi_num_tokens), "%")
        elif self.purpose == 'FINETUNE':
            original_metadata = self.load_metadata()
            self.original_w2idx = original_metadata['w2idx']
            self.original_idx2w = original_metadata['idx2w']
            self.idx2w, self.w2idx, _, self.update_num_tokens = self.index_token( self.input_tokens, vocab_size= self.vocab_size, REUSE=True)


        print('\n >> Zero Padding')
        self.idx_input = self.zero_pad(self.input_tokenized, self.w2idx, upperbound=self.limit_length)
        print("===original===\n", self.input_tokenized[120:123])
        print("===idx_input===\n", self.idx_input[120:123])

        print('\n >> Number of sentences with unk')
        below_half, above_half = self.ratio_of_unk(self.idx_input)
        print("Sentence Ratio: num of unk below half num of words:", below_half, "%")
        print("Sentence Ratio: num of unk above half num of words:", above_half, "%")


        print('\n >> Save numpy arrays to disk')
        if self.purpose == 'TRAIN':
            np.save('DATASET/emotion_classifier/train/'+ self.filename.replace('.csv','_train_idx_input.npy'), self.idx_input)
        elif self.purpose == 'FINETUNE':
            np.save('DATASET/emotion_classifier/finetune/'+ self.filename.replace('.csv','_finetune_idx_input.npy'), self.idx_input)
        
        # save the necessary dictionaries
        new_metadata = {
                        'w2idx' : self.w2idx,
                        'idx2w' : self.idx2w,
                        'emoticons': self.emoticons,
                        }

        # write to disk : data control dictionaries
        with open('DATASET/emotion_classifier/'+ self.purpose.lower() + '/'+ self.metadata, 'wb') as f:
                pickle.dump(new_metadata, f)
