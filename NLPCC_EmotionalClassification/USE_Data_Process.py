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
from collections import defaultdict
import Model_Data_Process

# for use, not fine
metadata, idx_input = Model_Data_Process.load_data('FineTune_metadata.pkl', 'FineTune_idx_input.npy')
w2idx = metadata['w2idx']

UNK = 'unk'
# FILENAME = 'FineTune_Data_Jonathenlee.csv'
# FILENAME = 'TEST/jay_lyrics.csv'
FILENAME = 'TEST/'+sys.argv[1]
INDEX_INPUT = 'USE_idx_input.npy'
limit_length = 30

def read_lines(filename):
	sentence_list = []
	with open(filename, 'r') as csvfile:
		csvfile = csv.reader(csvfile)
		sentence_list = []
		for line in csvfile:
			sentence_list.append(line[0])       # only sentence
	return sentence_list

def segmentation_to_token(sequence):
	new_sequence = []

	for line in sequence:
		list_tokens = []
		words = jieba.cut(line, cut_all= False)
		temp = ",".join(words).strip()
		temp_list = temp.split(',')
		for token in temp_list:
			if len(token)>0 and token != " ":
				list_tokens.append(token)
		new_sequence.append(list_tokens)
	
	return new_sequence

def reduce_size(TokenizedLines, upperbound):
	new_input_tokenized = []
	for num in range(len(TokenizedLines)):
		if len(TokenizedLines[num]) < upperbound:
			new_input_tokenized.append(TokenizedLines[num])
	return new_input_tokenized

def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))

def zero_pad(qtokenized, w2idx, upperbound):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, upperbound], dtype=np.int32) 

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, upperbound)

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
    return idx_q


lines = read_lines(filename=FILENAME)
input_tokenized = segmentation_to_token(lines)
input_tokenized = reduce_size(input_tokenized, limit_length)
idx_input = zero_pad(input_tokenized, w2idx,upperbound=limit_length)
# pdb.set_trace()

np.save(INDEX_INPUT, idx_input)

metadata = {
	'USE_input': idx_input,
	'USE_sentences': input_tokenized,
}


with open('USE_metadata.pkl', 'wb') as f:
	pickle.dump(metadata, f)
