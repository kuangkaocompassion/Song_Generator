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

# ===== Files =====
# FILENAME: Emoticons, Sentence
# UNK: Unknown word 
FILENAME = 'emoticons_sentence.csv'
VOCAB_SIZE = 20000
limit_length = 30
UNK = 'unk'

# ===== Conditions =====
# forbidden_symbol: symbols not allowed in sentence

forbidden_symbol = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\'0123456789！＠＃＄％＾＆＊（）＿＋＝『』｜「」><`。：，'

# lines: store all qualified lines
lines = []


# ===== Functions =====
# [pad_seq]: subfunction for "zero_pad"
# [index_token]: create "index2word", "word2index"
# [zero_pad]: adding 0(index to ' ') to sentence, whose length is under 10
# [count_unk]: count how many unknown word in a tokenized sentence
# [filter_lines]: test if the line is qualified; if it is, then return new_line; if not, return 0
# [segmentation_to_token]: make segmentation to list
# [read_lines]: read in emoticons and sentence from filename
# [reduce_size]: get rid off long sentence
# [process_data]: START PROCESSING
#  >Functions for other files to import "w2index", "index2word", "emoticons", "idx_input"
# **[load_data]**

def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))

def index_token(tokens , vocab_size):
	# get frequency distribution
    freq_dist = nltk.FreqDist(tokens)
    # pdb.set_trace()
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ token[0] for token in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, vocab

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


def filter_lines(line_seq, emoticon_seq):
	global forbidden_symbol
	new_line_seq = []
	new_emoticon_seq = []
	
	for line_num in range(len(line_seq)):
		new_line = ''.join([ ch for ch in line_seq[line_num] if ch not in forbidden_symbol ])
		if len(new_line)>0 and new_line != " ":
			new_line.rstrip()
			if '\u3000' in str(new_line):
				temp = new_line.split('\u3000')
				new_line = ''.join(temp)
			new_line_seq.append(new_line)
			new_emoticon_seq.append(emoticon_seq[line_num])

	return new_line_seq, new_emoticon_seq

def count_unk(sequence):
	total_num_of_words = 0
	total_num_of_unk = 0
	for line in sequence:
		for token in line:
			if token == 0:
				break
			else:
				total_num_of_words += 1
				if token == 1:
					total_num_of_unk += 1
	return float(total_num_of_unk*100)/float(total_num_of_words)


def segmentation_to_token(sequence):
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

def read_lines(filename):
	sentence_list = []
	emoticon_list = []
	with open(filename, 'r') as csvfile:
		csvfile = csv.reader(csvfile)
		sentence_list = []
		emoticon_list = []
		for line in csvfile:
			sentence_list.append(line[1])
			emoticon_list.append(int(line[0]))
	return sentence_list, emoticon_list

def reduce_size(TokenizedLines, emoticons, upperbound):
	new_input_tokenized = []
	new_emoticons = []
	for num in range(len(TokenizedLines)):
		if len(TokenizedLines[num]) < upperbound:
			new_input_tokenized.append(TokenizedLines[num])
			new_emoticons.append(emoticons[num])
	return new_input_tokenized, new_emoticons

def ration_of_unk(sentence_batch, length=limit_length):
	num_below_half = 0
	num_above_half = 0
	num_total_sentence = len(sentence_batch)
	for sentence in sentence_batch:
		count_no_meaning = 0
		for index_num in sentence:
			if index_num == 1:
				count_no_meaning += 1
		if count_no_meaning > 0 and count_no_meaning < float(limit_length)/2.0:
			num_below_half += 1
		if count_no_meaning > float(limit_length)/2.0:
			num_above_half += 1
	return float(num_below_half)*100/num_total_sentence, float(num_above_half)*100/num_total_sentence

def process_data():
	
	print('\n>> Read lines from file')
	lines, emoticons = read_lines(filename=FILENAME)

	print('\n>> Filter lines from lines')
	lines, emoticons = filter_lines(lines, emoticons)
	print("=====info: filtered lines=====")
	print("sample lines:", lines[0:3])

	print('\n>> Tokenize every lines')
	input_tokenized, input_tokens = segmentation_to_token(lines)

	print('\n>> Filtered out lines with too many tokens')
	input_tokenized, emoticons = reduce_size(input_tokenized, emoticons, limit_length)

	print('\n >> Index2words AND Word2Index')
	idx2w, w2idx, tokens_freq = index_token( input_tokens, vocab_size=VOCAB_SIZE)
	print("=====info: index2word, word2index=====")
	print("number of lines:", len(input_tokenized))
	print("number of tokens:", len(tokens_freq))
	print("sample tokenized lines:", input_tokenized[120: 123])
	print("sample tokens:", tokens_freq[0:20])

	print('\n >> Zero Padding')
	idx_input = zero_pad(input_tokenized, w2idx,upperbound=limit_length)
	print("===original===\n", input_tokenized[120:123])
	print("===idx_input===\n", idx_input[120:123])

	print('\n >> Ratio of sentences with "unk"')
	below_half, above_half = ration_of_unk(idx_input)
	print("number of unk below the total number of words:", below_half, "%")
	print("number of unk grestter than the total number of words:", above_half, "%")



	print('\n >> Save numpy arrays to disk')
	# save them
	np.save('idx_input.npy', idx_input)


	# let us now save the necessary dictionaries
	metadata = {
	        	'w2idx' : w2idx,
	        	'idx2w' : idx2w,
	        	'emoticons': emoticons,
	        	'tokens_freq': tokens_freq,
	        	}

	# write to disk : data control dictionaries
	with open('metadata.pkl', 'wb') as f:
		pickle.dump(metadata, f)

def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_input = np.load(PATH + 'idx_input.npy')
    return metadata, idx_input


if __name__ == '__main__':
    process_data()















