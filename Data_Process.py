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
limit_length = 10
UNK = 'unk'

# ===== Conditions =====
# forbidden_symbol: symbols not allowed in sentence
# unnessary_information: delete the sentence, if contaning any info in this list 

forbidden_symbol = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\'0123456789！＠＃＄％＾＆＊（）＿＋＝『』｜「」><`。：，'

unnessary_information = ["作詞", "作曲", "主題曲","編曲","更多更詳盡", "主唱","主題曲"
					     , "混音師", "混音室", "周杰倫", "羅大佑", "韋禮安",
					     "唐志中", "監製", "李宗盛", "汪峰", "齊秦", "陳奕迅"]


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
    # index2word
    index2word = ['_'] + [UNK] + [ i for i in tokens ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index

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
	global forbidden_symbol, unnessary_information
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

	print("=====info: tlkenized lines=====")
	print("Number of sentences:", len(input_tokenized))
	print("Number of tokens:", len(input_tokens))
	print(input_tokenized[120:123])

	print('\n >> Index2words AND Word2Index')
	idx2w, w2idx = index_token( input_tokens, vocab_size=VOCAB_SIZE)
	print("=====info: index2word, word2index=====")
	
	

	print('\n >> Zero Padding')
	idx_input = zero_pad(input_tokenized, w2idx,upperbound=limit_length)
	print("===original===\n", input_tokenized[120:123])
	print("===idx_input===\n", idx_input[120:123])
	print("---"*15)

	print('\n >> Number of sentences with many zero')

	num_ten = 0
	num_twenty = 0
	for sentence in idx_input:
		count_space = 0
		for index_num in sentence:
			if index_num == 0:
				count_space += 1
		if count_space > 0 and count_space < 5:
			num_ten += 1
		if count_space >5:
			num_twenty += 1
	print("0 ~ 5zeros:", num_ten,";", float(num_ten)*100/float(len(idx_input)),"%")
	print("5 up zeros:", num_twenty, ";",float(num_twenty)*100/float(len(idx_input)),"%")


	print('\n >> Save numpy arrays to disk')
	# save them
	np.save('idx_input.npy', idx_input)


	# let us now save the necessary dictionaries
	metadata = {
	        	'w2idx' : w2idx,
	        	'idx2w' : idx2w,
	        	'emoticons': emoticons,
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















