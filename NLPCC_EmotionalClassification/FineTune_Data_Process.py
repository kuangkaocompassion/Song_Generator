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
import Model_Data_Process
from collections import defaultdict

# ===== Files =====
# FILENAME: Emoticons, Sentence
# UNK: Unknown word 
FILENAME = 'FineTune_Data_Jonathenlee.csv'
METADATA = 'FineTune_metadata.pkl'
INDEX_INPUT = 'FineTune_idx_input.npy'

original_metadata, idx_input = Model_Data_Process.load_data('Model_metadata.pkl', 'Model_idx_input.npy')
original_idx2w = original_metadata['idx2w']
exp_info = original_metadata['exp_info']

# get original 
original_metadata = 'Model_metadata.pkl'

# anger, disgust, fear, happiness, like, sadness, surprise
EMOTION_DIC = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'like':4, 'sadness':5, 'surprise':6}
FILTERED_EMO = ['fear', 'surprise']
VOCAB_SIZE = 5000
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
    global original_idx2w
    # get frequency distribution
    freq_dist = nltk.FreqDist(tokens)
    # pdb.set_trace()
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(len(freq_dist))
    # index2word
    index2word = [ token[0] for token in vocab ]
    original_idx2w += index2word
    original_idx2w = list(set(original_idx2w))
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(original_idx2w)] )
    return original_idx2w, word2index, vocab, len(freq_dist)

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
			emoticon_list.append(EMOTION_DIC[line[0]])
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

def filter_emotions(sequence, emotion, filter):
	new_sequence = []
	new_emoticons = []
	for num in range(len(sequence)):
		status = 1
		for emo in filter:
			if emotion[num] == EMOTION_DIC[emo]:
				status = 0
		if status:
			new_sequence.append(sequence[num])
			new_emoticons.append(emotion[num])
	return new_sequence, new_emoticons

def process_data():
	
	print('\n>> Read lines from file')
	lines, emoticons = read_lines(filename=FILENAME)

	print('\n>> INFO about DataSet')
	freq_emotion = nltk.FreqDist(emoticons)
	for emotion in EMOTION_DIC.keys():
		print(emotion,":", freq_emotion[EMOTION_DIC[emotion]])
	
	print('\n>> Filter lines from lines')
	lines, emoticons = filter_lines(lines, emoticons)
	lines, emoticons = filter_emotions(lines, emoticons, filter=FILTERED_EMO)
	print("=====info: filtered lines=====")
	print("sample lines:", lines[0:3])

	print('\n>> Tokenize every lines')
	input_tokenized, input_tokens = segmentation_to_token(lines)
	# pdb.set_trace()
	print('\n>> Filtered out lines with too many tokens')
	input_tokenized, emoticons = reduce_size(input_tokenized, emoticons, limit_length)
	# pdb.set_trace()

	print('\n >> Index2words AND Word2Index')
	idx2w, w2idx, tokens_freq, origi_num_tokens = index_token( input_tokens, vocab_size=VOCAB_SIZE)
	print("=====info: index2word, word2index=====")
	print("number of lines:", len(input_tokenized), "original:15690")
	print("number of reduced tokens:", len(tokens_freq))
	print("sample tokenized lines:", input_tokenized[120: 123])
	print("ration of tokens left:", float(len(tokens_freq))*100/float(origi_num_tokens), "%")
	# print("tokens:", tokens_freq)
	exp_info['number_of_lines'] = len(input_tokenized)
	exp_info['tokens_left'] = float(len(tokens_freq))*100/float(origi_num_tokens)
	exp_info['least_frequency'] = tokens_freq[-1][1]

	print('\n >> Zero Padding')
	idx_input = zero_pad(input_tokenized, w2idx,upperbound=limit_length)
	print("===original===\n", input_tokenized[120:123])
	print("===idx_input===\n", idx_input[120:123])

	print('\n >> Number of sentences with unk')

	below_half, above_half = ration_of_unk(idx_input)
	print("Sentence Ratio: num of unk below half num of words:", below_half, "%")
	print("Sentence Ratio: num of unk above half num of words:", above_half, "%")
	exp_info['ration_unk_below_half'] = below_half
	exp_info['ratio_unk_above_half'] = above_half


	print('\n >> Save numpy arrays to disk')
	np.save(INDEX_INPUT, idx_input)


	# save the necessary dictionaries
	metadata = {
	        	'w2idx' : w2idx,
	        	'idx2w' : idx2w,
	        	'emoticons': emoticons,
	        	'tokens_freq': tokens_freq,
	        	'exp_info': exp_info,
	        	}

	# write to disk : data control dictionaries
	with open(METADATA, 'wb') as f:
		pickle.dump(metadata, f)
	
# def load_data(PATH='', METADATA, INDEX_INPUT):
#     # read data control dictionaries
#     with open(PATH + METADATA, 'rb') as f:
#         metadata = pickle.load(f)
#     # read numpy arrays
#     idx_input = np.load(PATH + INDEX_INPUT)
#     return metadata, idx_input


def load_data(METADATA, INDEX_INPUT,PATH=''):
    # read data control dictionaries
    with open(PATH + METADATA, 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_input = np.load(PATH + INDEX_INPUT)
    return metadata, idx_input

if __name__ == '__main__':
    process_data()











