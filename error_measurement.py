# This Python file uses the following encoding: utf-8
import os, sys
import csv
from Data_Process import load_data

"""
Find out all the sentences in category7
"""
with open('emoticons_sentence.csv', 'r') as csvfile, open('category_7.csv', 'w') as csvfile2:
	csvfile_r = csv.reader(csvfile)
	csvfile2_w = csv.writer(csvfile2)
	for line in csvfile_r:
		if line[0] == '7':
			csvfile2_w.writerow(line)

"""
Find out all the category with "愛" in its sentences
"""
metadata, idx_input = load_data(PATH='')
emoticons = metadata['emoticons']
training_data = idx_input

for sentence_num in range(len(training_data)):
	
