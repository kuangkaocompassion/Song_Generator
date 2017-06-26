import collections
import os
import sys
import pdb
import csv
import argparse

import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
from hanziconv import HanziConv 			# TW -> CN unicode
from process import Data_Process
from EmoClassify import Emo_Classifier

# example:
# python3 main.py -p TRAIN -n Weibo_data.csv -m char/token
# python3 main.py -p FINETUNE -n FineTune_Data_Jay-1.csv -of Weibo_data -m char/token
# python3 main.py -p USE -n jay_lyrics_notag.csv -m char/token -nf jay_lyrics_notag_withEMO.csv

def default_check(args):
	if (args.method==''):
		args.method = 'char'
	if (args.newfile==''):
		args.newfile = 'test.csv'
	return args

def set_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument("-p","--purpose", help="Purpose: TRAIN, USE, FINETUNE", type=str)
	parser.add_argument("-n", "--name", help="name: process target csv file", type=str)
	parser.add_argument("-m", "--method", help="process method: token, char", type=str)
	parser.add_argument("-of", "--original_file", help="name: original training file", type=str)
	parser.add_argument("-cmd", "--commandinput", help="input by command: 1 or 0", type=int)
	parser.add_argument("-e", "--epochs", help="training/finetune epochs: int", type=int)
	parser.add_argument("-nf", "--newfile", help="result filename of USE: csv", type=str)
	args = parser.parse_args()
	args = default_check(args)
	return args

def get_user_input():	
	if (args.commandinput == 1):
		csvin = open('TEST/userinput.csv', 'w')
		csvin_writer = csv.writer(csvin)
		sentence = input("\nPlease input sentences: (end if enter 'q')\n>>> ")
		while (sentence != 'q'):
			if sentence=='':
				sentence = input(">>> ")
			sentence = HanziConv.toSimplified(sentence)
			# for char in sentence:
				# csvin_writer.writerow([char])
			csvin_writer.writerow([str(sentence)])
			sentence = input(">>> ")
		csvin.close()

### MAIN ###	

args = set_argparse()
if (args.commandinput==1):
	get_user_input()
	args.name = 'userinput.csv'
	args.newfile = 'userinput_out.csv'

print("\n>> Data Process")
data_object = Data_Process(limit_length=30, args=args)
if (args.purpose == "USE"):
	data_object.use_process_data()
else:
	data_object.train_process_data()


print("\n>> Emotion Classification")
emo_object = Emo_Classifier(args=args)
emo_object.USE_emotion_classifier()
print("\nend")

