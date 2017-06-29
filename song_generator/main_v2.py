import tensorflow as tf
import numpy as np
import pdb
import argparse
from EmoClassify import EmotionClassifier
from EmoLyrics import EmoLyricsModel

def set_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--purpose", help="purpose: TRAIN, FINETUNE", type=str)
    parser.add_argument("-n", "--name", help="name: process target csv file", type=str)    
    args = parser.parse_args()
    return args
# args = set_argparse()
# with tf.variable_scope('emotion_classifier'):
#     model_EC = EmotionClassifier(args)

# with tf.variable_scope('song_generator'):
#     model_ELM = EmoLyricsModel('USE')
# pdb.set_trace()

args = set_argparse()
# restore emotion classifier model
with tf.variable_scope('emotion_classifier'):
    model_EC = EmotionClassifier(args)
saver_EC = tf.train.Saver([v for v in tf.global_variables() if 'emotion_classifier' in v.name])
session_EC = tf.Session()
saver_EC.restore(session_EC, 'CKPT/emotion_classifier/emo_classify_finetune_model-90')

# restore song generator model
with tf.variable_scope('song_generator'):
    model_ELM = EmoLyricsModel('TRAIN')
# pdb.set_trace()
saver_ELM = tf.train.Saver([v for v in tf.global_variables() if 'song_generator' in v.name])
session_ELM = tf.Session()
saver_ELM.restore(session_ELM, 'CKPT/song_generator/emo_lyrics_model-75')

pdb.set_trace()
