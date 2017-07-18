import tensorflow as tf
import numpy as np
import pdb
import argparse
import jieba
import pickle
from EmoClassify import EmotionClassifier
from EmoLyrics import EmoLyricsModel
from EmoLyrics import ELM_DataGenerator

# python3 main_v2.py -p USE -t emotion

# purpose: 
def set_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--purpose", help="purpose: TRAIN, FINETUNE, USE", type=str)
    parser.add_argument("-t","--type", help="type: emotion, baseline", type=str)
    parser.add_argument("-n", "--name", help="name: process target csv file", type=str)    
    args = parser.parse_args()
    return args

def LoadMetadata(path):
    with open(path, 'rb') as f:
        metadata = pickle.load(f)
    return metadata

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

args = set_argparse()
# restore emotion classifier model
with tf.variable_scope('emotion_classifier'):
    model_EC = EmotionClassifier(args)
saver_EC = tf.train.Saver([v for v in tf.global_variables() if 'emotion_classifier' in v.name])
session_EC = tf.Session()
saver_EC.restore(session_EC, 'CKPT/emotion_classifier/emo_classify_finetune_model-90')

# restore song generator model
with tf.variable_scope('song_generator'):
    model_ELM = EmoLyricsModel(args)
# pdb.set_trace()
saver_ELM = tf.train.Saver([v for v in tf.global_variables() if 'song_generator' in v.name])
session_ELM = tf.Session()
saver_ELM.restore(session_ELM, 'CKPT/song_generator/emo_lyrics_model-45')

# classifier: dictionary
metadata = LoadMetadata('DATASET/emotion_classifier/finetune/FineTune_Data_Jay-1_finetune_metadata.pkl')
token2id = metadata['w2idx']
id2token = metadata['idx2w']
token2id_key = list(token2id.keys())

# generator: dictionary
char2id = ELM_DataGenerator.LoadObj('emo_char2id_dict')
id2char = ELM_DataGenerator.LoadObj('emo_id2char_dict')
# pdb.set_trace()
for line_num in range(30):
    if line_num == 0:
        sentence = u'你要离开我知道很简单'
    
    classify_sentence = segmentation_to_token([sentence])[0]
    
    for num,token in enumerate(classify_sentence):
        if token not in token2id_key:
            classify_sentence[num] = 'unk'

    classify_idx = [token2id[t] for t in classify_sentence]
    classify_idx_append = classify_idx
    classify_idx_append += [0 for num in range(30 - len(classify_idx))]
    
    classify_idx_append = np.array(classify_idx_append)
    classify_input = np.reshape(classify_idx_append, [-1, 30, 1])

    feed_dict = {
                 model_EC.x: classify_input
                }
    emo_dist = session_EC.run([model_EC.softmax_result], feed_dict) 
    sentence_idx = [char2id[c] for c in sentence]
    
    sentence_idx += [0 for i in range( 30 - len(sentence_idx))]
    y_fake = [0 for i in range(30)]
    feed_dict_ = {
                 model_ELM.x_input: [sentence_idx],
                 model_ELM.y_input: [y_fake],
                 model_ELM.x_emo_dist: emo_dist[0],
                }
    output_idx = session_ELM.run([model_ELM.DecoWord_idx], feed_dict_)
    output_sentence = [id2char[c] for c in output_idx[0]]
    output_sentence_w = ''.join(output_sentence)
    
    print(output_sentence_w)
    sentence = output_sentence_w
    # pdb.set_trace()




