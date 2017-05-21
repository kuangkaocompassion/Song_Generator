from itertools import repeat
import csv
"""
train.tsv: original file for training
emoticons_sentence.csv: only write out emoticons and sentences
"""
with open('train.tsv','r') as tsvin, open('emoticons_sentence.csv', 'w') as csvout:
    tsvin_r = csv.reader(tsvin, delimiter='\t')
    csvout_r = csv.writer(csvout)

    for row in tsvin_r:
    	csvout_r.writerow([row[1], row[2]])

tsvin.close()
csvout.close()


















