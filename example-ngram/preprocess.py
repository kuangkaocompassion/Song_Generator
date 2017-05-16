#!/usr/bin/env python3
import sys
import argparse

#import nltk


candidate = { i:'EMOTICON_+'+str(i) for i in range(1,41) }

def load_dataset(infile, transform=None, tokenizer=None):
    for line in infile:

        rid, emot, text = line.strip().split('\t', maxsplit=2)
        if tokenizer:
            text = HanziConv.toSimplified(text)
            text = list( w for w in tokenizer.cut(text) if w != ' ' )

            try:
                pos  = text.index('EMOTICON')

                if transform:
                    text[pos] = candidate[int(emot)]

                yield rid, text, pos
            except ValueError: # no EMOTICON
                print("oops, sent #{} no emoticon!".format(rid), file=stderr)
        else:
            yield rid, text

if __name__ == '__main__':

    import jieba
    from hanziconv import HanziConv
    tokenizer = jieba.Tokenizer()
    tokenizer.tmp_dir = "."

    if len(sys.argv) > 1:
        for rid, text, pos in load_dataset(sys.stdin, transform=False, tokenizer=tokenizer):
            for cand, cand_string in sorted(candidate.items()):
                text[pos] = cand_string
                print(' '.join(text))
    else:
        for rid, text, pos in load_dataset(sys.stdin, transform=True, tokenizer=tokenizer):
            print(' '.join(text))

