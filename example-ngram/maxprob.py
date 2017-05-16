#!/usr/bin/env python3
import sys
import heapq

from preprocess import load_dataset

if __name__ == '__main__':


    n = 3
    pfile = open(sys.argv[2])
    print("Id,Emoticon")
    for rid, _ in load_dataset(open(sys.argv[1])):
        pred = [ 
            str(x[0]) for x in 
                sorted(
                    ((i,float(next(pfile))) for i in range(1, 41))
                ,key=lambda x:x[1])
        ]

        print( "{rid},{prediction}".format(
            rid=rid,
            prediction=" ".join(reversed(pred))
        ))

        

