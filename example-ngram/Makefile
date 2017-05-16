
KENLM := ./kenlm/bin
ORDER := 3



.PHONY: all clean clean-all prep

.SECONDARY:

all: prediction.csv
prep: kenlm pypackages


kenlm:
	wget -O - http://kheafield.com/code/kenlm.tar.gz | tar xz
	cd kenlm && ./bjam -j4
pypackages:
	python3 -mpip install jieba
	python3 -mpip install hanziconv
###

prediction.csv: 
	#word segmentation + subsitute EMOTICON for the labeled candidate
	python3 preprocess.py < ../train.tsv > corpus-train.txt
	#word segmentation + subsitute EMOTICON for all candidates
	python3 preprocess.py -test < ../test.tsv > corpus-test.txt
	#build language model
	$(KENLM)/lmplz -o $(ORDER) --text corpus-train.txt --arpa lm.arpa
	#convert language model into binary format, for faster loading
	$(KENLM)/build_binary lm.arpa lm.bin
	#evaluate the probabilities of all possible sentences
	$(KENLM)/query -s lm.bin < corpus-test.txt | cut -d' ' -f2 > test.prob
	#sort & output the prediction for every 40 lines
	python3 maxprob.py ../test.tsv test.prob > $@
###

clean:
	rm -rf lm*.arpa lm*.bin test*.prob prediction*.csv test*.prob
clean-all: clean
	rm -rf corpus-train.txt corpus-test.txt *.cache
	rm -rf kenlm
	rm -rf __pycache__
	
