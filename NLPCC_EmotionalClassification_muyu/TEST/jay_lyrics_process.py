import csv

textin = open('周杰伦歌词大全.txt', 'r')
textout = open('jay_lyrics.csv', 'w')
textout_write = csv.writer(textout)
while True:
	list_line = textin.readline().split()
	for line in list_line:
		textout_write.writerow([line])
textin.close()
textout.close()