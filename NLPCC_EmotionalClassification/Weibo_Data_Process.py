from lxml import etree
import csv

# Two Files:
# File 1: 'Training_data_for_Emotion_Classification.xml'
# File 2: 'Training_data_for_Emotion_Expression_Identification.xml'
doc = etree.parse('Training_data_for_Emotion_Classification.xml')
doc2 = etree.parse('Training_data_for_Emotion_Expression_Identification.xml')

xm_string = etree.tostring(doc)
xm_string_2 = etree.tostring(doc2)
xml_data_tree = etree.fromstring(xm_string)
xml_data_tree_2 = etree.fromstring(xm_string_2)

csv_out = open('Training_data_for_Emotion_Classification.csv', 'w')
csv_out_w = csv.writer(csv_out)

for paragraph in xml_data_tree:
	for sentence in paragraph:
		if sentence.attrib['opinionated'] == 'Y':
			csv_out_w.writerow([sentence.attrib['emotion-1-type'], sentence.text])

for paragraph in xml_data_tree_2:
	for sentence in paragraph:
		if sentence.attrib['opinionated'] == 'Y':
			csv_out_w.writerow([sentence.attrib['emotion-1-type'], sentence.text])

csv_out.close()