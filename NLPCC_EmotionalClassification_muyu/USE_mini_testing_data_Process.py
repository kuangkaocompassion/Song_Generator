from lxml import etree
import csv

doc = etree.parse('mini_testing_data.xml')

xm_string = etree.tostring(doc)
xml_data_tree = etree.fromstring(xm_string)

csv_out = open('mini_testing_data.csv', 'w')
csv_out_w = csv.writer(csv_out)

for paragraph in xml_data_tree:
	for sentence in paragraph:
		csv_out_w.writerow([sentence.text])

csv_out.close()