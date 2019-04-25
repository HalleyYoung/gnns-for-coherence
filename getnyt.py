import os
from bs4 import BeautifulSoup
import nltk
import pickle

def concat(xss):
	new = []
	for xs in xss:
		new.extend(xs)
	return new

a = list(os.walk("1987"))

sents = []
sent_counts = {}
sent_count = 0
text_count = 0

for (fold, sub_dirs, files) in a:
	for file in files:
		if file[-3:] == "xml":
			f = open(fold + "/" + file).read()
			soup = BeautifulSoup(f, 'xml')
			ps = soup.findAll("p")
			for val in ps:
				text = val.text
				sent_text = nltk.sent_tokenize(text)
				if len(sent_text) >= 4 and str(sent_text[-1])[-1] == "." and all([len(i) > 20 and "." not in i[:10] and "LEAD" not in i[:10] for i in sent_text]):
					print("in " + str(text_count))
					sents.append(sent_text)
					sent_counts[text_count] = range(sent_count, sent_count + len(sent_text))
					sent_count = sent_count + len(sent_text)
					text_count += 1
					if sent_count > 80000:
						f = open("nyt.sent", "w+")
						f.write("\n".join(concat(sents)).replace("''", '"'))
						f.close()

						pickle.dump(sent_counts, open("sent_counts.pcl", "w+"))

						exit(0)


