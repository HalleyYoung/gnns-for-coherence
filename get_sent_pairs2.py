import sys
# import torch
# from torch.nn.functional import softmax
# from pytorch_pretrained_bert import BertTokenizer, BertModel, \
#      BertForMaskedLM,BertForNextSentencePrediction
import os
from os import listdir
from os.path import isfile, join
import datetime
import pickle
import random

## Get sentences from NYT texts
def get_sent_pairs(nyt_file):
    docs = dict()
    with open(nyt_file, 'r', encoding='utf8') as f:
        doc_lines = dict()
        par_lines = []
        cur_doc, cur_par = -1, -1
        for i, line in enumerate(f):
            ls = line.strip().split("\t")

            if len(ls) == 1:
                print(ls)
                print(i)

            ## If line is start of a new document, save previous document
            if cur_doc != ls[1]:
                if cur_doc != -1:
                    doc_lines[cur_par] = par_lines
                    docs[cur_doc] = doc_lines

                doc_lines, par_lines = dict(), [ls[4]]
                cur_doc = ls[1]
                cur_par = int(ls[2])

            ## If line is start of a new paragraph, save previous paragraph
            elif cur_par != int(ls[2]):
                if cur_par != -1:
                    doc_lines[cur_par] = par_lines

                par_lines = [ls[4]]
                cur_par = int(ls[2])

            else:
                par_lines.append(ls[4])
    
    ## Extract original and reversed sentence pairs from dictionary
    first_sent = []
    prev_doc = False
    op, rev_p, ran_p, fake_p = [], [], [], []
    for doc_id, doc_dict in docs.items():

        paragraphs = [k for k in doc_dict.values() if len(k) > 1]
        if len(paragraphs) > 2 and prev_doc:
            fake_pars = [i for i in prev_doc.values() if len(i) > 1]
            for par_id, paragraph in doc_dict.items():
                if len(paragraph) > 3:
                    for i in range(2,len(paragraph)-1):
                        print(len(first_sent))
                        first_sent.append(paragraph[i])
                        op.append(paragraph[i + 1])
                        rev_p.append(paragraph[i - 1])
                        other_ps = [i for i in paragraphs if i[0] != paragraph[0] and len(i) > 1]
                        other_ps = random.choice(other_ps)
                        ran_p.append(other_ps[random.randint(1, len(other_ps) - 1)])
                        fake_ps =  random.choice(fake_pars)
                        fake_p.append(fake_ps[random.randint(1, len(fake_ps) - 1)])

        if len(paragraphs) > 0:
            prev_doc = doc_dict
           
    return op, rev_p, ran_p, fake_p, first_sent
    


if __name__ == '__main__':
    nyt_file = sys.argv[1]

    start = datetime.datetime.now()

    orig_sents, rev_sents, rand_sents, fake_sents, first_sent = get_sent_pairs(nyt_file)
    with open("./rand_sents.pcl", 'wb') as f:
        pickle.dump(rand_sents, f)
    with open("./fake_sents.pcl", 'wb') as f:
        pickle.dump(fake_sents, f)   
    with open("./rev_sents.pcl", 'wb') as f:
        pickle.dump(rev_sents, f)   
    with open("./orig_sents.pcl", 'wb') as f:
        pickle.dump(orig_sents, f)    
    with open("./first_sent.pcl", "wb") as f:
        pickle.dump(first_sent, f) 
    # with open(intermediate_folder + "/orig_probs.pkl", 'wb') as f:
    #     pickle.dump(orig_probs, f)
    # with open(intermediate_folder + "/orig_probs.pkl", 'wb') as f:
    #     pickle.dump(orig_probs, f)
    # with open(intermediate_folder + "/orig_probs.pkl", 'wb') as f:
    #     pickle.dump(orig_probs, f)

    finish = datetime.datetime.now()
    print("Total time: " + str(finish-start))
    
'''
python3 get_sent_pairs.py nyt_clean_sents_all.txt 
'''
    
