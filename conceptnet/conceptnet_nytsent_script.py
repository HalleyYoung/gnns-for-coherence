
# coding: utf-8
import pandas as pd
import pickle
from time import time
import nltk
from nltk.corpus import stopwords
# lines[i] = pickle.load(open("sent_counts.pcl"))

stopwords = set(stopwords.words('english'))

with open("sent_counts.pcl", 'rb') as f:
    d = pickle.load(f, encoding='latin1') 

with open("conceptNet", 'rb') as f:
    d_nondd = pickle.load(f, encoding='latin1') 


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def filter_p_only(curr_p_edges, curr_p_nodes):
    new_edges = set()
    curr_p_edges = list(curr_p_edges)
    curr_p_nodes = list(curr_p_nodes)
    for edge in curr_p_edges:
        rel = edge[0]
        start = edge[1]
        end = edge[2]
        if start != end and start not in stopwords and start in curr_p_nodes and end not in stopwords and end in curr_p_nodes:
            new_edges.add(edge)
        elif start not in stopwords and start in curr_p_nodes and end not in curr_p_nodes:
            for i in range(len(curr_p_nodes)):
                endl = end.split("_")
                if curr_p_nodes[i] != start and curr_p_nodes[i] not in stopwords and "_" in end and curr_p_nodes[i] in endl:
                    print("end!!         curr_p_nodes[i], edge:\t", curr_p_nodes[i], edge)
                    new_edges.add(edge)
        elif end not in stopwords and end in curr_p_nodes and start not in curr_p_nodes:
            for i in range(len(curr_p_nodes)):
                startl = start.split("_")
                if curr_p_nodes[i] != end and curr_p_nodes[i] not in stopwords and "_" in start and curr_p_nodes[i] in startl:
                    print("start!!         curr_p_nodes[i], edge:\t", curr_p_nodes[i], edge)
                    new_edges.add(edge)
    return list(new_edges)
    

# curr_p_edges = []
# curr_p_nodes = []
# passed = set()
passed = {}
curr_p_edges = set()
curr_p_nodes = set()

with open("nyt.sent.new", 'r') as f1, open('conceptNet_nyt_edges_swfree.txt', 'w') as f2, open('words_not_in_cn', 'wb') as f3:
    lines = f1.readlines()
  
    # senid = -1
    pid = 0
    i = 0
    # w = ""

    l = len(lines)
    # while True:
    t1 = time()
    while i < l:
        line = lines[i].replace("/", " ").replace("?", "").replace("[", "").replace("]", "").replace("<", "").replace(">", "").replace("*", "").replace("&", "").replace("!", "").replace("(", "").replace(")", "").replace("-", " ").replace(",", "").replace(".", "").replace("'s ", " ").replace("'","").replace('"','').lower()
        # tokens = nltk.word_tokenize(line)
        tokens = line.split()
        print("     i, pid, d[pid]", i, pid, d[pid])
        print("tokens", tokens)
        if i not in d[pid] and i in d[pid+1]:
            pid += 1

            print("len(curr_p_nodes), curr_p_nodes", len(curr_p_nodes), curr_p_nodes, "\n")

            new_curr_p_edges = filter_p_only(curr_p_edges, curr_p_nodes)
            ################################
            # for i in range(len(edges_by_p)):
            f2.write("p" + str(pid) + "\t" + "numEdges " + str(len(new_curr_p_edges)) + "\n")
            for x in new_curr_p_edges:
                f2.write(str(x))
                f2.write("\n")
            f2.write("\n")
            ################################

            # curr_p_edges = []
            # curr_p_nodes = []
            curr_p_edges = set()
            curr_p_nodes = set()

        for w in tokens:
            if not hasNumbers(w):
                try:
                    curr_p_edges = curr_p_edges.update(set(d_nondd[w]))
                    curr_p_nodes.add(w)
                    # curr_p_edges.extend(d_nondd[w])
                    # curr_p_nodes.append(w)
                except:
                    print(w, " Not in ConceptNet!!!", "\n")
                    # passed.add(w)
                    if w not in passed:
                        passed[w] = 1
                    else:
                        passed[w] += 1
                    print(i, time()-t1)
                    pass

        i += 1
        # print(i, time()-t1)

    # print(len(passed), list(passed))
    print("len(passed), sum(passed.values()), passed", len(passed), sum(passed.values()), passed)
    pickle.dump(passed, f3)



# {'by', 'ourselves', 'they', 'for', 'am', "didn't", 'having', 'and', 'which', 'after', 'all', 'theirs', 'm', 'couldn', "shan't", "doesn't", 'our', 'over', 'who', 'about', 'ma', 'most', 'ain', 're', 'once', 'before', 'a', 'isn', 'those', 'myself', 'hadn', 'some', 'we', 'if', 'them', 'few', 'yours', 'at', 'above', 't', 'd', 'this', 'not', 've', 'will', "don't", "wouldn't", "weren't", 'weren', 'just', 'did', 'on', 'i', 'through', "shouldn't", 'between', 'didn', 'mightn', 'in', 'again', 'doesn', 'against', 'whom', 'does', 'until', 'of', 'his', 'is', 'the', 'into', 's', 'y', 'too', 'then', 'him', 'was', "it's", "she's", 'out', 'should', 'now', "aren't", 'don', "wasn't", 'yourself', 'yourselves', "needn't", 'were', 'to', 'their', 'an', "you'd", "mightn't", 'more', "you'll", 'shan', 'ours', 'mustn', 'herself', 'can', 'her', 'be', 'are', 'been', 'where', 'very', 'such', "isn't", 'during', 'because', 'what', 'll', 'its', "hadn't", 'shouldn', 'same', 'haven', 'here', 'up', 'had', 'me', 'doing', 'himself', 'no', 'hasn', 'while', "haven't", 'or', 'own', 'o', 'my', 'she', 'so', 'have', 'won', 'do', "won't", 'it', 'he', 'both', 'has', 'how', 'any', "mustn't", 'itself', 'you', 'down', 'when', 'why', "should've", 'being', "you're", 'hers', 'themselves', 'as', 'off', "couldn't", 'with', 'wasn', 'than', 'other', 'below', "hasn't", "that'll", 'but', 'your', 'there', 'aren', 'wouldn', 'that', 'further', 'only', 'from', 'nor', 'needn', 'each', 'under', "you've", 'these'}
