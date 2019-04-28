
# coding: utf-8
import pandas as pd
import pickle
from time import time

# lines[i] = pickle.load(open("sent_counts.pcl"))

with open("sent_counts.pcl", 'rb') as f:
    d = pickle.load(f, encoding='latin1') 

with open("conceptNet", 'rb') as f:
    d_nondd = pickle.load(f, encoding='latin1') 


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def filter_p_only(curr_p_edges, curr_p_nodes):
    new_edges = set()
    for edge in curr_p_edges:
        rel = edge[0]
        start = edge[1]
        end = edge[2]
        if start in curr_p_nodes and end in curr_p_nodes:
            new_edges.add(edge)
    return list(new_edges)
    

# edges_by_p = []
# node_by_p = []
curr_p_edges = []
curr_p_nodes = []
passed = set()

with open("nyt.txt.sent.all.amr-semeval.parsed", 'r') as f1, open('conceptNet_edges_AHH.txt', 'w') as f2:
    lines = f1.readlines()
  
    senid = -1
    pid = 0
    i = 0

    w = ""

    l = len(lines)
    # while True:
    t1 = time()
    while i < l:
        if lines[i] == "\n":
            i += 1
            continue
        # print("i", i)
        # print("     i, senid, pid", i, senid, pid)
        if lines[i][:7] == "# ::id ":
            ####################################
            #split the running!
            senid = int(lines[i].split()[-1]) - 1
            # senid += 1
            ####################################

            
            i += 2
            if senid not in d[pid] and senid in d[pid+1]:
                pid += 1
                # node_by_p.append(curr_p_nodes)


                # print("i, senid, pid", i, senid, pid)
                # print("len(curr_p_nodes), curr_p_nodes", len(curr_p_nodes), curr_p_nodes[:5], "\n")
                # # print("len(curr_p_edges), curr_p_edges", len(curr_p_edges), curr_p_edges[:5], "\n")

                new_curr_p_edges = filter_p_only(curr_p_edges, curr_p_nodes)
                # edges_by_p.append(new_curr_p_edges)
                # print("len(new_curr_p_edges), new_curr_p_edges", len(new_curr_p_edges), new_curr_p_edges[:5])
                # print()
                ################################
                # for i in range(len(edges_by_p)):
                f2.write("p" + str(pid) + "\t" + "numEdges " + str(len(new_curr_p_edges)) + "\n")
                for x in new_curr_p_edges:
                    f2.write(str(x))
                    f2.write("\n")
                f2.write("\n")
                ################################

                curr_p_edges = []
                curr_p_nodes = []
            continue

        li = lines[i].split()
        word = li[-1].strip()
        word = word.replace(")","")

        if '"' in word:
            word = word.replace('"','')

        chs = word.split("-")
        j = len(chs) - 1
        while j >= 0:
            tp = chs[j]

            if not hasNumbers(tp):
                w = tp
                # if "." in w:
                w = w.replace(".","")
                    # ind = w.index(".")
                    # ind = w.rfind(".")
                    # w = w[ind+1:]
                w = w.lower()

                # print("w", w)

                try:
                    curr_p_edges.extend(d_nondd[w])
                    curr_p_nodes.append(w)
                except:
                    try:
                        ww = w[1:]
                        curr_p_edges.extend(d_nondd[ww])
                        curr_p_nodes.append(ww)
                    except:
                        print(w, " Not in ConceptNet!!!", "\n")
                        passed.add(w)
                        print(i, time()-t1)
                        pass

            j -= 1
        i += 1
        # print(i, time()-t1)

    print(len(passed), list(passed))


