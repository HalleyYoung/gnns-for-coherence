
# coding: utf-8

# In[1]:


# from google.colab import drive
# # This will prompt for authorization.
# drive.mount('/content/drive')


# In[2]:


# cd drive/My\ Drive/CIS700-project2


# In[3]:


import pandas as pd
import pickle


# In[4]:


# conceptnet = pd.read_csv("conceptnet-assertions-5.6.0.csv",error_bad_lines=False)
# conceptnet.head()

# a = pickle.load(open("sent_counts.pcl"))

with open("sent_counts.pcl", 'rb') as f:
    d = pickle.load(f, encoding='latin1') 


# In[5]:


# d[0]


# In[13]:


import requests
# obj = requests.get('http://api.conceptnet.io/c/en/role').json()
# obj.keys()


# In[14]:


# len(obj['edges'])
# obj['edges'][2]


# In[15]:


# tup = e["@id"].split(",")
def findTuples(w):
    obj = requests.get('http://api.conceptnet.io/c/en/' + w).json()
    edges = obj['edges']
    # print("len(edges)",len(edges))
    results = []
    for e in edges:
        if "language" not in e['start'].keys() or "language" not in e['end'].keys():
            continue     
        if e['start']["language"] != "en" or e['end']["language"] != "en":
            continue
        results.append((e['rel']["label"], e['start']["label"], e['end']["label"]))
    return results


# In[16]:


# findTuples("stiles")


# In[17]:


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def filter_p_only(curr_p_edges, curr_p_nodes):
    new_edges = []
    for edge in curr_p_edges:
        rel = edge[0]
        start = edge[1]
        end = edge[2]
        if start in curr_p_nodes and end in curr_p_nodes:
            new_edges.append(edge)
    return new_edges
    


# In[22]:
# pids = [i for i in range(1000)]
# pids = pids[97]

edges_by_p = []
node_by_p = []
curr_p_edges = []
curr_p_nodes = []

with open("nyt.txt.sent.all.amr-semeval.parsed", 'r') as f1:
    lines0 = f1.readlines()
  
    lines = lines0[:100000]

    senid = -1
    pid = 0
    i = 0
    w = ""

    while i < len(lines):
        if lines[i] == "\n":
            i += 1
            continue
        print("i", i)
        if lines[i][:7] == "# ::id ":
            #split the running!
            # senid = int(line[i].split()[-1]) - 1
            senid += 1
            i += 2
            if senid not in d[pid] and senid in d[pid+1]:
                pid += 1
                node_by_p.append(curr_p_nodes)


                print("i, senid, pid", i, senid, pid)
                print("len(curr_p_nodes), curr_p_nodes", len(curr_p_nodes), curr_p_nodes)
                print("len(curr_p_edges), curr_p_edges", len(curr_p_edges), curr_p_edges[:5])

                new_curr_p_edges = filter_p_only(curr_p_edges, curr_p_nodes)
                edges_by_p.append(new_curr_p_edges)
                print("len(new_curr_p_edges), new_curr_p_edges", len(new_curr_p_edges), new_curr_p_edges[:5])
                print()

                curr_p_edges = []
                curr_p_nodes = []
            continue

        li = lines[i].split()
        word = li[-1].strip()
        word = word.replace(")","")
    #     print("word", word)

        if '"' in word:
            word = word.replace('"','')

        chs = word.split("-")
        j = len(chs) - 1
        while j >= 0:
            tp = chs[j]

            if not hasNumbers(tp):
                w = chs[j]

                print("w", w)            
                curr_p_edges.extend(findTuples(w))
                print()
                curr_p_nodes.append(w)
            j -= 1
        i += 1
  


        
    


# In[23]:


with open('conceptNet_edges_new.txt', 'w') as f2:
    for i in range(len(edges_by_p)):
        f2.write("p" + str(i) + "\n")
        for x in edges_by_p[i]:
            f2.write(str(x))
            f2.write("\n")
        f2.write("\n")

