from time import time
def csv2dict():
    i = 0
    with open("conceptnet-assertions-5.6.0.csv","r") as f1, open('conceptNet_all.txt', 'w') as f2:
        while True:
            t1 = time()
            a = f1.readline()
            aa = a.split('\t')[0]


            try:
                b0 = aa.split("/,/c/")[0]
                b1 = aa.split("/,/c/")[1]
                b2 = aa.split("/,/c/")[2]

                c0 = b0.split("/")
                c1 = b1.split("/")
                c2 = b2.split("/")
        

                try:
                    rel = c0[-1]
                    w1 = c1[1]
                    w2 = c2[1]
                except:
                    print(aa)
                    print(b0, "\t", c0)
                    print(b1, "\t", c1)
                    print(b2, "\t", c2)
                    print()

            except:
                print(time() - t1, aa)
                continue
        

      
            if c1[0] == "en" and c2[0] == "en":
                i += 1
           
                f2.write('\t'.join([rel, w1, w2]))
                f2.write("\n")
            
    #         if i > 15:
    #           break
 

csv2dict()


# # In[ ]:


# import pandas as pd
# import pickle


# # In[ ]:


# with open("sent_counts.pcl", 'rb') as f0:
#     d = pickle.load(f0, encoding='latin1') 
# # import requests
# # tup = e["@id"].split(",")
# def findTuples(w):
#     obj = requests.get('http://api.conceptnet.io/c/en/' + w).json()
#     edges = obj['edges']
#     # print("len(edges)",len(edges))
#     results = []
#     for e in edges:
#         if "language" not in e['start'].keys() or "language" not in e['end'].keys():
#             continue     
#         if e['start']["language"] != "en" or e['end']["language"] != "en":
#             continue
#         results.append((e['rel']["label"], e['start']["label"], e['end']["label"]))
#     return results


# # In[ ]:


# i = 0
# with open("conceptnet-assertions-5.6.0.csv","r") as f:
#   while True:
#     a =f.readline()
#     b = a.split(",")[1]
#     c = b.split("/")
#     if c[1] == "en" or c[2] == "en":
#       i += 1
#       print(a)
#       if i > 5:
#         break


# # In[ ]:




# def hasNumbers(inputString):
#     return any(char.isdigit() for char in inputString)

# def filter_p_only(curr_p_edges, curr_p_nodes):
#     new_edges = []
#     for edge in curr_p_edges:
#         rel = edge[0]
#         start = edge[1]
#         end = edge[2]
#         if start in curr_p_nodes and end in curr_p_nodes:
#             new_edges.append(edge)
#     return new_edges
    


# # In[22]:
# # pids = [i for i in range(1000)]
# # pids = pids[97]



# # In[ ]:


# edges_by_p = []
# node_by_p = []
# curr_p_edges = []
# curr_p_nodes = []

# with open("nyt.txt.sent.all.amr-semeval.parsed", 'r') as f1, open('conceptNet_edges_100000.txt', 'w') as f2:
#     lines0 = f1.readlines()
  
#     lines = lines0[:100000]
#     # print(lines[:5])

#     senid = -1
#     pid = 0
#     i = 0
#     # senid = 1277
#     # pid = 202
#     # i = 21016
#     w = ""

#     # tplen = len(lines) + 21016
#     tplen = len(lines)

#     while i < tplen:
#         if lines[i] == "\n":
#             i += 1
#             continue
#         print("i", i)
#         if lines[i][:7] == "# ::id ":
#             #split the running!
#             # senid = int(line[i].split()[-1]) - 1
#             senid += 1
#             i += 2
#             if senid not in d[pid] and senid in d[pid+1]:
#                 pid += 1
#                 node_by_p.append(curr_p_nodes)


#                 print("i, senid, pid", i, senid, pid)
#                 print("len(curr_p_nodes), curr_p_nodes", len(curr_p_nodes), curr_p_nodes)
#                 print("len(curr_p_edges), curr_p_edges", len(curr_p_edges), curr_p_edges[:5])

#                 new_curr_p_edges = filter_p_only(curr_p_edges, curr_p_nodes)
#                 edges_by_p.append(new_curr_p_edges)
#                 print("len(new_curr_p_edges), new_curr_p_edges", len(new_curr_p_edges), new_curr_p_edges[:5])
#                 print()
#                 ################################
#                 # for i in range(len(edges_by_p)):
#                 f2.write("p" + str(pid) + "\n")
#                 for x in new_curr_p_edges:
#                     f2.write(str(x))
#                     f2.write("\n")
#                 f2.write("\n")
#                 ################################

#                 curr_p_edges = []
#                 curr_p_nodes = []
#             continue

#         li = lines[i].split()
#         word = li[-1].strip()
#         word = word.replace(")","")
#     #     print("word", word)

#         if '"' in word:
#             word = word.replace('"','')

#         chs = word.split("-")
#         j = len(chs) - 1
#         while j >= 0:
#             tp = chs[j]

#             if not hasNumbers(tp):
#                 w = chs[j]

#                 print("w", w)            
#                 curr_p_edges.extend(findTuples(w))
#                 print()
#                 curr_p_nodes.append(w)
#             j -= 1
#         i += 1

