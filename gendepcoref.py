import pickle
import networkx as nx
import numpy as np
import sparray
import spacy
nlp = spacy.load('en_core_web_lg')

import neuralcoref
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')

def cleanup(st):
	if "-" in st:
		st = st.split("-")
		new = []
		for word in st:
			cleaned = "".join([i for i in word if i.isalpha()])
			if len(cleaned) > 0:
				new.append(cleaned)
		if len(new) > 0:
			return "-".join(new)
		else:
			return "OOV"
	if any([i in st for i in "0123456789"]):
		return "num"
	else:
		return "".join([i for i in st if i.isalpha() or i == '-'])


with open("sent_counts.pcl", 'rb') as f:
    sents_par = pickle.load(f, encoding='latin1') 

lines = [i.strip("\n") for i in open("nyt.sent").readlines()][:10000]

def concat(xss):
	new = []
	for xs in xss:
		new.extend(xs)
	return new


dep_parses = []
coref_parses = []
nlp_rep = []
for (index, line) in enumerate(lines):
	print(index)
	a = nlp(line)
	json = a.to_json()
	dep_parses.append(str(json["tokens"]))
	if a._.has_coref:
		coref_parses.append("[" + ", ".join(["[" + ", ".join([str(i[-1].i) for i in j]) + "]" for j in a._.coref_clusters]) + "]")
	else:
		coref_parses.append("[]")
	nlp_rep.append(str([x for x in a]))

f = open("dep_parses.json", "w+")
f.write("\n".join([str(i) for i in dep_parses]))
f.close()


g = open("corefs.txt", "w+")
g.write("\n".join(coref_parses))
g.close()

h = open("nlprep.txt", "w+")
h.write("\n".join(nlp_rep))
h.close()

"""
for sent_graph in sents[1:]:
	nodes = []
	node_maps = {}
	lines = sent_graph.split("\n")[2:-1]
	G = nx.DiGraph()
	prev_node = lines[0].split(" ")[-1]
	if "-" in prev_node:
		prev_node = cleanup(prev_node)
	prev_node_x = lines[0].split("(x")[1].split(" ")[0]
	path = [prev_node]
	node_maps[prev_node_x] = prev_node
	nodes.append(prev_node)
	for line in lines[1:]:
		if line == "":
			continue
		indents = line.count("\t")
		path = path[:indents + 1]
		prev_node = path[-1]
		if len(line.split(" ")) > 2:
			try:
				cur_node = cleanup(line.split(" ")[-1])
				cur_node_x = line.split("(")[1].split(" ")[0][1:].split("(")[0]
				cur_edge = line[line.index(":") + 1:].split(" ")[0]
				G.add_edge(prev_node, cur_node, label=cur_edge)
				node_maps[cur_node_x] = cur_node
				nodes.append(cur_node)
				path.append(cur_node)
				edge_types.add(cur_edge)
			except:
				cur_edge = line[line.index(":") + 1:].split(" ")[0]
				cur_node = cleanup(line[line.index(":") + 1:].split(" ")[1])
				nodes.append(cur_node)
				path.append(cur_node)
				edge_types.add(cur_edge)
				G.add_edge(prev_node, cur_node, label=cur_edge)
		else:
			try:
				cur_node_x = line.split("(")[1][1:]
				G.add_edge(prev_node, node_maps[cur_node_x], label=cur_edge)
				path.append(node_maps[cur_node_x])
				edge_types.add(cur_edge)
			except:
				cur_edge = line[line.index(":") + 1:].split(" ")[0]
				cur_node = cleanup(line[line.index(":") + 1:].split(" ")[1])
				if cur_node in node_maps:
					G.add_edge(prev_node, node_maps[cur_node], label=cur_edge)
				else:
					G.add_edge(prev_node, cur_node, label=cur_edge)
					nodes.append(cur_node)
				path.append(cur_node)
				edge_types.add(cur_edge)
				
	sent_graphs.append(G)
	sent_nodes.append(nodes)

edge_types = list(edge_types)

graphs = []
k = 0
for par in sents_par:
	k += 1
	print(k)
	s = sents_par[par]
	if any(i >= len(sents) for i in s):
		break
	graph_sents = [sent_graphs[i] for i in s]
	graph_verts = [sent_nodes[i] for i in s]
	verts = []
	for i in range(len(graph_verts)):
		for j in range(len(graph_verts[i])):
			verts.append(graph_verts[i][j] + "-" + str(i))
	adj = np.zeros((len(verts), len(verts), len(edge_types) + 1))
	for i in range(len(graph_sents)):
		for (v1, v2, label) in graph_sents[i].edges.data("label"):
			label = edge_types.index(label)
			v1_ind = sum(map(len, graph_verts[:i])) + graph_verts[i].index(v1)
			v2_ind = sum(map(len, graph_verts[:i])) + graph_verts[i].index(v2)
			adj[v1_ind, v2_ind, label] = 1.0
	for i in range(len(graph_sents) - 1):
		adj[sum(map(len, graph_verts[:i])), sum(map(len, graph_verts[:i + 1])), len(edge_types)] = 1.0
	graphs.append((verts, sparray.sparray(adj.shape, adj)))

pickle.dump(graphs, open("graphs.pcl", "w+"))
"""
