import networkx
import pickle
import itertools
import gensim
import gensim.downloader as api

glove_model = api.load('glove-twitter-200')

corefs = open("corefs.txt").readlines()
par = pickle.load(open("sent_counts.pcl"))
dep_lines = open("dep_parses.json").readlines()
text = open("nyt.sent").readlines()
frameslines = open("predicted-args.conll").readlines()
single_frames = [{}]
for line in frameslines:
	try:
		a = line.split("\t")
		single_frames[-1][a[1].lower()] = (a[-2], a[-1])
	except:
		single_frames.append({})

frames = [[single_frames[0]]]
for i in range(1, len(single_frames)):
	if single_frames[i].keys() == single_frames[i - 1].keys():
		frames[-1].append(single_frames[i])
	else:
		frames.append([single_frames[i]])





sent_edges = []
edge_list = set([])
for i in range(4400):
	sent = text[i]
	dep = eval(dep_lines[i].strip("\n"))
	frame = frames[i]
	nodes = {}
	edges = set([])
	if len(corefs[i]) > 1:
		coref = eval(corefs[i])
	else:
		coref = []

	all_frame_nodes = []
	
	for (ind, f) in enumerate(frame):
		frame_inst_nodes = set([])
		for word in f:
			if len(f[word][0]) > 2:
				frame_inst_nodes.add(f[word][0] + "-" + str(ind))
			if len(f[word][1]) > 2:
				frame_inst_nodes.add(f[word][1] + "-" + str(ind))
			if len(f[word][0]) > 2 and len(f[word][1]) > 2:
				edges.add((f[word][0] + "-" + str(ind), f[word][1] + "-" + str(ind), "framenet-pair"))
		for (x,y) in list(itertools.product(frame_inst_nodes, frame_inst_nodes)):
			if x != y:
				edges.add((x,y,"same-frame"))
			

	for (ind, val) in enumerate(dep):
		word = sent[val["start"]:val["end"]]
		nodes[val["id"]] = word + "-" + str(ind) + "-" + str(i)

	start_word = nodes[0]
	edges.add((start_word[0], start_word[0], "par-start"))
	for (ind, val) in enumerate(dep):
		word = sent[val["start"]:val["end"]] 
		head = val["head"]
		#edges.add((head, word, val["dep"]))
		edges.add((nodes[val["id"]], nodes[val["head"]], val["dep"]))
		if len(word) > 3:
			for (ind, f) in enumerate(frame):
				if word in f:
					if len(f[word][0]) > 2:
						edges.add((nodes[val["id"]], f[word][0] + "-" + str(ind), "has-frame0"))
					if len(f[word][1]) > 2:
						edges.add((nodes[val["id"]], f[word][0] + "-" + str(ind), "has-frame1"))
	for k in range(1, len(dep)):
		edges.add((nodes[k - 1], nodes[k], "consec"))

	for coref_group in coref:
		coref_words = [nodes[k] for k in coref_group]
		for (x,y) in itertools.product(coref_words, coref_words):
			if x != y:
				edges.add((x,y,"coref"))
	sent_edges.append(edges)

graphs_by_par = []
for para in par:
	if max(par[para]) < 4400:
		graphs_by_par.append(sent_edges[i] for i in par[para])

for paragraph in graphs_by_par:
	par_edges = set([])
	for (edges, ind) in enumerate(paragraph):
		for (x,y,label) in edges:
			if label != "par-start":
				par_edges.add((x + "-" + str(ind), y + "-" + str(ind), label))
	starting_edges = [filter(lambda i: i[2] == "par-start", edges)[0] for edges in paragraph]
	for i in range(1, len(starting_edges)):
		par_edges.add(starting_edges[i - 1] + "-" + str(i - 1), starting_edges[i] + "-" + str(i), "consec-sent")
	nodes = set([i[0] for i in par_edges]).union(set([i[1] for i in par_edges]))
	prod = itertools.product(nodes, nodes):
	for (x,y) in prod:
		x_bas = x.split("-")[0]
		y_bas = x.split("-")[0]
		for edge in getEdges(x_bas, y_bas):
			par_edges.add(x,y,edge)
		if x_bas == y_bas:
			par_edges.add(x,y,"same-word")
		if glove_model.similarity(x_bas, y_bas) > 0.65:
			par_edges.add(x,y,"cosine-sim")







