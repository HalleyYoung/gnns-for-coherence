import spacy
import allennlp
from allennlp.predictors.predictor import Predictor
import pickle
import itertools
import numpy as np
import random
from nltk.stem import WordNetLemmatizer 
import gensim.downloader as api


glove_model = api.load('glove-twitter-200')



#initializing nlp tools
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
lemmatizer = WordNetLemmatizer() 

concept_edges_ = open("conceptNet_nyt_edges_swfree.txt").readlines()#pickle.load(open("concept_edges.pcl"))
concept_edges = []
for line in concept_edges_:
	if line[0] == "p" and "\t" in line[2:6]:
		concept_edges.append([])
	else:
		try:
			concept_edges[-1].append(eval(line.strip("\n")))
		except:
			pass
nlp = spacy.load('en_core_web_lg')

import neuralcoref
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')

edge_types = []
graphs = []

def glove_sim(x,y):
	try:
		return glove_model.similarity(x,y)
	except:
		return -1
base_lemmas = {}


#get conceptnet edges
def getEdges(a, b, paragraph_ind):
	global base_lemmas
	if a in base_lemmas:
		a_lem = base_lemmas[a]
	else:
		a_lem = lemmatizer.lemmatize(a)
		base_lemmas[a] = a_lem
	if b in base_lemmas:
		b_lem = base_lemmas[b]
	else:
		b_lem = lemmatizer.lemmatize(b)
		base_lemmas[b] = b_lem
	#wn_edges = getWordNet(a_lem, b_lem)
	current_edges = []#[(i, "forward") for i in concept_edges[paragraph_ind].setdefault((unicode(a_lem), unicode(b_lem)), [])] + [(i, "backward") for i in concept_edges[paragraph_ind].setdefault((unicode(b_lem), unicode(a_lem)), [])]
	for (r,x,y) in concept_edges[paragraph_ind]:
		try:
			if x.split("_")[-1] == a_lem and y.split("_")[-1] == b_lem:
				current_edges.append((r, "forward"))
			elif x.split("_")[-1] == b_lem and y.split("_")[-1] == a_lem:
				current_edges.append((r, "backward"))
		except:
			pass
	return current_edges



with open("sent_counts.pcl", 'rb') as f:
	par = pickle.load(f, encoding='latin1')

text = open("nyt.sent.new").readlines()

#get approximate gold-standard coherence score
def getScore(perm):
	score = 0
	for i in range(len(perm)):
		for j in range(i + 1, len(perm)):
			if perm[i] < perm[j]:
				score += 1
	for i in range(1, len(perm)):
		if perm[i - 1] == perm[i] - 1:
			score += 2
	return score


graphs = []
for (par_ind, paragraph) in par.items():
	print(par_ind)
	a = 0#andom.randint(1, len(paragraph) - 2)
	paragraph_sents = [text[i].strip("\n") for i in paragraph]

	edges = set([])
	if random.uniform(0,1) < 0.5:
		sentences_permed = [paragraph_sents[i] for i in [a, a + 1]] #some permutation of first 4 sentences
		label = 0 #gold standard score
	else:
		sentences_permed = [paragraph_sents[i] for i in [a + 1, a]] #some permutation of first 4 sentences
		label = 1
	edges = set([])
	nlp_perm = nlp(" ".join(sentences_permed))
	toks = list(nlp_perm) #words

	#get dependency parse edges
	dep = nlp_perm.to_json()["tokens"]
	nodes = {}
	for (ind, val) in enumerate(dep):
		word = nlp_perm.text[val["start"]:val["end"]]
		nodes[val["id"]] = word + "-" + str(ind)
	for (ind, val) in enumerate(dep):
		word = nlp_perm.text[val["start"]:val["end"]]
		edges.add((nodes[val["id"]], nodes[val["head"]], val["dep"]))

	#get coref edges
	if nlp_perm._.has_coref:
		clusters = nlp_perm._.coref_clusters
		for coref_group in clusters:
			coref_words = [nodes[k[-1].i] for k in coref_group]
			for (x,y) in itertools.product(coref_words, coref_words):
				if x != y:
					edges.add((x,y,"coref"))
	#get consecutive word edges
	for i in range(1, len(toks)):
		edges.add((nodes[toks[i-1].i], nodes[toks[i].i], "consec_word"))
		
		#if str(toks[i]) == "." and i < len(toks) - 1:
		#	edges.add((nodes[toks[i - 1].i], nodes[toks[i + 1].i], "consec_sent"))
	#get consecutive sentence edges
	sents = list(nlp_perm.sents)	
	for i in range(len(sents) - 1):
		toks_in_i = [nodes[k.i] for k in list(sents[i])]
		toks_in_j = [nodes[k.i] for k in list(sents[i + 1])]
		prod = list(itertools.product(toks_in_i, toks_in_j))
		for (i,j) in prod:
			edges.add((i,j,"consec_sent"))

	#get srl edges
	"""
	for sentence in nlp_perm.sents:
		a = predictor.predict(sentence=sentence.text)
		verbs = a["verbs"]
		for verb_ in verbs:
			description = verb_["description"]
			opens = [i for i in range(len(description)) if description[i] == "["]
			closes = [i for i in range(len(description)) if description[i] == "]"]
			nodes_args = []
			for i in range(len(opens)):
				try:
					[arg_name, arg_val] = description[opens[i] + 1:closes[i]].split(": ")
					root = list(nlp(arg_val).sents)[0].root
					nodes_val = [j for j in nodes.values() if j.split("-")[0] == str(root)]
					nodes_args.append((arg_name, nodes_val))
				except: pass
					
			prod = list(itertools.product(nodes_args, nodes_args))
			for ((arg1, ns1), (arg2, ns2)) in prod:
				if arg1 != arg2:
					for n1 in ns1:
						for n2 in ns2:
							edges.add((n1, n2, arg1 + "-" + arg2))
	"""

	#get conceptnet/lemma edges
	for (nodeA, nodeB) in list(itertools.product(nodes.values(), nodes.values())):
		x_bas = nodeA.split("-")[0]
		y_bas = nodeB.split("-")[0]
		if x_bas != y_bas and len(x_bas) > 3 and len(y_bas) > 3:
			for (edge_type, direc) in getEdges(x_bas, y_bas, par_ind):
				if edge_type == "forward":
					edges.add((nodeA,nodeB,edge_type))
				else:
					edges.add((nodeB,nodeA,edge_type))

			if glove_sim(x_bas, y_bas) > 0.6:
				edges.add((nodeA,nodeB,"glove-sim"))
				edges.add((nodeB,nodeA,"glove-sim"))

		if lemmatizer.lemmatize(x_bas) == lemmatizer.lemmatize(y_bas):
			edges.add((nodeA,nodeB,"same-word"))
			edges.add((nodeB,nodeA,"same-word"))

	nodes = list(nodes.values())

	#get vector of nodes
	nodes_mat = np.ones((len(nodes), 200))

	for (ind_node, node_) in enumerate(nodes):
		try:
			nodes_mat[ind_node] = glove_model[base_lemmas[node_.split("-")[0].lower()]]
		except:
			try:
				nodes_mat[ind_node] = glove_model[node_.split("-")[0].lower()]
			except:
				try:
					x = int(node_.split("-")[0].lower())
					nodes_mat[ind_node] = glove_model["year"]
				except:
					pass

	print("tot edges " + str(len(edges)) + " tot nodes " + str(len(nodes)))

	#get vector of edges
	for (x,y,label_) in edges:
		if label_ not in edge_types:
			edge_types.append(label_)
	edge_mat = np.zeros((len(edges), 3))

	for (edge_ind,  (x,y,label_)) in enumerate(edges):
			edge_mat[edge_ind] = (nodes.index(x),nodes.index(y),edge_types.index(label_))
	graphs.append((nodes_mat, edge_mat, label))
	
	if len(graphs) < 10000:
		#pickle.dump(edge_types, open("edge_types2sent.pcl", "wb"))
		pickle.dump(graphs, open("graphs2sentnosrl.pcl", "wb"))
	else:
		#pickle.dump(edge_types[10000:20000], open("edge_types2sent2.pcl", "wb"))
		pickle.dump(graphs[10000:20000], open("graphs2sentnosrl.pcl", "wb"))