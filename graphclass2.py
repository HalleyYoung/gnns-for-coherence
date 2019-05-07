
import networkx as nx
import pickle

###############################################################################
# Form a graph mini-batch
# -----------------------
# To train neural networks more efficiently, a common practice is to **batch**
# multiple samples together to form a mini-batch. Batching fixed-shaped tensor
# inputs is quite easy (for example, batching two images of size :math:`28\times 28`
# gives a tensor of shape :math:`2\times 28\times 28`). By contrast, batching graph inputs
# has two challenges:
#
# * Graphs are sparse.
# * Graphs can have various length (e.g. number of nodes and edges).
#
# To address this, DGL provides a :func:`dgl.batch` API. It leverages the trick that
# a batch of graphs can be viewed as a large graph that have many disjoint
# connected components. Below is a visualization that gives the general idea:
#
# .. image:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/batch/batch.png
#     :width: 400pt
#     :align: center
#
# We define the following ``collate`` function to form a mini-batch from a given
# list of graph and label pairs.

import dgl

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

###############################################################################
# The return type of :func:`dgl.batch` is still a graph (similar to the fact that
# a batch of tensors is still a tensor). This means that any code that works
# for one graph immediately works for a batch of graphs. More importantly,
# since DGL processes messages on all nodes and edges in parallel, this greatly
# improves efficiency.
#
# Graph Classifier
# ----------------
# The graph classification can be proceeded as follows:
#
# .. image:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/batch/graph_classifier.png
#
# From a batch of graphs, we first perform message passing/graph convolution
# for nodes to "communicate" with others. After message passing, we compute a
# tensor for graph representation from node (and edge) attributes. This step may
# be called "readout/aggregation" interchangeably. Finally, the graph
# representations can be fed into a classifier :math:`g` to predict the graph labels.
#
# Graph Convolution
# -----------------
# Our graph convolution operation is basically the same as that for GCN (checkout our 
# `tutorial <https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html>`_). The only difference is
# that we replace :math:`h_{v}^{(l+1)} = \text{ReLU}\left(b^{(l)}+\sum_{u\in\mathcal{N}(v)}h_{u}^{(l)}W^{(l)}\right)` by
# :math:`h_{v}^{(l+1)} = \text{ReLU}\left(b^{(l)}+\frac{1}{|\mathcal{N}(v)|}\sum_{u\in\mathcal{N}(v)}h_{u}^{(l)}W^{(l)}\right)`.
# The replacement of summation by average is to balance nodes with different
# degrees, which gives a better performance for this experiment.
#
# Note that the self edges added in the dataset initialization allows us to
# include the original node feature :math:`h_{v}^{(l)}` when taking the average.

import dgl.function as fn
import torch
import torch.nn as nn




# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation, is_input_layer = False):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(num_rels, in_feats, out_feats)
        self.in_feat = 200
        self.out_feats = 50
        self.activation = activation
        self.weight = nn.Parameter(torch.Tensor(num_rels, 50,
                                                50)) 
        self.fc1 = nn.Linear(200, 50)
        self.bias = nn.Parameter(torch.ones(out_feats))
        #nn.init.xavier_uniform_(self.bias)
        nn.init.xavier_uniform_(self.weight,gain=nn.init.calculate_gain('relu'))


        def message_func(edges):
                w = self.weight[edges.data['rel_type']]
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                #msg = msg * edges.data['norm']
                return {'m': msg}

        self.message_func = message_func
        def apply_func(nodes):
            h = nodes.data['h']
            h = h + self.bias
            h = self.activation(h)
            return {'h': h}
        self.apply_func = apply_func

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        for i in range(3):
            g.send(g.edges(), self.apply_mod.message_func)
            g.recv(g.nodes(), reduce, self.apply_mod.apply_func)
        return g.ndata.pop('h')

###############################################################################
# Readout and Classification
# --------------------------
# For this demonstration, we consider initial node features to be their degrees.
# After two rounds of graph convolution, we perform a graph readout by averaging
# over all node features for each graph in the batch
#
# .. math::
#
#    h_g=\frac{1}{|\mathcal{V}|}\sum_{v\in\mathcal{V}}h_{v}
#
# In DGL, :func:`dgl.mean_nodes` handles this task for a batch of
# graphs with variable size. We then feed our graph representations into a
# classifier with one linear layer to obtain pre-softmax logits.

import torch.nn.functional as F


class Regressor(nn.Module):
    def __init__(self, in_feats=200, hidden_dim=50):
        super(Regressor, self).__init__()
        self.gcn1 = GCN(hidden_dim, hidden_dim, F.relu)
        self.gcn2 = GCN(hidden_dim, hidden_dim, F.relu)
        self.classify1 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.classify2 = nn.Linear(int(hidden_dim/2), 1)
        self.sigmoid = nn.Sigmoid()
        self.reduce_dim = nn.Linear(in_feats, hidden_dim)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h = self.reduce_dim(g.ndata["x"])
        h = self.gcn1(g,h)
        h = self.gcn2(g,h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.sigmoid(self.classify2(self.classify1((hg)))).view(32)

###############################################################################
# Setup and Training
# ------------------
# We create a synthetic dataset of :math:`400` graphs with :math:`10` ~
# :math:`20` nodes. :math:`320` graphs constitute a training set and
# :math:`80` graphs constitute a test set.

import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
# Create training and test sets.
#trainset = MiniGCDataset(320, 10, 20)
#testset = MiniGCDataset(80, 10, 20)
num_rels = 0

dataset = pickle.load(open("graphs2.pcl", "rb"))
random.shuffle(dataset)
trainset = []
for (nodes, edges, label) in dataset[:1000]:
    graph = dgl.DGLGraph()
    graph.add_nodes(len(nodes))
    edges_ = range(len(edges))#random.sample(range(len(edges)), min(500, len(edges)))
    graph.add_edges([edges[i][0] for i in edges_], [edges[i][1] for i in edges_])
    #for k in range(len(edges)):
    #    graph.edges[k].data["rel_type"] = torch.from_numpy(np.ones((1))*int(edges[k][2]))
    graph.ndata["x"] = torch.from_numpy(nodes).float()
    graph.edata["rel_type"] = torch.from_numpy(np.array([int(edges[i][2]) for i in edges_])).long()
    trainset.append((graph, label))
    if num_rels < max([edges[i][2] for i in edges_]):
        num_rels = int(max([edges[i][2] for i in edges_]))
# Use PyTorch's DataLoader and the collate function
# defined before.
data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
                         collate_fn=collate)

# Create model
model = Regressor(200, 50)
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

epoch_losses = []
for epoch in range(80):
    epoch_loss = 0
    for iter, (bg, label) in enumerate(data_loader):
        print("in iter")
        prediction = model(bg)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        print(loss.item())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)

###############################################################################
# The learning curve of a run is presented below:

###############################################################################
# The trained model is evaluated on the test set created. Note that for deployment
# of the tutorial, we restrict our running time and you are likely to get a higher
# accuracy (:math:`80` % ~ :math:`90` %) than the ones printed below.
