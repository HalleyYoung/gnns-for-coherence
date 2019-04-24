"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/MichSchli/RelationPrediction
Difference compared to MichSchli/RelationPrediction
* report raw metrics instead of filtered metrics
"""

import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dgl.contrib.data import load_data
import pickle
from dgl import DGLGraph
import dgl



parser = argparse.ArgumentParser(description='RGCN')
parser.add_argument("--dropout", type=float, default=0.2,
        help="dropout probability")
parser.add_argument("--n-hidden", type=int, default=500,
        help="number of hidden units")
parser.add_argument("--gpu", type=int, default=-1,
        help="gpu")
parser.add_argument("--lr", type=float, default=1e-2,
        help="learning rate")
parser.add_argument("--n-bases", type=int, default=100,
        help="number of weight blocks for each relation")
parser.add_argument("--n-layers", type=int, default=2,
        help="number of propagation rounds")
parser.add_argument("--n-epochs", type=int, default=6000,
        help="number of minimum training epochs")
parser.add_argument("--eval-batch-size", type=int, default=500,
        help="batch size when evaluating")
parser.add_argument("--regularization", type=float, default=0.01,
        help="regularization weight")
parser.add_argument("--grad-norm", type=float, default=1.0,
        help="norm to clip gradient to")
parser.add_argument("--graph-batch-size", type=int, default=30000,
        help="number of edges to sample in each iteration")
parser.add_argument("--graph-split-size", type=float, default=0.5,
        help="portion of edges used as positive sample")
parser.add_argument("--negative-sample", type=int, default=10,
        help="number of negative samples per positive sample")
parser.add_argument("--evaluate-every", type=int, default=500,
        help="perform evaluation every n epochs")

args = parser.parse_args()



class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if self.bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr

class RGCNBasisLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNBasisLayer, self).__init__(in_feat, out_feat, bias, activation)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels,
                                                    self.num_bases))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))

    def propagate(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(
                                    self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def msg_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['type'] * self.in_feat + edges.src['id']
                return {'msg': embed.index_select(0, index) * edges.data['norm']}
        else:
            def msg_func(edges):
                w = weight.index_select(0, edges.data['type'])
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}

        g.update_all(msg_func, fn.sum(msg='msg', out='h'), None)

class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop,
                                             dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0

        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        # assuming in_feat and out_feat are both divisible by num_bases
        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(
                    -1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}

class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        for layer in self.layers:
            layer(g)
        return g.ndata.pop('h')

import utils

class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = self.embedding(node_id)

class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, self_loop=True, dropout=self.dropout)

class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g):
        return self.rgcn.forward(g)

    def evaluate(self, g):
        # get embedding and relation weight without grad
        embedding = self.forward(g)
        return embedding, self.w_relation

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        embedding = self.forward(g)
        score = self.calc_score(embedding, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embedding)
        return predict_loss + self.reg_param * reg_loss


with open("edge_types.pcl", 'rb') as f:
    edge_types_list = pickle.load(f, encoding='latin1') 

with open("edges_lists.pcl", "rb") as f:
    edge_data = pickle.load(f, encoding='latin1') 



edge_src = [[i[0] for i in edge_dat] for edge_dat in edge_data]
edge_dst = [[i[1] for i in edge_dat] for edge_dat in edge_data]
edge_types_ = [[i[2] for i in edge_dat] for edge_dat in edge_data]

num_nodes = [max(max(edge_src[i]), max(edge_dst[i])) + 1 for i in range(len(edge_src))]
num_rels = len(edge_types_list)

edge_types = [torch.from_numpy(np.array(edge_type)) for edge_type in edge_types_]
edge_norms = [torch.from_numpy(np.zeros(len(edge_s))).unsqueeze(1) for edge_s in edge_src]


###############################################################################
# Create graph and model
# ~~~~~~~~~~~~~~~~~~~~~~~

# configurations
n_hidden = 16 # number of hidden units
n_bases = 50 # use number of relations as number of bases
n_hidden_layers = 0 # use 1 input layer, 1 output layer, no hidden layer
n_epochs = 25 # epochs to train
lr = 0.01 # learning rate
l2norm = 0 # L2 norm coefficient

# create graph
gs = []

max_nodes = max(num_nodes)
for i in range(len(edge_src)):
    g = DGLGraph()
    g.add_nodes(max_nodes)
    g.add_edges(edge_src[i], edge_dst[i])
    g.edata.update({'rel_type': edge_types[i], 'norm': edge_norms[i]})
    gs.append(g)

g = dgl.batch(gs)

model = LinkPredict(max_nodes,
                    2,
                    num_rels,
                    num_bases=50,
                    num_hidden_layers=0,
                    dropout=0.2,
                    use_cuda=False,
                    reg_param=args.regularization)

## validation and testing triplets
#valid_data = torch.LongTensor(valid_data)
#test_data = torch.LongTensor(test_data)

# build test graph
"""
test_graph, test_rel, test_norm = utils.build_test_graph(
    num_nodes, num_rels, train_data)
test_deg = test_graph.in_degrees(
            range(test_graph.number_of_nodes())).float().view(-1,1)
test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
test_rel = torch.from_numpy(test_rel)
test_norm = torch.from_numpy(test_norm).view(-1, 1)
test_graph.ndata.update({'id': test_node_id, 'norm': test_norm})
test_graph.edata['type'] = test_rel

if use_cuda:
    model.cuda()

# build adj list and calculate degrees for sampling
"""
# optimizer
triplets = [(edge_src[i], edge_types_[i], edge_dst[i]) for i in range(len(gs))]
triplets = [[(triple[0][i], triple[1][i], triple[2][i]) for i in range(len(triple[0]))] for triple in triplets]
adj_lists = []
degrees_lists = []
for triple in triplets:
    adj_list, degrees = utils.get_adj_and_degrees(max_nodes, triple)
    adj_lists.append(adj_list)
    degrees_lists.append(degrees)


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

model_state_file = 'model_state.pth'
forward_time = []
backward_time = []

# training loop
print("start training...")

epoch = 0
best_mrr = 0


while epoch < 10000:
    model.train()
    epoch += 1

    # perform edge neighborhood sampling to generate training graph and data
    index = random.randint(250,450)
    print(index)
    print(len(triplets[index]))

    g, node_id, edge_type, node_norm, data, labels = \
        utils.generate_sampled_graph_and_labels(
            triplets[index], 1000, 0.5,
            num_rels, adj_lists[index], degrees_lists[index], 100)
    print("Done edge sampling")

    # set node/edge feature
    node_id = torch.from_numpy(node_id).view(-1, 1)
    edge_type = torch.from_numpy(edge_type)
    node_norm = torch.from_numpy(node_norm).view(-1, 1)
    data, labels = torch.from_numpy(data), torch.from_numpy(labels)
    deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': node_norm})
    g.edata['type'] = edge_type

    t0 = time.time()
    loss = model.get_loss(g, data, labels)
    t1 = time.time()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
    optimizer.step()
    t2 = time.time()

    forward_time.append(t1 - t0)
    backward_time.append(t2 - t1)
    print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
          format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))

    optimizer.zero_grad()

print("training done")
print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))