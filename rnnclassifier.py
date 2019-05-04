"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
matplotlib
numpy
"""
from torch.utils.data import Dataset, DataLoader

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import spacy
import pickle
import itertools
import random

# torch.manual_seed(1)    # reproducible
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


# Hyper Parameters
TIME_STEP = 200      # rnn time step
INPUT_SIZE = 200      # rnn input size
LR = 0.02           # learning rate

# show data
nlp = spacy.load('en_core_web_lg')
with open("sent_counts.pcl", 'rb') as f:
    par = pickle.load(f, encoding='latin1')

seqs = []
text = open("nyt.sent.new").readlines()
for (par_ind, paragraph) in par.items():
    print(par_ind)
    a = list(itertools.permutations(range(4)))
    paragraph_sents = [text[i].strip("\n") for i in paragraph]
    for (perm_ind, permutation) in enumerate(random.sample(a, 6)):
        label = getScore(permutation)
        seq = []
        sentences_permed = [paragraph_sents[i] for i in permutation]
        nlp_perm = nlp(" ".join(sentences_permed))
        seq = np.zeros((200,200))
        for (ind, word) in enumerate(list(nlp_perm)):
            try:
                seq[ind] = (glove_model[word.text.split("-")[0]])
            except:
                pass
        seqs.append((np.array(seq), getScore(permutation)))

trainloader = DataLoader(([i[0] for i in seqs], [i[1] for i in seqs]), batch_size=32)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=32,     # rnn hidden unit
            num_layers=5,       # number of rnn layer
            batch_first=True,
            bidirectional=True   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)
        self.h_state = nn.Parameter()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        #outs = []    # save all predictions
        #for time_step in range(r_out.size(1)):    # calculate output for each time step
        #    outs.append(self.out(r_out[:, time_step, :]))
        #return torch.stack(outs, dim=1), h_state
        return self.sigmoid(self.out(r_out))
        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # outs = outs.view(-1, TIME_STEP, 1)
        # return outs, h_state
        
        # or even simpler, since nn.Linear can accept inputs of any dimension 
        # and returns outputs with same dimension except for the last
        # outs = self.out(r_out)
        # return outs

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()

#h_state = None      # for initial hidden state
"""
plt.figure(1, figsize=(12, 5))
plt.ion()           # continuously plot
"""
for (batch_idx, (x,y)) in trainloader:
    #start, end = step * np.pi, (step+1)*np.pi   # time range
    # use sin predicts cos
    #steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)  # float32 for converting torch FloatTensor
    #x_np = np.sin(steps)
    #y_np = np.cos(steps)

    prediction, h_state = rnn(x)# h_state)   # rnn output
    # !! next step is important !!        # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, y)         # calculate loss
    optimizer.zero_grad()                   # clear gradients for this training step
    loss.backward()                         # backpropagation, compute gradients
    optimizer.step()                        # apply gradients

    # plotting
    #plt.plot(steps, y_np.flatten(), 'r-')
    #plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    #plt.draw(); plt.pause(0.05)

#plt.ioff()
#plt.show()