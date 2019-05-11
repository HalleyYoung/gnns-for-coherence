"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
matplotlib
numpy
"""
from torch.utils.data import Dataset, DataLoader
import gensim.downloader as api

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
TIME_STEP = 100      # rnn time step
INPUT_SIZE = 200      # rnn input size
LR = 0.04           # learning rate

# show data
glove_model = api.load('glove-twitter-200')


nlp = spacy.load('en_core_web_lg')
with open("sent_counts.pcl", 'rb') as f:
    par = pickle.load(f, encoding='latin1')

seqs = []
text = open("nyt.sent.new").readlines()
for (par_ind, paragraph) in list(par.items()):
    print(par_ind)
    a = random.randint(1, len(paragraph) - 2)
    b = a + 1#random.randint(a + 1, len(paragraph))
    paragraph_sents = [text[i].strip("\n") for i in paragraph]

    nlp_perm_a = nlp(paragraph_sents[a])
    seq_a = np.zeros((70,200))
    for (ind, word) in enumerate(list(nlp_perm_a)):
        try:
            seq_a[ind] = (glove_model[word.text.split("-")[0]])
        except:
            pass
    nlp_perm_b = nlp(paragraph_sents[b])
    seq_b = np.zeros((70,200))
    for (ind, word) in enumerate(list(nlp_perm_b)):
        try:
            seq_b[ind] = (glove_model[word.text.split("-")[0]])
        except:
            pass
    if random.uniform(0,1) < 0.5:
        seqs.append((seq_a,seq_b,0))
    else:
        seqs.append((seq_b,seq_a,1))

random.shuffle(seqs)




#seqs_ = [(torch.from_numpy(i[0]).float(), i[1]) for i in random.sample(seqs,30000)]
#for i in range(12):
#    pickle.dump(seqs[1000*i:1000*i+1000], open("rnnseqs" + str(i) + ".pcl", "wb"))

trainloader = DataLoader([(torch.from_numpy(i[0]).float(), torch.from_numpy(i[1]).float(), i[2]) for i in seqs], batch_size=32)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn1 = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=50,     # rnn hidden unit
            num_layers=5,       # number of rnn layer
            batch_first=True,
            bidirectional=True   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.rnn2 = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=50,     # rnn hidden unit
            num_layers=5,       # number of rnn layer
            batch_first=True,
            bidirectional=True   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out1 = nn.Linear(1000, 50)
        self.out2 = nn.Linear(50, 2)
        self.h_0 = nn.Parameter(torch.rand(5*2, 32, 50))
        self.c_0 = nn.Parameter(torch.rand(5*2, 32, 50))
        nn.init.xavier_uniform_(self.h_0)
        nn.init.xavier_uniform_(self.c_0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)

        r_out1, (h_out1, c_out1) = self.rnn1(x, (self.h_0, self.c_0))
        r_out2, (h_out2, c_out2) = self.rnn2(x, (self.h_0, self.c_0))

        h_out1 = h_out1.permute(1,0,2).contiguous().view(32,-1)
        h_out2 = h_out2.permute(1,0,2).contiguous().view(32,-1)
        h_out = torch.stack((h_out1, h_out2), dim=1).view(32,-1)
        #, 1)
        #print(h_out.shape)
        #h_out = h_out.view(32, -1)


        #outs = []    # save all predictions   
        #for time_step in range(r_out.size(1)):    # calculate output for each time step
        #    outs.append(self.out(r_out[:, time_step, :]))
        #return torch.stack(outs, dim=1), h_state
        return self.sigmoid(self.out2(self.out1(h_out))).view((32,2))
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
#loss_func = nn.BinaryCrossEntropy()
loss_func = nn.CrossEntropyLoss()
#h_state = None      # for initial hidden state
"""
plt.figure(1, figsize=(12, 5))
plt.ion()           # continuously plot
"""
ten_loss = 0
for epoch in range(10):
    for (batch_idx, (x,y,label)) in enumerate(trainloader):
        if x.shape[0] != 32:
            continue
        #start, end = step * np.pi, (step+1)*np.pi   # time range
        # use sin predicts cos
        #steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)  # float32 for converting torch FloatTensor
        #x_np = np.sin(steps)
        #y_np = np.cos(steps)
        #label = label.float()
        prediction = rnn(x,y)# h_state)   # rnn output
        # !! next step is important !!        # repack the hidden state, break the connection from last iteration

        loss = loss_func(prediction, label)
        ten_loss += loss.item()         # calculate loss
        optimizer.zero_grad()                   # clear gradients for this training step
        loss.backward()                         # backpropagation, compute gradients
        optimizer.step()  
                              # apply gradients
        if batch_idx % 20 == 19:
            print("Loss: " + str(ten_loss))
            ten_loss = 0
            torch.save(rnn.state_dict(), "rnn.pth")


    # plotting
    #plt.plot(steps, y_np.flatten(), 'r-')
    #plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    #plt.draw(); plt.pause(0.05)

#plt.ioff()
#plt.show()
