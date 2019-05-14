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

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import random
import sys

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
LR = 0.01           # learning rate

# show data
glove_model = api.load('glove-twitter-200')


nlp = spacy.load('en_core_web_lg')

sent_types = ["rand_sents", "fake_sents", "rev_sents"]

#exec(open("rnnclassifier6.py").read())
class LSTMClassifier(nn.Module):

    def __init__(self, hidden_dim=50, label_size=2, batch_size=32, use_gpu=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self.lstm = nn.LSTM(200, hidden_dim)
        self.hidden2label = nn.Sequential(nn.Linear(2*hidden_dim, 10), nn.Linear(10, label_size))
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, x,y):
        x = x.permute(1,0,2)
        y = y.permute(1,0,2)
        lstm_out1, self.hidden = self.lstm(x, self.hidden)
        lstm_out2, self.hidden = self.lstm(y, self.hidden)
        a = lstm_out1[-1]
        b = lstm_out2[-1]
        a = torch.cat((lstm_out1[-1], lstm_out2[-1]), dim=1)
        y  = self.hidden2label(a)
        return y

rnn = LSTMClassifier()

first_sents = pickle.load(open("first_sent.pcl", "rb"))
orig_sents = pickle.load(open("orig_sents.pcl", "rb"))
sent_type = sys.argv[1]
dataset = pickle.load(open(sent_type + ".pcl", "rb"))
seqs = []
print(len(first_sents))
for i_ in range(20000):
    print(i_)
    if random.uniform(0,1) < 0.5:
        nlp_perm_a = nlp(first_sents[i_] + "  " + orig_sents[i_])
        nlp_perm_b = nlp(first_sents[i_] + "  " + dataset[i_])
        label = 0
    else:
        nlp_perm_b = nlp(first_sents[i_] + "  " + orig_sents[i_])
        nlp_perm_a = nlp(first_sents[i_] + "  " + dataset[i_])
        label = 1
    seq_a = np.zeros((30,200))
    for (ind, word) in enumerate(list(nlp_perm_a)):
        try:
            seq_a[ind] = (glove_model[word.text.split("-")[0].lower()])
        except:
            pass
    
    seq_b = np.zeros((30,200))
    for (ind, word) in enumerate(list(nlp_perm_b)):
        try:
            seq_b[ind] = (glove_model[word.text.split("-")[0].lower()])
        except: pass
            #print(word.text.split("-")[0].lower())
    seqs.append((seq_a, seq_b, label))
random.shuffle(seqs)
pickle.dump(seqs, open(sent_type + "seq.pcl", "wb"))
#data[sent_type] = seqs

trainloader = DataLoader([(torch.from_numpy(i[0]).float(), torch.from_numpy(i[1]).float(), i[2]) for i in seqs[:15000]], batch_size=32,shuffle=True)
testloader = DataLoader([(torch.from_numpy(i[0]).float(), torch.from_numpy(i[1]).float(), i[2]) for i in seqs[15000:]], batch_size=32, shuffle=True)



optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
#loss_func = nn.BinaryCrossEntropy()
loss_func = nn.CrossEntropyLoss()
ten_loss = 0
for epoch in range(100):
    for (batch_idx, (x,y, label)) in enumerate(trainloader):
        print(batch_idx)
        if x.shape[0] != 32 or batch_idx > 100:
            continue
        label = label.long()
        prediction = rnn(x,y)# h_state)   # rnn output
        # !! next step is important !!        # repack the hidden state, break the connection from last iteration
        #print(prediction)
        loss = loss_func(prediction, label)
        ten_loss += loss.item()         # calculate loss
        optimizer.zero_grad()                   # clear gradients for this training step
        loss.backward(retain_graph=True)                         # backpropagation, compute gradients
        optimizer.step()  
                              # apply gradients
        if batch_idx % 20 == 19:
            print("Loss: " + str(ten_loss))
            print("TOT: " + str(len([i for i in range(32) if (label[i] == 1 and prediction[i][1] > prediction[i][0]) or (label[i] == 0 and prediction[i][0] > prediction[i][1])])))
            ten_loss = 0
            torch.save(rnn.state_dict(), "rnn" + sent_type + "6.pth")
    rnn.eval()
    num_tot = 0
    num_correct = 0
    for (batch_idx, (x,y,label)) in enumerate(testloader):
        if batch_idx > 100:
            continue
        prediction = rnn(x,y)
        num_correct += len([i for i in range(32) if (label[i] == 1 and prediction[i][1] > prediction[i][0]) or (label[i] == 0 and prediction[i][0] > prediction[i][1])])
        num_tot += 32
    open("tot" + sent_type + ".txt", "w+").write("Percent correct: " + str(num_tot/(num_correct + 0.0)))

#to run: python3 rnnclassifiersenttypes.py [rev_sents | fake_sents | rand_sents]
