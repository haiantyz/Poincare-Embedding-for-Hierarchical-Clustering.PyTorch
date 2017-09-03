import numpy as np
import torch
import torch.nn as nn
from math_utils import *
from data import get_data

def train():
    # hyponym lambda functions
    hypo = lambda s: s.hyponyms()

    print('There are total of %i nouns in wordnet.' % len(all_noun))

    num_synsets = len(all_list)
    shape = [num_synsets, num_synsets]
    
    # load index, which are links [u,v] if u is a hyponym of v.
    try:
        links = np.load('links.npy')
        noun_hypernym_list = np.load('noun_hypernym_list.npy')
    except IOError:
        links, noun_hypernym_list = get_data(noun_list)
        
    # shuffling always help training
    np.random.shuffle(links)

    values = np.full((len(links)), 1)
    #transitive_closure = 
    # input = set of vocabs one hot vector
    # obtain the full transitive closure matrix
    
    # split into train, val, test by holding out some links    
    if linkPrediction:
        test_split = int(0.9 * num_synsets)
        val_split = int(0.8 * num_synsets)
        train_list = links[:val_split]
        val_list = links[val_split:test_split]
        test_list = links[test_split:]
        
    # init embed layer with  U(-0.001, 0.001)
    
    ndims = 5
    embedding = nn.Embedding(num_synsets, ndims)
    
    
    # inverse metric tensor
   
    # dtheta formula

    # loss function

    # dd 

    # update
   for p in model.parameters():
         p.data.add_(-lr, metric_tensor * p.grad.data)


