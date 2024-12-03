from data_loader import *
from model import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# print("\nbrowsed news: ", browsed_news_train.shape
#       , "\ncandidate news: ", candidate_news_train.shape
#       , "\nclicked news: ", clicked_news_train.shape)


### Test full model
print('\n -----COMPLETE MODEL------') 

batch_size = 3

model_final = NRMS(embed_size=300, heads=15, word_embedding_matrix=glove_vectors, attention_dim=200)

browsed_news_batch = browsed_news_train[:batch_size, :, :] #[Batch size, all news, 26 words]
candidate_news_batch = candidate_news_train[:batch_size, :, :] #[Batch size, all news, 26 words]
clicked_news_batch = clicked_news_train[:batch_size, :] #[Batch size, all news]

# Forward pass for the entire batch
click = model_final(browsed_news_batch, candidate_news_batch)

print('\n -----CLICK------') 
print(click)

