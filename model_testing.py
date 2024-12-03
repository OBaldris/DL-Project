
from data_loader import *
from model import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

print("\nbrowsed news: ", browsed_news_train.shape
      , "\ncandidate news: ", candidate_news_train.shape
      , "\nclicked news: ", clicked_news_train.shape)


batch_size = 16

### Test news encoder
print("\n ------NEWS ENCODER------")

word_embedding_matrix = glove_vectors  # Assume glove_vectors are loaded and of correct size
attention_dim = 200
# Instantiate the model
news_encoder = NewsEncoder(embed_size=300, heads=15, word_embedding_matrix=word_embedding_matrix, attention_dim=attention_dim)

# Random input

x = browsed_news_train[:batch_size, 1, :] #[Batch size, 1 news, 26 words]

output = news_encoder(x)

print("input shape:", x.shape)
print("output shape:", output.shape) # News encoder works fine




### Test user encoder
batch_size = 3

print('\n ------USER ENCODER------')

user_encoder = UserEncoder(embed_size=300, heads=15, attention_dim=200)

x = browsed_news_train[:batch_size, :, :] #[Batch size, all news, 26 words]

e = [news_encoder(news) for news in x] # Apply the news encoder to each news article
e = torch.stack(e, dim=0)


output = user_encoder(e)

print("input shape:", e.shape)
print("output shape:", output.shape) # User encoder works fine

### Test full model
print('\n -----COMPLETE MODEL------') 

batch_size = 3

model_final = NRMS(embed_size=300, heads=15, word_embedding_matrix=glove_vectors, attention_dim=200)

browsed_news_batch = browsed_news_train[:batch_size, :, :] #[Batch size, all news, 26 words]
candidate_news_batch = candidate_news_train[:batch_size, :, :] #[Batch size, all news, 26 words]
clicked_news_batch = clicked_news_train[:batch_size, :] #[Batch size, all news]

# Forward pass for the entire batch
click = model_final(browsed_news_batch, candidate_news_batch)

print(f"\nInput shape: Browsed = {browsed_news_batch.shape}, Candidate = {candidate_news_batch.shape}")
print(f"Output shape: Click = {click.shape}") # Full model works fine

print("\nSum of probabilities:", torch.sum(click, dim=1)) # Probabilities sum to 1
print(f"\nClicked indices: True = {torch.argmax(click, dim=1)}, Predicted = {torch.argmax(clicked_news_batch, dim=1)}") # Predicted indices match true indices