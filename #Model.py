#DL Project - Model

import sys
import os

print("Current working directory:", os.getcwd())
print("Python sys.path:", sys.path)

sys.path.append(os.getcwd())

from Data_loader import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


#1. MULTI HEAD SELF ATTENTION CLASS
#we are not making a class but using pytorch function
#EXAMPLE
#multihead_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
#output, attention_weights = multihead_attention(x, x, x)


#2. ADDITIVE ATTENTION CLASS
#initialization of Vw and qw is glorot uniform as it is in the official code
#initialization of vw is zeros as it is in the official code

#embed_size: dim of the input embedding (300 from paper)
#attention_dim: dim of the attention space (200 from paper)

#h:torch.Tensor, input tensor of shape [batch_size, seq_length, embed_size] from the multihead attention, represents each word
#forward returns: torch.Tensor, Context vector of shape [batch_size, embed_size].

class AdditiveAttention(nn.Module):
    def __init__(self, embed_size, attention_dim):
        
        super(AdditiveAttention, self).__init__()
        self.V_w = nn.Linear(embed_size, attention_dim)  #weight and bias (vw)
        self.q_w = nn.Parameter(torch.zeros(attention_dim))

        #initializations
        nn.init.xavier_uniform_(self.V_w.weight) 
        nn.init.zeros_(self.V_w.bias)
        nn.init.xavier_uniform_(self.q_w.unsqueeze(0)) 

    def forward(self, h):
        
        #projection into attention space and tanh after
        projection=self.V_w(h) + self.v_w  #[batch_size, seq_length, attention_dim]
        tanh_output=torch.tanh(projection)  #[batch_size, seq_length, attention_dim]

        #attention scores and softmax to normalize scores and get weights
        attention_scores=torch.matmul(tanh_output, self.q_w)  #[batch_size, seq_length]
        attention_weights=F.softmax(attention_scores, dim=1)  #[batch_size, seq_length]

        #weighted sum of h
        attention_weights = attention_weights.unsqueeze(-1)  #[batch_size, seq_length, 1]
        r_vector=torch.sum(attention_weights * h, dim=1)  #[batch_size, embed_size]

        return r_vector



#3. NEWS ENCODER
#1st layer: word embedding 
#2nd layer: multi head self attention
#3rd layer: additive attention (aggergated info from words to the news article)

#embed_size: 300
#heads: 16
#word_embedding_matrix: glove_vectors from load_glove function
#attention_dim: 200
class NewsEncoder(nn.Module):
    def __init__(self, embed_size, heads, word_embedding_matrix,attention_dim):
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word_embedding_matrix, freeze=False)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
        self.additive_attention = AdditiveAttention(embed_size, attention_dim)
        
    def forward(self, x):
        embedding = self.embedding(x)
        attn_output, _ = self.multi_head_attention(embedding, embedding, embedding)
        news_representation = self.additive_attention(attn_output)
        return news_representation
    


### Test news encoder
word_embedding_matrix = glove_vectors
attention_dim = 200
# Instantiate the model
encoder = NewsEncoder(embed_size=300, heads=5, word_embedding_matrix=word_embedding_matrix,attention_dim=200)

# Random input
x = input_data_train.loc[5, 'candidate_news']

tensor_list = [torch.tensor(sublist) for sublist in x]
x = pad_sequence(tensor_list, batch_first=True, padding_value=0)

# Forward pass
output = encoder(x)
print("output size", output.shape)

print("output", output)
    


