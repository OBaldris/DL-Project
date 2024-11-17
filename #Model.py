#DL Project - Model

import torch
import torch.nn as nn



#1. MULTI HEAD SELF ATTENTION CLASS
#we are not making a class but using pytorch function
#EXAMPLE
# Define model parameters
embed_size = 8  # Embedding size (d_model)
num_heads = 2   # Number of attention heads
seq_length = 5  # Sequence length (number of tokens)
batch_size = 3  # Batch size
# Initialize the multi-head attention layer
multihead_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
# Example input
x = torch.rand(batch_size, seq_length, embed_size)  # Input embeddings [batch_size, seq_length, embed_size]
# Apply self-attention
output, attention_weights = multihead_attention(x, x, x)


#2. ADDITIVE ATTENTION CLASS
#x is the hidden states from the multihead self attention layer
class AdditiveAttention(nn.Module):
    def __init__(self, embed_size):
        super(AdditiveAttention, self).__init__()
        self.attention = nn.Linear(embed_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        scores = self.attention(self.tanh(x))  # [batch_size, sequence_length, 1]
        weights = self.softmax(scores)         # [batch_size, sequence_length, 1]
        output = torch.sum(weights * x, dim=1)  # Weighted sum: [batch_size, embed_size]
        return output














#3. NEWS ENCODER
#1st layer: word embedding 
#2nd layer: multi head self attention
#3rd layer: additive attention (aggergated info from words to the news article)

import torch
import torch.nn as nn
import torch.nn.functional as F

#embed_size: 300
#heads: 16
#word_embedding_matrix: glove_vectors from load_glove function
class NewsEncoder(nn.Module):
    def __init__(self, embed_size, heads, word_embedding_matrix):
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word_embedding_matrix, freeze=False)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
        self.fc = nn.Linear(embed_size, embed_size)
        
    def forward(self, x):
        embedding = self.embedding(x)
        attn_output, _ = self.multi_head_attention(embedding, embedding, embedding)
        news_representation = torch.tanh(self.fc(attn_output.mean(dim=1)))
        return news_representation
