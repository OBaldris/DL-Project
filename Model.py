from Data_loader import *

print(input_data_train.columns)
print(input_data_train.head())

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

#---------- News Encoder -----------
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
    
### Test news encoder
word_embedding_matrix = glove_vectors
# Instantiate the model
encoder = NewsEncoder(embed_size=300, heads=5, word_embedding_matrix=word_embedding_matrix)

# Random input
x = input_data_train.loc[5, 'candidate_news']

tensor_list = [torch.tensor(sublist) for sublist in x]
x = pad_sequence(tensor_list, batch_first=True, padding_value=0)

# Forward pass
output = encoder(x)
print("output size", output.shape)
