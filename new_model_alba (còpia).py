
from Data_loader import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence



class AdditiveAttention(nn.Module):
    def __init__(self, embed_size, attention_dim):
        super(AdditiveAttention, self).__init__()
        self.V_w = nn.Linear(embed_size, attention_dim)  # weight and bias (V_w)
        self.q_w = nn.Parameter(torch.zeros(attention_dim))

        # Initializations
        nn.init.xavier_uniform_(self.V_w.weight)
        nn.init.zeros_(self.V_w.bias)
        nn.init.xavier_uniform_(self.q_w.unsqueeze(0))

    def forward(self, h):
        # Projection into attention space and tanh after
        projection = self.V_w(h)  # [batch_size, M, attention_dim]  - OK
        tanh_output = torch.tanh(projection)  # [batch_size, M, attention_dim]  - OK

        # Attention scores and softmax to normalize scores and get weights
        attention_scores = torch.matmul(tanh_output, self.q_w)  # [batch_size, M] - OK
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, M] - OK
        
        # Weighted sum of h
        attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, seq_length, 1] - OK
        
        r_vector = torch.sum(attention_weights * h, dim=-2)  # [batch_size, embed_size] - OK
        
        return r_vector



class NewsEncoder(nn.Module):
    def __init__(self, embed_size, heads, word_embedding_matrix, attention_dim):
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word_embedding_matrix, freeze=False)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
        self.additive_attention = AdditiveAttention(embed_size, attention_dim)

    def forward(self, x):
        # Get the word embeddings for each word in the title
        embedding = self.embedding(x)
        print(f"Embedding shape: {embedding.shape}")  # Should be [batch_size, M, embed_size] - OK
        
        # Apply multi-head attention
        attn_output, _ = self.multi_head_attention(embedding, embedding, embedding)
        print(f"Attention output shape: {attn_output.shape}")  # Should be [batch_size, M, embed_size] - OK
        
        # Apply additive attention to get a fixed-size representation
        news_representation = self.additive_attention(attn_output)
        print(f"News representation shape: {news_representation.shape}")  # Should be [batch_size, embed_size] - OK
        
        return news_representation



### Test news encoder
word_embedding_matrix = glove_vectors  # Assume glove_vectors are loaded and of correct size
attention_dim = 200
# Instantiate the model
encoder = NewsEncoder(embed_size=300, heads=5, word_embedding_matrix=word_embedding_matrix, attention_dim=attention_dim)

# Random input
x = input_data_train.loc[5, 'candidate_news']  # Assuming x is a list of candidate news titles

tensor_list = [torch.tensor(sublist) for sublist in x]
#tensor_list = [
#    torch.tensor([3, 8, 2, 9, 1]),  # Tensor for News article 1
#   torch.tensor([7, 4, 6, 3, 5])   # Tensor for News article 2
#]
x1 = tensor_list[0]
x2 = tensor_list[1]
x3 = tensor_list[2]

# Forward pass for each title
output1 = encoder(x1)
print("output1 size", output1.shape)
#print("output1", output1)

output2 = encoder(x2)
print("output2 size", output2.shape)
#print("output2", output2)

output3 = encoder(x3)
print("output3 size", output3.shape)
#print("output3", output3)




print('------USER ENCODER------')

class UserEncoder(nn.Module):
    def __init__(self, embed_size, heads, attention_dim):
        super(UserEncoder, self).__init__()

        # First layer: Multi-head attention
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
        
        # Second layer: Additive attention
        self.additive_attention = AdditiveAttention(embed_size=embed_size, attention_dim=attention_dim)


    def forward(self, news_representations):
        # Apply multi-head attention
        attn_output, _ = self.multi_head_attention(news_representations, news_representations, news_representations)
        print("Shape after multi-head attention:", attn_output.shape)  # [batch_size, N, h * embed_size]

        # Apply additive attention
        user_representation = self.additive_attention(attn_output)
        print("Shape of user representation:", user_representation.shape)  # [batch_size, embed_size]

        return user_representation


### Test user encoder
# Instantiate the UserEncoder
user_encoder = UserEncoder(embed_size=300, heads=5, attention_dim=200)

# Example input: Encoded news representations from the NewsEncoder
# Shape: [num_titles, embed_size]
# List of tensors (assuming they have the same shape)
news_representations = [output1, output2, output3]

# Convert list of tensors into a tensor
tensor_list = [sublist.clone().detach() for sublist in news_representations]

# Stack them into a single tensor with shape [batch_size, embed_size]
tensor_input = torch.stack(tensor_list, dim=0)  # shape will be [3, 300]
# news_representations = output

# Forward pass
user_representation = user_encoder(tensor_input)

# Output
print("User representation shape:", user_representation.shape)



print('------COMPLETE MODEL------')

class NRMS(nn.Module):
    def __init__(self, embed_size, heads, word_embedding_matrix, attention_dim):
        super(NRMS, self).__init__()
        self.news_encoder = NewsEncoder(embed_size, heads, word_embedding_matrix, attention_dim)
        self.user_encoder = UserEncoder(embed_size, heads, attention_dim)
    
    def forward(self, browsed_news, candidate_news):

        #News representation - r vector
        #from candidate news
        candidate_news_repr = self.news_encoder(candidate_news)

        #User representation - u vector
        #1. News representation of browsed news
        browsed_news_repr = [self.news_encoder(news) for news in browsed_news]
        browsed_news_repr = torch.stack(browsed_news_repr, dim=1) #list of tensors
        #2. User representation from representation of browsed news
        user_repr = self.user_encoder(browsed_news_repr)
        
        #Click probability
        click_probability = torch.sigmoid(torch.sum(user_repr * candidate_news_repr, dim=1))
        
        return click_probability



### Test MODEL
# Instantiate the UserEncoder
model_final = NRMS(embed_size=300, heads=5, word_embedding_matrix=glove_vectors, attention_dim=200)

browsed_news = torch.stack(tensor_list, dim=0)
candidate_news = torch.stack(tensor_list, dim=0)

# Forward pass
click = model_final(browsed_news,candidate_news)

# Output
print("User representation shape:", user_representation.shape)