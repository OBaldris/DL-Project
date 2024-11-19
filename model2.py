from Data_loader import *

#print('input_data_train.columns')
#print(input_data_train.columns)
#print('input_data_train.head()')
#print(input_data_train.head())

#print candidate_news of the first user
#print(input_data_train.loc[0, "candidate_news"])


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence



class AdditiveAttention(nn.Module):
    def _init_(self, embed_size, attention_dim):
        
        super(AdditiveAttention, self)._init_()
        self.V_w = nn.Linear(embed_size, attention_dim)  #weight and bias (vw)
        self.q_w = nn.Parameter(torch.zeros(attention_dim))

        #initializations
        nn.init.xavier_uniform_(self.V_w.weight) 
        nn.init.zeros_(self.V_w.bias)
        nn.init.xavier_uniform_(self.q_w.unsqueeze(0)) 

    def forward(self, h):
        
        #projection into attention space and tanh after
        projection=self.V_w(h) #[batch_size, seq_length, attention_dim]
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
    def _init_(self, embed_size, heads, word_embedding_matrix,attention_dim):
        super(NewsEncoder, self)._init_()
        self.embedding = nn.Embedding.from_pretrained(word_embedding_matrix, freeze=False)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
        self.additive_attention = AdditiveAttention(embed_size, attention_dim)
        
    def forward(self, x):
        embedding = self.embedding(x)
        print(embedding.shape)
        attn_output, _ = self.multi_head_attention(embedding, embedding, embedding)
        print(attn_output.shape)
        news_representation = self.additive_attention(attn_output)
        print(news_representation.shape)

        return news_representation
    

### Test news encoder
word_embedding_matrix = glove_vectors
attention_dim = 200
# Instantiate the model
encoder = NewsEncoder(embed_size=300, heads=5, word_embedding_matrix=word_embedding_matrix,attention_dim=200)

# Random input
x = input_data_train.loc[5, 'candidate_news']
# print first title of x


tensor_list = [torch.tensor(sublist) for sublist in x]
x1 = tensor_list[0]
print(x1)
x2 = tensor_list[1]
x3 = tensor_list[2]

#x = pad_sequence(tensor_list, batch_first=True, padding_value=0)

# Forward pass
output1 = encoder(x1)
print("output size", output1.shape)
print("output", output1)

output2 = encoder(x2)
print("output size", output2.shape)
print("output", output2)

output3 = encoder(x3)
print("output size", output3.shape)
print("output", output3)



#---------- OLD News Encoder -----------
#class NewsEncoder(nn.Module):
 #   def _init_(self, embed_size, heads, word_embedding_matrix):
#        super(NewsEncoder, self)._init_()
    #     self.embedding = nn.Embedding.from_pretrained(word_embedding_matrix, freeze=False)
    #     self.multi_head_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
    #     self.fc = nn.Linear(embed_size, embed_size)
        
    # def forward(self, x):
    #     embedding = self.embedding(x)
    #     attn_output, _ = self.multi_head_attention(embedding, embedding, embedding)
    #     news_representation = torch.tanh(self.fc(attn_output.mean(dim=1)))
    #     return news_representation
    


print('------USER ENCODER------')

class AdditiveAttention2(nn.Module):
    def _init_(self, embed_size, attention_dim):
        super(AdditiveAttention2, self)._init_()
        
        # Parameters for attention mechanism
        self.V_n = nn.Linear(embed_size, attention_dim)  # Vn for projection
        self.q_n = nn.Parameter(torch.zeros(attention_dim))  # query vector for attention
        
        # Initializations
        nn.init.xavier_uniform_(self.V_n.weight)
        nn.init.zeros_(self.V_n.bias)
        nn.init.xavier_uniform_(self.q_n.unsqueeze(0))  # query vector

    def forward(self, h):
        """
        h: tensor of shape [batch_size, N, embed_size]
        where N is the number of browsed news, and embed_size is the size of the news representations.
        """
        
        # Projection of h_i^n through V_n and addition of v_n
        projection = self.V_n(h)  # shape: [batch_size, N, attention_dim]
        tanh_output = torch.tanh(projection)  # Apply tanh
        print("Shape of tanh_output:", tanh_output.shape)


        # Compute attention scores a_i^n = q_n^T * tanh(V_n * h_i^n + v_n)
        attention_scores = torch.matmul(tanh_output, self.q_n)  # shape: [batch_size, N]
        print("Shape of attention_scores:", attention_scores.shape)

        # Apply softmax to get attention weights alpha_i^n

        # !!!!! HERE I CHANGED THIS TO DIM=-1, BUT IN THE OTHER IS DIM=1
        attention_weights = F.softmax(attention_scores, dim=1)  # shape: [batch_size, N]
        # !!!!!!!
        print("Shape of attention_weights:", attention_weights.shape)

        # Attention weights should be expanded to match h_i^n shape for element-wise multiplication
        attention_weights = attention_weights.unsqueeze(-1)  # shape: [batch_size, N, 1]
        
        print("Shape of attention_weights:", attention_weights.shape)
        
        # Weighted sum of h_i^n to get final user representation
        u_vector = torch.sum(attention_weights * h, dim=1)  # shape: [batch_size, embed_size]

        print("Shape of u_vector:", u_vector.shape)

        return u_vector



class UserEncoder(nn.Module):
    def _init_(self, embed_size, heads, attention_dim):
        super(UserEncoder, self)._init_()

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
news_representations = encoder(x)  # Here, x is the padded input from the previous step
# news_representations = output

# Forward pass
user_representation = user_encoder(news_representations)

# Output
print("User representation shape:", user_representation.shape)



#---------- OLD User Encoder -----------
# class UserEncoder(nn.Module):
#     def _init_(self, embed_size, heads):
#         super(UserEncoder, self)._init_()
#         self.multi_head_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
#         self.fc = nn.Linear(embed_size, embed_size)

#     def forward(self, news_representations):
#         """
#         news_representations: Tensor of shape [num_titles, embed_size], 
#                               where num_titles is the number of browsed news articles for a user.
#         """
#         # Apply multi-head self-attention
#         attn_output, _ = self.multi_head_attention(news_representations, news_representations, news_representations)

#         # Mean Pooling: Aggregate attended news representations
#         user_representation = torch.tanh(self.fc(attn_output.mean(dim=0)))

#         return user_representation