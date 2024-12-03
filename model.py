
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
        #nn.init.xavier_uniform_(self.q_w)
        nn.init.uniform_(self.q_w, a=-0.01, b=0.01)

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
    

# Tokenized title --> r vector
class NewsEncoder(nn.Module):
    def __init__(self, embed_size, heads, word_embedding_matrix, attention_dim):
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word_embedding_matrix, freeze=False, padding_idx=0) #pad_idx=0
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
        self.additive_attention = AdditiveAttention(embed_size, attention_dim)

    def forward(self, x):
        # Get the word embeddings for each word in the title
        embedding = self.embedding(x)
        #print(f"Embedding shape: {embedding.shape}")  # Should be [batch_size, M, embed_size] - OK
        
        # Generate attention mask (True for padding tokens, False for real tokens)
        attention_mask = (x == 0).float()  #pad_idx=0  # [batch_size, seq_length]
        key_padding_mask = attention_mask.bool()  # Mask where True means padding

        # Apply multi-head attention
        attn_output, _ = self.multi_head_attention(embedding, embedding, embedding, key_padding_mask=key_padding_mask)
        #print(f"Attention output shape: {attn_output.shape}")  # Should be [batch_size, M, embed_size] - OK
        
        # Apply additive attention to get a fixed-size representation
        news_representation = self.additive_attention(attn_output)
        #print(f"News representation shape: {news_representation.shape}")  # Should be [batch_size, embed_size] - OK
        
        return news_representation


# r vectors --> u vector
class UserEncoder(nn.Module):
    def __init__(self, embed_size, heads, attention_dim):
        super(UserEncoder, self).__init__()

        # First layer: Multi-head attention
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
        
        # Second layer: Additive attention
        self.additive_attention = AdditiveAttention(embed_size=embed_size, attention_dim=attention_dim)


    def forward(self, news_representations):
        # Generate attention mask (True for padding tokens, False for real tokens)
        attention_mask = torch.isnan(news_representations).all(dim=2).float()  #pad_idx=0  # [batch_size, seq_length]
        key_padding_mask = attention_mask  # Mask where True means padding

        # Apply multi-head attention
        attn_output, _ = self.multi_head_attention(news_representations, news_representations, news_representations, key_padding_mask=key_padding_mask)
        #print("Shape after multi-head attention:", attn_output.shape)  # [batch_size, N, h * embed_size]

        # Apply additive attention
        user_representation = self.additive_attention(attn_output)
        #print("Shape of user representation:", user_representation.shape)  # [batch_size, embed_size]

        return user_representation




class NRMS(nn.Module):
    def __init__(self, embed_size, heads, word_embedding_matrix, attention_dim):
        super(NRMS, self).__init__()
        self.news_encoder = NewsEncoder(embed_size, heads, word_embedding_matrix, attention_dim)
        self.user_encoder = UserEncoder(embed_size, heads, attention_dim)
    
    def forward(self, browsed_news, candidate_news):

        #News representation - candidate r vectors
        #1. News representation of candidate news
        #the output must be a matrix of r vectors, on efor each candidate news (300xn)
        candidate_news_repr = [self.news_encoder(news) for news in candidate_news]
        candidate_news_repr = torch.stack(candidate_news_repr, dim=1) #list of tensors
        candidate_news_repr = candidate_news_repr.transpose(0, 1) #swap the dimensions
        #print(f"candidate_news_repr shape: {candidate_news_repr.shape}")

        #User representation - u vector
        #the output has to be a vector u, only one for a set of browsed news (1x300)
        #1. News representation of browsed news
        browsed_news_repr = [self.news_encoder(news) for news in browsed_news]
        browsed_news_repr = torch.stack(browsed_news_repr, dim=1) #list of tensors
        browsed_news_repr = browsed_news_repr.transpose(0, 1) #swap the dimensions
        #print(f"browsed_news_repr shape: {browsed_news_repr.shape}")
        #2. User representation from representation of browsed news
        user_repr = self.user_encoder(browsed_news_repr)
        user_repr = user_repr.unsqueeze(1) 
        
        #Click probability
        #vector (batch x n or n x batch)
        #print("\n")
        #print(f"user_repr shape: {user_repr.shape}")
        #print(f"candidate_news_repr shape: {candidate_news_repr.shape}")
        #click_probability = candidate_news_repr @ user_repr.transpose(0, 1)
        click_probability = torch.bmm(candidate_news_repr, user_repr.transpose(1, 2)).squeeze(-1)
        click_probability = F.softmax(click_probability, dim=1)
        
        return click_probability