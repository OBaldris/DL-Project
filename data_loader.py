

"""
'browsed_news' tensor.Shape = [batch_size, max_num_browsed, embed_size]
'candidate_news' tensor.Shape = [batch_size, max_num_candidates, embed_size]
'clicked_news' tensor.Shape = [batch_size, max_num_candidates]

'input_data_train' and 'input_data_validation' dataframes:
|     'browsed_news'    |  'candidate_news'    | 'article_ids_clicked'| 'clicked_idx' |
|  [[#,#,#], [#,#],...] | [[#,#,#], [#,#],...] |     [[#, #, #, ...]] | [0, 1, 0, ...]|

"""
from functions import *
import pandas as pd
from itertools import islice
from pprint import pprint
import torch
from pathlib import Path
import zipfile
import os
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np

#1. DOWNLOAD DATA----------------------------------------------
file_path = "../Data/ebnerd_demo"
print("Dataset: ebnerd_demo")

df_behaviors_train = pd.read_parquet(file_path + '/train' + '/behaviors.parquet')
df_behaviors_validation = pd.read_parquet(file_path + '/validation' + '/behaviors.parquet')

df_history_train = pd.read_parquet(file_path + '/train' + '/history.parquet')
df_history_validation = pd.read_parquet(file_path + '/validation' + '/history.parquet')

df_articles = pd.read_parquet(file_path + '/articles.parquet')

#2. GET RELEVANT FEATURES---------------------------------------
                   
                   ### DF_BEHAVIORS ###
#interested in 'UserID'='UserID', 'InviewArticleIds' = 'CandidateNews', 'ClickedArticleIDs' = 'ClickedArticleIDs'
df_behaviors_train = df_behaviors_train[['user_id','article_ids_inview','article_ids_clicked']]
df_behaviors_validation = df_behaviors_validation[['user_id','article_ids_inview','article_ids_clicked']]
df_behaviors_train=df_behaviors_train.dropna().rename(columns={'article_ids_inview': 'candidate_news'})
df_behaviors_validation=df_behaviors_validation.dropna().rename(columns={'article_ids_inview': 'candidate_news'})

# We need to one-hot encode the clicked articles

df_behaviors_train['clicked_idx'] = df_behaviors_train.apply(lambda row: one_hot_encode(row['candidate_news'], row['article_ids_clicked']), axis=1)
df_behaviors_validation['clicked_idx'] = df_behaviors_validation.apply(lambda row: one_hot_encode(row['candidate_news'], row['article_ids_clicked']), axis=1)

#Now behaviors has the shape and data that we want --> it will be our main dataset

# print('New behaviors: \n', df_behaviors_train.head()) 
#!!You can have several entries per user!! --> Repeated 'user_id' values
duplicates = df_behaviors_validation['user_id'].duplicated().any()
#print('Duplicate users in behaviors? ', duplicates)

                    ### DF_HISTORY ###
#interested in 'UserID' = 'UserID', 'ArticleIDs' = 'BrowsedNews'
df_history_train = df_history_train[['user_id','article_id_fixed']]
df_history_validation = df_history_validation[['user_id','article_id_fixed']]
df_history_train=df_history_train.dropna().rename(columns={'article_id_fixed': 'browsed_news'})
df_history_train = df_history_train.dropna(subset=['browsed_news'])

df_history_validation=df_history_validation.dropna().rename(columns={'article_id_fixed': 'browsed_news'})
df_history_validation = df_history_validation.dropna(subset=['browsed_news'])

#print('\n New history: \n', df_history_train.head())
#All user entries are different --> No repeated 'user_id' values
duplicates = df_history_validation['user_id'].duplicated().any()
#print('Duplicate users in history? ', duplicates)

                    ### DF_ARTICLES ###
#interested in 'ArticleID' and 'Title'
df_articles = df_articles[['article_id','title','subtitle']]
df_articles_temp = df_articles[['article_id']].copy()
df_articles_temp['title'] = df_articles['title'] + ' ' + df_articles['subtitle']
df_articles = df_articles_temp


#3. GLOVE TOKENIZATION, EMBEDDING AND PADDING------------------------------------
# Define the save/load paths for GloVe
glove_save_path = "../Data/glove_vectors.pt"

# Load or save GloVe vectors
# Call the function
glove_vocabulary, glove_vectors = load_glove_vectors()

# Add special tokens
special_tokens = ["<|pad|>", "<|start|>", "<|unknown|>"]
glove_vocabulary = special_tokens + glove_vocabulary
glove_vectors = torch.cat([torch.randn_like(glove_vectors[: len(special_tokens)]), glove_vectors])
pad_idx = glove_vocabulary.index("<|pad|>")
# Print summary
#print(f"GloVe Vocabulary Size: {len(glove_vocabulary)}")
#print(f"GloVe Vectors Shape: {glove_vectors.shape}")

def glove_tok(sentence):
    token_ids = glove_tokenizer.encode(sentence, add_special_tokens=False)
    if isinstance(token_ids, tokenizers.Encoding):
        token_ids = token_ids.ids
    return token_ids
def tokenize_and_pad(sentences, pad_idx):
    # Tokenize each sentence
    tokenized = [glove_tok(sentence) for sentence in sentences]

    # Convert to tensors
    token_tensors = [torch.tensor(tokens, dtype=torch.long) for tokens in tokenized]

    # Pad the sequences
    
    padded_sequences = pad_sequence(token_tensors, batch_first=True, padding_value=pad_idx)

    return padded_sequences

# tokenizer for GloVe (at word level)
glove_tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab={v:i for i,v in enumerate(glove_vocabulary)}, unk_token="<|unknown|>"))
glove_tokenizer.normalizer = tokenizers.normalizers.BertNormalizer(strip_accents=False)
glove_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

#input: titles (candidates_titles, clicked_titles or browsed_titles)
#for each sentence in the list, the function tokenizes it by giving a numerical ID to each word

#Padding
#list of sentences


#4. INPUT DATAFRAMES -------------------------------------------------------------

#4.1 CREATE DICTIONARIES
#DICT 1: ARTICLE ID AND ITS TOKENIZATION (same for train and validation)
to_tokenize_a=df_articles['title']
tokenized_articles=tokenize_and_pad(to_tokenize_a,pad_idx)
df_articles['title']=tokenized_articles.tolist()

#4.2 CUT THE TITLE TO n WORDS
max_words_articles = 20

#plot_title_size_distribution(df_articles)
len_before = len(df_articles)

# Truncate articles to 20 words
df_articles.loc[:,'title'] = df_articles['title'].apply(lambda tokens: truncate_to_n_tokens(tokens, max_words_articles))

#plot_title_size_distribution(df_articles)

articles_dict= df_articles.set_index('article_id')['title'].to_dict()



#input: list of articles id (from the data frame)
def map_tokenized_titles(article_ids_list):
    return [articles_dict[article_id] for article_id in article_ids_list if article_id in articles_dict]
#DICT 2: USER ID AND ITS HISTORY ALREADY TOKENIZED (different for train and validation)
df_history_train['browsed_news'] = df_history_validation['browsed_news'].apply(map_tokenized_titles)
history_dict_train = df_history_train.set_index('user_id')['browsed_news'].to_dict()

df_history_validation['browsed_news'] = df_history_validation['browsed_news'].apply(map_tokenized_titles)
history_dict_validation = df_history_validation.set_index('user_id')['browsed_news'].to_dict()

#STEP 2: DATA FRAME WITH TOKEN IDS
#1ST COLUMN: USER ID
#2ND COLUMN: LIST OF ARTICLES INVIWED
#3RD COLUMN: ART CLICKED

input_data_train = df_behaviors_train.copy()
input_data_train['candidate_news'] = input_data_train['candidate_news'].apply(map_tokenized_titles)
input_data_train['article_ids_clicked'] = input_data_train['article_ids_clicked'].apply(map_tokenized_titles)

input_data_validation = df_behaviors_validation.copy()
input_data_validation['candidate_news'] = input_data_validation['candidate_news'].apply(map_tokenized_titles)
input_data_validation['article_ids_clicked'] = input_data_validation['article_ids_clicked'].apply(map_tokenized_titles)

# STEP 3: USER BROWSED HISTORY
input_data_train['user_id'] = input_data_train['user_id'].map(history_dict_train)
input_data_validation['user_id'] = input_data_validation['user_id'].map(history_dict_validation)

input_data_train = input_data_train.rename(columns={'user_id': 'browsed_news'})
input_data_validation = input_data_validation.rename(columns={'user_id': 'browsed_news'})



#MAKE SURE TO DROP NA
input_data_train = input_data_train.dropna()
input_data_validation = input_data_validation.dropna()

<<<<<<< HEAD
#7. Extra padding and conversion to tensors
<<<<<<< HEAD:data_loader.py
# Function to truncate or filter data
def truncate_or_filter(input_data):
=======


# STATISTICS
# Calculate statistics for browsed_news and candidate_news in the train and validation datasets
stats_train = calculate_statistics(input_data_train, dataset_name="Train Dataset", plot_distributions=False)
stats_validation = calculate_statistics(input_data_validation, dataset_name="Validation Dataset", plot_distributions=False)



#5. TRUNCATE OR FILTER DATA------------------------------------------------------------
max_num_browsed = 20
max_num_candidates = 20


def truncate_or_filter(input_data, trunc_num_candidates=10, trunc_num_browsed=10):
>>>>>>> origin/maria
    """
    Truncate browsed_news and candidate_news to `trunc_num` items each,
    while ensuring that the clicked news is part of the candidate_news.
    Users with fewer than `trunc_num` browsed_news or candidate_news are removed.
    """
    def truncate_news(row):
        # Ensure clicked news is among the top trunc_num candidate news
        candidate_news = row['candidate_news']
        clicked_idx = row['clicked_idx']
        clicked_news = [candidate_news[i] for i, click in enumerate(clicked_idx) if click == 1]
        
        if len(clicked_news) == 0:
            return None  # Remove rows without clicked news
        
        # Truncate candidate_news to ensure clicked_news is included
        truncated_candidates = clicked_news[:1] + [news for news in candidate_news if news not in clicked_news][:trunc_num_candidates - 1]
        
        # Check if truncation resulted in exactly trunc_num items
        if len(truncated_candidates) < trunc_num_candidates:
            return None  # Remove rows with insufficient candidate news

        row['candidate_news'] = truncated_candidates
        row['clicked_idx'] = [1 if news in clicked_news else 0 for news in truncated_candidates]
        
        # Truncate browsed_news to trunc_num items
        if len(row['browsed_news']) < trunc_num_browsed:
            return None  # Remove rows with insufficient browsed news
        row['browsed_news'] = row['browsed_news'][:trunc_num_browsed]
        
        return row
    
    # Apply truncation and filter rows
    truncated_data = input_data.apply(truncate_news, axis=1)
    # Remove rows that returned None
    truncated_data = truncated_data.dropna().reset_index(drop=True)
    
    if not isinstance(truncated_data, pd.DataFrame):
        # Convert the filtered Series back to a DataFrame
        truncated_data = pd.DataFrame(truncated_data.tolist(), columns=input_data.columns)

    return truncated_data



# Apply truncation and filtering on training data
input_data_train_truncated = truncate_or_filter(input_data_train, max_num_candidates, max_num_browsed)
# Apply truncation and filtering on validation data
input_data_validation_truncated = truncate_or_filter(input_data_validation, max_num_candidates, max_num_browsed)


# Convert to tensors after truncation
# Convert each column to numpy arrays
browsed_news_array = np.array(input_data_train_truncated['browsed_news'].tolist())
candidate_news_array = np.array(input_data_train_truncated['candidate_news'].tolist())
clicked_idx_array = np.array(input_data_train_truncated['clicked_idx'].tolist())

# Convert numpy arrays to PyTorch tensors
browsed_news_train = torch.tensor(browsed_news_array, dtype=torch.long)
candidate_news_train = torch.tensor(candidate_news_array, dtype=torch.long)
clicked_news_train = torch.tensor(clicked_idx_array, dtype=torch.float)


# Convert each column to numpy arrays
browsed_news_array = np.array(input_data_validation_truncated['browsed_news'].tolist())
candidate_news_array = np.array(input_data_validation_truncated['candidate_news'].tolist())
clicked_idx_array = np.array(input_data_validation_truncated['clicked_idx'].tolist())

# Convert numpy arrays to PyTorch tensors
browsed_news_validation = torch.tensor(browsed_news_array, dtype=torch.long)
candidate_news_validation = torch.tensor(candidate_news_array, dtype=torch.long)
clicked_news_validation = torch.tensor(clicked_idx_array, dtype=torch.long)





#8. DATA LOADER--------------------------------------------------------------

class NewsRecommendationDataset(Dataset):
    def __init__(self, browsed_news, candidate_news, clicked_idx, embed_size):
        """
        Initialize dataset with preprocessed data.
        :param browsed_news: Tensor of browsed news articles [batch_size, max_num_browsed, embed_size]
        :param candidate_news: Tensor of candidate news articles [batch_size, max_num_candidates, embed_size]
        :param clicked_idx: Tensor of clicked labels [batch_size, max_num_candidates]
        :param embed_size: Embedding size for each article
        """
        self.browsed_news = browsed_news
        self.candidate_news = candidate_news
        self.clicked_idx = clicked_idx
        self.embed_size = embed_size

    def __len__(self):
        return len(self.browsed_news)

    def __getitem__(self, idx):
        """
        Return a single user's data for batching.
        """
        browsed_news = self.browsed_news[idx]
        candidate_news = self.candidate_news[idx]
        clicked_idx = self.clicked_idx[idx]
        return {
            'browsed_news': browsed_news,
            'candidate_news': candidate_news,
            'clicked_idx': clicked_idx
        }
    


# Assuming the truncated data has already been converted to tensors
train_dataset = NewsRecommendationDataset(
    browsed_news=browsed_news_train,
    candidate_news=candidate_news_train,
    clicked_idx=clicked_news_train,
    embed_size=glove_vectors.shape[1]  # Embedding size from GloVe
)

# Validation Dataset
validation_dataset = NewsRecommendationDataset(
    browsed_news=browsed_news_validation,
    candidate_news=candidate_news_validation,
    clicked_idx=clicked_news_validation,
    embed_size=glove_vectors.shape[1]
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

<<<<<<< HEAD
=======
def tensor_pad(column):
    column = [torch.tensor(sublist) for sublist in column]
    column = pad_sequence(column, batch_first=True, padding_value=0)
    return column

browsed_news_train = tensor_pad(input_data_train['browsed_news'])
candidate_news_train = tensor_pad(input_data_train['candidate_news'])
clicked_news_train = tensor_pad(input_data_train['clicked_idx'])

browsed_news_validation = tensor_pad(input_data_validation['browsed_news'])
candidate_news_validation = tensor_pad(input_data_validation['candidate_news'])
clicked_news_validation = tensor_pad(input_data_validation['clicked_idx'])
>>>>>>> 51f34a79d88775878446fbb553e8331d94eae20b:Data_loader.py
=======
>>>>>>> origin/maria
