"""
'browsed_news' tensor.Shape = [batch_size, max_num_browsed, embed_size]
'candidate_news' tensor.Shape = [batch_size, max_num_candidates, embed_size]
'clicked_news' tensor.Shape = [batch_size, max_num_candidates]

'input_data_train' and 'input_data_validation' dataframes:
|     'browsed_news'    |  'candidate_news'    | 'article_ids_clicked'| 'clicked_idx' |
|  [[#,#,#], [#,#],...] | [[#,#,#], [#,#],...] |             [[#, #]] | [0, 1, 0, ...]|

"""
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

#1. DOWNLOAD DATA----------------------------------------------
file_path = "../Data/ebnerd_demo"

df_behaviors_train = pd.read_parquet(file_path + '/train' + '/behaviors.parquet')
df_behaviors_validation = pd.read_parquet(file_path + '/validation' + '/behaviors.parquet')

df_history_train = pd.read_parquet(file_path + '/train' + '/history.parquet')
df_history_validation = pd.read_parquet(file_path + '/validation' + '/history.parquet')

df_articles = pd.read_parquet(file_path + '/articles.parquet')

#2. GET RELEVANT FEATURES---------------------------------------
# ~ DF_BEHAVIORS
#interested in 'UserID' = 'UserID', 'InviewArticleIds' = 'CandidateNews', 'ClickedArticleIDs' = 'ClickedArticleIDs'
df_behaviors_train = df_behaviors_train[['user_id','article_ids_inview','article_ids_clicked']]
df_behaviors_validation = df_behaviors_validation[['user_id','article_ids_inview','article_ids_clicked']]
df_behaviors_train=df_behaviors_train.dropna().rename(columns={'article_ids_inview': 'candidate_news'})
df_behaviors_validation=df_behaviors_validation.dropna().rename(columns={'article_ids_inview': 'candidate_news'})

#------- We need to one-hot encode the clicked articles -------
def one_hot_encode(candidate_news, clicked_news):
    return [1 if num == clicked_news[0] else 0 for num in candidate_news]

df_behaviors_train['clicked_idx'] = df_behaviors_train.apply(lambda row: one_hot_encode(row['candidate_news'], row['article_ids_clicked']), axis=1)
df_behaviors_validation['clicked_idx'] = df_behaviors_validation.apply(lambda row: one_hot_encode(row['candidate_news'], row['article_ids_clicked']), axis=1)

#Now behaviors has the shape and data that we want --> it will be our main dataset
print('New behaviors: \n', df_behaviors_train.head()) 
#!!You can have several entries per user!! --> Repeated 'user_id' values
duplicates = df_behaviors_validation['user_id'].duplicated().any()
#print('Duplicate users in behaviors? ', duplicates)

# ~ DF_HISTORY
#interested in 'UserID' = 'UserID', 'ArticleIDs' = 'BrowsedNews'
df_history_train = df_history_train[['user_id','article_id_fixed']]
df_history_validation = df_history_validation[['user_id','article_id_fixed']]
df_history_train=df_history_train.dropna().rename(columns={'article_id_fixed': 'browsed_news'})
df_history_validation=df_history_validation.dropna().rename(columns={'article_id_fixed': 'browsed_news'})
#print('\n New history: \n', df_history_train.head())
#All user entries are different --> No repeated 'user_id' values
duplicates = df_history_validation['user_id'].duplicated().any()
#print('Duplicate users in history? ', duplicates)

# ~ DF_ARTICLES
#interested in 'ArticleID' and 'Title'
df_articles = df_articles[['article_id','title']]

#3. GLOVE TOKENIZATION AND EMBEDDING--------------------

# Define the save/load paths for GloVe
glove_save_path = "../Data/glove_vectors.pt"

# Load or save GloVe vectors
def load_glove_vectors(filename="glove.6B.300d.txt", save_path=glove_save_path):
    """Load GloVe vectors, saving them for reuse if not already saved."""
    if os.path.exists(save_path):
        print(f"Loading GloVe vectors from saved file: {save_path}")
        data = torch.load(save_path, weights_only=True)
        return data["vocabulary"], data["vectors"]

    print("Downloading and processing GloVe vectors for the first time...")
    # Download and extract GloVe
    path = Path(hf_hub_download(repo_id="stanfordnlp/glove", filename="glove.6B.zip"))
    target_file = path.parent / filename
    if not target_file.exists():
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(path.parent)
        if not target_file.exists():
            raise ValueError(f"Target file `{target_file.name}` not found.")

    # Parse GloVe vectors
    vocabulary = []
    vectors = []
    with open(target_file, "r", encoding="utf8") as f:
        for l in tqdm(f.readlines(), desc=f"Parsing {target_file.name}..."):
            word, *vector = l.split()
            vocabulary.append(word)
            vectors.append(torch.tensor([float(v) for v in vector]))
    vectors = torch.stack(vectors)

    # Save the processed GloVe data
    torch.save({"vocabulary": vocabulary, "vectors": vectors}, save_path)
    print(f"GloVe vectors saved to: {save_path}")
    return vocabulary, vectors

# Call the function
glove_vocabulary, glove_vectors = load_glove_vectors()

# Add special tokens
special_tokens = ["<|start|>", "<|unknown|>", "<|pad|>"]
glove_vocabulary = special_tokens + glove_vocabulary
glove_vectors = torch.cat([torch.randn_like(glove_vectors[: len(special_tokens)]), glove_vectors])
pad_idx = glove_vocabulary.index("<|pad|>")
# Print summary
#print(f"GloVe Vocabulary Size: {len(glove_vocabulary)}")
#print(f"GloVe Vectors Shape: {glove_vectors.shape}")

# tokenizer for GloVe (at word level)
import tokenizers
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

glove_tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab={v:i for i,v in enumerate(glove_vocabulary)}, unk_token="<|unknown|>"))
glove_tokenizer.normalizer = tokenizers.normalizers.BertNormalizer(strip_accents=False)
glove_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

#input: titles (candidates_titles, clicked_titles or browsed_titles)
#for each sentence in the list, the function tokenizes it by giving a numerical ID to each word
def glove_tok(sentence):
    token_ids = glove_tokenizer.encode(sentence, add_special_tokens=False)
    if isinstance(token_ids, tokenizers.Encoding):
        token_ids = token_ids.ids
    return token_ids


#Padding
#list of sentences
def tokenize_and_pad(sentences, pad_idx):
    # Tokenize each sentence
    tokenized = [glove_tok(sentence) for sentence in sentences]

    # Convert to tensors
    token_tensors = [torch.tensor(tokens, dtype=torch.long) for tokens in tokenized]

    # Pad the sequences
    
    padded_sequences = pad_sequence(token_tensors, batch_first=True, padding_value=pad_idx)

    return padded_sequences

#input: list of articles id (from the data frame)
def map_tokenized_titles(article_ids_list):
    return [articles_dict[article_id] for article_id in article_ids_list if article_id in articles_dict]

#4. FINAL DATASET------------------------------------

#4.1 CREATE DICTIONARIES
#DICT 1: ARTICLE ID AND ITS TOKENIZATION (same for train and validation)
to_tokenize_a=df_articles['title']
tokenized_articles=tokenize_and_pad(to_tokenize_a,pad_idx)
df_articles['title']=tokenized_articles.tolist()
articles_dict= df_articles.set_index('article_id')['title'].to_dict()


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
#input_data_train['article_ids_clicked'] = input_data_train['article_ids_clicked'].apply(map_tokenized_titles)

input_data_validation = df_behaviors_validation.copy()
input_data_validation['candidate_news'] = input_data_validation['candidate_news'].apply(map_tokenized_titles)
#input_data_validation['article_ids_clicked'] = input_data_validation['article_ids_clicked'].apply(map_tokenized_titles)

# STEP 3: USER BROWSED HISTORY
input_data_train['user_id'] = input_data_train['user_id'].map(history_dict_train)
input_data_validation['user_id'] = input_data_validation['user_id'].map(history_dict_validation)

input_data_train = input_data_train.rename(columns={'user_id': 'browsed_news'})
input_data_validation = input_data_validation.rename(columns={'user_id': 'browsed_news'})


#6. MAKE SURE TO DROP NA
input_data_train = input_data_train.dropna()
input_data_validation = input_data_validation.dropna()

#7. Extra padding and conversion to tensors
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
