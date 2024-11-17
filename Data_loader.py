
"""
# LOAD THE VALIDATION AND TRAINING DATA AND CONVERT INTO:
df_behaviors: Train and validation DF containing 3 columns
|'user_id'|'candidate_news'|'article_ids_clicked'|

df_history: Train and validation DF containing 2 columns
|'user_id'|'browsed_news'|

ID_articles: A table containing:
|'article_id'|'title'|

"""
#DL Project
import pandas as pd
from itertools import islice
from pprint import pprint

#1. DOWNLOAD DATA
file_path = "../Data/ebnerd_demo"

df_behaviors_train = pd.read_parquet(file_path + '/train' + '/behaviors.parquet')
df_behaviors_validation = pd.read_parquet(file_path + '/validation' + '/behaviors.parquet')

df_history_train = pd.read_parquet(file_path + '/train' + '/history.parquet')
df_history_validation = pd.read_parquet(file_path + '/validation' + '/history.parquet')

df_articles = pd.read_parquet(file_path + '/articles.parquet')




#2. GET RELEVANT FEATURES
#DF_BEHAVIORS
#interested in 'UserID' = 'UserID', 'InviewArticleIds' = 'CandidateNews', 'ClickedArticleIDs' = 'ClickedArticleIDs'
df_behaviors_train = df_behaviors_train[['user_id','article_ids_inview','article_ids_clicked']]
df_behaviors_validation = df_behaviors_validation[['user_id','article_ids_inview','article_ids_clicked']]
df_behaviors_train=df_behaviors_train.dropna().rename(columns={'article_ids_inview': 'candidate_news'})
df_behaviors_validation=df_behaviors_validation.dropna().rename(columns={'article_ids_inview': 'candidate_news'})
#Now behaviors has the shape and data that we want --> it will be our main dataset
print('New behaviors: \n', df_behaviors_train.head()) 
    #!!You can have several entries per user!! --> Repeated 'user_id' values
duplicates = df_behaviors_validation['user_id'].duplicated().any()
print('Duplicate users in behaviors? ', duplicates)

#DF_HISTORY
#interested in 'UserID' = 'UserID', 'ArticleIDs' = 'BrowsedNews'
df_history_train = df_history_train[['user_id','article_id_fixed']]
df_history_validation = df_history_validation[['user_id','article_id_fixed']]
df_history_train=df_history_train.dropna().rename(columns={'article_id_fixed': 'browsed_news'})
df_history_validation=df_history_validation.dropna().rename(columns={'article_id_fixed': 'browsed_news'})
print('\n New history: \n', df_history_train.head())
    #All user entries are different --> No repeated 'user_id' values
duplicates = df_history_validation['user_id'].duplicated().any()
print('Duplicate users in history? ', duplicates)

#DF_ARTICLES
#interested in 'ArticleID' and 'Title'
df_articles = df_articles[['article_id','title']]




#3. GLOVE TOKENIZATION AND EMBEDDING
from typing import Tuple, List
import torch
from pathlib import Path
import zipfile
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import rich
from rich import print as rprint
import tokenizers
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

#load Glove vectors from a file
def load_glove_vectors(filename = "glove.6B.300d.txt") -> Tuple[List[str], torch.Tensor]:
    """Load the GloVe vectors. See: `https://github.com/stanfordnlp/GloVe`"""
    path = Path(hf_hub_download(repo_id="stanfordnlp/glove", filename="glove.6B.zip"))
    target_file = path.parent / filename
    if not target_file.exists():
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(path.parent)

        if not target_file.exists():
            print(f"Available files:")
            for p in path.parent.iterdir():
                print(p)
            raise ValueError(f"Target file `{target_file.name}` can't be found. Check if `{filename}` was properly downloaded.")

    # parse the vocabulary and the vectors
    vocabulary = []
    vectors = []
    with open(target_file, "r", encoding="utf8") as f:
        for l in tqdm(f.readlines(), desc=f"Parsing {target_file.name}..." ):
            word, *vector = l.split()
            vocabulary.append(word)
            vectors.append(torch.tensor([float(v) for v in vector]))
    vectors = torch.stack(vectors)
    return vocabulary, vectors

# prepare data for the later cells
glove_vocabulary, glove_vectors = load_glove_vectors()
rich.print(f"glove_vocabulary: type={type(glove_vocabulary)}, length={len(glove_vocabulary)}")
rich.print(f"glove_vectors: type={type(glove_vectors)}, shape={glove_vectors.shape}, dtype={glove_vectors.dtype}")

# add special tokens
special_tokens = ['<|start|>', '<|unknown|>', '<|pad|>']
glove_vocabulary = special_tokens + glove_vocabulary
glove_vectors = torch.cat([torch.randn_like(glove_vectors[:len(special_tokens)]), glove_vectors])

# tokenizer for GloVe (at word level)
glove_tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab={v:i for i,v in enumerate(glove_vocabulary)}, unk_token="<|unknown|>"))
glove_tokenizer.normalizer = tokenizers.normalizers.BertNormalizer(strip_accents=False)
glove_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

vocab_size = glove_tokenizer.get_vocab_size()

#input: titles (candidates_titles, clicked_titles or browsed_titles)
#for each sentence in the list, the function tokenizes it by giving a numerical ID to each word
def glove_tok(sentence):
    token_ids = glove_tokenizer.encode(sentence, add_special_tokens=False)
    if isinstance(token_ids, tokenizers.Encoding):
        token_ids = token_ids.ids
    return token_ids


#input: list of articles id (from the data frame)
def map_tokenized_titles(article_ids_list):
    return [articles_dict[article_id] for article_id in article_ids_list if article_id in articles_dict]




#4. FINAL DATASET

#4.1 CREATE DICTIONARIES
#DICT 1: ARTICLE ID AND ITS TOKENIZATION (same for train and validation)
df_articles['title_tokens'] = df_articles['title'].apply(glove_tok)
articles_dict = df_articles.set_index('article_id')['title_tokens'].to_dict()
#DICT 2: USER ID AND ITS HISTORY ALREADY TOKENIZED (different for train and validation)
df_history_train['browsed_news'] = df_history_train['browsed_news'].apply(map_tokenized_titles)
history_dict_train=df_history_train.set_index('user_id')['browsed_news'].to_dict()
pprint(history_dict_train)

df_history_validation['browsed_news'] = df_history_validation['browsed_news'].apply(map_tokenized_titles)
history_dict_validation=df_history_validation.set_index('user_id')['browsed_news'].to_dict()
pprint(history_dict_validation)



#STEP 2: DATA FRAME WITH TOKEN IDS
#1ST COLUMN: USER ID
#2ND COLUMN: LIST OF ARTCILES INVIWED
#3RD COLUMN: ART CLICKED



input_data_train=df_behaviors_train.copy()
input_data_train['candidate_news'] = input_data_train['candidate_news'].apply(map_tokenized_titles)
input_data_train['article_ids_clicked'] = input_data_train['article_ids_clicked'].apply(map_tokenized_titles)

input_data_validation=df_behaviors_validation.copy()
input_data_validation['candidate_news'] = input_data_validation['candidate_news'].apply(map_tokenized_titles)
input_data_validation['article_ids_clicked'] = input_data_validation['article_ids_clicked'].apply(map_tokenized_titles)



#STEP 3: USER BROWSED HISTORY
#????????????
