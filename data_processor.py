
import pandas as pd
import torch
from pathlib import Path
import zipfile
from tqdm import tqdm
from typing import Tuple, List, Dict
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from pprint import pprint
import rich


class NewsDataProcessor:
    def __init__(self, file_path: str):
        """
        Initializes the NewsDataProcessor with a given file path.
        
        Args:
            file_path (str): Path to the dataset folder.
        """
        self.file_path = file_path
        self.df_behaviors_train = None
        self.df_behaviors_validation = None
        self.df_history_train = None
        self.df_history_validation = None
        self.df_articles = None
        self.glove_vocabulary = None
        self.glove_vectors = None
        self.glove_tokenizer = None
        self.articles_dict = None
        self.history_dict_train = None
        self.history_dict_validation = None

    def load_glove_vectors(self, filename: str = "glove.6B.300d.txt") -> Tuple[List[str], torch.Tensor]:
        """Load the GloVe vectors."""
        path = Path(self.file_path)
        target_file = path / filename
        if not target_file.exists():
            glove_zip = path / "glove.6B.zip"
            if not glove_zip.exists():
                raise FileNotFoundError(f"GloVe file `{glove_zip}` not found. Please ensure it exists in `{self.file_path}`.")
            with zipfile.ZipFile(glove_zip, 'r') as zip_ref:
                zip_ref.extractall(path)

        vocabulary = []
        vectors = []
        with open(target_file, "r", encoding="utf8") as f:
            for l in tqdm(f.readlines(), desc=f"Parsing {target_file.name}..."):
                word, *vector = l.split()
                vocabulary.append(word)
                vectors.append(torch.tensor([float(v) for v in vector]))
        vectors = torch.stack(vectors)
        return vocabulary, vectors

    def glove_tok(self, sentence: str) -> List[int]:
        """Tokenizes a sentence using the GloVe tokenizer."""
        token_ids = self.glove_tokenizer.encode(sentence, add_special_tokens=False)
        return token_ids.ids if isinstance(token_ids, tokenizers.Encoding) else token_ids

    def map_tokenized_titles(self, article_ids_list: List[str]) -> List[List[int]]:
        """Maps tokenized titles based on article IDs."""
        return [self.articles_dict[article_id] for article_id in article_ids_list if article_id in self.articles_dict]

    def process_data(self) -> Tuple[pd.DataFrame, Dict[str, List[int]], Dict[str, List[int]]]:
        """
        Processes the data and returns the final dataset components.

        Returns:
            Tuple containing:
            - df_behaviors_train: DataFrame with behaviors for training.
            - history_dict_train: Dictionary of user history for training.
            - history_dict_validation: Dictionary of user history for validation.
        """
        # Load data
        self.df_behaviors_train = pd.read_parquet(f"{self.file_path}/train/behaviors.parquet")
        self.df_behaviors_validation = pd.read_parquet(f"{self.file_path}/validation/behaviors.parquet")
        self.df_history_train = pd.read_parquet(f"{self.file_path}/train/history.parquet")
        self.df_history_validation = pd.read_parquet(f"{self.file_path}/validation/history.parquet")
        self.df_articles = pd.read_parquet(f"{self.file_path}/articles.parquet")

        # Process behaviors data
        self.df_behaviors_train = self.df_behaviors_train[['user_id', 'article_ids_inview', 'article_ids_clicked']].dropna().rename(columns={'article_ids_inview': 'candidate_news'})
        self.df_behaviors_validation = self.df_behaviors_validation[['user_id', 'article_ids_inview', 'article_ids_clicked']].dropna().rename(columns={'article_ids_inview': 'candidate_news'})

        # Process history data
        self.df_history_train = self.df_history_train[['user_id', 'article_id_fixed']].dropna().rename(columns={'article_id_fixed': 'browsed_news'})
        self.df_history_validation = self.df_history_validation[['user_id', 'article_id_fixed']].dropna().rename(columns={'article_id_fixed': 'browsed_news'})

        # Process articles data
        self.df_articles = self.df_articles[['article_id', 'title']]

        # Load GloVe vectors
        self.glove_vocabulary, self.glove_vectors = self.load_glove_vectors()

        # Add special tokens
        special_tokens = ['<|start|>', '<|unknown|>', '<|pad|>']
        self.glove_vocabulary = special_tokens + self.glove_vocabulary
        self.glove_vectors = torch.cat([torch.randn(len(special_tokens), self.glove_vectors.shape[1]), self.glove_vectors])

        # Initialize tokenizer
        self.glove_tokenizer = Tokenizer(models.WordLevel(vocab={v: i for i, v in enumerate(self.glove_vocabulary)}, unk_token="<|unknown|>"))
        self.glove_tokenizer.normalizer = normalizers.BertNormalizer(strip_accents=False)
        self.glove_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Create dictionaries
        self.df_articles['title_tokens'] = self.df_articles['title'].apply(self.glove_tok)
        self.articles_dict = self.df_articles.set_index('article_id')['title_tokens'].to_dict()

        self.df_history_train['browsed_news'] = self.df_history_train['browsed_news'].apply(self.map_tokenized_titles)
        self.history_dict_train = self.df_history_train.set_index('user_id')['browsed_news'].to_dict()

        self.df_history_validation['browsed_news'] = self.df_history_validation['browsed_news'].apply(self.map_tokenized_titles)
        self.history_dict_validation = self.df_history_validation.set_index('user_id')['browsed_news'].to_dict()

        return self.df_behaviors_train, self.history_dict_train, self.history_dict_validation