import os
from pathlib import Path
import zipfile
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from huggingface_hub import hf_hub_download
import tokenizers
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
import matplotlib.pyplot as plt
import seaborn as sns


def one_hot_encode(candidate_news, clicked_news):
    return [1 if num == clicked_news[0] else 0 for num in candidate_news]



glove_save_path = "../Data/glove_vectors.pt"
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



def count_non_zero_tokens(tokens):
    return sum(1 for token in tokens if token != 0)



def plot_title_size_distribution(df, title_column='title'):

    # Calculate the number of non-zero tokens for each title
    df.loc[:,'title_size'] = df[title_column].apply(count_non_zero_tokens)

    # Plot the distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['title_size'], bins=10, kde=True, color='blue', edgecolor='black')

    # Add labels and title
    plt.xlabel('Title Size (number of non-zero tokens)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Title Sizes (Non-zero Tokens)', fontsize=14)

    # Show the plot
    plt.show()


#-----------OLD DIDNT TRUNCATE----------------
# def truncate_to_n_tokens(tokens, n):
#     # Keep only the first n non-zero tokens
#     non_zero_tokens = [token for token in tokens if token != 0]
#     return non_zero_tokens[:n] + [0] * (len(tokens) - len(non_zero_tokens[:n]))  # pad with zeros if needed


#-----------NEW TRUNCATES-------------
def truncate_to_n_tokens(tokens, n):
    # Keep only the first n non-zero tokens
    non_zero_tokens = [token for token in tokens if token != 0]
    
    # If there are fewer than n non-zero tokens, pad with zeros to make it n tokens
    if len(non_zero_tokens) < n:
        return non_zero_tokens + [0] * (n - len(non_zero_tokens))
    
    # Otherwise, return the first n non-zero tokens
    return non_zero_tokens[:n]




def calculate_statistics(input_data, dataset_name="Dataset", plot_distributions=False):
    # Calculate lengths of browsed_news and candidate_news
    browsed_news_lengths = input_data['browsed_news'].apply(len)
    candidate_news_lengths = input_data['candidate_news'].apply(len)

    # Compute statistics
    stats = {
        'browsed_news': {
            'min': browsed_news_lengths.min(),
            'max': browsed_news_lengths.max(),
            'mean': browsed_news_lengths.mean(),
            'std': browsed_news_lengths.std(),
        },
        'candidate_news': {
            'min': candidate_news_lengths.min(),
            'max': candidate_news_lengths.max(),
            'mean': candidate_news_lengths.mean(),
            'std': candidate_news_lengths.std(),
        }
    }

    # Print statistics for better visibility
    print("\n")
    print(f"Statistics for {dataset_name}:")
    print(f"Browsed News - Min: {stats['browsed_news']['min']}, Max: {stats['browsed_news']['max']}, "
          f"Mean: {stats['browsed_news']['mean']:.2f}, Std: {stats['browsed_news']['std']:.2f}")
    print(f"Candidate News - Min: {stats['candidate_news']['min']}, Max: {stats['candidate_news']['max']}, "
          f"Mean: {stats['candidate_news']['mean']:.2f}, Std: {stats['candidate_news']['std']:.2f}\n")
    
    # Plot distributions
    if not plot_distributions:
        return stats
    
 # Plot the distributions in subplots
    plt.figure(figsize=(14, 6))

    # Subplot for browsed_news lengths
    plt.subplot(1, 2, 1)
    sns.histplot(browsed_news_lengths, bins=10, kde=True, color='blue', edgecolor='black')
    plt.xlabel('Browsed News Length (Number of Tokens)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of Browsed News Lengths ({dataset_name})', fontsize=14)

    # Subplot for candidate_news lengths
    plt.subplot(1, 2, 2)
    sns.histplot(candidate_news_lengths, bins=10, kde=True, color='green', edgecolor='black')
    plt.xlabel('Candidate News Length (Number of Tokens)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of Candidate News Lengths ({dataset_name})', fontsize=14)

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

    return stats



def negative_sampling(click_prob, labels,K):

    #get click prob of positive samples
    #one per batch 
    no_batches, no_candidate_news = click_prob.size()
    positive_index = torch.arange(no_batches), torch.argmax(labels, dim=1)
    positive_sample = click_prob[positive_index]

    #get click prob of negative samples 
    #more than one per batch
    mask=torch.ones_like(click_prob, dtype=torch.bool)
    mask[positive_index] = False
    negative_samples = click_prob[mask].view(no_batches, -1)

    #select K random negative samples
    if K > no_candidate_news:
        raise ValueError("K cannot be larger than the size of the tensor.")
        
    #use randperm instead of randint so that we dont have repetitions
    random_negative_indices = torch.randperm(no_candidate_news)[:K] 
    #neg samples for all users (using the same indexes)
    negative_samples = click_prob[:, random_negative_indices]  # [batch_size, K]

    #compute posterior prob for the possitive sample
    exp_pos=torch.exp(positive_sample)  # [batch_size]
    exp_neg=torch.exp(negative_samples)  # [batch_size, K]
    sum_exp_neg=torch.sum(exp_neg, dim=1)  # [batch_size]
    pi_positive=exp_pos/(exp_pos + sum_exp_neg)  # [batch_size]

    #average loss across the batch
    loss=-torch.log(pi_positive).mean()

    return loss
        