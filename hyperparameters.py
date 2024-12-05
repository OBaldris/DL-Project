file_path = "../Data/ebnerd_demo"

max_words_articles = 15
max_num_browsed = 200
max_num_candidates = 25

num_epochs = 25  # Number of epochs for testing
batch_size = 128  # Small batch size
subset_size = 128  # Use only a small subset of the dataset
K = 4  # Number of negative samples

embed_size = 300
heads = 15
attention_dim=200
learning_rate = 0.01
weight_decay = 1e-5

"""
Remember to save before running 
"""