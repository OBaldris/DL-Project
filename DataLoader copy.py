
from torch.utils.data import Dataset, DataLoader
from model import *
from Data_loader import *
import matplotlib.pyplot as plt

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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)



print('Starting training...')

# Lightweight training loop for debugging
# num_epochs = 20  # Number of epochs for testing
# batch_size = 64  # Small batch size
# subset_size = len(train_loader.dataset  # Use only a small subset of the dataset
# K = 4  # Number of negative samples

num_epochs = 10  # Number of epochs for testing
batch_size = 32  # Small batch size
subset_size = 128  # Use only a small subset of the dataset
K = 4  # Number of negative samples

# Subset the train_loader for quick testing
small_train_dataset = torch.utils.data.Subset(train_loader.dataset, range(subset_size))
small_train_loader = torch.utils.data.DataLoader(small_train_dataset, batch_size=batch_size, shuffle=True)

# Instantiate the separate encoders
news_encoder = NewsEncoder(embed_size=300, heads=4, word_embedding_matrix=glove_vectors, attention_dim=128)
user_encoder = UserEncoder(embed_size=300, heads=4, attention_dim=128)

# Optimizer for both encoders
optimizer = torch.optim.Adam(list(news_encoder.parameters()) + list(user_encoder.parameters()), lr=0.001)

# Initialize a list to store epoch losses
epoch_losses = []


for epoch in range(num_epochs):
    news_encoder.train()
    user_encoder.train()
    epoch_loss = 0.0

    for batch in small_train_loader:
        # Extract batch data
        user_histories = batch['browsed_news']  # [batch_size, num_browsed, num_words]
        candidate_news = batch['candidate_news']  # [batch_size, num_candidates, num_words]
        labels = batch['clicked_idx']  # [batch_size]

        # Encode browsed news using the news_encoder
        browsed_news_encoded = torch.stack(
            [news_encoder(news) for news in user_histories], dim=0
        )  # [batch_size, num_browsed, embed_size] - OK

        # Aggregate encoded browsed news into user embedding
        user_embedding = user_encoder(browsed_news_encoded)  # [batch_size, embed_size] - OK

        # Encode candidate news
        candidate_news_encoded = torch.stack(
            [news_encoder(news) for news in candidate_news], dim=0
        )  # [batch_size, num_candidates, embed_size] - OK

        # Get dimensions
        batch_size, num_candidates, embed_size = candidate_news_encoded.size()
        # batch_size, 10, 300

        # Get the indices of the clicked news
        positive_indices = torch.argmax(labels, dim=1)  # [batch_size] - OK

        # Extract embeddings of the clicked (positive) news
        positive_embeddings = candidate_news_encoded[
            torch.arange(batch_size), positive_indices
        ]  # [batch_size, embed_size] - OK

        # Create a mask to exclude the positive (clicked) news
        mask = torch.ones(num_candidates, dtype=torch.bool, device=candidate_news_encoded.device)
        mask[positive_indices] = False  # Mask out the clicked news

        # Extract the non-clicked news
        non_clicked_news = candidate_news_encoded[:, mask]  # [batch_size, num_candidates-1, embed_size] -OK

        # Randomly select K negatives (non-clicked news) for each user
        K = 4
        negative_indices = torch.randint(0, non_clicked_news.size(1), (batch_size, K), device=non_clicked_news.device)
        negative_embeddings = non_clicked_news[
            torch.arange(batch_size).unsqueeze(1), negative_indices
        ]  # [batch_size, K, embed_size] - OK

        # Concatenate positive and negative embeddings
        all_embeddings = torch.cat(
            [positive_embeddings.unsqueeze(1), negative_embeddings], dim=1
        )  # [batch_size, K+1, embed_size] - OK

        # Compute scores
        scores = torch.bmm(all_embeddings, user_embedding.unsqueeze(2)).squeeze(2)  # [batch_size, K+1] - OK
        probabilities = F.softmax(scores, dim=1)  # [batch_size, K+1]

        # Loss computation (negative log-likelihood for positive samples)
        positive_probs = probabilities[:, 0]  # Positive sample probabilities
        loss = -torch.log(positive_probs).mean()  # Mean negative log-likelihood

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()

    # Store average loss for the epoch
    avg_loss = epoch_loss / len(small_train_loader)
    epoch_losses.append(avg_loss)

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(small_train_loader)}")

print('End of training')


# Plot the loss function
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()