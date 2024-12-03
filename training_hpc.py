from model import *
from data_loader import *
import argparse
import matplotlib.pyplot as plt

print('Starting training...')

# Lightweight training loop for debugging
# num_epochs = 20  # Number of epochs for testing
# batch_size = 64  # Small batch size
# subset_size = len(train_loader.dataset  # Use only a small subset of the dataset
# K = 4  # Number of negative samples

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('--num_epochs', type=int, default=5000, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--subset_size', type=int, default=0, help='Subset size of the dataset for training')
parser.add_argument('--K', type=int, default=4, help='Number of negative samples')
args = parser.parse_args()

num_epochs = args.num_epochs
batch_size = args.batch_size
subset_size = args.subset_size
K = args.K

print(f'Arguments: num_epochs={num_epochs}, batch_size={batch_size}, subset_size={subset_size}, K={K}')

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# Subset the train_loader for quick testing
if subset_size > 0:
    small_train_dataset = torch.utils.data.Subset(train_loader.dataset, range(subset_size))
    small_train_loader = torch.utils.data.DataLoader(small_train_dataset, batch_size=batch_size, shuffle=True)
else:
    small_train_loader = train_loader

# Model
nrms_model = NRMS(embed_size=300, heads=4, word_embedding_matrix=glove_vectors, attention_dim=128).to(device)

# Optimizer for model
optimizer = torch.optim.Adam(nrms_model.parameters(), lr=0.001)

# Initialize a list to store epoch losses
epoch_losses = []

for epoch in range(num_epochs):
    
    nrms_model.train()
    epoch_loss = 0.0

    for batch in small_train_loader:
        # Extract batch data and move to device
        user_histories = batch['browsed_news'].to(device)  # [batch_size, num_browsed, num_words]
        candidate_news = batch['candidate_news'].to(device)  # [batch_size, num_candidates, num_words]
        labels = batch['clicked_idx'].to(device)  # [batch_size]

        #print(f"Size of user_histories: {user_histories.size()}")
        #print(f"Size of candidate_news: {candidate_news.size()}")        

        # Get click prob
        click_prob = nrms_model(user_histories, candidate_news)

        # Get click prob of positive samples
        # One per batch 
        no_batches, no_candidate_news = click_prob.size()
        positive_index = torch.arange(no_batches), torch.argmax(labels, dim=1)
        positive_sample = click_prob[positive_index]

        # Get click prob of negative samples 
        # More than one per batch
        mask = torch.ones_like(click_prob, dtype=torch.bool)
        mask[positive_index] = False
        negative_samples = click_prob[mask].view(no_batches, -1)

        # Select K random negative samples
        if K > no_candidate_news:
            raise ValueError("K cannot be larger than the size of the tensor.")
        
        # Use randperm instead of randint so that we don't have repetitions
        random_negative_indices = torch.randperm(no_candidate_news)[:K] 
        # Neg samples for all users (using the same indexes)
        negative_samples = click_prob[:, random_negative_indices]  # [batch_size, K]

        # Compute posterior prob for the positive sample
        exp_pos = torch.exp(positive_sample)  # [batch_size]
        exp_neg = torch.exp(negative_samples)  # [batch_size, K]
        sum_exp_neg = torch.sum(exp_neg, dim=1)  # [batch_size]
        pi_positive = exp_pos / (exp_pos + sum_exp_neg)  # [batch_size]

        # Average loss across the batch
        loss = -torch.log(pi_positive).mean()
        
        # Backward pass and optimization
        # Clear previous gradients
        optimizer.zero_grad()  
        loss.backward()   
        # Update parameters    
        optimizer.step()  

        # Save loss value to calculate for the whole epoch
        epoch_loss += loss.item()

    # Store average loss for the epoch
    avg_loss = epoch_loss / len(small_train_loader)
    epoch_losses.append(avg_loss)

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(small_train_loader)}")

#save model to disk
torch.save(nrms_model.state_dict(), 'nrms_model.pth')

#save loss values to disk
with open('epoch_losses.txt', 'w') as f:
    for item in epoch_losses:
        f.write("%s\n" % item)

print('End of training')
# Save the loss function plot to a file
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.savefig('training_loss_plot.png')
