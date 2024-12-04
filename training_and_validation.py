from model import *
from data_loader import *
from functions import *
import matplotlib.pyplot as plt
from hyperparameters import *


print('Starting training with validation...')

# Lightweight training loop for debugging


# Subset the train_loader for quick testing
small_train_dataset = torch.utils.data.Subset(train_loader.dataset, range(subset_size))
small_train_loader = torch.utils.data.DataLoader(small_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Subset the validation_loader for quick testing
small_val_dataset = torch.utils.data.Subset(validation_loader.dataset, range(subset_size))
small_val_loader = torch.utils.data.DataLoader(small_val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nrms_model = NRMS(embed_size=embed_size, heads=heads, word_embedding_matrix=glove_vectors, attention_dim=attention_dim).to(device)

# Optimizer for model
optimizer = torch.optim.Adam(nrms_model.parameters(), lr=learning_rate)

# Initialize lists to store epoch losses
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    
    # Training phase
    nrms_model.train()
    train_epoch_loss = 0.0

    for batch in small_train_loader:
        # Extract batch data
        user_histories = batch['browsed_news'].to(device)  # [batch_size, num_browsed, num_words]
        candidate_news = batch['candidate_news'].to(device)  # [batch_size, num_candidates, num_words]
        labels = batch['clicked_idx'].to(device)  # [batch_size]

        #print(f"Size of user_histories: {user_histories.size()}")
        #print(f"Size of candidate_news: {candidate_news.size()}")        

        # Get click probabilities and compute loss
        click_prob = nrms_model(user_histories, candidate_news)
        loss = negative_sampling(click_prob, labels, K)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss for the epoch
        train_epoch_loss += loss.item()

    # Average training loss for the epoch
    train_avg_loss = train_epoch_loss / len(small_train_loader)
    train_losses.append(train_avg_loss)

    # Validation phase
    nrms_model.eval()
    val_epoch_loss = 0.0

    with torch.no_grad():
        for batch in small_val_loader:
            # Extract batch data
            user_histories = batch['browsed_news']  # [batch_size, num_browsed, num_words]
            candidate_news = batch['candidate_news']  # [batch_size, num_candidates, num_words]
            labels = batch['clicked_idx']  # [batch_size]

            # Get click probabilities and compute loss
            click_prob = nrms_model(user_histories, candidate_news)
            loss = negative_sampling(click_prob, labels, K)

            # Accumulate validation loss
            val_epoch_loss += loss.item()

    # Average validation loss for the epoch
    val_avg_loss = val_epoch_loss / len(small_val_loader)
    val_losses.append(val_avg_loss)

    print(f"Epoch {epoch + 1}, Training Loss: {train_avg_loss}, Validation Loss: {val_avg_loss}")

print('End of training')

# Plot the loss function
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, marker='o', label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
