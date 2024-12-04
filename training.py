from model import *
from data_loader import *
from neg_sample import *
import matplotlib.pyplot as plt


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

#model
nrms_model=NRMS(embed_size=300, heads=15, word_embedding_matrix=glove_vectors, attention_dim=200)
# Optimizer for model
optimizer = torch.optim.Adam(nrms_model.parameters(), lr=0.001)

# Initialize a list to store epoch losses
epoch_losses = []


for epoch in range(num_epochs):
    
    nrms_model.train()
    epoch_loss = 0.0

    for batch in small_train_loader:
        # Extract batch data
        user_histories = batch['browsed_news']  # [batch_size, num_browsed, num_words]
        candidate_news = batch['candidate_news']  # [batch_size, num_candidates, num_words]
        labels = batch['clicked_idx']  # [batch_size]

        print(f"Size of user_histories: {user_histories.size()}")
        print(f"Size of candidate_news: {candidate_news.size()}")        

        #get click prob
        click_prob=nrms_model(user_histories,candidate_news)

        loss=negative_sampling(click_prob, labels,K)
        
        #backward pass and optimization
        #clear previous gradients
        optimizer.zero_grad()  
        loss.backward()   
        #update parameters    
        optimizer.step()  

        #save loss value to calculate for the whole epoch
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