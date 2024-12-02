
from model import *
from data_loader2 import *
import matplotlib.pyplot as plt


print('Starting training...')

# Lightweight training loop for debugging
num_epochs = 10  # Number of epochs for testing
batch_size = 32  # Small batch size
subset_size = 128  # Use only a small subset of the dataset
K = 4  # Number of negative samples

# num_epochs = 10  # Number of epochs for testing
# batch_size = 32  # Small batch size
# subset_size = 128  # Use only a small subset of the dataset
# K = 4  # Number of negative samples

# Subset the train_loader for quick testing
small_train_dataset = torch.utils.data.Subset(train_loader.dataset, range(subset_size))
small_train_loader = torch.utils.data.DataLoader(small_train_dataset, batch_size=batch_size, shuffle=True)

#model
nrms_model=NRMS(embed_size=300, heads=15, word_embedding_matrix=fasttext_vectors, attention_dim=128)

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

        #print(f"Size of user_histories: {user_histories.size()}")
        #print(f"Size of candidate_news: {candidate_news.size()}")        

        #get click prob
        click_prob=nrms_model(user_histories,candidate_news)

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
        K = 6
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