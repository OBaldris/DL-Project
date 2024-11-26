from Data_loader import *
from Final_Model import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

print("input_data_train: ", input_data_train.head())
print("\nbrowsed news: ", browsed_news_train.shape
      , "\ncandidate news: ", candidate_news_train.shape
      , "\nclicked news: ", clicked_news_train.shape)


# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available! Your model will use the GPU.")
else:
    print("CUDA is not available! Your model will run on the CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Set the device to CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model_final = NRMS(embed_size=300, heads=15, word_embedding_matrix=glove_vectors, attention_dim=200)

# Move the model to the appropriate device (GPU or CPU)
model_final.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_final.parameters(), lr=0.001)

# Training loop
num_epochs = 100
batch_size = 64

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}")
    total_loss = 0
    
    for i in range(0, len(browsed_news_train), batch_size):
        # Fetch batch data
        browsed_news_batch = browsed_news_train[i:i+batch_size, :, :]
        candidate_news_batch = candidate_news_train[i:i+batch_size, :, :]
        clicked_news_batch = clicked_news_train[i:i+batch_size, :]

        # Forward pass
        click = model_final(browsed_news_batch, candidate_news_batch)

        # Compute loss
        loss = criterion(click, torch.argmax(clicked_news_batch, dim=1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print batch progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i//batch_size+1}/{len(browsed_news_train)//batch_size}], Loss: {loss.item():.4f}")
    
    # Print epoch loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss/len(browsed_news_train):.6f}")
    
print("Training finished")