


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch in train_loader:  # Assuming DataLoader for batching
        user_histories, candidate_news, labels = batch
        
        # 1. Forward pass
        user_embeddings = model.user_encoder(user_histories)
        candidate_news_embeddings = model.news_encoder(candidate_news)
        
        # Positive and Negative samples
        positive_scores = torch.matmul(user_embeddings, candidate_news_embeddings_positive.T)
        negative_scores = torch.matmul(user_embeddings, candidate_news_embeddings_negative.T)
        
        # Compute loss
        all_scores = torch.cat([positive_scores, negative_scores], dim=0)
        loss = torch.nn.CrossEntropyLoss()(all_scores, labels)
        
        # 2. Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 3. Update weights
        optimizer.step()
        
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")




model.eval()  # Set to evaluation mode
with torch.no_grad():
    for batch in val_loader:
        user_histories, candidate_news, labels = batch
        
        user_embeddings = model.user_encoder(user_histories)
        candidate_news_embeddings = model.news_encoder(candidate_news)
        
        predictions = torch.matmul(user_embeddings, candidate_news_embeddings.T)
        # Calculate AUC, nDCG, etc., here


#torch.save(model.state_dict(), "nrms_model.pth")