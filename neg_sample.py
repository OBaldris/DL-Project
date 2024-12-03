import torch

#labels = batch['clicked_idx']  # [batch_size]
#click_prob=nrms_model(user_histories,candidate_news)
#K=4

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
        