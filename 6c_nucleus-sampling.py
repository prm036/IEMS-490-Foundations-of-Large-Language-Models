import torch

def sample_top_p(probs, p):
    """
    Implement nucleus sampling for a pre-trained language model.
    Args:
        probs: The probability distribution from the model.
        p: The nucleus probability threshold.
    Returns:
        The sampled token.

    Nucleus sampling dynamically selects the smallest set of tokens that 
    cumulatively exceed the threshold probability p.  The initial probabilities 
    are sorted in descending order.  The cumulative sum of these sorted 
    probabilities is computed.  A mask is created to identify tokens where 
    the cumulative sum minus the current probability exceeds the threshold. 
    The probabilities of the masked tokens are zeroed out.  The remaining 
    unmasked probabilities are renormalized.  A token is sampled from the 
    newly filtered distribution.  The sampled token's index is mapped back 
    to the original vocabulary index using the gather function.  
    """
    # 1. Sort probabilities in descending order
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    
    # 2. Compute the cumulative sum of sorted probabilities
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    
    # 3. Create a mask for tokens where (cumulative_sum - current_prob) > p
    mask = (probs_sum - probs_sort) > p
    
    # 4. Zero out the masked tokens
    probs_sort[mask] = 0.0
    
    # 5. Renormalize the remaining probabilities
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    
    # 6. Sample from the filtered distribution
    next_token_sorted = torch.multinomial(probs_sort, num_samples=1)
    
    # 7. Map the sampled index back to the original vocabulary index using gather
    next_token = torch.gather(probs_idx, -1, next_token_sorted)
    
    return next_token