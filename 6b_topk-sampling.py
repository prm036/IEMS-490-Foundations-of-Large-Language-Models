import torch
import torch.nn.functional as F

def top_k_sampling(logits, k=50, temperature=1.0):
    """
    Implement top-k sampling for a pre-trained language model.
    Args:
        logits: The logits from the model.
        k: The number of top-k tokens to consider.
        temperature: The temperature to use for sampling.
    Returns:
        The sampled token.

    This function restricts the next-token distribution to the k most 
    probable tokens. Higher temperatures yield more random outcomes.  
    The logits are divided by the provided temperature.  The top-k logits 
    and their corresponding indices are selected.  A softmax function is 
    applied exclusively over these top-k logits to ensure a valid probability 
    distribution.  The next token is sampled from this distribution using 
    torch.multinomial.  The sampled index is mapped back to its original 
    vocabulary index using the gather function.  
    """
    # 1. Divide logits by temperature
    logits = logits / temperature
    
    # 2. Select the top-k logits and their indices
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    
    # 3. Apply softmax over only the top-k logits to get a valid distribution
    probs = F.softmax(top_k_logits, dim=-1)
    
    # 4. Sample from this distribution using torch.multinomial
    next_token_top_k = torch.multinomial(probs, 1)
    
    # 5. Map the sampled index back to the original vocabulary index
    next_token = torch.gather(top_k_indices, -1, next_token_top_k)
    
    return next_token