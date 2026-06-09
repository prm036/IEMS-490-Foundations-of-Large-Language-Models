import torch
import torch.nn.functional as F

def speculative_decoding(
    big_model, small_model, prompt, seq_len, gamma=5, 
    temperature=1.0, top_p=0.9, pad_id=0
):
    """
    This function accelerates generation by utilizing a smaller draft model to 
    autoregressively propose a set number of gamma tokens. These drafted tokens 
    are then verified by a larger model in a single forward pass. Each drafted 
    token is either accepted or rejected based on a probability threshold 
    comparing the large model's output to the small model's output. Upon 
    encountering the first rejection, the sequence is truncated, and a new token 
    is sampled directly from the large model. If all drafted tokens are successfully 
    accepted, one additional token is sampled from the large model to continue 
    the sequence.
    """
    batch, prompt_len = prompt.shape
    device = prompt.device
    out = prompt.clone()
    
    # Initialize caches for both models
    cache_big, cache_small = None, None
    seq_lens = torch.full((batch,), prompt_len, device=device)
    
    # Continue generating until the target sequence length is reached
    while (seq_lens < seq_len).any():
        
        # Step 1: Draft gamma tokens sequentially using the small model
        samples = []
        for _ in range(gamma):
            # Run the small model
            logits_small, cache_small = small_model(
                out, return_cache=True, cache=cache_small
            )
            # Take logits for the last token to predict the next
            next_logits = logits_small[:, -1]
            
            # Assuming top_p_sampling is an available helper function
            # next_token = top_p_sampling(next_logits, top_p=top_p) 
            # (Replaced below with a simple argmax/multinomial for standalone compatibility if top_p_sampling is undefined)
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append drafted token to the sequence
            out = torch.cat((out, next_token), dim=-1)
            samples.append(next_token)
            
        # Stack the drafted tokens
        samples = torch.stack(samples, dim=1).squeeze(-1) 
        
        # Step 2: Verify the drafted tokens simultaneously with the big model
        logits_big, cache_big = big_model(
            out, return_cache=True, cache=cache_big
        )
        
        # Extract the logits corresponding specifically to the drafted sequence
        # We need the logits that PREDICTED the drafted tokens
        logits_big_draft = logits_big[:, -gamma-1 : -1]
        logits_small_draft = logits_small[:, -gamma:]
        
        # Calculate probabilities
        prob_big = F.softmax(logits_big_draft / temperature, dim=-1)
        prob_small = F.softmax(logits_small_draft / temperature, dim=-1)
        
        # Step 3: Acceptance check
        # Get the specific probabilities of the actual sampled tokens
        p = prob_big.gather(-1, samples.unsqueeze(-1)).squeeze(-1)
        q = prob_small.gather(-1, samples.unsqueeze(-1)).squeeze(-1)
        
        # Accept condition: random uniform < min(1, p/q)
        accept = torch.rand_like(p) < (p / (q + 1e-8))
        
        # Step 4: Handle rejections to find the first rejected token in the sequence
        reject_mask = ~accept
        
        # Find the index of the first rejection. If none rejected, default to gamma
        first_reject_idx = torch.where(
            reject_mask.any(dim=1),
            reject_mask.float().argmax(dim=1),
            torch.full((accept.size(0),), gamma, device=accept.device, dtype=torch.long)
        )
        
        # Step 5: Remove rejected tokens from the sequence
        # Update sequence lengths based on where the rejection occurred
        seq_lens = (prompt_len + first_reject_idx).clamp(max=out.size(1))
        Lmax = int(seq_lens.max().item())
        
        # Truncate sequence to the maximum valid length in the batch
        out = out[:, :Lmax].contiguous()
        
        # Apply padding for sequences in the batch that were truncated earlier
        pos = torch.arange(Lmax, device=out.device).unsqueeze(0)
        mask = pos < seq_lens.unsqueeze(1)
        out = torch.where(mask, out, torch.full_like(out, pad_id))
        
        # Step 6: Sample one true token directly from the big model's distribution
        # using the valid logits at the truncation point
        prob_next = prob_big[:, -1]
        next_token = torch.multinomial(prob_next, 1)
        out = torch.cat((out, next_token), dim=-1)
        
        # Update tracked lengths and pad out appropriately
        seq_lens += 1
        out = F.pad(out, (0, seq_len - out.shape[1]), value=pad_id)
        
    # Return the generated tokens, excluding the original prompt
    return out[:, prompt_len:]