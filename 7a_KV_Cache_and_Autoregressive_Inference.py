import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# Implements a Key-Value (KV) cache to improve autoregressive 
# inference efficiency within a Transformer model. Without a cache, keys and 
# values for all previous tokens would need to be recomputed at every generation 
# step. By storing previously computed K and V tensors in pre-allocated buffers, 
# the model only computes them for new tokens and concatenates them with the 
# cached data. The forward_inference function manages this pass, including 
# building a causal mask for prefill stages which consists of an upper-triangular 
# -inf matrix prepended with zeros for already cached positions
# ==========================================

class KVCache(nn.Module):
    def __init__(self, batch_size, seq_length, n_kv_heads, head_dim, dtype, device):
        super().__init__()
        # Pre-allocate key and value cache buffers.
        cache_shape = (batch_size, seq_length, n_kv_heads, head_dim)
        
        # Use register_buffer so these tensors are part of the module state 
        # but are not treated as trainable parameters.
        self.register_buffer(
            "cache_k", torch.zeros(cache_shape, dtype=dtype, device=device)
        )
        self.register_buffer(
            "cache_v", torch.zeros(cache_shape, dtype=dtype, device=device)
        )

    def update(self, start_pos, xk, xv):
        """
        Write new keys/values into the cache and return all cached K, V up to the current position.
        """
        seqlen = xk.size(1)
        
        # 1. & 2. Write new xk and xv into the cache at the correct sequence positions
        self.cache_k[:, start_pos : start_pos + seqlen] = xk
        self.cache_v[:, start_pos : start_pos + seqlen] = xv
        
        # 3. Return the full cache up to the new current position
        return (
            self.cache_k[:, : start_pos + seqlen],
            self.cache_v[:, : start_pos + seqlen]
        )


def forward_inference(model, tokens, start_pos):
    """
    Transformer forward pass for inference utilizing the KV cache.
    """
    _bsz, seqlen = tokens.shape
    
    # 1. Embed the tokens
    h = model.tok_embeddings(tokens)
    
    # 2. Slice RoPE (Rotary Position Embeddings) frequencies for the current window
    freqs_cis = model.freqs_cis[start_pos : start_pos + seqlen]
    
    # 3. Build the causal mask
    mask = None
    if seqlen > 1: # Indicates prefill phase
        # Create an upper triangular matrix filled with -inf
        mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1)
        
        # Prepend zeros of width start_pos to account for positions already in the KV cache
        mask = torch.hstack([
            torch.zeros((seqlen, start_pos), device=tokens.device), 
            mask
        ]).type_as(h)
        
    # 4. Pass through all transformer layers
    for layer in model.layers:
        h = layer(h, start_pos, freqs_cis, mask)
        
    # 5. Apply final normalization
    h = model.norm(h)
    
    # 6. Compute final output logits
    output = model.output(h).float()
    
    return output