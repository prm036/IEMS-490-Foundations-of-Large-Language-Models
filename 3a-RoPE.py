import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# RoPE encodes token positions by rotating pairs of dimensions in the query and 
# key vectors. This technique is widely used in modern Large Language Models like 
# Llama. The precompute_freqs_cis function calculates and stores the rotation 
# frequencies (sine and cosine tables) for all positions to save compute during 
# runtime. The apply_rotary_emb function splits the embedding dimensions into two 
# halves, applies a rotational transformation, and concatenates the output
# ==========================================

def precompute_freqs_cis(head_dim, seq_len, base=10000.0):
    """
    Precompute cosine and sine tables for RoPE.
    """
    # 1. & 2. Compute inverse frequencies for the given base
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    
    # 3. Create a position tensor from 0 to sequence length
    t = torch.arange(seq_len, dtype=torch.float32)
    
    # 4. Compute the outer product to get the rotational frequencies 
    # Resulting shape: (seq_len, head_dim/2)
    freqs = torch.outer(t, inv_freq) 
    
    # 5. Extract cosine and sine values
    cos, sin = freqs.cos(), freqs.sin()
    
    # 6. Reshape into (1, seq_len, 1, head_dim/2) to allow for broadcasting
    # across batch and head dimensions during the forward pass
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    
    return cos, sin

def apply_rotary_emb(x, cos, sin):
    """
    Apply rotary positional embeddings to input tensor.
    """
    assert x.ndim == 4
    
    # 1. Find the split point (half of the head dimension)
    d = x.shape[3] // 2 
    
    # 2. Split the last dimension into two halves
    x1, x2 = x[..., :d], x[..., d:]
    
    # 3. & 4. Apply the rotary transformation
    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos
    
    # 5. Concatenate the rotated halves back together along the last dimension
    return torch.cat([y1, y2], dim=3)
