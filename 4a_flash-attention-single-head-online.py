import numpy as np
import math

"""
Implement a chunked online attention mechanism (similar to FlashAttention) for a single head.
    The function processes key-value blocks in chunks, maintaining running statistics
    (max, sum-of-exp, and weighted output) to avoid materializing the full attention matrix.
    Support an optional causal mask.
"""

def attention_online(q, k, v, chunk_size=1024, causal=False):
    """
    Online (chunked) attention for single-head, batch_size=1.
    """
    # Get dimensions
    Lq, D = q.shape
    Lk, Dv = v.shape
    
    scale = 1.0 / math.sqrt(D)
    
    # Maintain running statistics
    m = np.full((Lq, 1), -np.inf) # Running max
    l = np.zeros((Lq, 1))         # Running sum of exp
    o = np.zeros((Lq, Dv))        # Running numerator
    
    # Iterate through key-value matrices in chunks
    for i in range(0, Lk, chunk_size):
        k_block = k[i:i + chunk_size]
        v_block = v[i:i + chunk_size]
        
        # 1. Compute scores
        scores = (q @ k_block.T) * scale
        
        # 2. If causal, mask out future positions with -inf
        if causal:
            # Create grids of sequence indices for queries and the current key block
            q_pos = np.arange(Lq)[:, None]
            k_pos = np.arange(i, i + k_block.shape[0])[None, :]
            
            # Mask out where the key position comes after the query position
            mask = k_pos > q_pos
            scores[mask] = -np.inf
            
        # 3. Update m_new = max(m, block_max)
        block_max = np.max(scores, axis=-1, keepdims=True)
        m_new = np.maximum(m, block_max)
        
        # 4. Compute exp_scores
        exp_scores = np.exp(scores - m_new)
        
        # 5. Rescale old stats: alpha = exp(m - m_new)
        # We use np.where to prevent NaN corruption when m and m_new are both -inf 
        # (which happens on fully masked rows in causal attention)
        alpha = np.where(m_new == -np.inf, 0.0, np.exp(m - m_new))
        
        # 6. Update l = l * alpha + sum(exp_scores)
        l = l * alpha + np.sum(exp_scores, axis=-1, keepdims=True)
        
        # 7. Update o = o * alpha + exp_scores @ v_block
        o = o * alpha + (exp_scores @ v_block)
        
        # Update the running max for the next chunk iteration
        m = m_new
        
    # Return o / l
    # Handle division by zero for any rows that were completely causally masked out
    final_output = np.zeros_like(o)
    valid_rows = (l != 0).squeeze(-1)
    final_output[valid_rows] = o[valid_rows] / l[valid_rows]
    
    return final_output