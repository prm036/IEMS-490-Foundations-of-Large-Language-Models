# ==========================================
# Grouped-Query Attention is an optimization technique where the number of 
# key-value (KV) heads is smaller than the number of query heads, effectively 
# reducing the memory bandwidth bottleneck of multi-head attention. Each KV 
# head is shared by a "group" of query heads, calculated as n_heads / n_kv_heads. 
# The repeat_kv function duplicates the KV heads to match the query head count 
# during the forward pass. The Attention class projects the queries, keys, and 
# values separately , applies ROPE exclusively to queries and keys , scales 
# the KV heads using repeat_kv , and finally computes the scaled dot-product 
# attention along with any causal masking
# ==========================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def repeat_kv(x, n_rep):
    """
    Repeat KV heads to match query head count for Grouped-Query Attention.
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    
    # If there's 1-to-1 mapping, no repetition is needed
    if n_rep == 1:
        return x
        
    # 1. Expand a new dimension for repeating
    # 2. Expand the tensor to duplicate the heads
    # 3. Reshape back to collapse the duplicated heads into the n_kv_heads dimension
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=None):
        super().__init__()
        self.n_heads = n_heads
        # Default to standard Multi-Head Attention if n_kv_heads is not specified
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        
        # Calculate how many times each KV head needs to be repeated
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads

        # Linear projections for Query, Key, Value, and Output
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, freqs_cos, freqs_sin, mask=None):
        bsz, seqlen, _ = x.shape

        # 1. & 2. Project x to get queries, keys, and values, then reshape 
        # to separate the heads
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # 3. Apply Rotary Positional Embeddings to queries and keys
        xq = apply_rotary_emb(xq, freqs_cos, freqs_sin)
        xk = apply_rotary_emb(xk, freqs_cos, freqs_sin)

        # 4. Repeat KV heads to match the number of query heads (GQA logic)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 5. Transpose dimensions to (batch, heads, seqlen, head_dim) for batch matmul
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 6. Compute raw attention scores (scaled dot-product)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 7. Apply causal mask (if provided) to prevent looking ahead in autoregressive generation
        if mask is not None:
            scores = scores + mask

        # 8. Apply softmax to get probabilities, then multiply by values
        # Cast to float for precision during softmax, then revert to original dtype
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)

        # 9. Transpose back to (batch