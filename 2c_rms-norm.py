# ==========================================
# Implements Root Mean Square Layer Normalization, an architecture 
# choice used in modern models like Llama. Unlike standard LayerNorm, 
# RMSNorm does not center the activations by subtracting the mean. Instead, 
# it normalizes the input strictly by its root mean square and applies 
# a learnable scale weight. The implementation handles necessary 
# floating-point conversions to ensure numerical stability during the calculation
# ==========================================

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        """
        Root Mean Square Layer Normalization.
        """
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (weight), initialized to ones
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Computes the root mean square normalization.
        RMSNorm(x) = x * rsqrt(mean(x^2) + eps)
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 1. Cast x to float32 for numerical stability during division/squares
        # 2. Apply the RMS normalization
        # 3. Cast back to the original input data type
        output = self._norm(x.float()).type_as(x)
        
        # 4. Multiply by the learnable weight parameter
        return output * self.weight