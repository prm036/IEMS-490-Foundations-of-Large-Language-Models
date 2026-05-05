"""
Implement a Sparse Mixture of Experts (MoE) layer:
(a) Top-k Router: A gating network that selects the top-k experts per token.
- Compute logits via a linear layer.
- Select top-k, mask the rest to -inf, then apply softmax to re-normalize.
(b) SparseMoE: The full MoE layer.
- Route each token to its top-k experts using the router.
- For each expert, identify which tokens are assigned to it.
- Compute expert output, weight by gating scores, and accumulate.
"""

import numpy as np

class TopkRouter:
    def __init__(self, n_embed, num_experts, top_k):
        """
        n_embed: input embedding dimension
        num_experts: number of experts
        top_k: how many experts each token selects
        """
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Initialize linear layer mapping n_embed -> num_experts
        self.weight = np.random.randn(n_embed, num_experts) * 0.02
        self.bias = np.zeros(num_experts)

    def forward(self, mh_output):
        # mh_output: (batch, seq, n_embed) input tensor

        # 1. Compute logits via a linear layer
        logits = mh_output @ self.weight + self.bias

        # 2. Get top_k logits and indices
        # argsort sorts ascending, so we slice the last top_k and reverse to get descending order
        indices = np.argsort(logits, axis=-1)[..., -self.top_k:][..., ::-1]
        top_k_logits = np.take_along_axis(logits, indices, axis=-1)

        # 3. Create a tensor of -inf, scatter top_k logits back
        router_output = np.full_like(logits, -np.inf)
        np.put_along_axis(router_output, indices, top_k_logits, axis=-1)

        # 4. Apply softmax to get routing weights (with numerical stability)
        max_logits = np.max(router_output, axis=-1, keepdims=True)
        exp_logits = np.exp(router_output - max_logits)
        routing_weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return routing_weights, indices

class Expert:
    # Standard Feed-Forward Network commonly used for experts in generative models
    def __init__(self, n_embed):
        self.w1 = np.random.randn(n_embed, n_embed * 4) * 0.02
        self.w2 = np.random.randn(n_embed * 4, n_embed) * 0.02

    def forward(self, x):
        # Apply a ReLU activation between layers
        return np.maximum(0, x @ self.w1) @ self.w2

class SparseMoE:
    def __init__(self, n_embed, num_experts, top_k):
        self.num_experts = num_experts
        
        # Initialize router and expert list
        self.router = TopkRouter(n_embed, num_experts, top_k)
        self.experts = [Expert(n_embed) for _ in range(num_experts)]

    def forward(self, x):
        # x: (batch, seq, n_embed)
        routing_weights, indices = self.router.forward(x)

        # Initialize the final output tensor with zeros
        final_output = np.zeros_like(x)

        # For each expert i:
        for i in range(self.num_experts):
            
            # 1. Create mask where expert i is in the top-k indices
            # indices shape: (batch, seq, top_k)
            expert_mask = (indices == i)

            # Collapse the top_k dimension to find which tokens use this expert
            token_mask = np.any(expert_mask, axis=-1) # shape: (batch, seq)

            # Skip processing if no tokens are routed to this expert
            if not np.any(token_mask):
                continue 

            # 2. Select input tokens assigned to expert i
            expert_inputs = x[token_mask] # shape: (num_assigned_tokens, n_embed)

            # 3. Run through expert
            expert_outputs = self.experts[i].forward(expert_inputs)

            # Extract the gating scores for the assigned tokens
            # routing_weights shape: (batch, seq, num_experts)
            gating_scores = routing_weights[token_mask, i, np.newaxis] 
            
            # Weight by gating scores
            weighted_output = expert_outputs * gating_scores

            # 4. Accumulate into final output
            final_output[token_mask] += weighted_output

        return final_output