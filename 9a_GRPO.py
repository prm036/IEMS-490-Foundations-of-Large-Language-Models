"""
GRPO (Group Relative Policy Optimization):
GRPO is designed to align language models efficiently by removing the need for 
a separate value model (critic) used in PPO.

sequences_log_probs: Extracts the token-level log probabilities from the
model's logits using negative cross-entropy, effectively shifting the logits 
and labels for autoregressive training.

approx_kl_divergence: Calculates an unbiased, sample-based approximation of 
the KL divergence ($e^r - r - 1$) between the reference model and the active 
policy to prevent the model from drifting too far from its original distribution.

GRPOLoss: Computes the clipped surrogate objective function combined with the 
KL penalty, applying an action mask to ensure padding tokens do not affect 
the gradient.
"""

import torch
import torch.nn.functional as F

def sequences_log_probs(model, sequence_ids, attention_mask):
    """
    Compute per-token log probabilities for each sequence.
    """
    # 1. Forward pass to get logits
    # Assuming the model returns an object with a .logits attribute
    outputs = model(sequence_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # 2. Shift logits and labels for next-token prediction
    # logits shape: (B, seq_len - 1, vocab_size)
    # labels shape: (B, seq_len - 1)
    logits = logits[:, :-1, :].contiguous()
    labels = sequence_ids[:, 1:].contiguous()
    
    # 3. Use F.cross_entropy with reduction='none'
    # Flatten tensors to use standard cross_entropy formatting (N, C)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), 
        labels.view(-1), 
        reduction='none', 
        ignore_index=-100
    )
    
    # Negate the loss to get log probabilities
    log_probs = -loss
    
    # 4. Reshape back to sequence format
    # Note: If input was (batch_size * K, seq_len), this returns (batch_size * K, seq_len - 1)
    # The caller can reshape to (batch_size, K, seq_len - 1) using .view(batch_size, K, -1)
    log_probs = log_probs.view(labels.size(0), labels.size(1))
    
    return log_probs


def approx_kl_divergence(log_probs, log_probs_ref, action_mask):
    """
    Approximate KL divergence: exp(r) - r - 1
    where r = log_probs_ref - log_probs.
    """
    # Calculate the log ratio (r)
    r = log_probs_ref - log_probs
    
    # Compute the approximate KL divergence formula
    kl = torch.exp(r) - r - 1.0
    
    # Apply action_mask to ignore padding tokens (zeros out masked positions)
    kl = kl * action_mask
    
    return kl


def GRPOLoss(advantages, logprobs_new, logprobs_old, logprobs_ref, action_mask, eps=0.2, kl_weight=0.01):
    """
    GRPO loss with PPO-style clipping and KL penalty.
    """
    # 1. Compute KL divergence penalty
    kl = approx_kl_divergence(logprobs_new, logprobs_ref, action_mask)
    
    # 2. Compute the policy ratio
    ratio = torch.exp(logprobs_new - logprobs_old)
    
    # 3. Compute unclipped surrogate (surr1)
    surr1 = ratio * advantages
    
    # 4. Compute clipped surrogate (surr2)
    surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
    
    # 5. Compute total per-token loss: -min(surr1, surr2) + kl_weight * KL
    loss_per_token = -torch.min(surr1, surr2) + (kl_weight * kl)
    
    # 6. Mask out padding tokens and normalize over the valid action tokens
    masked_loss = loss_per_token * action_mask
    loss = masked_loss.sum() / action_mask.sum()
    
    return loss