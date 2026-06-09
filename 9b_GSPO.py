"""
GSPO (Sequence-Level Policy Optimization):

Standard PPO uses token-level advantage ratios, which can sometimes misalign
with the fact that rewards are often given at the end of a whole sequence.
GSPO addresses this by computing a sequence-level importance ratio (the mean
of the token-level log probability differences) and combining it with
token-level gradient flow using a stop-gradient operator (.detach() in PyTorch).
It then applies standard PPO-style clipping to this combined sequence-level ratio.

compute_policy_loss_gspo: Computes the clipped surrogate objective function
combined with the KL penalty, applying an action mask to ensure padding tokens
do not affect the gradient.
"""
import torch

def compute_policy_loss_gspo(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    clip_ratio_low=0.2,
    clip_ratio_high=0.2,
):
    """
    GSPO policy loss with sequence-level importance ratios.
    """
    # 1. Compute the negative approximate KL (difference in log probabilities)
    negative_approx_kl = log_prob - old_log_prob
    
    # 2. Compute sequence-level ratio
    # Get the valid sequence lengths for each item in the batch
    seq_lengths = response_mask.sum(dim=-1)
    
    # Average the neg_kl over the valid tokens in the sequence
    neg_kl_seq = (negative_approx_kl * response_mask).sum(dim=-1) / seq_lengths
    
    # 3. Combined token-level ratio (with stop-gradient)
    # log(s_{i,t}) = log_prob - sg[log_prob] + sg[neg_kl_seq]
    # We use .detach() to apply the stop-gradient (sg) operator in PyTorch
    log_ratio = log_prob - log_prob.detach() + neg_kl_seq.unsqueeze(-1).detach()
    
    # Clamp the log_ratio to prevent numerical instability (max=10 as given) before exponentiating
    ratio = torch.exp(torch.clamp(log_ratio, max=10.0))
    
    # 4. Clipped surrogate computation
    # Note: advantages is often negated when framing as a standard minimization loss
    loss1 = -advantages * ratio
    loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    
    # We take the max of the negative advantages (equivalent to -min for positive advantages)
    loss = torch.max(loss1, loss2)
    
    # 5. Aggregate: mean over tokens per sequence, then mean over sequences.
    # First, mean over valid tokens in each sequence
    loss_per_seq = (loss * response_mask).sum(dim=-1) / seq_lengths
    
    # Second, mean across the entire batch of sequences
    final_loss = loss_per_seq.mean()
    
    return final_loss