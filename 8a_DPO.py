import torch
import torch.nn.functional as F

# DPO simplifies the alignment process by bypassing the need for a separate reward model. 
# The get_log_prob function computes sequence-level log probabilities by shifting logits and labels 
# (teacher forcing) and ignoring prompt tokens marked with -100. The DPO loss is then calculated 
# using the difference in log probabilities between a policy model and a frozen reference model for 
# both chosen and rejected responses.

def get_log_prob(logits, labels):
    """
    Compute sequence-level log probability from token logits.
    """
    # 1. Shift labels and logits for autoregressive teacher forcing
    labels = labels[:, 1:]
    logits = logits[:, :-1, :]
    
    # 2. Create a mask where labels equal -100 (prompt tokens to ignore)
    logits_mask = labels == -100
    
    # 3. Replace -100 in labels with 0 so torch.gather doesn't throw an out-of-bounds error
    labels[labels == -100] = 0
    
    # 4. Gather the log_softmax probabilities at the actual label positions
    per_token_logps = torch.gather(
        F.log_softmax(logits, dim=-1), 
        dim=2, 
        index=labels.unsqueeze(2)
    ).squeeze(2)
    
    # 5. Mask out the ignored tokens (set their log probs to 0) and sum over the sequence length
    per_token_logps[logits_mask] = 0
    
    return per_token_logps.sum(-1)

def compute_dpo_loss(policy_logits, reference_logits, labels, beta=0.1):
    """
    Computes the DPO loss given logits from a policy and reference model.
    Assumes the batch is stacked: [chosen_samples, rejected_samples].
    """
    # 1. Get log probs for policy and reference models
    policy_logps = get_log_prob(policy_logits, labels)
    reference_logps = get_log_prob(reference_logits, labels)
    
    # 2. Split into chosen and rejected halves
    bs = policy_logps.shape[0] // 2
    
    chosen_policy_logps = policy_logps[:bs]
    rejected_policy_logps = policy_logps[bs:]
    
    chosen_reference_logps = reference_logps[:bs]
    rejected_reference_logps = reference_logps[bs:]
    
    # 3. Calculate DPO loss using the Bradley-Terry preference formulation
    loss = -F.logsigmoid(
        beta * (
            (chosen_policy_logps - chosen_reference_logps) - 
            (rejected_policy_logps - rejected_reference_logps)
        )
    )
    
    # 4. Return the mean loss over the batch
    return loss.mean()