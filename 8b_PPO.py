""" 
PPO is a reinforcement learning algorithm that optimizes a policy while 
ensuring it doesn't deviate too far from the old policy. The compute_gae 
function calculates Generalized Advantage Estimation (GAE), which balances bias 
and variance using Temporal Difference (TD) errors and an exponential decay 
factor. The ppo_update function computes the clipped surrogate policy loss to 
prevent destructively large policy updates, alongside a clipped value function 
loss to stabilize training.
"""

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

def compute_gae(rewards, dones, values, gamma, lam):
    """
    Compute Generalized Advantage Estimation (GAE) and returns.
    """
    T = len(rewards)
    adv = torch.zeros(T, device=rewards.device)
    lastgae = 0.0
    
    # Iterate backwards through the trajectory
    for t in reversed(range(T)):
        # Handle the boundary condition for the next value
        next_val = values[t+1] if t+1 < T else 0.0
        
        # Calculate Temporal Difference (TD) error
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        
        # Accumulate the exponentially discounted advantages
        lastgae = delta + gamma * lam * (1 - dones[t]) * lastgae
        adv[t] = lastgae
        
    # Returns are the estimated advantages plus the baseline value estimates
    ret = adv + values
    return adv, ret

def ppo_update(
    net, optimizer, data, epochs=4, batch_size=64,
    clip_eps=0.2, vf_clip_eps=0.2, ent_coef=0.01,
    vf_coef=0.5, max_grad_norm=0.5
):
    """
    PPO update step with a clipped surrogate objective and value clipping.
    """
    obs, acts, old_logps, advs, rets, old_vals = data
    N = len(obs)
    
    # 1. Normalize advantages to stabilize training variance
    advs = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8)
    idxs = np.arange(N)
    
    for _ in range(epochs):
        np.random.shuffle(idxs)
        
        # Minibatch updates
        for start in range(0, N, batch_size):
            mb = idxs[start : start + batch_size]
            b_obs, b_acts = obs[mb], acts[mb]
            b_old_logps, b_advs = old_logps[mb], advs[mb]
            b_rets, b_old_vals = rets[mb], old_vals[mb]
            
            # 2. Compute new log probs and entropy from the current policy network
            dist = net.policy(b_obs)
            new_logps = dist.log_prob(b_acts)
            entropy = dist.entropy().mean()
            
            # 3. Compute Policy Loss (Clipped Surrogate)
            ratio = torch.exp(new_logps - b_old_logps)
            surr1 = ratio * b_advs
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_advs
            pi_loss = -torch.min(surr1, surr2).mean()
            
            # 4. Compute Value Loss with clipping
            new_vals = net.value(b_obs)
            v_clipped = b_old_vals + (new_vals - b_old_vals).clamp(-vf_clip_eps, vf_clip_eps)
            v_loss = 0.5 * torch.max(
                (new_vals - b_rets).pow(2), 
                (v_clipped - b_rets).pow(2)
            ).mean()
            
            # 5. Total composite loss
            loss = pi_loss + vf_coef * v_loss - ent_coef * entropy
            
            # Backpropagation step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()