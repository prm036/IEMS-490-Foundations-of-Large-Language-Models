import numpy as np
import torch
import torch.nn as nn

# ==========================================
# The code implements three versions of the softmax function using NumPy. 
# The standard numerically-stable version subtracts the maximum value along 
# the axis before exponentiating and normalizing to prevent overflow. The log-softmax 
# variant utilizes the log-sum-exp trick to maintain numerical stability in 
# logarithmic space. Finally, the online softmax computes the probabilities in 
# a single streaming pass by keeping track of a running maximum and a scaled sum, 
# which is highly memory-efficient
# ==========================================

def softmax(x, axis=-1):
    """
    Numerically stable standard softmax.
    """
    x = np.asarray(x)
    
    # 1. Subtract the max along the specified axis for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    
    # 2. Exponentiate the shifted values
    exp_shifted = np.exp(shifted)
    
    # 3. Normalize by the sum of exponentials
    softmax_vals = exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)
    
    return softmax_vals

def log_softmax(x, axis=-1):
    """
    Log-softmax using the log-sum-exp trick for numerical stability.
    """
    # 1. Subtract the max along the axis
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    
    # 2. Compute log-sum-exp: logsumexp(x) = max + log(sum(exp(x - max)))
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)
    logsumexp = np.log(sum_exp)
    
    # 3. Return the log softmax: x_shifted - logsumexp
    return shifted - logsumexp

def softmax_online(x):
    """
    Online softmax for a 1D array computing the result in a single pass.
    """
    m = -np.inf # Running max initialized to negative infinity
    s = 0.0     # Running scaled sum initialized to zero
    
    for v in x:
        # Update the scaled sum using the previous max and the current value
        s = s * np.exp(m - max(m, v)) + np.exp(v - max(m, v))
        # Update the running max
        m = max(m, v)
        
    # Finally, compute the softmax for the entire array
    return np.exp(x - m) / s
