# ==========================================
# This function builds the Adam optimization algorithm from scratch. 
# It performs a defined number of update steps using an initial parameter vector 
# and a gradient function. The algorithm tracks biased first and second raw 
# moment estimates, applies bias correction to both, and then updates the parameters
# ==========================================

import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, 
                   beta2=0.999, epsilon=1e-8, num_iterations=10):
    """
    Adam optimizer implementation from scratch.
    """
    x = x0
    # Initialize 1st moment (m) and 2nd raw moment (v) vectors to zero
    m, v = np.zeros_like(x), np.zeros_like(x)
    
    # Perform optimization steps
    for it in range(1, num_iterations + 1):
        # Retrieve gradients for the current parameters
        g = grad(x)
        
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * g
        
        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (g ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** it)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2 ** it)
        
        # Update parameters
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
    return x