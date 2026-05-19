import numpy as np

def relu(x):
    # Applies the Rectified Linear Unit function element-wise.
    # Returns x if x > 0, else 0.
    return np.maximum(0, x)

def relu_grad(x):
    # Derivative of ReLU: returns 1 for positive elements, 0 otherwise.
    return (x > 0).astype(x.dtype)

def softmax(x):
    # Numerically stable softmax. Subtracting the max value prevents 
    # exponential overflow while keeping the resulting probabilities mathematically identical.
    x_max = np.max(x, axis=-1, keepdims=True)
    exps = np.exp(x - x_max)
    # Normalize by the sum of exponentials to get a valid probability distribution.
    return exps / np.sum(exps, axis=-1, keepdims=True)

def cross_entropy_loss(probs, y_onehot):
    # Calculates the negative log-likelihood. 
    # Added 1e-12 (epsilon) inside the log to prevent NaN errors in case a probability is exactly 0.
    return -np.sum(y_onehot * np.log(probs + 1e-12))

def one_hot(y, num_classes):
    # Helper function to convert integer labels into one-hot encoded vectors.
    v = np.zeros(num_classes)
    v[y] = 1
    return v

def mlp_forward(x, W1, b1, W2, b2):
    # Forward pass of a 2-layer MLP
    z1 = x @ W1 + b1      # Linear transformation 1
    h = relu(z1)          # Non-linear activation
    z2 = h @ W2 + b2      # Linear transformation 2 (logits)
    probs = softmax(z2)   # Convert logits to probabilities
    
    # Store intermediate values in a tuple; these are required for the chain rule in backprop
    cache = (x, z1, h, z2) 
    return probs, cache
    
def mlp_backward(probs, y_onehot, cache, W1, b1, W2, b2):
    # Unpack the cached activations from the forward pass
    x, z1, h, z2 = cache
    
    # --- OUTPUT LAYER GRADIENTS ---
    # Key insight: The derivative of Cross-Entropy Loss with Softmax w.r.t the logits (z2) 
    # simplifies cleanly to (probabilities - actual_labels).
    dz2 = probs - y_onehot 
    
    # Gradient for W2: Since inputs are 1D vectors, we use the outer product
    # of the hidden state (h) and the incoming gradient (dz2).
    dW2 = np.outer(h, dz2)  
    
    # Gradient for b2: Simply the incoming gradient itself.
    db2 = dz2               
    
    # --- HIDDEN LAYER GRADIENTS ---
    # Backpropagate the gradient from the logits (z2) to the hidden state (h)
    # using the transpose of W2.
    dh = dz2 @ W2.T         
    
    # Backpropagate through the ReLU activation function using element-wise multiplication.
    dz1 = dh * relu_grad(z1) 
    
    # Gradient for W1: Outer product of the