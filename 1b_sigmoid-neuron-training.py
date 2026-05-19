import numpy as np

def sigmoid(x):
    # Safe sigmoid implementation
    # Using np.clip can help prevent overflow warnings in np.exp for very large negative numbers
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

def train_neuron(features, labels, initial_weights, initial_bias,
                 learning_rate, epochs):
    # Initialize weights and bias
    weights = np.array(initial_weights, dtype=float)
    bias = float(initial_bias)
    features = np.array(features)
    labels = np.array(labels)
    
    mse_values = []
    
    # Define N for the gradient calculations
    N = len(labels)

    for _ in range(epochs):
        # 1. Forward Pass: Compute predictions [cite: 71, 72]
        z = np.dot(features, weights) + bias
        predictions = sigmoid(z)
        
        # 2. Compute MSE [cite: 73]
        mse = np.mean((predictions - labels) ** 2)
        mse_values.append(mse)  # Keeping full precision is standard practice
        
        # 3. Compute Gradients using the Chain Rule [cite: 74]
        # Mathematical breakdown:
        # d(MSE)/d(pred) = (2/N) * (predictions - labels)
        # d(pred)/d(z) = predictions * (1 - predictions)
        # d(z)/d(w) = features,  d(z)/d(b) = 1
        
        # Combining the first two parts of the chain rule (dL/dz)
        errors = predictions - labels
        dz = errors * predictions * (1 - predictions)
        
        # Matrix multiplication to sum the gradients across the batch for each weight
        weight_gradients = (2 / N) * np.dot(features.T, dz)
        
        # Summing the gradients across the batch for the bias
        bias_gradient = (2 / N) * np.sum(dz)
        
        # 4. Update weights and bias [cite: 74]
        weights -= learning_rate * weight_gradients
        bias -= learning_rate * bias_gradient
        
    # RETURN OUTSIDE THE LOOP! 
    # We return the trained parameters and the history of the loss.
    return weights, bias, mse_values