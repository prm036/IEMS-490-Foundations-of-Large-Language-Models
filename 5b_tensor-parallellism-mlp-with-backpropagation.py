'''
This section implements a two-layer Multi-Layer Perceptron (MLP) 
employing Column-Row Tensor Parallelism using torch and threading 
to simulate GPU shards. The network architecture follows the 
equation  y = relu(x*W_1 + b_1)*W_2 + b_2.  

The first layer's weights and biases (W_1 and b_1) are 
distributed via column parallelism, whereas the second layer's 
weights (W_2) are distributed via row parallelism. During the 
backward pass, gradients for local shards are computed 
independently, and the bias gradient is all-reduced across 
the simulated GPUs.  
'''

import torch
import threading

def relu(x):
    return torch.maximum(torch.zeros_like(x), x)

def relu_grad(x):
    return (x > 0).float()

class Col_Row_TP_MLP_Twolayer:
    def __init__(self):
        self.num_gpus = 4
        self.dim_1 = 32
        self.dim_2 = 128
        self.dim_3 = 2
        
        # Initialize full weights
        W1 = torch.randn(self.dim_1, self.dim_2)
        b1 = torch.zeros(self.dim_2)
        W2 = torch.randn(self.dim_2, self.dim_3)
        b2 = torch.zeros(self.dim_3)
        
        # Split W1, b1 by columns (dim=1 for W1, dim=0 for b1)
        self.W1_shards = list(torch.chunk(W1, self.num_gpus, dim=1))
        self.b1_shards = list(torch.chunk(b1, self.num_gpus, dim=0))
        
        # Split W2 by rows (dim=0 for W2)
        self.W2_shards = list(torch.chunk(W2, self.num_gpus, dim=0))
        
        # b2 is not split by rows; each GPU adds a fraction of the bias 
        # so the final sum matches the mathematical definition.
        self.b2 = b2

    def forward_rank(self, x, w1, b1, w2, b2, results, index):
        # Single GPU forward: out1 = x @ w1 + b1
        out1 = x @ w1 + b1
        out1_relu = relu(out1)
        
        # out2 = out1_relu @ w2 + b2
        # (Dividing b2 by num_gpus prevents it from being added 4 times during the all-reduce step)
        out2 = out1_relu @ w2 + (b2 / self.num_gpus)
        
        # Store intermediate results for backward
        results[index] = {
            'out1': out1,
            'out1_relu': out1_relu,
            'out2': out2
        }

    def forward(self, x):
        results = [None] * self.num_gpus
        threads = []
        
        # Launch forward_rank on each GPU (thread)
        for i in range(self.num_gpus):
            t = threading.Thread(
                target=self.forward_rank,
                args=(x, self.W1_shards[i], self.b1_shards[i], 
                      self.W2_shards[i], self.b2, results, i)
            )
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        out1_list = [res['out1'] for res in results]
        out1_relu_list = [res['out1_relu'] for res in results]
        out2_list = [res['out2'] for res in results]
        
        # All-reduce (sum) the out2 across GPUs
        out2_summed = sum(out2_list)
        
        return out1_list, out1_relu_list, out2_summed

    def backward_rank(self, x, y, out1, out1_relu, out2, w1, b1, w2, b2, results, index):
        # Single GPU backward using MSE loss gradient: d_out2 = out2 - y
        d_out2 = out2 - y
        
        # Compute gradients for w2, b2, w1, b1 using chain rule
        d_w2 = out1_relu.T @ d_out2
        d_b2 = d_out2.sum(dim=0)
        
        d_out1_relu = d_out2 @ w2.T
        d_out1 = d_out1_relu * relu_grad(out1)
        
        d_w1 = x.T @ d_out1
        d_b1 = d_out1.sum(dim=0)
        
        results[index] = {
            'd_w1': d_w1,
            'd_b1': d_b1,
            'd_w2': d_w2,
            'd_b2': d_b2
        }

    def backward(self, x, y, out1_list, out1_relu_list, out2_summed):
        results = [None] * self.num_gpus
        threads = []
        
        # Launch backward_rank on each GPU (thread)
        for i in range(self.num_gpus):
            t = threading.Thread(
                target=self.backward_rank,
                args=(x, y, out1_list[i], out1_relu_list[i], out2_summed, 
                      self.W1_shards[i], self.b1_shards[i], self.W2_shards[i], 
                      self.b2, results, i)
            )
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        self.d_w1_list = [res['d_w1'] for res in results]
        self.d_b1_list = [res['d_b1'] for res in results]
        self.d_w2_list = [res['d_w2'] for res in results]
        
        # All-reduce (sum) the bias2 gradient
        self.d_b2_summed = sum([res['d_b2'] for res in results])

    def train(self, x, y, epochs, lr=0.01):
        batch_size = x.shape[0]
        
        for epoch in range(epochs):
            # Forward
            out1_list, out1_relu_list, out2_summed = self.forward(x)
            
            # Backward
            self.backward(x, y, out1_list, out1_relu_list, out2_summed)
            
            # Update weights (Remember to divide gradients by batch size)
            with torch.no_grad():
                for i in range(self.num_gpus):
                    self.W1_shards[i] -= lr * (self.d_w1_list[i] / batch_size)
                    self.b1_shards[i] -= lr * (self.d_b1_list[i] / batch_size)
                    self.W2_shards[i] -= lr * (self.d_w2_list[i] / batch_size)
                
                self.b2 -= lr * (self.d_b2_summed / batch_size)