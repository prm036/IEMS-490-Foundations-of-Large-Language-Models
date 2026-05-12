'''
The following implementations utilize numpy to partition and 
compute matrix multiplications across simulated GPUs based 
on three distinct strategies: column parallel, row parallel, 
and data parallel.  In Column Parallelism, the weight matrix 
is split along its columns (axis=1), and the final outputs 
are concatenated. For Row Parallelism, the input matrix is 
split by columns while the weight matrix is split by rows, 
requiring an all-reduce (sum) of the partial results. 
Finally, in Data Parallelism, the batch dimension of the 
input matrix is split across GPUs while each GPU retains 
a full copy of the weights.  
'''
import numpy as np

def tensor_parallel_column(shared_a, weight_b, gpu_nums):
    # Split B along columns (axis=1) across GPUs
    b_splits = np.array_split(weight_b, gpu_nums, axis=1)
    
    # Each GPU: Y_i = A @ B_i
    y_partials = [shared_a @ b_i for b_i in b_splits]
    
    # Final: Y = concat(Y_0, Y_1, ...) along columns
    y = np.concatenate(y_partials, axis=1)
    return y

def tensor_parallel_row(shared_a, weight_b, gpu_nums):
    # Split A along columns (axis=1) and B along rows (axis=0)
    a_splits = np.array_split(shared_a, gpu_nums, axis=1)
    b_splits = np.array_split(weight_b, gpu_nums, axis=0)
    
    # Each GPU: Y_i = A_i @ B_i
    y_partials = [a_i @ b_i for a_i, b_i in zip(a_splits, b_splits)]
    
    # Final: Y = sum(Y_0, Y_1, ...)
    y = sum(y_partials)
    return y

def data_parallel(shared_a, weight_b, gpu_nums):
    # Split A along the batch dimension (axis=0)
    a_splits = np.array_split(shared_a, gpu_nums, axis=0)
    
    # Each GPU holds a full copy of B. Each GPU: Y_i = A_i @ B
    y_partials = [a_i @ weight_b for a_i in a_splits]
    
    # Final: Y = concat(Y_0, Y_1, ...) along batch dim
    y = np.concatenate(y_partials, axis=0)
    return y