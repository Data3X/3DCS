import numpy as np
import matplotlib.pyplot as plt

""" 
This file contains functions to apply smooth transitions to the input traces, 
which can be useful for tapering the begin of data before applying optimization algorithms. 
The main function is `smooth_transition`, 
which takes traces and applies a specified transition type (linear, sigmoid, polynomial, or exponential).
"""

def smooth_transition(traces, start_idx=50, end_idx=70, transition_type='sigmoid'):
    T, K = traces.shape
    
    assert 0 <= start_idx < end_idx < T, "bad index"
    transition_length = end_idx - start_idx + 1
    
    result = traces.copy()
    
    for k in range(K):
        original_series = traces[:, k]
        
        result[:start_idx, k] = 0
        
        target_value = original_series[end_idx] 
        
        if transition_type == 'linear':
            result[start_idx:end_idx+1, k] = np.linspace(0, target_value, transition_length)
        
        elif transition_type == 'sigmoid':
            result[start_idx:end_idx+1, k] = sigmoid_transition(
                start_idx, end_idx, target_value, 0
            )
            
        elif transition_type == 'polynomial':
            result[start_idx:end_idx+1, k] = polynomial_transition(
                start_idx, end_idx, target_value, 0
            )
            
        elif transition_type == 'exponential':
            result[start_idx:end_idx+1, k] = exponential_transition(
                start_idx, end_idx, target_value, 0
            )
    
    return result

def sigmoid_transition(start_idx, end_idx, target_value, start_value=0):
    transition_length = end_idx - start_idx + 1
    t = np.linspace(-6, 6, transition_length)
    sigmoid = 1 / (1 + np.exp(-t))
    
    transition_values = start_value + (target_value - start_value) * sigmoid
    return transition_values

def polynomial_transition(start_idx, end_idx, target_value, start_value=0):
    transition_length = end_idx - start_idx + 1
    x = np.linspace(0, 1, transition_length)
    transition_curve = 3 * x**2 - 2 * x**3
    
    transition_values = start_value + (target_value - start_value) * transition_curve
    return transition_values

def exponential_transition(start_idx, end_idx, target_value, start_value=0):
    transition_length = end_idx - start_idx + 1
    x = np.linspace(0, 1, transition_length)
    transition_curve = 1 - np.exp(-4 * x)
    
    transition_values = start_value + (target_value - start_value) * transition_curve
    return transition_values


def plot_transition_comparison(original_matrix, smoothed_matrix, k=0):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(original_matrix[:, k], 'b-', linewidth=2)
    plt.axvline(x=50, color='r', linestyle='--', alpha=0.7)
    plt.axvline(x=70, color='g', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(smoothed_matrix[:, k], 'r-', linewidth=2)
    plt.axvline(x=50, color='r', linestyle='--', alpha=0.7)
    plt.axvline(x=70, color='g', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    transition_range = slice(45, 75) 
    plt.plot(range(45, 75), original_matrix[transition_range, k], 'b-',  linewidth=2, alpha=0.7)
    plt.plot(range(45, 75), smoothed_matrix[transition_range, k], 'r-', linewidth=2)
    plt.axvline(x=50, color='r', linestyle='--', alpha=0.7)
    plt.axvline(x=70, color='g', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    T, K = 100, 5
    np.random.seed(42)
    
    original_matrix = np.zeros((T, K))
    for k in range(K):
        trend = np.linspace(0, 10 + k*2, T) + np.random.normal(0, 0.3, T)
        original_matrix[:, k] = trend

    transition_types = ['linear', 'sigmoid', 'polynomial', 'exponential']
    
    for trans_type in transition_types:
        smoothed_matrix = smooth_transition(
            original_matrix, 
            start_idx=50, 
            end_idx=70, 
            transition_type=trans_type
        )
        
        transition_diff = np.diff(smoothed_matrix[50:71, 0])
        
        if trans_type == 'linear':
            plot_transition_comparison(original_matrix, smoothed_matrix, k=0)
    

    

