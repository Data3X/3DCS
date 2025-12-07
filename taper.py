import numpy as np
import matplotlib.pyplot as plt

def smooth_transition_2d(matrix, start_idx=50, end_idx=70, transition_type='sigmoid'):
    T, K = matrix.shape
    
    assert 0 <= start_idx < end_idx < T, "bad index"
    transition_length = end_idx - start_idx + 1
    
    result = matrix.copy()
    
    for k in range(K):
        original_series = matrix[:, k]
        
        result[:start_idx, k] = 0
        
        target_value = original_series[end_idx] 
        
        if transition_type == 'sigmoid':
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

def smooth_transition_2d_vectorized(matrix, start_idx=50, end_idx=70, transition_type='sigmoid'):
    T, K = matrix.shape
    transition_length = end_idx - start_idx + 1
    
    result = matrix.copy()
    
    result[:start_idx, :] = 0
    
    if transition_type == 'sigmoid':
        t = np.linspace(-6, 6, transition_length)
        transition_curve = 1 / (1 + np.exp(-t))
    elif transition_type == 'polynomial':
        x = np.linspace(0, 1, transition_length)
        transition_curve = 3 * x**2 - 2 * x**3
    elif transition_type == 'exponential':
        x = np.linspace(0, 1, transition_length)
        transition_curve = 1 - np.exp(-4 * x)
    
    start_values = 0 
    target_values = matrix[end_idx, :]
    
    for i, t_idx in enumerate(range(start_idx, end_idx + 1)):
        alpha = transition_curve[i]
        result[t_idx, :] = start_values + (target_values - start_values) * alpha
    
    return result

def advanced_smooth_transition_2d(matrix, start_idx=50, end_idx=70, 
                                 transition_func=None, transition_params=None):
    T, K = matrix.shape
    
    if transition_func is None:
        def transition_func(x, params):
            steepness = params.get('steepness', 6)
            t = np.linspace(-steepness/2, steepness/2, len(x))
            return 1 / (1 + np.exp(-t))
    
    if transition_params is None:
        transition_params = {}
    
    result = matrix.copy()
    result[:start_idx, :] = 0
    
    transition_length = end_idx - start_idx + 1
    x = np.arange(transition_length)
    
    transition_curve = transition_func(x, transition_params)
    
    target_values = matrix[end_idx, :]
    
    for i, t_idx in enumerate(range(start_idx, end_idx + 1)):
        alpha = transition_curve[i]
        result[t_idx, :] = 0 + (target_values - 0) * alpha
    
    return result

def plot_2d_transition_comparison(original_matrix, smoothed_matrix, k=0):
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

    transition_types = ['sigmoid', 'polynomial', 'exponential']
    
    for trans_type in transition_types:
        smoothed_matrix = smooth_transition_2d_vectorized(
            original_matrix, 
            start_idx=50, 
            end_idx=70, 
            transition_type=trans_type
        )
        
        transition_diff = np.diff(smoothed_matrix[50:71, 0])
        
        if trans_type == 'sigmoid':
            plot_2d_transition_comparison(original_matrix, smoothed_matrix, k=0)
    
    
    def linear_transition(x, params):
        return np.linspace(0, 1, len(x))
    
    linear_smoothed = advanced_smooth_transition_2d(
        original_matrix, 
        start_idx=50, 
        end_idx=70,
        transition_func=linear_transition
    )
