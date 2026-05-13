import numpy as np
from scipy.fftpack import dct, idct
import torch

def flatten_method_2d(H: int, W: int, pos: np.ndarray, indices_1d: np.ndarray):
    """flatten a 2D image into a 1D vector using a specified method (row-wise or column-wise).

    Args:
        H (int): The height of the 2D image
        W (int): The width of the 2D image
        pos (np.ndarray): The positions of the observed pixels
        indices_1d (np.ndarray): The 1D indices of the observed pixels
    Returns:
        np.ndarray: The flattened 1D vector, shape (H*W,)
    """
    K = pos.shape[0]
    posh, posw = pos[:, 0], pos[:, 1] # [K]
    idx = posh * W + posw # [K]
    
    sensing_matrix = np.zeros((K, H * W), dtype=np.float32)
    
    for i, idx in enumerate(idx):
        basis = np.zeros(H * W)
        basis[idx] = 1
        sensing_matrix[i, :] = idct(basis, norm='ortho')
    
    A = sensing_matrix[:, indices_1d].T # [C, K]
    B = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-8)
    
    k_idx_out = np.arange(K).astype(np.float32)
    coords = np.column_stack([posh, posw, k_idx_out])  # Get 2D coordinates of sampled pixels, shape (K, 3)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A = torch.from_numpy(A).to(dtype=torch.float32, device=device)
    B = torch.from_numpy(B).to(dtype=torch.float32, device=device)
    coords = torch.from_numpy(coords).to(dtype=torch.float32, device=device)
    
    return A, B, coords