import torch
from CS3D_main import compute_reduced_dictionary_GPU

def soft_thresholding(x, threshold):
    """Soft thresholding operator, proximal operator for L1 regularization"""
    return torch.sign(x) * torch.maximum(torch.abs(x) - threshold, torch.zeros_like(x))

def direct_solve(ob, pos, H, W, T, indices_3d, max_iter=1000, tol=1e-6):
    """
    Solve using ISTA: minimize ||A.T @ xk - shaped_ob||^2 + lambda * ||xk||_1
    
    Args:
        ob: Observations (T, K)
        pos: Position information
        H, W, T: Dimension parameters
        indices_3d: 3D indices
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    """
    device = ob.device
    T, K = ob.shape
    L = H * W
    
    # Flatten observations (using Fortran order simulation via transpose)
    shaped_ob = ob.T.flatten()  # PyTorch uses C order, transpose to simulate F order
    
    # Compute 1D indices
    indices_1d = indices_3d[:, 0] * H * W + indices_3d[:, 1] * W + indices_3d[:, 2]
    
    # Obtain dictionary matrices (assumed to return torch tensors)
    A, B, coords_3d = compute_reduced_dictionary_GPU(T, H, W, K, pos, indices_3d)
    C, _ = A.shape
    
    # ISTA parameters
    lambda_ = 0.01
    AT = A.T  # precompute transpose
    
    # Compute Lipschitz constant (step size)
    ATA = AT @ A
    L_lip = torch.linalg.eigvalsh(ATA)[-1].item()
    step_size = 1.0 / (L_lip + 1e-8)
    
    # Initialization
    xk = torch.zeros(C, device=device, dtype=ob.dtype)
    
    # ISTA iterations
    print("Starting ISTA optimization...")
    prev_obj = float('inf')
    
    for iteration in range(max_iter):
        # Gradient step: xk - step_size * gradient
        residual = AT @ xk - shaped_ob
        gradient = A @ residual
        xk_temp = xk - step_size * gradient
        
        # Proximal step: soft thresholding
        xk = soft_thresholding(xk_temp, lambda_ * step_size)
        
        # Compute objective function (check every 50 iterations)
        if iteration % 50 == 0:
            residual = AT @ xk - shaped_ob
            data_term = torch.sum(residual ** 2).item()
            reg_term = lambda_ * torch.sum(torch.abs(xk)).item()
            obj_val = data_term + reg_term
            
            print(f"Iter {iteration}: obj={obj_val:.6e}, data={data_term:.6e}, reg={reg_term:.6e}")
            
            # Check convergence
            if abs(prev_obj - obj_val) < tol:
                print(f"Converged at iteration {iteration}")
                break
            prev_obj = obj_val
    
    # Reconstruct 3D tensor
    xr = torch.zeros((T, H, W), device=device, dtype=ob.dtype)
    
    for i, idx in enumerate(indices_1d):
        tidx = idx // L
        hidx = (idx % L) // W
        widx = idx % W
        xr[tidx, hidx, widx] = xk[i]
    
    # Compute residual norm
    ori_ctn = torch.linalg.norm(shaped_ob)
    final_residual = AT @ xk - shaped_ob
    ctn = torch.linalg.norm(final_residual)
    print(f'ori_ctn:{ori_ctn:.3e}, final_ctn:{ctn:.3e}')
    
    return xr


# If faster convergence is needed, use FISTA (Fast ISTA)
def direct_solve_fista(ob, pos, H, W, T, indices_3d, max_iter=1000, tol=1e-6):
    """FISTA solver with faster convergence rate"""
    device = ob.device
    T, K = ob.shape
    L = H * W
    
    shaped_ob = ob.T.flatten()
    indices_1d = indices_3d[:, 0] * H * W + indices_3d[:, 1] * W + indices_3d[:, 2]
    
    A, B, coords_3d = compute_reduced_dictionary_GPU(T, H, W, K, pos, indices_3d)
    C, _ = A.shape
    
    lambda_ = 0.01
    AT = A.T
    ATA = AT @ A
    L_lip = torch.linalg.eigvalsh(ATA)[-1].item()
    step_size = 1.0 / (L_lip + 1e-8)
    
    # FISTA initialization
    xk = torch.zeros(C, device=device, dtype=ob.dtype)
    yk = xk.clone()
    tk = 1.0
    
    print("Starting FISTA optimization...")
    prev_obj = float('inf')
    
    for iteration in range(max_iter):
        xk_old = xk.clone()
        
        # Compute gradient at yk
        residual = AT @ yk - shaped_ob
        gradient = A @ residual
        
        # Gradient step + proximal step
        xk = soft_thresholding(yk - step_size * gradient, lambda_ * step_size)
        
        # Momentum update
        tk_new = (1 + torch.sqrt(torch.tensor(1 + 4 * tk**2))) / 2
        yk = xk + ((tk - 1) / tk_new) * (xk - xk_old)
        tk = tk_new.item()
        
        # Check convergence
        if iteration % 50 == 0:
            residual = AT @ xk - shaped_ob
            data_term = torch.sum(residual ** 2).item()
            reg_term = lambda_ * torch.sum(torch.abs(xk)).item()
            obj_val = data_term + reg_term
            
            print(f"Iter {iteration}: obj={obj_val:.6e}")
            
            if abs(prev_obj - obj_val) < tol:
                print(f"Converged at iteration {iteration}")
                break
            prev_obj = obj_val
    
    # Reconstruct
    xr = torch.zeros((T, H, W), device=device, dtype=ob.dtype)
    for i, idx in enumerate(indices_1d):
        tidx = idx // L
        hidx = (idx % L) // W
        widx = idx % W
        xr[tidx, hidx, widx] = xk[i]
    
    ori_ctn = torch.linalg.norm(shaped_ob)
    ctn = torch.linalg.norm(AT @ xk - shaped_ob)
    print(f'ori_ctn:{ori_ctn:.3e}, final_ctn:{ctn:.3e}')
    
    return xr