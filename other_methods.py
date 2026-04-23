from CS3D_main import compute_reduced_dictionary_CPU_mpi
import numpy as np
import cvxpy as cp

# direct solve the optimization problem using cvxpy, which is a convex optimization solver.
def direct_solve(ob: np.ndarray, pos: np.ndarray, H: int, W: int, T: int, indices_3d: np.ndarray):
    T,K = ob.shape
    L = H*W
    shaped_ob = ob.flatten(order='F')
    indices_1d = indices_3d[:,0]*H*W + indices_3d[:,1]*W + indices_3d[:,2]
    A = compute_reduced_dictionary_CPU_mpi(T, H, W, K, pos, indices_3d, req='A')
    C, _ = A.shape
    xr = np.zeros((T,H,W))
    xk = cp.Variable(C)
    lambda_ = 0.01
    objective = cp.Minimize(cp.sum_squares(A.T @ xk - shaped_ob) + lambda_ * cp.norm1(xk))
    problem = cp.Problem(objective)
    problem.solve()
    xk = xk.value
        
    for i, idx in enumerate(indices_1d):
        tidx = idx // L
        hidx = (idx % L) // W
        widx = idx % W
        xr[tidx, hidx, widx] = xk[i]
        
    ori_ctn = np.linalg.norm(shaped_ob)
    ctn = np.linalg.norm(A.T @ xk - shaped_ob)
    print(f'ori_ctn:{ori_ctn:.3e}, final_ctn:{ctn:.3e}')
    return xr    
