import torch

def OMP(A, B, rk, shaped_ob, iters, err, sparsity_list, device):
    norm = torch.linalg.norm
    I_mat = torch.eye(iters + 1, device=device)
    Skpos = torch.zeros(iters + 1, dtype=torch.long, device=device)
    mask = torch.ones(A.shape[0], dtype=torch.bool, device=device)
    
    P = A.shape[1]
    Sk_buffer = torch.zeros((P, iters + 1), dtype=torch.float32, device=device)
    
    ctn = norm(rk)
    for s in range(iters + 1):
        res = B @ rk
        res[~mask] = 0
        maxindex = torch.argmax(torch.abs(res))
        Skpos[s] = maxindex
        mask[maxindex] = False
        
        Sk_buffer[:, s] = A[maxindex, :]
        Sk_slice = Sk_buffer[:, :s + 1]
        
        Asm = Sk_slice.T @ Sk_slice
        
        L_chol, info = torch.linalg.cholesky_ex(Asm)
        if info.item() > 0:
            lam = 1e-6 * torch.trace(Asm)
            L_chol = torch.linalg.cholesky(Asm + lam * I_mat[:s+1, :s+1])
            
        A_T_y = (Sk_slice.T @ shaped_ob).unsqueeze(1)
        xk = torch.cholesky_solve(A_T_y, L_chol)
        rk = shaped_ob - (Sk_slice @ xk).squeeze(1)
        ctn = torch.norm(rk)
        
        if ctn < err:
            return xk.squeeze(), Skpos, s, ctn.item()
        elif s >= sparsity_list[0]:
            print(f"Sparsity reached: {s}")
            sparsity_list.pop(0)
            
    print("Maximum iterations reached without convergence.")
    return xk.squeeze(), Skpos, s, ctn.item()
            
def SAMP(A, B, rk, shaped_ob, iters, err, sparsity_list, device):
    P = A.shape[1]
    max_dim = A.shape[0]
    
    I_mat = torch.eye(max_dim, device=device)
    norm = torch.linalg.norm
    Gamma_mask = torch.zeros(max_dim, dtype=torch.bool, device=device)

    S, L = 1, 1    
    sparsity_queue = list(sparsity_list)
    y = shaped_ob.reshape(-1, 1)

    for t in range(iters + 1):
        res = B @ rk
        _, idx = torch.topk(torch.abs(res), k=L)
        Gamma_mask[idx] = True
        Ck = Gamma_mask.nonzero(as_tuple=False).squeeze(-1) # (|Ck|,)
        At = A[Ck, :].T
        Atmp = At.T @ At
        
        L_chol, info = torch.linalg.cholesky_ex(Atmp)
        if info.item() > 0:
            lam = 1e-6 * torch.trace(Atmp)
            dim = Atmp.shape[0]
            L_chol = torch.linalg.cholesky(Atmp + lam * I_mat[:dim, :dim])
            
        theta_t = torch.cholesky_solve(At.T @ y, L_chol).reshape(-1)  # (|Ck|,)
        
        _, maxindex = torch.topk(torch.abs(theta_t).squeeze(), k=L)
        if maxindex.dim() == 0:
            maxindex = maxindex.unsqueeze(0)
        AtL = At[:, maxindex]
        Gamma_tL = Ck[maxindex]
        
        AtLT_AtL = AtL.T @ AtL
        L_chol_sub, info_sub = torch.linalg.cholesky_ex(AtLT_AtL)
        if info_sub > 0:
            lam_sub = 1e-6 * torch.trace(AtLT_AtL)
            dim_sub = AtLT_AtL.shape[0]
            L_chol_sub = torch.linalg.cholesky(AtLT_AtL + lam_sub * I_mat[:dim_sub, :dim_sub])

        xk_new = torch.cholesky_solve(AtL.T @ y, L_chol_sub).reshape(-1)
        rk_new = shaped_ob - AtL @ xk_new
        
        ctn = norm(rk)
        ctn_new = norm(rk_new)
        xk = theta_t[maxindex]
        
        if ctn_new >= ctn:
            L = L + S
        elif ctn_new <= err:
            rk = rk_new
            print(f"Converged at iteration {t}, L={L}")
            break
        else:
            rk = rk_new
            if sparsity_queue and L >= sparsity_queue[0]:
                sparsity_queue.pop(0)
                print(f"Sparsity checkpoint reached: L={L}, error={ctn_new.item():.4f}")
            if t + 1 < iters:
                pass
    
    final_error = norm(rk).item()
    print(f'Final sparsity: {len(Gamma_tL)}, Final error: {final_error:.4f}')
    return xk, Gamma_tL, t, L, ctn.item()