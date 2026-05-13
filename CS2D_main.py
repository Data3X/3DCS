import torch
import time
import numpy as np
from scipy.fftpack import idct
from tools import get_1d_idct_matrix, normalize
from CS2D_tools import compare_2D_results, main_spectrum_statistics_2D, main_spectrum_statistics_1D, gene_mask_2D, show_synthetic_data_results_2D
from greedy_methods_GPU import OMP, SAMP
from flatten_method_2d import flatten_method_2d
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def compute_reduced_dictionary_2D_GPU(H: int, W: int, K: int, pos_np: np.ndarray, indices_2d_np: np.ndarray, 
                                      pos_type: str = 'float', device: str = 'cuda'):
    C = indices_2d_np.shape[0]

    BFH = torch.from_numpy(idct(np.eye(H), axis=0, norm='ortho')).to(dtype=torch.float32, device=device)
    BFW = torch.from_numpy(idct(np.eye(W), axis=0, norm='ortho')).to(dtype=torch.float32, device=device)

    pos = torch.as_tensor(pos_np, dtype=torch.float32, device=device)
    indices_2d = torch.as_tensor(indices_2d_np, dtype=torch.long, device=device)

    h_idx = indices_2d[:, 0] # [C]
    w_idx = indices_2d[:, 1] # [C]
    
    posh, posw = pos[:, 0], pos[:, 1] # [K]

    if pos_type == 'int':
        posh_long, posw_long = posh.long(), posw.long()
        val_H = BFH[posh_long.unsqueeze(1), h_idx]  # [K, C]
        val_W = BFW[posw_long.unsqueeze(1), w_idx]  # [K, C]
    else:
        def interpolate_basis(basis, p, idx):
            # p: [K], idx: [C]
            p0 = p.floor().long().clamp(0, basis.size(0) - 1)
            p1 = (p0 + 1).clamp(0, basis.size(0) - 1)
            weight = (p - p0.float()).unsqueeze(1) # [K, 1]

            v0 = basis[p0.unsqueeze(1), idx] # [K, C]
            v1 = basis[p1.unsqueeze(1), idx] # [K, C]
            return v0 * (1 - weight) + v1 * weight

        val_H = interpolate_basis(BFH, posh, h_idx) # [K, C]
        val_W = interpolate_basis(BFW, posw, w_idx) # [K, C]

    # A[c, k] = Basis_H(pos_h[k], idx_h[c]) * Basis_W(pos_w[k], idx_w[c])
    val_HW = val_H * val_W  # [K, C]
    A = val_HW.T  # [C, K]

    norms = torch.linalg.norm(A, axis=1, keepdims=True)
    B = A / (norms + 1e-8)

    k_idx_out = torch.arange(K, device=device).float()
    coords_2d = torch.stack([posh, posw, k_idx_out], dim=-1) # [K, 3]

    return A, B, coords_2d


def CS2D_for_synthetic_data(event_for_spectrum_statistics: np.ndarray, sampling_ratio: float = 0.05, method: str = 'OMP', reload: bool = False, plot: bool = False, use_all_freqs: bool = False, flatten_to_1d: bool = False):
    H, W = 64, 64
    
    full_video = normalize(event_for_spectrum_statistics[50:350, :, :, 0]) 
    image = full_video[100, 32:H+32, 32:W+32]

    if not reload:
        ob, ob_pos, test_pos = gene_mask_2D(image, H, W, K=10, sample_ratio=sampling_ratio, mask_type='int') 
        np.save(r'CS2D_compare_results\ob_2d_reloading.npy', ob)
        np.save(r'CS2D_compare_results\ob_pos_2d_reloading.npy', ob_pos)
        np.save(r'CS2D_compare_results\test_pos_2d_reloading.npy', test_pos)
    else:
        ob = np.load(r'CS2D_compare_results\ob_2d_reloading.npy')
        ob_pos = np.load(r'CS2D_compare_results\ob_pos_2d_reloading.npy')
        test_pos = np.load(r'CS2D_compare_results\test_pos_2d_reloading.npy')
        
    K = ob.shape[0]

    if not use_all_freqs:
        if not reload:
            indices_2d = main_spectrum_statistics_2D(full_video, image, plot=False)
            indices_1d = main_spectrum_statistics_1D(full_video, image, plot=False)
            np.save(r'CS2D_compare_results\indices_2d_reloading.npy', indices_2d)
            np.save(r'CS2D_compare_results\indices_1d_reloading.npy', indices_1d)
        else:
            indices_2d = np.load(r'CS2D_compare_results\indices_2d_reloading.npy')
            indices_1d = np.load(r'CS2D_compare_results\indices_1d_reloading.npy')
    else:
        row = np.arange(H)
        col = np.arange(W)
        grid_y, grid_x = np.meshgrid(row, col, indexing='ij')
        indices_2d = np.column_stack([grid_y.ravel(), grid_x.ravel()])
        indices_2d = indices_2d.reshape(-1, 2)
        indices_1d = indices_2d[:, 0] * W + indices_2d[:, 1]

    if flatten_to_1d:
        indices_2d = np.zeros((len(indices_1d), 2), dtype=np.int32)
        indices_2d[:, 0] = indices_1d // W
        indices_2d[:, 1] = indices_1d % W
        A, B, coords_2d = flatten_method_2d(H, W, ob_pos, indices_1d)
    else:
        A, B, coords_2d = compute_reduced_dictionary_2D_GPU(H, W, K, ob_pos, indices_2d)

    print("Performing 2D Reconstruction...")
    rec_image = recovery_2D_GPU(ob, H, W, ob_pos, indices_2d, A=A, B=B, coords_2d=coords_2d, method=method)

    compare_2D_results(image, rec_image, pos=ob_pos, test_pos=test_pos, save=True)
    
    np.save(r'CS2D_compare_results\rec_image_theo.npy', rec_image)
    np.save(r'CS2D_compare_results\ori_image_theo.npy', image)
    
    show_synthetic_data_results_2D(rec_image, image, test_pos)
    
    return rec_image

def recovery_2D_GPU(ob_np: np.ndarray, H: int, W: int, pos: np.ndarray, indices_2d: np.ndarray, 
                    A, B, coords_2d, iters: int = 1000, device: str = 'cuda', method: str = 'SAMP'):
    
    norm = torch.linalg.norm
    K = ob_np.shape[0]
    Sq = H * W
    
    k_indices = coords_2d[:, 2].long()

    ob = torch.as_tensor(ob_np, dtype=torch.float32, device=device)
    shaped_ob = ob[k_indices]
    
    rk = shaped_ob.clone()
    err = 1e-3 * norm(rk)
    print(f's: 0 L: 0, ctn: {norm(rk):.4f}')
    
    indices_1d = indices_2d[:, 0] * W + indices_2d[:, 1]
    indices_1d = torch.as_tensor(indices_1d, device=device, dtype=torch.long)

    iters = K
    sparsity_list = [i*10 for i in range(1, (iters // 10) + 2)]
    
    D_H = get_1d_idct_matrix(H, device)
    D_W = get_1d_idct_matrix(W, device)

    def recover_to_image(xk, Skpos, t, L):
        Skpos_1d = indices_1d[Skpos]
        
        xf = torch.zeros((H, W), dtype=torch.float32, device=device)
        
        hidx = Skpos_1d // W
        widx = Skpos_1d % W
        
        xf[hidx, widx] = xk.to(xf.dtype)
        
        rec_image = torch.einsum('hb, bw -> hw', D_H, xf)
        rec_image = torch.einsum('hw, wc -> hc', rec_image, D_W.T)
        
        print(f's: {t} L: {L}, ctn: {ctn:.4f}')
        return rec_image.to('cpu').numpy()
    
    if method == 'OMP':
        xk, Skpos, s, ctn = OMP(A, B, rk, shaped_ob, iters, err, sparsity_list, device)
        return recover_to_image(xk, Skpos, s, s)

    elif method == 'SAMP':
        xk, Gamma_tL, t, L, ctn = SAMP(A, B, rk, shaped_ob, iters, err, sparsity_list, device)
        return recover_to_image(xk, Gamma_tL, t, L)

if __name__ == '__main__':
    event_path = r"D:\research\3DCS\upload\synthetic_event0.npy"
    # loading the data for real data experiment
    channel = 0
    event_for_spectrum_statistics = np.load(event_path)
    
    start = time.time()
    CS2D_for_synthetic_data(event_for_spectrum_statistics, sampling_ratio=0.04, method='SAMP', reload=True, plot=False, use_all_freqs=False, flatten_to_1d=True)
    end = time.time()
    print(f'used_time: {end - start:.1f} s')