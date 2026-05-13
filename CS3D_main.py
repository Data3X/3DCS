import numpy as np
from scipy.fftpack import idct
from plot_tools import animate_comparaison_for_two_videos, main_spectrum_statistics, animate_a_video
from tools import get_1d_idct_matrix, gene_mask, shuffle, joint, get_sta_pos, normalize
from greedy_methods_GPU import OMP, SAMP
import taper
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import RegularGridInterpolator
import time

def dct_basis_via_idct(H, W, T, u, v, w, norm='ortho'):
    coeff = np.zeros((H, W, T))
    coeff[u, v, w] = 1.0

    basis = idct(idct(idct(coeff, axis=0, norm=norm), axis=1, norm=norm), axis=2, norm=norm)
    return basis

def show_real_data_results(T: int, H: int, W: int, rec_video: np.ndarray, ob: np.ndarray, sta_info, 
                           test_sta: np.ndarray, full_pos: np.ndarray, plot: bool = False):
    """present the real data results by comparing the reconstructed video with the observations at the test positions,
    which can be used to evaluate the performance of the reconstruction algorithm on real data.

    Args:
        T (int): the number of time steps
        H (int): the height of the spatial grid
        W (int): the width of the spatial grid
        rec_video (np.ndarray): the reconstructed video, shape (T, H, W)
        ob (np.ndarray): the observations, shape (T, K)
        sta_info (list): information about the stations
        test_sta (np.ndarray): the test stations recordings, shape (T, s)
        full_pos (np.ndarray): the positions of all the stations, shape (K+s, 2)
    """
    k = test_sta.shape[1] # the number of test stations
    np.save('rec_video.npy',rec_video)
    # rec_video = np.load('rec_video.npy')
    if plot:
        animate_a_video(rec_video, full_pos[:-k], normalize=False, save=True, color=ob)

    h_coords,w_coords,t_coords = np.arange(H),np.arange(W),np.arange(T)
    interp_func = RegularGridInterpolator(
        (h_coords, w_coords),
        np.zeros((H,W)))

    delta = 1
    t = np.linspace(0, T*delta, T)
    # s = test_sta.shape[1] 
    s = 30 # the number of test stations we want to show in the comparison figure, we will select the first s stations in test_sta for comparison
    K = ob.shape[1]
    fig, axes = plt.subplots(s, 1, figsize=(10, 2*s))
    plt.subplots_adjust(left=0.2)
    maxv1 = 1.2*np.max(np.abs(test_sta)) # the maximum absolute value among the test stations, used for setting the y-axis limits in the comparison figure
    for i in range(s):
        pos = full_pos[K+i]
        results = []
        for j in range(T):
            interp_func.values = rec_video[j,:,:]
            results.append(interp_func(pos, method='linear'))
        results = np.hstack(results) 
        axes[i].plot(t, test_sta[:,i], color='black')
        axes[i].plot(t, results, color='red')
        # axes[i].plot(t, results-test_sta[:,i], color='green')
        xs = round(pos[0]*330/64,2)
        ys = round(pos[1]*330/64,2)
        axes[i].text(0.98, 0.9, f'{sta_info[K+i][0]}.{sta_info[K+i][1]}({ys}km, {xs}km)', 
                transform=axes[i].transAxes,
                verticalalignment='top',
                horizontalalignment='right', fontsize=20)
        axes[i].tick_params(axis='x', labelsize=20)
        axes[i].tick_params(axis='y', labelsize=20)
        fig.supylabel('Amplitude (m/s)',fontsize=30)
        if i!=s-1:
            axes[i].set_xticks([])
        else:
            axes[i].set_xlabel('Seconds after earthquake', fontsize=30)
        axes[i].set_ylim(-maxv1, maxv1)
        
    plt.savefig(r'real_data_results/test_comparison.jpg')

    plt.figure(figsize=(10, 5))
    plt.scatter(full_pos[:K, 1], full_pos[:K, 0], 
                marker='^', color='black', 
                s=100, label='Selected Stations')

    plt.scatter(full_pos[K:, 1], full_pos[K:, 0], 
                marker='^', color='red', 
                s=100, label='Test Stations')

    plt.xlim(0, 63)
    plt.ylim(0, 31)
    plt.xticks([0, 19.4, 38.79, 58.18], ['0km', '100km', '200km', '300km'], fontsize=20)
    plt.yticks([0, 19.4], ['', '100km'], fontsize=20)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(r'real_data_results/stations.jpg')

    fig, axes = plt.subplots(s, 1, figsize=(10, 2*s))
    plt.subplots_adjust(left=0.2)
    maxv2 = 1.2*np.max(np.abs(ob[:,:s]))
    for i in range(s):
        pos = full_pos[i]
        results = []
        for j in range(T):
            interp_func.values = rec_video[j,:,:]
            results.append(interp_func(pos, method='linear'))
        results = np.hstack(results) 
        axes[i].plot(t, ob[:,i], color='black')
        axes[i].plot(t, results, color='red')
        # axes[i].plot(t, results-ob[:,i], color='green')
        xs = round(pos[0]*330/64,2)
        ys = round(pos[1]*330/64,2)
        axes[i].text(0.98, 0.9, f'{sta_info[i][0]}.{sta_info[i][1]}({ys}km, {xs}km)', 
                transform=axes[i].transAxes,
                verticalalignment='top',
                horizontalalignment='right', fontsize=20)
        axes[i].set_ylim(-maxv2, maxv2)
        axes[i].tick_params(axis='x', labelsize=20)
        axes[i].tick_params(axis='y', labelsize=20)
        fig.supylabel('Amplitude (m/s)',fontsize=30)
        if i!=s-1:
            axes[i].set_xticks([])
        else:
            axes[i].set_xlabel('Seconds after earthquake',fontsize=30)
    plt.savefig(r'real_data_results/err_comparison.jpg')
    
def show_synthetic_data_results(rec_video: np.ndarray, ori_video: np.ndarray, test_pos: np.ndarray):
    """present the synthetic data results by comparing the reconstructed video with the original video at the test positions, 
    which can be used to evaluate the performance of the reconstruction algorithm on synthetic data.

    Args:
        rec_video (np.ndarray): the reconstructed video, shape (T, H, W)
        ori_video (np.ndarray): the original video, shape (T, H, W)
        test_pos (np.ndarray): the positions of the test stations, shape (K, 2)
        K (int): the number of test stations
    """
    T, H, W = rec_video.shape
    h_coords, w_coords, t_coords = np.arange(H), np.arange(W), np.arange(T)
    interp_func = RegularGridInterpolator(
        (h_coords, w_coords),
        np.zeros((H,W)))

    delta = 1
    t = np.linspace(0, T*delta, T)
    k = test_pos.shape[0]
    fig, axes = plt.subplots(k, 1, figsize=(8, 2*k))
    axes = np.asarray(axes)
    plt.subplots_adjust(left=0.2)
    maxv = np.max(np.abs(ori_video))
    for i in range(k):
        rec_results = []
        ori_results = []
        pos = test_pos[i]
        for j in range(T):
            interp_func.values = rec_video[j,:,:]
            rec_results.append(interp_func(pos, method='linear'))
            interp_func.values = ori_video[j,:,:]
            ori_results.append(interp_func(pos, method='linear'))
        ori_results = np.array(ori_results)
        rec_results = np.array(rec_results)
        axes[i].plot(t, ori_results, color='black')
        axes[i].plot(t, rec_results, color='red')
        # axes[i].plot(t, rec_results-ori_results, color='green')
        xs = round(pos[0]*100/64,2)
        ys = round(100-pos[1]*100/64,2)
        axes[i].text(0.98, 0.9, f'({xs}km, {ys}km)', 
                transform=axes[i].transAxes,
                verticalalignment='top',
                horizontalalignment='right', fontsize=20)
        axes[i].tick_params(axis='x', labelsize=20)
        axes[i].tick_params(axis='y', labelsize=20)
        if i!=k-1:
            axes[i].set_xticks([])
        else:
            axes[i].set_xlabel('Seconds after earthquake',fontsize=30)
        
        fig.supylabel('Amplitude (mm/s)',fontsize=30)
        axes[i].set_ylim(-maxv, maxv)
    plt.savefig(r'synthetic_data_results/test_comparison.jpg')

def compute_reduced_dictionary_GPU(T: int, H: int, W: int, K: int, pos_np: np.ndarray, indices_3d_np: np.ndarray, pos_type: str = 'float',
                                   device: str = 'cuda', apply_random_time_sampling: bool = False, sample_rate: float = 0.5):
    """
    compute the reduced dictionary using indices_3d by parallel computing on GPU,
    which can be used for direct optimization algorithms like Lasso or OMP
    
    Args:
        T (int): the number of time steps
        H (int): the height of the spatial grid
        W (int): the width of the spatial grid
        K (int): the number of observations
        pos (np.ndarray): the positions of the station locations, shape (K, 2)
        indices_3d (np.ndarray): the indices of the reduced dictionary
        pos_type (str, optional): the type of positions. Defaults to 'float'.
        device (str, optional): the device to use for computation. Defaults to 'cuda'.

    Returns:
        (A, B): the computed dictionaries
    """
    C = indices_3d_np.shape[0]
    S = int(T * sample_rate) if apply_random_time_sampling else T
    P = K*S

    BFH = torch.from_numpy(idct(np.eye(H), axis=0, norm='ortho')).to(dtype=torch.float32, device=device)
    BFW = torch.from_numpy(idct(np.eye(W), axis=0, norm='ortho')).to(dtype=torch.float32, device=device)
    BFT = torch.from_numpy(idct(np.eye(T), axis=0, norm='ortho')).to(dtype=torch.float32, device=device)

    pos = torch.as_tensor(pos_np, dtype=torch.float32, device=device)
    indices_3d = torch.as_tensor(indices_3d_np, dtype=torch.long, device=device)

    t_idx = indices_3d[:, 0]
    h_idx = indices_3d[:, 1]
    w_idx = indices_3d[:, 2]
    
    posh, posw = pos[:, 0], pos[:, 1]

    if pos_type == 'int':
        posh_long, posw_long = posh.long(), posw.long()
        val_H = BFH[posh_long.unsqueeze(1), h_idx]  # [K, C]
        val_W = BFW[posw_long.unsqueeze(1), w_idx]  # [K, C]
    else:
        def interpolate_basis(basis, p, idx):
            p0 = p.floor().long().clamp(0, basis.size(0) - 1)
            p1 = (p0 + 1).clamp(0, basis.size(0) - 1)
            weight = (p - p0.float()).unsqueeze(1)

            v0 = basis[p0.unsqueeze(1), idx]
            v1 = basis[p1.unsqueeze(1), idx]
            return v0 * (1 - weight) + v1 * weight

        val_H = interpolate_basis(BFH, posh, h_idx) # [K, C]
        val_W = interpolate_basis(BFW, posw, w_idx) # [K, C]

    val_HW = val_H * val_W  # [K, C]

    if apply_random_time_sampling:
        rand_matrix = torch.rand(K, T, device=device)
        post = torch.topk(rand_matrix, S, dim=1).indices # [K, S]
        post, _ = post.sort(dim=1)
    else:
        post = torch.arange(T, device=device).repeat(K, 1) # [K, T]

    w_out = posh.repeat_interleave(S)
    h_out = posw.repeat_interleave(S)
    t_out = post.reshape(-1) # [P]
    k_idx_out = torch.arange(K, device=device).repeat_interleave(S)
    coords_3d = torch.stack([w_out, h_out, t_out.float(), k_idx_out.float()], dim=-1)
    
    val_HW_expanded = val_HW.repeat_interleave(S, dim=0)
    val_T_sampled = BFT[t_out.unsqueeze(1), t_idx]
    A = (val_HW_expanded * val_T_sampled).T # [C, P]

    norms = torch.linalg.norm(A, axis=1, keepdims=True)
    B = A / (norms + 1e-8)
    return A, B, coords_3d

def recovery_GPU(ob_np: np.ndarray, H: int, W: int, T: int, pos: np.ndarray, indices_3d: np.ndarray, 
                 A=None, B=None, coords_3d=None, iters: int = 1000, device: str = 'cuda', method: str = 'SAMP'):
    
    norm = torch.linalg.norm
    _, K = ob_np.shape
    Sq = H * W
    C = Sq * T
    
    if coords_3d is None:
        A, B, coords_3d = compute_reduced_dictionary_GPU(T, H, W, K, pos, indices_3d)
    t_indices = coords_3d[:, 2].long()
    k_indices = coords_3d[:, 3].long()

    ob = torch.as_tensor(ob_np, dtype=torch.float32, device=device)
    shaped_ob = ob[t_indices, k_indices]
    
    rk = shaped_ob.clone()
    err = 1e-3 * norm(rk)
    xr = torch.zeros((T, H, W), dtype=torch.float32, device=device)
    
    indices_1d = indices_3d[:, 0] * H * W + indices_3d[:, 1] * W + indices_3d[:, 2]
    indices_1d = torch.tensor(indices_1d, device=device)

    if not iters:
        iters = K * T

    sparsity_list = [0, 20, 50, 100, 200, 300, 500, 800, 1000, 100000]
    
    D_T = get_1d_idct_matrix(T, device)
    D_H = get_1d_idct_matrix(H, device)
    D_W = get_1d_idct_matrix(W, device)

    def recover_to_video(xk, Skpos, t, L):
        Skpos_1d = indices_1d[Skpos]
        xr.zero_()
        
        tidx = torch.div(Skpos_1d, Sq, rounding_mode='floor')
        hidx = torch.div(Skpos_1d % Sq, W, rounding_mode='floor')
        widx = Skpos_1d % W
        
        xr[tidx, hidx, widx] = xk.to(xr.dtype)
        
        rec_video = torch.einsum('wc, abc -> abw', D_W, xr)
        rec_video = torch.einsum('hb, abw -> ahw', D_H, rec_video)
        rec_video = torch.einsum('ta, ahw -> thw', D_T, rec_video)
        
        print(f's: {t} L: {L}, ctn: {ctn:.4f}')
        return rec_video
    
    if method == 'OMP':
        xk, Skpos, s, ctn = OMP(A, B, rk, shaped_ob, iters, err, sparsity_list, device)
        return recover_to_video(xk, Skpos, s, s)

    elif method == 'SAMP':
        xk, Gamma_tL, t, L, ctn = SAMP(A, B, rk, shaped_ob, iters, err, sparsity_list, device)
        return recover_to_video(xk, Gamma_tL, t, L)

def CS3D_for_synthetic_data(event_for_spectrum_statistics: np.ndarray, sampling_ratio: float = 0.05, method: str = 'OMP', reload: bool = False, plot: bool = False):
    """
    perform the CS3D reconstruction for synthetic data, 
    which can be used to evaluate the performance of the reconstruction algorithm on synthetic data, 
    and the reconstructed video will be compared with the original video at the test positions.   
    """
    T_clip = 64
    overlap_len = 8
    clip_num = 4
    # the total length of the video after jointing the clips, which is determined by the clip length, the number of clips and the overlap length
    T = T_clip + (clip_num - 1) * (T_clip - overlap_len) 
    H, W = 64, 64
    full_video = normalize(event_for_spectrum_statistics[50:450,:,:,0])
    video = full_video[32:T+32,32:H+32,32:W+32]

    if not reload:
        ob, ob_pos, test_pos = gene_mask(video, H, W, 10, sample_ratio=sampling_ratio, mask_type='float') # random
        np.save(r'synthetic_data_results\ob_for_reloading.npy', ob)
        np.save(r'synthetic_data_results\ob_pos_for_reloading.npy', ob_pos)
        np.save(r'synthetic_data_results\test_pos_for_reloading.npy', test_pos)

    else:
        ob = np.load(r'synthetic_data_results\ob_for_reloading.npy')
        ob_pos = np.load(r'synthetic_data_results\ob_pos_for_reloading.npy')
        test_pos = np.load(r'synthetic_data_results\test_pos_for_reloading.npy')
        
    K = ob.shape[1]
    rec_video_list = []
    indices_3d = main_spectrum_statistics(full_video[50:306], full_video[:T_clip,:H,:W], step=1, plot=False)
    A, B, coords_3d = compute_reduced_dictionary_GPU(T_clip, H, W, K, ob_pos, indices_3d)
    for i in range(clip_num):
        print(f"clip {i}")
        start_frame = i*(T_clip - overlap_len)
        rec_video = recovery_GPU(ob[start_frame:start_frame+T_clip], H, W, T_clip, ob_pos, indices_3d, A=A, B=B, coords_3d=coords_3d, method=method)
        rec_video_list.append(rec_video)
    rec_video = joint(rec_video_list, overlap_len)
    # rec_video = np.load(r'cs3D_theo_data\rec_video_theo.npy')
    if plot:
        animate_comparaison_for_two_videos(video, rec_video, pos=ob_pos, test_pos=test_pos, save=True)
    np.save(r'synthetic_data_results\rec_video_theo.npy', rec_video)
    show_synthetic_data_results(rec_video, video, test_pos)

def CS3D_for_real_data(event_for_spectrum_statistics: np.ndarray, allpoints: np.ndarray, full_pos: np.ndarray, 
                       sta_info: np.ndarray, sampling_ratio: float = 0.05, method: str = 'OMP', reload: bool = False, plot: bool = False):
    """perform  the CS3D reconstruction for real data, 
    which can be used to evaluate the performance of the reconstruction algorithm on real data, 
    and the reconstructed video will be compared with the observations at the test positions, 
    and the comparison results will be presented in show_real_data_results function. 

    Args:
        event_for_spectrum_statistics (np.ndarray): the event data used for spectrum statistics, shape (T_total, H_total, W_total)
        allpoints (np.ndarray): the observed points, shape (T, K)
        full_pos (np.ndarray): the full positions of the observed points, shape (K, 2)
        sta_info (np.ndarray): the station information, shape (K,)
        reload (bool, optional): whether to reload the test data. Defaults to False.
    """
    H, W, T_clip = 32, 64, 64
    full_video = normalize(event_for_spectrum_statistics[50:306,:,:,0])
    video = full_video[80:80+T_clip,32:32+H,32:32+W]
    allpoints = taper.smooth_transition(allpoints)

    keep = int(sampling_ratio * H * W)
    if not reload:
        ob, test_sta, full_pos, sta_info = shuffle(allpoints, full_pos, sta_info, keep=keep) # random
    else:
        test_sta_info = np.loadtxt(r"real_data_results\test_sta_info.txt", dtype=str)
        ob, test_sta, full_pos, sta_info = get_sta_pos(sta_info, test_sta_info, allpoints, full_pos)
        
    T,K = ob.shape
    overlap_len = 8
    clip_num = 16
    rec_video_list = []
    indices_3d = main_spectrum_statistics(full_video, video, step=1, plot=False)
    A, B, coords_3d = compute_reduced_dictionary_GPU(T_clip, H, W, K, full_pos[:K], indices_3d_np=indices_3d)
    
    for i in range(clip_num):
        print(f"clip {i}")
        start_frame = i*(T_clip - overlap_len)
        rec_video = recovery_GPU(ob[start_frame:start_frame+T_clip], H, W, T_clip, full_pos[:K], indices_3d, A=A, B=B, coords_3d=coords_3d, method=method)
        rec_video_list.append(rec_video)
        
    rec_video = joint(rec_video_list, overlap_len)
    np.save(r'real_data_results\rec_video.npy', rec_video)
    # rec_video = np.load('rec_video.npy')
    show_real_data_results(T, H, W, rec_video, ob, sta_info, test_sta, full_pos, plot=plot)

if __name__ == '__main__':
    """ run the CS3D reconstruction at this dictionary, 
    you can change the path to run the reconstruction for your own data, 
    and you can also choose to run the reconstruction for synthetic data or real data by commenting the corresponding lines.
    """
    event_path = r"D:\research\3DCS\upload\synthetic_event0.npy"
    # loading the data for real data experiment
    channel = 0
    allpoints = np.load(r'real_data_results\all_sta_data.npy')[:,:,channel]
    full_pos = np.load(r'real_data_results\sta_pos_in_grid.npy')
    sta_info = np.loadtxt(r"real_data_results\sta_info.txt", dtype=str)
    event_for_spectrum_statistics = np.load(event_path)
    
    start = time.time()
    # CS3D_for_real_data(event_for_spectrum_statistics, allpoints, full_pos, sta_info, sampling_ratio=0.04, method='SAMP', reload=False, plot=False)
    CS3D_for_synthetic_data(event_for_spectrum_statistics, sampling_ratio=0.04, method='OMP', reload=False, plot=False)
    end = time.time()
    print(f'used_time: {end - start:.1f} s')

    
