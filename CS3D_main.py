import numpy as np
from scipy.fftpack import idct,dct
from plot_tools import animate_comparaison_for_two_videos, main_spectrum_statistics, animate_a_video
from tools import gene_mask, shuffle, joint, get_sta_pos, normalize
import taper
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed
from tqdm import tqdm
import time

def show_real_data_results(T: int, H: int, W: int, rec_video: np.ndarray, ob: np.ndarray, sta_info: list, test_sta: np.ndarray, full_pos: np.ndarray):
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
    maxv1 = 1.2*np.max(abs(test_sta)) # the maximum absolute value among the test stations, used for setting the y-axis limits in the comparison figure
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
                s=100, label='Not-selected Sattions')

    plt.xlim(0, 63)
    plt.ylim(0, 31)
    plt.xticks([0, 19.4, 38.79, 58.18], ['0km', '100km', '200km', '300km'], fontsize=20)
    plt.yticks([0, 19.4], ['', '100km'], fontsize=20)
    # plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(r'real_data_results/stations.jpg')

    fig, axes = plt.subplots(s, 1, figsize=(10, 2*s))
    plt.subplots_adjust(left=0.2)
    maxv2 = 1.2*np.max(abs(ob[:,:s]))
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
    plt.subplots_adjust(left=0.2)
    maxv = np.max(abs(ori_video))
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

    
def compute_reduced_dictionary_CPU_mpi(T: int, H: int, W: int, pos: np.ndarray, indices_3d: np.ndarray, req: str = 'AB', pos_type: str = 'float'):
    """
    compute the reduced dictionary using indices_3d by parallel computing on CPU,
    which can be used for direct optimization algorithms like Lasso or OMP
    
    Args:
        T (int): the number of time steps
        H (int): the height of the spatial grid
        W (int): the width of the spatial grid
        pos (np.ndarray): the positions of the station locations
        indices_3d (np.ndarray): the indices of the reduced dictionary
        req (str, optional): the requested output dictionaries. Defaults to 'AB'.
        pos_type (str, optional): the type of positions. Defaults to 'float'.

    Returns:
        A or (A, B): the computed dictionaries A and/or B
    """
    C,K = indices_3d.shape[0], pos.shape[0]
    norm = np.linalg.norm
    P = K*T
    A = np.zeros((C, P), dtype=np.float32)
    BFH = idct(np.eye(H,dtype=np.float32), axis=0, norm='ortho')
    BFW = idct(np.eye(W,dtype=np.float32), axis=0, norm='ortho')
    BFT = idct(np.eye(T,dtype=np.float32), axis=1, norm='ortho').reshape(1,T,T)
    
    if pos_type=='int':
        posh,posw = pos[:, 0],pos[:, 1]
        for i,idx in enumerate(indices_3d):
            if i%10000 == 0:
                print(i)
            u1 = BFH[:,idx[1]]
            u2 = u1[:, None] * BFW[None, :, idx[2]]
            u3 = u2[:, :, None] * BFT[0,idx[0],:]
            A[i,:] = u3[posh, posw, :].flatten()
    
    elif pos_type=='float':
        def process_one(idx):
            h_coords,w_coords = np.arange(H),np.arange(W)
            interp_func = RegularGridInterpolator(
                (h_coords, w_coords),
                np.zeros((H,W)))
            data = np.zeros((K, T))
            u1 = BFH[:, idx[1]]
            u2 = u1[:, None] * BFW[None, :, idx[2]]
            u3 = u2[:, :, None] * BFT[0, idx[0], :]
            for t in range(T):
                interp_func.values = u3[:,:,t]
                data[:, t] = interp_func(pos, method='cubic')
            return data.ravel()
        results = Parallel(n_jobs=-1)(delayed(process_one)(idx) for idx in tqdm(indices_3d, desc= f"Processing"))
        A = np.stack(results, axis=0)
        
    np.save('A.npy', A)
    if req=='AB':
        norms = norm(A, axis=1, keepdims=True)
        B = A / norms
        np.save('B.npy',B)
        return A, B
    elif req=='A':
        return A

def compute_reduced_dictionary_GPU(T: int, H: int, W: int, K: int, pos: np.ndarray, indices_3d: np.ndarray, req: str = 'AB', device: str = 'cuda'):
    """
    compute the reduced dictionary using indices_3d by parallel computing on GPU,
    which can be used for direct optimization algorithms like Lasso or OMP
    
    Args:
        T (int): the number of time steps
        H (int): the height of the spatial grid
        W (int): the width of the spatial grid
        pos (np.ndarray): the positions of the station locations
        indices_3d (np.ndarray): the indices of the reduced dictionary
        req (str, optional): the requested output dictionaries. Defaults to 'AB'.
        pos_type (str, optional): the type of positions. Defaults to 'float'.

    Returns:
        A or (A, B): the computed dictionaries A and/or B
    """
    C = indices_3d.shape[0]
    norm = torch.linalg.norm
    P = K*T

    BFH = torch.tensor(idct(np.eye(H), axis=0, norm='ortho'), dtype=torch.float32, device=device)
    BFW = torch.tensor(idct(np.eye(W), axis=0, norm='ortho'), dtype=torch.float32, device=device)
    BFT = torch.tensor(idct(np.eye(T), axis=0, norm='ortho'), dtype=torch.float32, device=device)

    pos = torch.tensor(pos, dtype=torch.long, device=device)
    indices_3d = torch.tensor(indices_3d, dtype=torch.long, device=device)
    A = torch.zeros((C, P), dtype=torch.float32, device=device)
    for i in range(C):
        if i%1000 == 0:
            print(i)
        t_idx, h_idx, w_idx = indices_3d[i]
        u1 = BFH[:,h_idx]
        u2 = u1[:, None] * BFW[:, w_idx]
        u3 = u2[:, :, None] * BFT[:, t_idx]
        A[i,:] = u3[pos[:, 0], pos[:, 1], :].flatten()

    if req=='AB':
        norms = norm(A, axis=1, keepdims=True)
        B = A / norms
        return A, B
    elif req=='A':
        return A

def compute_full_dic_GPU(T: int, H: int, W: int, K: int, pos: np.ndarray, device: str = 'cuda', pos_type: str = 'int'):
    """
    compute full dictionary A for the whole space, which can be used for direct optimization algorithms like Lasso or OMP, 
    but it is not memory efficient and time efficient, 
    so we recommend using the reduced dictionary computed by compute_reduced_dictionary_CPU_mpi or compute_reduced_dictionary_GPU instead.

    Args:
        T (int): the number of time steps
        H (int): the height of the spatial grid
        W (int): the width of the spatial grid
        K (int): the number of spatial locations
        pos (np.ndarray): the positions of the spatial locations
        device (str, optional): the device to use for computation. Defaults to 'cuda'.
        pos_type (str, optional): the type of positions. Defaults to 'int'.

    Returns:
        A: the full dictionary matrix, shape (H*W*T, K*T)
        B: the normalized dictionary matrix, shape (H*W*T, K*T)
    """
    norm = torch.linalg.norm
    P = K*T
    L = H*W
    C = L*T
    # using unitary dictionary to calculate the dictionary fastly
    BFH = torch.tensor(idct(np.eye(H), axis=0, norm='ortho'), dtype=torch.float32, device=device)
    BFW = torch.tensor(idct(np.eye(W), axis=0, norm='ortho'), dtype=torch.float32, device=device)
    BFT = torch.tensor(idct(np.eye(T), axis=0, norm='ortho'), dtype=torch.float32, device=device)
    if pos_type == 'float':
        pos = torch.tensor(pos, dtype=torch.float32, device=device)
    A = torch.zeros((C, P), dtype=torch.float32, device=device)
    for h in range(H):
        u1 = BFH[:,h]
        print(h)
        for w in range(W):
            u2 = u1[:, None] * BFW[:, w]
            for k in range(T):
                j = k*L + h*W + w
                u3 = u2[:, :, None] * BFT[:, k]
                A[j,:] = u3[pos[:, 0], pos[:, 1], :].flatten()
    norms = norm(A, axis=1, keepdims=True)
    B = A / norms
    return A, B

def recovery_CPU(ob: np.ndarray, pos: np.ndarray, H: int, W: int, T: int, indices_3d: np.ndarray, iters: int = 0):
    norm = np.linalg.norm
    T,K = ob.shape
    Sq = H*W
    C = Sq*T
    shaped_ob = ob.flatten(order='F')
    rk = shaped_ob.copy()
    err = 1e-3*norm(rk)
    xr = np.zeros((T,H,W))
    
    indices_1d = indices_3d[:,0]*H*W + indices_3d[:,1]*W + indices_3d[:,2]
    A, B = compute_reduced_dictionary_CPU_mpi(T, H, W, K, pos, indices_3d)

    mask = np.ones(A.shape[0], dtype=bool)
    Sk,Skpos = [], []
    
    if iters is None:
        iters = K*T-1
        
    for s in range(iters):
        if s%100 == 0:
            print(s)
        res = B @ rk
        res[~mask] = 0
        maxindex = np.argmax(np.abs(res))
        Skpos.append(maxindex)
        Sk.append(A[maxindex, :])
        mask[maxindex] = False
        Sk_slice = np.stack(Sk, axis=1)
        Asm = Sk_slice.T @ Sk_slice
        try:
            L_chol = np.linalg.cholesky(Asm)
        except:
            lam = np.sum(np.abs(Asm)) / (C * 10)
            L_chol = np.linalg.cholesky(Asm + lam * np.eye(s+1))
        y = np.linalg.solve(L_chol, Sk_slice.T @ shaped_ob)
        xk = np.linalg.solve(L_chol.T, y)
        rk = shaped_ob - Sk_slice @ xk
        ctn = norm(rk)
        if ctn < err:
            print(f's:{s}  err:{err:.3e}  ctn:{ctn:.3e}')
            break

    if indices_3d is not None:
        Skpos = indices_1d[Skpos]
    for i, idx in enumerate(Skpos):
        tidx = idx // Sq
        hidx = (idx % Sq) // W
        widx = idx % W
        xr[tidx, hidx, widx] = xk[i]

    print(f's:{s}  err:{err:.3e}  ctn:{ctn:.3e}')
    return xr

def recovery_GPU(ob: np.ndarray, H: int, W: int, T: int, pos: np.ndarray, indices_3d: np.ndarray, A=None, B=None, iters: int = 1000, device: str = 'cuda', method: str = 'SAMP'):
    """
    recover a video from the observations using the dictionary computed by compute_reduced_dictionary_GPU or compute_reduced_dictionary_CPU_mpi, 
    which can be used for direct optimization algorithms like Lasso or greedy algorithms like OMP, 
    but here we implement OMP and SAMP as examples to show how to use the computed dictionary for recovery, 
    and you can also implement other optimization algorithms like Lasso by using the computed dictionary.

    Args:
        ob (np.ndarray): the observations, shape (T, K)
        H (int): the height of the spatial grid
        W (int): the width of the spatial grid
        T (int): the number of time steps
        pos (np.ndarray): the positions of the station locations, shape (K, 2)
        indices_3d (np.ndarray): the indices of the reduced dictionary, shape (C, 3)
        A (np.ndarray, optional): the dictionary matrix A, shape (H*W*T, K*T). If None, it will be computed by compute_reduced_dictionary_CPU_mpi. Defaults to None.
        B (np.ndarray, optional): the dictionary matrix B, shape (H*W*T, K*T). If None, it will be computed by compute_reduced_dictionary_CPU_mpi. Defaults to None.
        iters (int, optional): the number of iterations for the recovery algorithm. Defaults to 1000.
        device (str, optional): the device to use for computation ('cuda' or 'cpu'). Defaults to 'cuda'.
        method (str, optional): the recovery method to use ('OMP' or 'SAMP'). Defaults to 'SAMP'.

    Returns:
        torch.Tensor: the recovered video, shape (T, H, W)
    """
    norm = torch.linalg.norm
    _,K = ob.shape
    Sq = H*W
    C = Sq*T
    shaped_ob = torch.tensor(ob.flatten(order='F'), dtype=torch.float32, device=device)
    rk = shaped_ob.clone()
    err = 1e-3*norm(rk)
    xr = torch.zeros((T,H,W), dtype=torch.float32, device=device)
    indices_1d = indices_3d[:,0]*H*W + indices_3d[:,1]*W + indices_3d[:,2]
    indices_1d = torch.tensor(indices_1d, device=device)
    if A is None:
        A, B = compute_reduced_dictionary_CPU_mpi(T,H,W,K,pos,indices_3d)
    A, B = torch.tensor(A, device=device, dtype=torch.float32), torch.tensor(B, device=device, dtype=torch.float32)

    if not iters:
        iters = K*T

    ctn = norm(rk)
    spa_list = [0,20,50,100,200,300,500,800,1000,100000]
    
    def recover_to_video(xk, Skpos, t, L):
        Skpos = indices_1d[Skpos]
        for i, idx in enumerate(Skpos):
            idx = idx.item()
            tidx = idx // Sq
            hidx = (idx % Sq) // W
            widx = idx % W
            xr[tidx, hidx, widx] = xk[i]
        coe = xr.cpu().numpy()
        rec_video = idct(idct(idct(coe,axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho')
        # mse = np.mean((rec_video - video)**2)
        # PSNR = 10*np.log10(maxv**2/mse)
        print(f's: {t} L: {L} ctn: {ctn}')
        return rec_video
        
    if method == 'OMP':
        mask = torch.ones(A.shape[0], dtype=torch.bool, device=device)
        Sk = []
        Skpos = []
        for s in range(iters+1):
            res = B @ rk
            res[~mask] = 0
            maxindex = torch.argmax(torch.abs(res)).item()
            Skpos.append(maxindex)
            Sk.append(A[maxindex, :])
            mask[maxindex] = False
            Sk_slice = torch.stack(Sk, dim=1)
            Asm = Sk_slice.T @ Sk_slice
            try:
                L_chol = torch.linalg.cholesky(Asm)
            except:
                lam = torch.sum(torch.abs(Asm)) / (C * 10)
                L_chol = torch.linalg.cholesky(Asm + lam * torch.eye(s+1, device=device))
            y = torch.linalg.solve(L_chol, (Sk_slice.T @ shaped_ob).unsqueeze(1))
            xk = torch.linalg.solve(L_chol.T, y)
            rk = shaped_ob - (Sk_slice @ xk).squeeze(1)
            ctn = norm(rk)
            if ctn < err:
                Skpos = torch.tensor(Skpos, device=device)
                break
            elif s >= spa_list[0]:
                recover_to_video(xk, torch.tensor(Skpos, device=device), s, s)
                spa_list.pop(0)
        return recover_to_video(xk, Skpos, s, s)
    
    elif method == 'SAMP':
        Gamma_t = torch.tensor([], dtype=torch.long,device=device)
        S = 1 # parameter for SAMP, which controls the number of new indices added to the support set in each iteration
        L = S # parameter 
        for t in range(iters*10):
            res = B @ rk
            _, maxindex = torch.topk(torch.abs(res), k=L)
            Ck = torch.unique(torch.cat([Gamma_t, maxindex]))
            At = A[Ck,:].T
            Atmp = At.T @ At
            try:
                L_chol = torch.linalg.cholesky(Atmp)
            except:
                lam = torch.sum(torch.abs(Atmp)) / (C * 10)
                L_chol = torch.linalg.cholesky(Atmp + lam * torch.eye(Atmp.shape[0], device=device))
            y = torch.linalg.solve(L_chol, (At.T @ shaped_ob).unsqueeze(1))
            theta_t = torch.linalg.solve(L_chol.T, y).squeeze(1)
            _,maxindex = torch.topk(torch.abs(theta_t), k=L, dim=0)
            AtL = At[:,maxindex]
            AtLT = AtL.T
            Gamma_tL = Ck[maxindex]
            rk_new = shaped_ob - AtL @ torch.linalg.solve(AtLT @ AtL, AtLT @ shaped_ob)
            ctn = norm(rk)
            if norm(rk_new) >= norm(rk):
                L = L + S
            elif norm(rk_new) <= err:
                break
            elif L < iters+11:
                if L >= spa_list[0]:
                    xk = theta_t[maxindex]
                    Skpos = Gamma_tL
                    recover_to_video(xk, Skpos, t, L)
                    spa_list.pop(0)
                Gamma_t = Gamma_tL
                rk = rk_new
                t = t + 1
            else:
                break
        xk = theta_t[maxindex]
        Skpos = Gamma_tL
        return recover_to_video(xk, Skpos, t, L)


def CS3D_for_synthetic_data(event_for_spectrum_statistics: np.ndarray):
    """
    perform the CS3D reconstruction for synthetic data, 
    which can be used to evaluate the performance of the reconstruction algorithm on synthetic data, 
    and the reconstructed video will be compared with the original video at the test positions.   
    """
    # the original video has 256 frames, but we only use 232 frames to make sure the size of the video is divisible by the clip size (64) and the overlap size (8)
    H, W, T = 64, 64, 232 
    full_video = normalize(event_for_spectrum_statistics[50:450,:,:,0])
    video = full_video[32:T+32,32:H+32,32:W+32]
    T_clip = 64
    overlap_len = 8
    clip_num = 4
    ob, ob_pos, test_pos = gene_mask(video, H, W, 10, sample_ratio=0.04, mask_type='float')
    rec_video_list = []
    indices_3d = main_spectrum_statistics(full_video[50:306], full_video[:T_clip,:H,:W], step=1, plot=True)
    A, B = compute_reduced_dictionary_CPU_mpi(T_clip, H, W, ob_pos, indices_3d)
    for i in range(clip_num):
        print(f"clip {i}")
        start_frame = i*(T_clip - overlap_len)
        rec_video = recovery_GPU(ob[start_frame:start_frame+T_clip], H, W, T_clip, ob_pos, indices_3d, A=A, B=B)
        rec_video_list.append(rec_video)
    rec_video = joint(rec_video_list, overlap_len)
    # rec_video = np.load(r'cs3D_theo_data\rec_video_theo.npy')
    animate_comparaison_for_two_videos(video, rec_video, pos=ob_pos, test_pos=test_pos, save=True)
    np.save(r'synthetic_data_results\rec_video_theo.npy', rec_video)
    show_synthetic_data_results(rec_video, video, test_pos)

def CS3D_for_real_data(event_for_spectrum_statistics: np.ndarray, allpoints: np.ndarray, full_pos: np.ndarray, sta_info: np.ndarray, reload: bool = False):
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

    if not reload:
        ob, test_sta, full_pos, sta_info = shuffle(allpoints, full_pos, sta_info, keep=50)
    else:
        test_sta_info = np.loadtxt(r"real_data_results\test_sta_info.txt", dtype=str)
        ob, test_sta, full_pos, sta_info = get_sta_pos(sta_info, test_sta_info, allpoints, full_pos)
        
    T,K = ob.shape
    overlap_len = 8
    clip_num = 16
    rec_video_list = []
    indices_3d = main_spectrum_statistics(full_video, video, step=1, plot=True)
    A, B = compute_reduced_dictionary_CPU_mpi(T_clip, H, W, full_pos[:K], indices_3d)
    
    for i in range(clip_num):
        print(f"clip {i}")
        start_frame = i*(T_clip - overlap_len)
        rec_video = recovery_GPU(ob[start_frame:start_frame+T_clip], H, W, T_clip, full_pos[:K], indices_3d, A=A, B=B)
        rec_video_list.append(rec_video)
        
    rec_video = joint(rec_video_list, overlap_len)
    np.save(r'real_data_results\rec_video.npy', rec_video)
    # rec_video = np.load('rec_video.npy')
    show_real_data_results(T, H, W, rec_video, ob, sta_info, test_sta, full_pos)

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
    CS3D_for_real_data(event_for_spectrum_statistics, allpoints, full_pos, sta_info)
    # CS3D_for_synthetic_data(event_for_spectrum_statistics)
    end = time.time()
    print(f'used_time: {end - start:.1f} s')

    
