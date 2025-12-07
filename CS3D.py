import numpy as np
from scipy.fftpack import idct,dct
from plot import animate_matrices,present_statistics,present_statistics_GPU,animate_a_video
import taper
import cvxpy as cp
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed
from tqdm import tqdm
import time

def generate_simple_indices_3d(T,H,W,n,nt):
    shape = (T//nt, H//n, W//n)
    indices = np.indices(shape)
    indices_3d = indices.transpose(1, 2, 3, 0).reshape(-1, 3)
    return indices_3d

def CS3Dreal_show(T,H,W,rec_video, ob, sta_info, test_points, full_pos):
    k = test_points.shape[1]
    np.save('rec_video.npy',rec_video)
    # rec_video = np.load('rec_video.npy')
    animate_a_video(rec_video, full_pos[:-k], normalize=False, save=True, color=ob)

    h_coords,w_coords,t_coords = np.arange(H),np.arange(W),np.arange(T)
    interp_func = RegularGridInterpolator(
        (h_coords, w_coords),
        np.zeros((H,W)))

    delta = 1
    t = np.linspace(0, T*delta, T)
    s = test_points.shape[1]
    s = 30
    K = ob.shape[1]
    fig, axes = plt.subplots(s, 1, figsize=(10, 2*s))
    plt.subplots_adjust(left=0.2)
    maxv1 = 1.2*np.max(abs(test_points))
    for i in range(s):
        pos = full_pos[K+i]
        results = []
        for j in range(T):
            interp_func.values = rec_video[j,:,:]
            results.append(interp_func(pos, method='linear'))
        results = np.hstack(results) 
        axes[i].plot(t, test_points[:,i], color='black')
        axes[i].plot(t, results, color='red')
        # axes[i].plot(t, results-test_points[:,i], color='green')
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
        
    plt.savefig(r'data/test_comparison.jpg')

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
    plt.savefig(r'data/stations.jpg')

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
    plt.savefig(r'data/err_comparison.jpg')
    

def CS3D_show(rec_video, ori_video, test_pos):
    T,H,W = rec_video.shape
    h_coords,w_coords,t_coords = np.arange(H),np.arange(W),np.arange(T)
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
    plt.savefig(r'cs3D_theo_data/test_comparison.jpg')


def normalize(video):
    return video/np.max(abs(video))

def gene_mask(video,H,W,K,sample_ratio=0.01,mask_type=float):
    L = H*W
    sample_num = int(L*sample_ratio)
    if mask_type == 'int':
        mask_per_frame = np.hstack([np.zeros(L - sample_num), np.ones(sample_num)])
        np.random.shuffle(mask_per_frame)
        mask_per_frame = mask_per_frame.reshape(H,W)
        ob_pos = np.argwhere(mask_per_frame == 1)
        ob = video[:,ob_pos[:,0],ob_pos[:,1]]
    elif mask_type == 'float':
        x = np.random.uniform(0, W-1, size=sample_num + K)
        y = np.random.uniform(0, H-1, size=sample_num + K)
        all_pos = np.column_stack((x, y))
        ob_pos,test_pos = all_pos[:-K],all_pos[-K:]
        T = video.shape[0]
        for i in range(T):
            h_coords,w_coords = np.arange(H),np.arange(W)
            interp_func = RegularGridInterpolator(
                (h_coords, w_coords),
                np.zeros((H,W)))
            ob = np.zeros((T, sample_num))
            for t in range(T):
                interp_func.values = video[t,:,:]
                ob[t, :] = interp_func(ob_pos, method='linear')
    return ob,ob_pos,test_pos
    
def calc_reduced_dic_mpi(T,H,W,pos,indices_3d, req='AB',pos_type='float'):
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
    else:
        return A

def calc_reduced_dic_GPU(T,H,W,K,pos,indices_3d, req='AB',device='cuda'):

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
    else:
        return A

def calc_dic_GPU(T,H,W,K,pos,device='cuda',pos_type='int'):
    norm = torch.linalg.norm
    P = K*T
    L = H*W
    C = L*T
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

def direct_solve(ob, pos, H, W, T, indices_3d):
    T,K = ob.shape
    L = H*W
    shaped_ob = ob.flatten(order='F')
    indices_1d = indices_3d[:,0]*H*W + indices_3d[:,1]*W + indices_3d[:,2]
    A = calc_reduced_dic_mpi(T,H,W,K,pos,indices_3d,req='A')
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

def OMP(ob, pos, H, W, T, indices_3d=None, iters=None):
    norm = np.linalg.norm
    T,K = ob.shape
    Sq = H*W
    C = Sq*T
    shaped_ob = ob.flatten(order='F')
    rk = shaped_ob.copy()
    err = 1e-3*norm(rk)
    xr = np.zeros((T,H,W))
    
    if indices_3d is not None:
        indices_1d = indices_3d[:,0]*H*W + indices_3d[:,1]*W + indices_3d[:,2]
        A, B = calc_reduced_dic_mpi(T,H,W,K,pos,indices_3d)
    else:
        A, B = calc_dic_GPU(T,H,W,K,pos)

    mask = np.ones(A.shape[0], dtype=bool)
    Sk = []
    Skpos = []
    if not iters:
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

def OMP_GPU(ob, H, W, T, pos, indices_3d, A=None, B=None, iters=1000, device='cuda', method='SAMP'):
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
        A, B = calc_reduced_dic_mpi(T,H,W,K,pos,indices_3d)
    A, B = torch.tensor(A, device=device, dtype=torch.float32), torch.tensor(B, device=device, dtype=torch.float32)

    if not iters:
        iters = K*T

    ctn = norm(rk)
    spa_list = [0,20,50,100,200,300,500,800,1000,100000]
    
    def recover(xk, Skpos, t, L):
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
                recover(xk, torch.tensor(Skpos, device=device), s, s)
                spa_list.pop(0)
        return recover(xk, Skpos, s, s)
    elif method == 'SAMP':
        Gamma_t = torch.tensor([], dtype=torch.long,device=device)
        S = 1
        L = S
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
                    recover(xk, Skpos, t, L)
                    spa_list.pop(0)
                Gamma_t = Gamma_tL
                rk = rk_new
                t = t + 1
            else:
                break
        xk = theta_t[maxindex]
        Skpos = Gamma_tL
        return recover(xk, Skpos, t, L)

def shuffle(a,pos,sta_info,Knew):
    K = a.shape[1]
    combined_coords = pos.copy() 
    random_indices = np.random.permutation(K)
    shuffled_data = a[:, random_indices]
    shuffled_coords = combined_coords[random_indices, :]
    shuffled_sta_info = sta_info[random_indices, :]
    return shuffled_data[:, :Knew], shuffled_data[:, Knew:], shuffled_coords, shuffled_sta_info

def smooth_function(video1, video2, window_type='linear'):
    T, H, W = video1.shape
    if window_type == 'linear':
        weights = np.linspace(0, 1, T)
    elif window_type == 'hann':
        n = np.arange(T)
        weights = 0.5 * (1 - np.cos(2 * np.pi * n / (T - 1)))
    elif window_type == 'hamming':
        n = np.arange(T)
        weights = 0.54 - 0.46 * np.cos(2 * np.pi * n / (T - 1))
    elif window_type == 'blackman':
        n = np.arange(T)
        weights = (0.42 - 0.5 * np.cos(2 * np.pi * n / (T - 1)) + 
                  0.08 * np.cos(4 * np.pi * n / (T - 1)))
    else:
        raise ValueError("不支持的窗函数类型")
    
    weights_3d = weights[:, np.newaxis, np.newaxis]
    reverse_weights_3d = weights[::-1][:, np.newaxis, np.newaxis]
    
    result = video1.copy()
    result = (weights_3d * video1 + reverse_weights_3d * video2)
    
    return result 

def joint(video_list, overlap_len):
    video_num = len(video_list)
    long_video = video_list[0].copy()[:-overlap_len]
    for i in range(1,video_num-1):
        long_video = np.concatenate([long_video, smooth_function(video_list[i-1][-overlap_len:], video_list[i][:overlap_len]), video_list[i][overlap_len:-overlap_len]], axis=0)
    long_video = np.concatenate([long_video, smooth_function(video_list[i][-overlap_len:], video_list[i+1][:overlap_len]), video_list[i+1][overlap_len:]], axis=0)
    return long_video

def get_sta_pos(sta_info, selected_sta_info, allpoints, full_pos):
    test_idx = []
    for row in selected_sta_info:
        match_indices = np.where((sta_info == row).all(axis=1))[0]
        if len(match_indices) > 0:
            test_idx.append(match_indices[0])
    all_indices = np.arange(sta_info.shape[0])
    test_idx = np.array(test_idx)
    keep_idx = np.setdiff1d(all_indices, test_idx)
    new_order = np.concatenate([keep_idx, test_idx])
    K = len(keep_idx)
    ob, test_points = allpoints[:,new_order[:K]], allpoints[:, new_order[K:]]
    full_pos = full_pos[new_order]
    sta_info = sta_info[new_order]
    
    return ob, test_points, full_pos, sta_info


if __name__ == '__main__':
    def CS3D_theo():
        start = time.time()
        H, W, T = 64, 64, 232
        full_video = normalize(np.load(r'D:\research\3DCS\a_event.npy')[50:450,:,:,0])
        # animate_a_video(full_video)
        video = full_video[32:T+32,32:H+32,32:W+32]
        T_clip = 64
        overlap_len = 8
        clip_num = 4
        ob,ob_pos,test_pos = gene_mask(video,H,W,10,sample_ratio=0.04, mask_type='float')
        rec_video_list = []
        indices_3d = present_statistics(full_video[50:306], full_video[:T_clip,:H,:W], step=1, plot=True)
        A, B = calc_reduced_dic_mpi(T_clip,H,W,ob_pos,indices_3d)
        for i in range(clip_num):
            print(f"clip {i}")
            start_frame = i*(T_clip - overlap_len)
            rec_video = OMP_GPU(ob[start_frame:start_frame+T_clip], H, W, T_clip, ob_pos, indices_3d, A=A, B=B)
            rec_video_list.append(rec_video)
        rec_video = joint(rec_video_list, overlap_len)
        # rec_video = np.load(r'cs3D_theo_data\rec_video_theo.npy')
        animate_matrices(video, rec_video, pos=ob_pos, test_pos=test_pos, save=True)
        np.save(r'cs3D_theo_data\rec_video_theo.npy', rec_video)
        CS3D_show(rec_video, video, test_pos)
        end = time.time()
        print(f'used_time: {end - start:.1f} s')

    def CS3D_real():
        start = time.time()
        H, W, T_clip = 32, 64, 64
        full_video = normalize(np.load(r'D:\research\3DCS\a_event.npy')[50:306,:,:,0])
        video = full_video[80:80+T_clip,32:32+H,32:32+W]
        channel = 0
        allpoints = np.load(r'data\all_ob_points.npy')[:,:,channel]
        allpoints = taper.smooth_transition_2d(allpoints)
        full_pos = np.load(r'data\shuffled_round_xy.npy')
        sta_info = np.loadtxt(r"data\sta_info.txt", dtype=str)
        if 1:
            ob, test_points, full_pos, sta_info = shuffle(allpoints, full_pos, sta_info, Knew=50)
        else:
            test_sta_info = np.loadtxt(r"data\test_sta_info.txt", dtype=str)
            ob, test_points, full_pos, sta_info = get_sta_pos(sta_info, test_sta_info, allpoints, full_pos)
        T,K = ob.shape
        overlap_len = 8
        clip_num = 16
        rec_video_list = []
        indices_3d = present_statistics(full_video, video, step=1, plot=True)
        A, B = calc_reduced_dic_mpi(T_clip,H,W,full_pos[:K],indices_3d)
        
        for i in range(clip_num):
            print(f"clip {i}")
            start_frame = i*(T_clip - overlap_len)
            rec_video = OMP_GPU(ob[start_frame:start_frame+T_clip], H, W, T_clip, full_pos[:K], indices_3d, A=A, B=B)
            rec_video_list.append(rec_video)
        rec_video = joint(rec_video_list, overlap_len)
        np.save(r'data\rec_video.npy', rec_video)
        # rec_video = np.load('rec_video.npy')
        CS3Dreal_show(T,H,W, rec_video, ob, sta_info, test_points, full_pos)
        end = time.time()
        print(f'used_time: {end - start:.1f} s')
        
    # CS3D_real()
    CS3D_theo()

    
