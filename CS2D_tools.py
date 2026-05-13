import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.fftpack import dct
import plotly.graph_objects as go

def compute_metrics_2D(original: np.ndarray, reconstructed: np.ndarray):
    img1 = np.asarray(original, dtype=np.float64)
    img2 = np.asarray(reconstructed, dtype=np.float64)

    mse = np.mean((img1 - img2) ** 2)

    rmse = np.sqrt(mse)

    if mse < 1e-12: 
        psnr = 100.0
    else:
        max_pixel = np.max(img1)
        if max_pixel == 0:
            max_pixel = 1.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "PSNR": psnr
    }
    
def compare_2D_results(A: np.ndarray, B: np.ndarray, pos=None, test_pos=None, save=False):
    C = B - A
    H, W = A.shape
    
    mse = np.mean((A - B) ** 2)
    max_val = np.max(np.abs(A))
    psnr = 10 * np.log10((max_val ** 2) / (mse + 1e-10)) if mse > 0 else 100
    
    amp = 1.0 / (max_val + 1e-8)
    A_norm, B_norm, C_norm = A * amp, B * amp, C * amp
    
    maxv, minv = 1, -1
    cmap = 'seismic'
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    fig.suptitle(f'2D Reconstruction Comparison\nMSE: {mse:.2e} | PSNR: {psnr:.2f} dB', fontsize=28)

    # A: Ground Truth
    im1 = ax1.imshow(A_norm, cmap=cmap, vmin=minv, vmax=maxv, interpolation='bicubic', origin='lower')
    ax1.set_title('Ground Truth', fontsize=30)
    
    # B: Reconstructed
    im2 = ax2.imshow(B_norm, cmap=cmap, vmin=minv, vmax=maxv, interpolation='bicubic', origin='lower')
    ax2.set_title('Reconstructed', fontsize=30)

    # C: Residual
    im3 = ax3.imshow(C_norm, cmap=cmap, vmin=minv, vmax=maxv, interpolation='bicubic', origin='lower')
    ax3.set_title('Residual (Error)', fontsize=30)

    for ax in [ax1, ax3]:
        if pos is not None:
            ax.scatter(pos[:, 1], pos[:, 0], color='black', s=100, marker='^', 
                       edgecolor='white', linewidth=0.5, label='Observation', alpha=0.8)
        if test_pos is not None:
            ax.scatter(test_pos[:, 1], test_pos[:, 0], color='lime', s=100, marker='o', 
                       edgecolor='black', linewidth=0.5, label='Test Points', alpha=0.8)

    cbar = fig.colorbar(im1, ax=[ax1, ax2, ax3], orientation='vertical', fraction=0.015, pad=0.04)
    cbar.set_label('Normalized Amplitude', fontsize=12)

    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([0, W//2, W-1])
        ax.set_yticks([0, H//2, H-1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
    if save:
        import os
        save_path = r"synthetic_data_results"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        full_name = os.path.join(save_path, "comparison_2d_results.png")
        plt.savefig(full_name, dpi=300, bbox_inches='tight')
        print(f"Success: Comparison image saved to {full_name}")
    
    plt.show()

def show_synthetic_data_results_2D(rec_image: np.ndarray, ori_image: np.ndarray, test_pos: np.ndarray):
    H, W = rec_image.shape
    h_coords, w_coords = np.arange(H), np.arange(W)
    
    interp_ori = RegularGridInterpolator((h_coords, w_coords), ori_image, method='linear')
    interp_rec = RegularGridInterpolator((h_coords, w_coords), rec_image, method='linear')

    ori_values = interp_ori(test_pos)
    rec_values = interp_rec(test_pos)
    
    k = test_pos.shape[0]
    
    results = compute_metrics_2D(ori_image, rec_image)
    mse = results['MSE']
    psnr = results['PSNR']

    print(f"--- 2D Reconstruction Results ---")
    print(f"Global MSE: {mse:.6f}")
    print(f"Global PSNR: {psnr:.2f} dB")

    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(k)
    width = 0.35
    
    ax.bar(x - width/2, ori_values, width, label='Original', color='black', alpha=0.7)
    ax.bar(x + width/2, rec_values, width, label='Reconstructed', color='red', alpha=0.7)
    
    ax.set_xlabel('Test Station Index', fontsize=15)
    ax.set_ylabel('Amplitude', fontsize=15)
    ax.set_title('Value Comparison at Test Positions', fontsize=18)
    ax.set_xticks(x)
    
    for i in range(k):
        xs = round(test_pos[i, 0] * 100 / H, 1)
        ys = round(test_pos[i, 1] * 100 / W, 1)
        ax.text(i, max(ori_values[i], rec_values[i]), f'({xs},{ys})', 
                ha='center', va='bottom', fontsize=8, rotation=45)

    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(r'synthetic_data_results/test_comparison_2d.jpg')
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.plot([ori_values.min(), ori_values.max()], [ori_values.min(), ori_values.max()], 'r--', linewidth=4)
    plt.scatter(ori_values, rec_values, color='blue', alpha=0.5, edgecolor='black', s=150)

    plt.xlabel('Original Value', fontdict={'fontsize': 22})
    plt.ylabel('Reconstructed Value', fontdict={'fontsize': 22})
    plt.title('Correlation at Test Points', fontdict={'fontsize': 24})
    plt.savefig(r'synthetic_data_results/correlation_scatter.jpg')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)  
    plt.show()

def gene_mask_2D(image: np.ndarray, H: int, W: int, K: int, sample_ratio: float = 0.01, mask_type: str = 'float'):
    L = H * W
    sample_num = int(L * sample_ratio)
    
    test_pos = np.array([])

    if mask_type == 'int':
        mask_flat = np.hstack([np.zeros(L - sample_num), np.ones(sample_num)])
        np.random.shuffle(mask_flat)
        mask_2d = mask_flat.reshape(H, W)
        
        ob_pos = np.argwhere(mask_2d == 1) # [sample_num, 2]
        
        ob = image[ob_pos[:, 0], ob_pos[:, 1]]
        
        remaining_pos = np.argwhere(mask_2d == 0)
        if len(remaining_pos) >= K:
            idx = np.random.choice(len(remaining_pos), K, replace=False)
            test_pos = remaining_pos[idx]

    elif mask_type == 'float':
        y = np.random.uniform(0, H - 1, size=sample_num + K)
        x = np.random.uniform(0, W - 1, size=sample_num + K)
        all_pos = np.column_stack((y, x))
        
        ob_pos = all_pos[:sample_num]
        test_pos = all_pos[sample_num:]
        
        h_coords = np.arange(H)
        w_coords = np.arange(W)
        interp_func = RegularGridInterpolator((h_coords, w_coords), image, method='linear')
        
        ob = interp_func(ob_pos)

    return ob, ob_pos, test_pos

def main_spectrum_statistics_2D(full_video: np.ndarray, window_image: np.ndarray, step: int = 1, plot: bool = False, save: bool = False):
    Wh, Ww = window_image.shape
    T, H, W = full_video.shape
    
    remain_num = int(0.01 * Wh * Ww) 
    rand_sampled_num = int(0.1 * (H - Wh * step + 1) * (W - Ww * step + 1))
    spectrum_statistics = np.zeros(Wh * Ww, dtype=np.int32)
    random_time = np.random.randint(50, 150)

    for i in range(rand_sampled_num):
        Hs = np.random.randint(H - Wh * step + 1)
        Ws = np.random.randint(W - Ww * step + 1)
        
        cliped_image = full_video[random_time, Hs:Hs+Wh*step:step, Ws:Ws+Ww*step:step]
        
        spectrum = dct(dct(cliped_image, axis=0, norm='ortho'), axis=1, norm='ortho')
        spectrum = np.abs(spectrum).flatten()
        
        indices = np.argpartition(spectrum, -remain_num)[-remain_num:]
        spectrum_statistics[indices] += 1
  
    spectrum_statistics_2d = spectrum_statistics.reshape((Wh, Ww)) / rand_sampled_num
    
    threshold = 0 
    indices_2d = np.argwhere(spectrum_statistics_2d > threshold)
    
    if plot:
        fig = go.Figure(data=go.Heatmap(
            z=spectrum_statistics_2d,
            colorscale='Rainbow',
            colorbar=dict(title='Occurrence Prob')
        ))

        fig.update_layout(
            title='2D Main Spectrum Distribution (DCT Domain)',
            xaxis_title='Width Frequency (W)',
            yaxis_title='Height Frequency (H)',
            width=600, height=500
        )
        fig.show()
        
        if save:
            fig.write_image("main_spectrum_distribution_2d.png")
            
    return indices_2d

def main_spectrum_statistics_1D(full_video: np.ndarray, window_image: np.ndarray, step: int = 1, plot: bool = False, save: bool = False):
    Wh, Ww = window_image.shape
    T, H, W = full_video.shape
    
    remain_num = int(0.01 * H*W)
    
    rand_sampled_num = int(0.1 * (H * W - W * step + 1))
    
    spectrum_statistics = np.zeros(H*W, dtype=np.int32)
    random_time = np.random.randint(50, 150)

    for i in range(rand_sampled_num):
        Hs = np.random.randint(H - Wh * step + 1)
        Ws = np.random.randint(W - Ww * step + 1)
        
        cliped_signal = full_video[random_time, Hs:Hs+Wh*step:step, Ws:Ws+Ww*step:step].flatten()
        
        spectrum = dct(cliped_signal, norm='ortho')
        spectrum_abs = np.abs(spectrum)
        
        indices = np.argpartition(spectrum_abs, -remain_num)[-remain_num:]
        spectrum_statistics[indices] += 1
  
    spectrum_prob_1d = spectrum_statistics / rand_sampled_num
    threshold = 0 
    indices_1d = np.where(spectrum_prob_1d > threshold)[0]
    
    if plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(W),
            y=spectrum_prob_1d,
            mode='lines+markers',
            marker=dict(size=4, color=spectrum_prob_1d, colorscale='Rainbow', showscale=True),
            line=dict(width=1),
            name='Occurrence Prob'
        ))

        fig.update_layout(
            title='1D Main Spectrum Distribution (DCT Domain)',
            xaxis_title='Frequency Index',
            yaxis_title='Occurrence Probability',
            width=800, height=400,
            template='plotly_white'
        )
        fig.show()
        
        if save:
            fig.write_image("main_spectrum_distribution_1d.png")
            
    return indices_1d

def plot_sparsity(spa1, spa2, err1, err2, ticks):
    plt.figure(figsize=(8, 6))
    plt.plot(spa1, err1, marker='o', markersize=12, label='loss of traditional method', linewidth=4)
    plt.plot(spa2, err2, marker='s', markersize=12, label='loss of our method', linewidth=4) 

    plt.xlabel('Sparsity', fontsize=24)
    plt.ylabel('Loss', fontsize=24)
    plt.title('Comparison of Two Losses vs Sparsity', fontsize=28)
    plt.legend(fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.xticks(ticks, fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    spa2 = [0, 10, 20, 30, 40, 50, 60, 70, 78]
    spa1 = [0, 10, 20, 30, 40, 64, 70, 78, 80, 95, 102]
    err2 = [3.4661, 1.5454, 0.5033, 0.3305, 0.2028, 0.1170, 0.0897, 0.0785, 0.0681]
    err1 = [3.4661, 1.8803, 1.1816, 0.9259, 0.6983, 0.6398, 0.5852, 0.5803, 0.5510, 0.5072, 0.4768]
    ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    plot_sparsity(spa1, spa2, err1, err2, ticks)