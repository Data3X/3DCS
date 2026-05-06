import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fftpack import dct,idct,dctn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def animate_a_video(video: np.ndarray, pos, normalize: bool = True, save: bool = False, color = None):
    """animate a video, which can be used to visualize the seismic wave propagation process,
    and the positions of the stations can also be visualized if provided.
    
    Args:
        video (np.ndarray): the video to be animated, shape (T, H, W)
        pos (np.ndarray, optional): the positions of the stations, shape (K, 2). Defaults to None.
        normalize (bool, optional): whether to normalize the video for better visualization. Defaults to True.
        save (bool, optional): whether to save the animation. Defaults to False.
        color (np.ndarray, optional): the color of the stations. Defaults to None.
    """ 
    T,H,W = video.shape
    fig, ax = plt.subplots()
    ax.set_ylim(0,H)
    maxv = np.max(np.abs(video))
    if normalize:
        video = video / maxv
        im = ax.imshow(video[0,:,:], cmap='seismic', vmin=-1, vmax=1,interpolation='bicubic', origin='lower')
    else:
        im = ax.imshow(video[0,:,:], cmap='seismic', vmin=-maxv, vmax=maxv,interpolation='bicubic', origin='lower')

    ax.set_xticks([0, 19.4, 38.79, 58.18]) 
    ax.set_xticklabels(['0km', '100km', '200km', '300km'],fontsize=20) 
    ax.set_yticks([0, 19.4]) 
    ax.tick_params(axis='y', rotation=90, labelrotation=90)
    ax.set_yticklabels(['', '100km'],fontsize=20) 

    # if color is not None:
    #     color = 0.5 + 0.5*color/np.max(abs(color))
        
    scatter = ax.scatter([], [], c=[], cmap='seismic', vmin=-maxv, vmax=maxv, s=50, edgecolors='black',linewidths=0.4)

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, aspect=37, shrink=2)
    cbar.set_label('Amplitude (m/s)', fontsize=20)

    def update(frame):
        im.set_array(video[frame,:,:])
        # if pos is not None:
        #     scatter.set_offsets(pos[:, ::-1])
        #     scatter.set_array(color[frame, :])
        ax.set_title(f"{5 + frame}s after earthquake", fontsize=20)
        return im,scatter
    ani = animation.FuncAnimation(fig, update, frames=T, interval=20, blit=False)
    ani.save(r"real_data_results\animation.mp4", writer="ffmpeg", fps=2)
    plt.show()

def main_spectrum_statistics(full_video: np.ndarray, window_video: np.ndarray, step: int = 1, plot: bool = False, save: bool = False):
    """get the distribution of the main spectrum for a video   

    Args:
        full_video (np.ndarray): the full video, shape (T, H, W)
        window_video (np.ndarray): the window video, shape (Wt, Wh, Ww)
        step (int, optional): the step size for the sliding window. Defaults to 1.
        plot (bool, optional): whether to plot the distribution of the main spectrum. Defaults to False.
        save (bool, optional): whether to save the plot of the distribution of the main spectrum. Defaults to False.

    Returns:
        indices_3d (np.ndarray): the indices of the main spectrum
    """
    Wt, Wh, Ww = window_video.shape
    T, H, W = full_video.shape
    remain_num = int(0.01*Wh*Wt*Ww) # the number of main spectrum we want to keep
    # the number of random samples we want to take from full_video to get the distribution of the main spectrum for window video
    rand_sampled_num  = int(0.01*(H-Wh*step+1)*(W-Ww*step+1)*(T-Wt*step+1))  
    spectrum_statistics = np.zeros(Wh*Ww*Wt, dtype=np.int32)

    for i in range(rand_sampled_num):
        Hs, Ws, Ts = np.random.randint(H-Wh*step+1), np.random.randint(W-Ww*step+1), np.random.randint(T-Wt*step+1)
        cliped_video = full_video[Ts:Ts+Wt*step:step, Hs:Hs+Wh*step:step, Ws:Ws+Ww*step:step]
        spectrum = abs(dct(dct(dct(cliped_video,axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho')).flatten()
        indices = np.argpartition(spectrum, -remain_num)[-remain_num:]
        spectrum_statistics[indices] = spectrum_statistics[indices] + 1
  
  
    # indices = np.argpartition(spectrum_statistics, -remain_num)[-remain_num:]
    spectrum_statistics_3d = spectrum_statistics.reshape((Wt,Wh,Ww))/rand_sampled_num
    final_remain_num = int(0.01*Wh*Wt*Ww)
    threshold = np.partition(spectrum_statistics, -final_remain_num)[-final_remain_num]
    threshold = 0
    indices_3d = np.argwhere(spectrum_statistics_3d > threshold)
    if plot:
        x, y, z = indices_3d.T
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=spectrum_statistics_3d[tuple(indices_3d.T)],              
                colorscale='Rainbow',        
                opacity=0.8,
                colorbar=dict(title='Probability of occurrence')
            )
        )])

        fig.update_layout(
            title='3D Scatter with Value Coloring',
            scene=dict(
                xaxis_title='T',
                yaxis_title='H',
                zaxis_title='W',
                xaxis=dict(range=[0, Wt]),  
                yaxis=dict(range=[0, Wh]), 
                zaxis=dict(range=[0, Ww]) 
            )
        )
        fig.show()
        if save:
            fig.write_image(r"main_spectrum_distribution.png")
            
    return indices_3d

def present_different_precents_for_recovering(video: np.ndarray):
    T,H,W = video.shape
    maxp = 100
    present_num = 4096
    percent_list = np.arange(present_num)*maxp/present_num
    spectrum_sample_list = np.array(percent_list*T*H*W/100, dtype=np.int64)
    rmse_list = np.zeros(present_num)
    coe = dct(dct(dct(video,axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho')
    abs_coe = abs(coe)
    threshold_list = np.sort(np.partition(abs_coe.ravel(), -spectrum_sample_list[-1])[-spectrum_sample_list])[::-1]
    recover_threshold_list = [90, 99, 99.9]
    recover_threshold_list_copy = recover_threshold_list.copy()
    threshold_point_list = []

    for i,threshold in enumerate(threshold_list):
        if i%100 == 0:
            print(i)
        mask = abs_coe >= threshold
        reduced_coe = np.where(mask, coe, 0)
        rec_video = idct(idct(idct(reduced_coe,axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho')

        rmse_list[i] = np.round((1 - np.sqrt(np.mean((rec_video - video) ** 2) / np.mean(video**2)))*100,4)
        # if rmse_list[i] > 90:
        #     animate_matrices(video, rec_video)
        if len(recover_threshold_list) > 0 and rmse_list[i] > recover_threshold_list[0]:
            threshold_point_list.append(i)
            recover_threshold_list.pop(0)

    plt.plot(percent_list,rmse_list)
    for i,threshold_point in enumerate(threshold_point_list):
        recover_threshold = recover_threshold_list_copy[i]
        plt.scatter(percent_list[threshold_point],rmse_list[threshold_point])
        plt.axhline(y=recover_threshold, color='red', linestyle='--', label=f'Threshold ({recover_threshold})')
        plt.text(float(percent_list[threshold_point]), float(rmse_list[threshold_point]) + 0.5, f'({percent_list[threshold_point]:.3f},{rmse_list[threshold_point]:.3f})',fontsize=12, ha='center', va='bottom')

    plt.xlabel('percents (%)')
    plt.ylabel('recoverd percents (%)')
    plt.show()


def present_main_spectrum_for_a_video(video: np.ndarray, percent: float = 0.01): 
    """present the distribution of the main spectrum for a video, which can be used to analyze the sparsity of the video in the frequency domain, and the main spectrum can be used to design the dictionary for the CS3D reconstruction.

    Args:
        video (np.ndarray): the video for which to present the main spectrum, shape (T, H, W)
        percent (float, optional): the percentage of the main spectrum to present. Defaults to 0.01.
    """
    T,H,W = video.shape
    coe = dct(dct(dct(video,axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho')
    abs_coe = abs(coe)
    N = int(percent*T*H*W)
    threshold = np.sort(np.partition(abs_coe.ravel(), -N))[-N]
    main_spectrum = np.argwhere(coe > threshold)
    x, y, z = main_spectrum.T
    color_value = abs_coe[main_spectrum[:,0],main_spectrum[:,1],main_spectrum[:,2]]

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=color_value,              
            colorscale='Viridis',        
            opacity=0.8,
            colorbar=dict(title='Value')
        )
    )])
    fig.update_layout(
        title='3D Scatter with Value Coloring',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[0, T]),  
            yaxis=dict(range=[0, H]), 
            zaxis=dict(range=[0, W]) 
        )
    )
    fig.show()

def animate_comparaison_for_two_videos(A: np.ndarray, B: np.ndarray, pos=None, test_pos=None, save=False):
    """animate the comparison for two videos A and B, which can be used to evaluate the performance 
    of the reconstruction algorithm by comparing the reconstructed video with the original video.

    Args:
        A (np.ndarray): the first video, shape (T, H, W)
        B (np.ndarray): the second video, shape (T, H, W)
        pos (np.ndarray, optional): the positions of the stations, shape (K, 2). Defaults to None.
        test_pos (np.ndarray, optional): the positions of the test points, shape (K, 2). Defaults to None.
        save (bool, optional): whether to save the animation. Defaults to False.

    Returns:
        animation.FuncAnimation: the animation object
    """
    C = B - A
    T, H, W = A.shape
    amp = 1/np.max(abs(A))
    A, B, C = A*amp, B*amp, C*amp
    maxv,minv = 1,-1
    cmap = 'seismic'
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Comparison', fontsize=18)

    im1 = ax1.imshow(A[0, :, :], cmap=cmap, vmin=minv, vmax=maxv,interpolation='bicubic', origin='lower')
    im2 = ax2.imshow(B[0, :, :], cmap=cmap, vmin=minv, vmax=maxv,interpolation='bicubic', origin='lower')
    im3 = ax3.imshow(C[0, :, :], cmap=cmap, vmin=minv, vmax=maxv,interpolation='bicubic', origin='lower')

    cbar = fig.colorbar(im1, ax=[ax1, ax2, ax3], orientation='vertical', fraction=0.025, pad=0.02, shrink=0.85)
    cbar.set_label('Amplitude (mm/s)', fontsize=20)

    if pos is not None:
        for ax in [ax3]:
            for (x, y) in pos:
                ax.scatter(x, y, color='red', s=60, marker='^', edgecolor='black', linewidth=1)
                
    if test_pos is not None:
        for ax in [ax3]:
            for (x, y) in test_pos:
                ax.scatter(x, y, color='green', s=100, marker='o', edgecolor='green', linewidth=1)

    def update(frame):
        fig.suptitle(f'Time Step: {frame + 1}/{T}', fontsize=20)
        im1.set_array(A[frame, :, :])
        im2.set_array(B[frame, :, :])
        im3.set_array(C[frame, :, :])
        return im1, im2, im3

    for ax in [ax1, ax2, ax3]:  
        ax.set_xticks([0, A.shape[2]//2, A.shape[2]-1]) 
        ax.set_xticklabels(['', '', ''], fontsize=20) 
        ax.set_yticks([0, A.shape[1]//2, A.shape[1]-1])  
        ax.set_yticklabels(['', '', ''], fontsize=20)

    ani = animation.FuncAnimation(fig, update, frames=T, interval=100, blit=False)
    if save:
        ani.save(r"synthetic_data_results\comparaison.mp4", writer="ffmpeg", fps=10)
    plt.show()

    