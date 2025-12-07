import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fftpack import dct,idct
import plotly.graph_objects as go

def animate_a_video(video, pos=None, normalize=True, save=False, color=None):
    T,H,W = video.shape
    fig, ax = plt.subplots()
    ax.set_ylim(0,H)
    maxv = np.max(abs(video))
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
    ani.save(r"data\animation.mp4", writer="ffmpeg", fps=2)
    plt.show()

def present_statistics(full_video, video, step=1, plot=True, save=False):
    Wt, Wh, Ww = video.shape
    T, H, W = full_video.shape
    remain_num = int(0.01*Wh*Wt*Ww)
    rand_sampled_num  = int(0.01*(H-Wh*step+1)*(W-Ww*step+1)*(T-Wt*step+1))
    # rand_sampled_num = 1
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

        # fig.show()
        # fig.write_image("output.png") 
    return indices_3d

def present_statistics_GPU(full_video, video, plot=True, save=False, device='cuda'):
    Wh, Ww, Wt = video.shape
    T, H, W = full_video.shape
    remain_num = int(0.1*Wh*Wt*Ww)
    rand_sampled_num  = int(0.01*(H-Wh+1)*(W-Ww+1)*(T-Wt+1))
    # rand_sampled_num = 1
    spectrum_statistics = np.zeros(Wh*Ww*Wt, dtype=np.int32)

    for i in range(rand_sampled_num):
        Hs, Ws, Ts = np.random.randint(H-Wh+1), np.random.randint(W-Ww+1), np.random.randint(T-Wt+1)
        cliped_video = full_video[Ts:Ts+Wt, Hs:Hs+Wh, Ws:Ws+Ww]
        spectrum = abs(dct(dct(dct(cliped_video,axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho')).flatten()
        indices = np.argpartition(spectrum, -remain_num)[-remain_num:]
        spectrum_statistics[indices] = spectrum_statistics[indices] + 1
  
  
    # indices = np.argpartition(spectrum_statistics, -remain_num)[-remain_num:]
    spectrum_statistics_3d = spectrum_statistics.reshape((Wt,Wh,Ww))
    final_remain_num = int(0.02*Wh*Wt*Ww)
    threshold = np.partition(spectrum_statistics, -final_remain_num)[-final_remain_num]
    # threshold = 0
    indices_3d = np.argwhere(spectrum_statistics_3d > threshold)
    if plot:
        x, y, z = indices_3d.T
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=spectrum_statistics_3d[tuple(indices_3d.T)],              
                colorscale='Rainbow',        
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
                xaxis=dict(range=[0, Wt]),  
                yaxis=dict(range=[0, Wh]), 
                zaxis=dict(range=[0, Ww]) 
            )
        )

        fig.show()
    return indices_3d

def present_dct_basicwave3D(video):
    pass

def present_wavefront3D(video):
    pass

def present_different_precents_for_recovering(video):
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
        plt.text(percent_list[threshold_point],rmse_list[threshold_point]+0.5,f'({percent_list[threshold_point]:.3f},{rmse_list[threshold_point]:.3f})',fontsize=12, ha='center', va='bottom')

    plt.xlabel('percents (%)')
    plt.ylabel('recoverd percents (%)')
    plt.show()

def ThreeD_Scatter_for_main_spectrum(video, percent=0.01):
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

def animate_matrices(A, B, pos=None, test_pos=None, save=False):
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
        ani.save(r"cs3D_theo_data\comparaison.mp4", writer="ffmpeg", fps=10)
    plt.show()

if __name__ == '__main__':
    # full_video = np.load(r'C:\Users\syr\Desktop\1L0ernmv82sy36mh.full.npy')[50:178,:,:,0]
    # T,H,W = full_video.shape
    # step = 1
    # video = full_video[0:T:step,0:H:step,0:W:step]
    # present_different_precents_for_recovering(video)
    
    # full_video = np.load(r'C:\Users\syr\Desktop\1L0ernmv82sy36mh.full.npy')[50:200,:,:,0]
    # T,H,W = full_video.shape
    # length = 64
    # video = full_video[0:length,0:length,0:length] 
    # present_statistics(full_video, video) 

    full_video = np.load(r'C:\Users\syr\Desktop\1L0ernmv82sy36mh.full.npy')[50:178,:,:,0]
    ThreeD_Scatter_for_main_spectrum(full_video, percent=0.00121)
    
    # animate_a_video(video, normalize=True)
    