import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_3comp_wavefield(
    data,
    time=None,
    h_axis=None,
    w_axis=None,
    prof_h_idx=None,
    prof_w_idx=None,
    t_slice=None,
    cmap='RdBu',
    vrange=None,
    component_names=('E', 'N', 'Z')
):
    """
    Create a single figure with three 3D subplots (E, N, Z) sharing ONE colour bar.
    Each subplot shows two orthogonal vertical cross‑sections of the wavefield:
    one at a fixed H index (time‑W plane) and one at a fixed W index (time‑H plane).

    Parameters
    ----------
    data : np.ndarray, shape (T, H, W, 3)
        Wavefield snapshots. T = time steps, H, W = spatial dimensions,
        3 components.
    time : 1D array, optional
        Time coordinates. If None, sample indices are used.
    h_axis : 1D array, optional
        Coordinates along the H dimension (length H). If None, sample indices.
    w_axis : 1D array, optional
        Coordinates along the W dimension (length W). If None, sample indices.
    prof_h_idx : int, optional
        Index of the fixed‑H profile (constant H). Default: H // 2.
    prof_w_idx : int, optional
        Index of the fixed‑W profile (constant W). Default: W // 2.
    t_slice : int, optional
        If provided, also plot a horizontal time‑slice (constant T).
    cmap : str
        Plotly colour scale name (e.g. 'RdBu', 'seismic').
    vrange : tuple (vmin, vmax) or None
        Colour scale limits. If None, the limits are set to ±max(|data|)
        (zero‑symmetric) so that positive/negative amplitudes are comparable
        across components.
    component_names : tuple of str
        Subplot titles, e.g. ('E', 'N', 'Z').

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure with three 3D subplots and a single shared colour bar.
    """
    # ---------- validation ----------
    if data.ndim != 4 or data.shape[3] != 3:
        raise ValueError("data must have shape (T, H, W, 3)")
    T, H, W, _ = data.shape
    if len(component_names) != 3:
        raise ValueError("component_names must contain exactly 3 entries")

    # ---------- default coordinates & profiles ----------
    if time is None:
        time = np.arange(T)
    if h_axis is None:
        h_axis = np.arange(H)
    if w_axis is None:
        w_axis = np.arange(W)
    if prof_h_idx is None:
        prof_h_idx = H // 2
    if prof_w_idx is None:
        prof_w_idx = W // 2

    # ---------- global colour range ----------
    if vrange is None:
        vabs = np.max(np.abs(data))
        vmin, vmax = -vabs, vabs
    else:
        vmin, vmax = vrange

    # ---------- create subplot grid (1 row × 3 cols) ----------
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=[f'{name} component' for name in component_names],
        horizontal_spacing=0.08
    )

    scene_names = ['scene1', 'scene2', 'scene3']
    coloraxis_name = 'coloraxis'   # all surfaces will refer to this

    # ---------- add surfaces for each component ----------
    for comp_idx, scene_name in enumerate(scene_names):
        field = data[:, :, :, comp_idx]   # shape (T, H, W)

        # ---- Profile 1: fixed H (time‑W plane, y = constant) ----
        slice_h = field[:, prof_h_idx, :]               # (T, W)
        W_grid, T_grid_h = np.meshgrid(w_axis, time)    # (T, W)
        H_const_h = np.full_like(W_grid, h_axis[prof_h_idx])

        fig.add_trace(
            go.Surface(
                x=W_grid,
                y=H_const_h,
                z=T_grid_h,
                surfacecolor=slice_h,
                colorscale=cmap,
                cmin=vmin, cmax=vmax,           # enforce global range
                coloraxis=coloraxis_name,       # use shared colour axis
                showscale=False,
                name=f'H={h_axis[prof_h_idx]:.2f}'
            ),
            row=1, col=comp_idx + 1
        )

        # ---- Profile 2: fixed W (time‑H plane, x = constant) ----
        slice_w = field[:, :, prof_w_idx]               # (T, H)
        H_grid, T_grid_w = np.meshgrid(h_axis, time)    # (T, H)
        W_const_w = np.full_like(H_grid, w_axis[prof_w_idx])

        fig.add_trace(
            go.Surface(
                x=W_const_w,
                y=H_grid,
                z=T_grid_w,
                surfacecolor=slice_w,
                colorscale=cmap,
                cmin=vmin, cmax=vmax,
                coloraxis=coloraxis_name,
                showscale=False,
                name=f'W={w_axis[prof_w_idx]:.2f}'
            ),
            row=1, col=comp_idx + 1
        )

        # ---- Optional horizontal time‑slice ----
        if t_slice is not None:
            slice_t = field[t_slice, :, :]
            H_grid_t, W_grid_t = np.meshgrid(h_axis, w_axis, indexing='ij')
            T_const_t = np.full_like(H_grid_t, time[t_slice])
            fig.add_trace(
                go.Surface(
                    x=W_grid_t,
                    y=H_grid_t,
                    z=T_const_t,
                    surfacecolor=slice_t,
                    colorscale=cmap,
                    cmin=vmin, cmax=vmax,
                    coloraxis=coloraxis_name,
                    showscale=False,
                    opacity=0.6,
                    name=f'T={time[t_slice]:.2f}'
                ),
                row=1, col=comp_idx + 1
            )

    # ---------- define the shared colour axis ----------
    fig.update_layout(
        coloraxis=dict(
            colorscale=cmap,
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(
                title='Amplitude',
                len=0.85,            # stretch across most of the height
                thickness=20,
                x=1.02,              # place to the right of the subplots
                y=0.5,
                yanchor='middle'
            )
        ),
        title='3-C wavefield: orthogonal time-space cross-sections',
        width=1500,
        height=600
    )

    # ---------- scene cosmetics ----------
    for scene_name in scene_names:
        fig.update_layout(**{
            f'{scene_name}.xaxis.title': 'Y',
            f'{scene_name}.yaxis.title': 'X',
            f'{scene_name}.zaxis.title': 'Z',
            # f'{scene_name}.zaxis.autorange': 'reversed',
            f'{scene_name}.aspectmode': 'manual',
            f'{scene_name}.aspectratio.x': 1,
            f'{scene_name}.aspectratio.y': 1,
            f'{scene_name}.aspectratio.z': 1.,
        })

    return fig


if __name__ == '__main__':
    # replace with your actual file path
    from CS3D_main import dct_basis_via_idct
    H, W, T = 64, 64, 64
    data1 = dct_basis_via_idct(H, W, T, u=0, v=2, w=4, norm='ortho')
    data = np.stack([data1, data1, data1], axis=-1)  # shape (H, W, T, 3)
    data = np.transpose(data, (2, 0, 1, 3))
    # data = np.load('upload/synthetic_event2.npy')[100:228]   # shape (T, H, W, 3)
    T, H, W, C = data.shape
    # If the array is large, downsample to improve rendering speed
    # data = data[::3, ::2, ::2, :]

    # Build the figure with a shared colour bar
    fig = plot_3comp_wavefield(
        data,
        time=np.arange(T),
        h_axis=np.arange(H),
        w_axis=np.arange(W),
        cmap='RdBu',
        t_slice=T//2,               # set to an index to add a time slice
        # vrange=(-0.5, 0.5)        # uncomment to force a specific colour range
    )

    fig.show()
    # fig.write_html('wavefield_shared_cbar.html')