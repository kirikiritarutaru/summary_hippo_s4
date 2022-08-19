# Referenced code
# https://github.com/HazyResearch/state-spaces/blob/main/src/models/hippo/visualizations.py

"""
Standalone implementation of HiPPO operators.
Contains experiments for the function reconstruction experiment in original HiPPO paper, as well as new animations from "How to Train Your HiPPO"
This file ports the notebook notebooks/hippo_function_approximation.ipynb, which is recommended if Jupyter is supported
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation

from hippo import HiPPO

sns.set(rc={
    "figure.dpi": 300,
    'savefig.dpi': 300,
    'animation.html': 'jshtml',
    'animation.embed_limit': 100,  # Max animation size in Mb
})
sns.set_style('ticks')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cwd = Path(__file__).parent


# Synthetic data generation
def whitesignal(period, dt, freq, rms=0.5, batch_shape=()):
    """
    Produces output signal of length period / dt, band-limited to frequency freq
    Output shape (*batch_shape, period/dt)
    Adapted from the nengo library
    """

    if freq is not None and freq < 1. / period:
        raise ValueError(
            f"Make ``{freq=} >= 1. / {period=}`` to produce a non-zero signal",)

    nyquist_cutoff = 0.5 / dt
    if freq > nyquist_cutoff:
        raise ValueError(
            f"{freq} must not exceed the Nyquist frequency for the given dt ({nyquist_cutoff:0.3f})")

    n_coefficients = int(np.ceil(period / dt / 2.))
    shape = batch_shape + (n_coefficients + 1,)
    sigma = rms * np.sqrt(0.5)
    coefficients = 1j * np.random.normal(0., sigma, size=shape)
    coefficients[..., -1] = 0.
    coefficients += np.random.normal(0., sigma, size=shape)
    coefficients[..., 0] = 0.

    set_to_zero = np.fft.rfftfreq(2 * n_coefficients, d=dt) > freq
    coefficients *= (1 - set_to_zero)
    power_correction = np.sqrt(1. - np.sum(set_to_zero, dtype=float) / n_coefficients)
    if power_correction > 0.:
        coefficients /= power_correction
    coefficients *= np.sqrt(2 * n_coefficients)
    signal = np.fft.irfft(coefficients, axis=-1)
    signal = signal - signal[..., :1]  # Start from 0
    return signal


# Animation code from HTTYH
def plt_lines(x, y, color, size, label=None):
    return plt.plot(x, y, color, linewidth=size, label=label)[0]


def update_lines(ln, x, y):
    ln.set_data(x, y)


def animate_hippo(
    method,
    T=5, dt=5e-4, N=64, freq=20.0,
    interval=100,
    plot_hippo=False, hippo_offset=0.0, label_hippo=False,
    plot_measure=False, measure_offset=-3.0, label_measure=False,
    plot_coeff=None, coeff_offset=3.0,
    plot_s4=False, s4_offset=6.0,
    plot_hippo_type='line', plot_measure_type='line', plot_coeff_type='line',
    size=1.0,
    plot_legend=True, plot_xticks=True, plot_box=True,
    plot_vline=False,
    animate_u=False,
    seed=2,
):
    np.random.seed(seed)

    vals = np.arange(0, int(T / dt) + 1)
    L = int(T / dt) + 1

    u = torch.FloatTensor(whitesignal(T, dt, freq=freq))
    u = F.pad(u, (1, 0))
    # add 3/4 of a sin cycle
    u = u + torch.FloatTensor(np.sin(1.5 * np.pi / T * np.arange(0, T + dt, dt)))
    u = u.to(device)

    hippo = HiPPO(method=method, N=N, dt=dt, T=T).to(device)
    coef_hippo = hippo(u).cpu().numpy()
    h_hippo = hippo.reconstruct(hippo(u)).cpu().numpy()
    u = u.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 4))

    if animate_u:
        ln_u = plt_lines([], [], 'k', size, label='Input $u(t)$')
    else:
        plt_lines(vals, u, 'k', size, label='Input $u(t)$')

    if plot_hippo:
        label_args = {'label': 'HiPPO reconstruction'} if label_hippo else {}
        ln = plt_lines([], [], size=size, color='red', **label_args)

    if plot_measure:
        label_args = {'label': 'HiPPO Measure'} if label_measure else {}
        ln_measure = plt_lines(vals, np.zeros(
            len(vals)) + measure_offset, size=size, color='green', **label_args)

    if plot_coeff is None:
        plot_coeff = []
    if isinstance(plot_coeff, int):
        plot_coeff = [plot_coeff]
    if len(plot_coeff) > 0:
        ln_coeffs = [
            plt_lines([], [], size=size, color='blue')
            for _ in plot_coeff
        ]
        plt_lines(
            [], [], size=size, color='blue', label='State $x(t)$'
        )  # For the legend

    # Y AXIS LIMITS
    if plot_measure:
        min_y = measure_offset
    else:
        min_y = np.min(u)

    if len(plot_coeff) > 0:
        max_u = np.max(u) + coeff_offset
    else:
        max_u = np.max(u)

    C = np.random.random(N)
    s4 = np.sum(coef_hippo * C, axis=-1)
    max_s4 = 0.0
    if plot_s4:
        ln_s4 = plt_lines(
            [], [], size=size, color='red', label='Output $y(t)$'
        )
        max_s4 = np.max(s4) + s4_offset

    if plot_vline:
        ln_vline = ax.axvline(0, ls='-', color='k', lw=1)

    if plot_legend:
        plt.legend(loc='upper left', fontsize='x-small')

    def init():
        left_endpoint = vals[0]
        ax.set_xlim(left_endpoint, vals[-1] + 1)
        ax.set_ylim(min_y, max(max_u, max_s4))
        ax.set_yticks([])
        if not plot_xticks:
            ax.set_xticks([])
        if not plot_box:
            plt.box(False)
        return []  # ln,

    def update(frame):
        if animate_u:
            xdata = np.arange(frame)
            ydata = u[:frame]
            update_lines(ln_u, xdata, ydata)

        m = np.zeros(len(vals))
        m[:frame] = hippo.measure_fn(np.arange(frame) * dt)[::-1]
        xdata = vals
        if plot_measure:
            update_lines(ln_measure, xdata, m + measure_offset)

        if plot_hippo:
            ydata = h_hippo[frame] + hippo_offset
            m2 = hippo.measure_fn(np.arange(len(ydata)) * dt)[::-1]
            # Remove reconstruction where measure is 0
            ydata[m2 == 0.0] = np.nan
            xdata = np.arange(frame - len(ydata), frame)
            update_lines(ln, xdata, ydata)

        if len(plot_coeff) > 0:
            for coeff, ln_coeff in zip(plot_coeff, ln_coeffs):
                update_lines(
                    ln_coeff, np.arange(frame), coef_hippo[:frame, coeff] + coeff_offset
                )
        if plot_s4:  # Only scale case; scale case should copy plot_hippo logic
            update_lines(ln_s4, np.arange(0, frame), s4[:frame] + s4_offset)

        if plot_vline:
            ln_vline.set_xdata([frame, frame])

        return []

    ani = FuncAnimation(
        fig, update,
        frames=np.arange(0, int(T * 1000 / interval) + 1) * int(interval / 1000 / dt),
        interval=interval, init_func=init, blit=True
    )

    return ani


if __name__ == '__main__':
    # Visualize HiPPO online reconstruction
    ani = animate_hippo(
        'legs',  # Try 'legt' or 'fourier'
        T=5, dt=5e-4, N=64, interval=100,
        # T=1, dt=1e-3, N=64, interval=200, # Faster rendering for testing
        size=1.0,

        animate_u=True,
        plot_hippo=True, hippo_offset=0.0, label_hippo=True,
        plot_s4=False, s4_offset=6.0,
        plot_measure=True, measure_offset=-3.0, label_measure=True,
        plot_coeff=[], coeff_offset=3.0,
        plot_legend=True, plot_xticks=True, plot_box=True,
        plot_vline=True,
    )
    ani.save(str(cwd / 'output' / 'hippo_legs.gif'))

    # Visualize S4
    ani = animate_hippo(
        'legs',  # Try 'legt' or 'fourier'
        T=5, dt=5e-4, N=64, interval=100,
        size=1.0,

        animate_u=True,
        plot_hippo=False, hippo_offset=0.0, label_hippo=True,
        plot_s4=True, s4_offset=6.0,
        plot_measure=False, measure_offset=-3.0, label_measure=True,
        plot_coeff=[0, 1, 2, 3], coeff_offset=3.0,
        plot_legend=True, plot_xticks=True, plot_box=True,
        plot_vline=True,
    )
    ani.save(str(cwd / 'output' / 's4_legs.gif'))
