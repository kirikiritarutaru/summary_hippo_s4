# Referenced code
# https://github.com/HazyResearch/state-spaces/blob/main/src/models/hippo/visualizations.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from box import Box

from hippo import HiPPO, HiPPOScale

device = (
    torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
)
cwd = Path(__file__).parent

conf = Box({
    'hippo': {
        'N': 64,  # 多項式の次数
    },
    'signal': {
        'T': 10,  # 入力する信号の長さ（秒数）
        'dt': 1e-3,  # サンプリング間隔
        'freq': 5.0,  # 信号の最低周波数
    },
    'exp': {
        'save_fig': True,
        'fig_path': None,
        'seed': 42,
    },
})


# Synthetic data generation
def whitesignal(period, dt, freq, rms=0.5, batch_shape=()):
    """
    Produces output signal of length period / dt, band-limited to frequency
    freq Output shape (*batch_shape, period/dt) Adapted from the nengo library
    """

    if freq is not None and freq < 1. / period:
        raise ValueError(
            f"Make ``{freq=} >= 1. / {period=}`` to produce a non-zero signal",
        )

    nyquist_cutoff = 0.5 / dt
    if freq > nyquist_cutoff:
        raise ValueError(
            f"{freq} must not exceed the Nyquist frequency "
            "for the given dt ({nyquist_cutoff:0.3f})"
        )

    n_coefficients = int(np.ceil(period / dt / 2.))
    shape = batch_shape + (n_coefficients + 1,)
    sigma = rms * np.sqrt(0.5)
    coefficients = 1j * np.random.normal(0., sigma, size=shape)
    coefficients[..., -1] = 0.
    coefficients += np.random.normal(0., sigma, size=shape)
    coefficients[..., 0] = 0.

    set_to_zero = np.fft.rfftfreq(2 * n_coefficients, d=dt) > freq
    coefficients *= (1 - set_to_zero)
    power_correction = np.sqrt(
        1. - np.sum(set_to_zero, dtype=float) / n_coefficients
    )
    if power_correction > 0.:
        coefficients /= power_correction
    coefficients *= np.sqrt(2 * n_coefficients)
    signal = np.fft.irfft(coefficients, axis=-1)
    signal = signal - signal[..., :1]  # Start from 0
    return signal


def plot(
        T=10, dt=1e-3, N=64, freq=3.0, save=False, seed=0,
        output_path=cwd / 'output'/'function_approximation.pdf',
):
    np.random.seed(seed)
    vals = np.arange(0.0, T, dt)

    figure_mosaic = """
    A
    B
    """
    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(16, 9))

    u = whitesignal(T, dt, freq=freq)
    u = torch.tensor(u, dtype=torch.float)
    u = u.to(device)

    axes['A'].set_title('White Signal')
    axes['A'].plot(vals, u.cpu(), 'k', linewidth=1.0)
    axes['B'].plot(vals, u.cpu(), 'k', linewidth=1.0)

    # Linear Time Invariant (LTI) methods x' = Ax + Bu
    lti_methods = [
        'legs',
        'legt',
        'fourier',
    ]

    for method in lti_methods:
        hippo = HiPPO(method=method, N=N, dt=dt, T=T).to(device)
        u_hippo = hippo.reconstruct(hippo(u))[-1].cpu()
        axes['B'].plot(vals[-len(u_hippo):], u_hippo, label=method, ls='--')

    # Original HiPPO-LegS, which uses time-varying SSM x' = 1/t [ Ax + Bu]
    # we call this "linear scale invariant"
    lsi_methods = ['legs']
    for method in lsi_methods:
        hippo = HiPPOScale(
            method=method, N=N, max_length=int(T / dt)
        ).to(device)
        u_hippo = hippo.reconstruct(hippo(u))[-1].cpu()
        axes['B'].plot(
            vals[-len(u_hippo):], u_hippo, label=method + ' (scaled)', ls=':'
        )

    axes['B'].set_title('HiPPO Reconstruction')
    axes['B'].set_xlabel('Time (Normalized)')
    axes['B'].legend()
    if save:
        fig.savefig(output_path, bbox_inches='tight')
    plt.show()


def plot_LTI():
    conf.exp.fig_path = cwd/'output'/'function_approximation_LTI.pdf'
    np.random.seed(conf.exp.seed)
    vals = np.arange(0.0, conf.signal.T, conf.signal.dt)

    figure_mosaic = """
    A
    B
    """
    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(16, 9))

    u = whitesignal(conf.signal.T, conf.signal.dt, freq=conf.signal.freq)
    u = torch.tensor(u, dtype=torch.float)
    u = u.to(device)

    axes['A'].set_title('White Signal')
    axes['A'].plot(vals, u.cpu(), 'k', linewidth=1.0)
    axes['B'].plot(vals, u.cpu(), 'k', linewidth=1.0)

    # Linear Time Invariant (LTI) methods x' = Ax + Bu
    lti_methods = [
        'legs',
        'legt',
        'fourier',
    ]

    for method in lti_methods:
        hippo = HiPPO(
            method=method,
            N=conf.hippo.N,
            dt=conf.signal.dt,
            T=conf.signal.T
        ).to(device)
        u_hippo = hippo.reconstruct(hippo(u))[-1].cpu()
        axes['B'].plot(vals[-len(u_hippo):], u_hippo, label=method, ls='--')

    axes['B'].set_title('HiPPO Reconstruction')
    axes['B'].set_xlabel('Time (Normalized)')
    axes['B'].legend()
    if conf.exp.save_fig:
        fig.savefig(conf.exp.fig_path, bbox_inches='tight')
    plt.show()


def plot_LSI():
    conf.exp.fig_path = cwd / 'output' / 'function_approximation_LSI.pdf'
    np.random.seed(conf.exp.seed)
    vals = np.arange(0.0, conf.signal.T, conf.signal.dt)

    figure_mosaic = """
    A
    B
    """
    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(16, 9))

    u = whitesignal(conf.signal.T, conf.signal.dt, freq=conf.signal.freq)
    u = torch.tensor(u, dtype=torch.float)
    u = u.to(device)

    axes['A'].set_title('White Signal')
    axes['A'].plot(vals, u.cpu(), 'k', linewidth=1.0)
    # axes['B'].plot(vals, u.cpu(), 'k', linewidth=1.0)

    # Original HiPPO-LegS, which uses time-varying SSM x' = 1/t [ Ax + Bu]
    # we call this "linear scale invariant"
    lsi_methods = ['legs']
    print(f'max_length: {int(conf.signal.T / conf.signal.dt)}')
    for method in lsi_methods:
        hippo = HiPPOScale(
            method=method,
            N=conf.hippo.N,
            max_length=int(conf.signal.T / conf.signal.dt)
        ).to(device)

        # print(f'hippo(u) size: {hippo(u).size()}')
        # print(
        #    f'hippo reconstruct size: {hippo.reconstruct(hippo(u)).size()}'
        # )

        # hippo(u) により、各時刻tにおけるルジャンドル多項式の係数が帰ってくる
        # 逐次的に推論するように修正？
        # 横軸のスケールがあってない気がする
        # TODO: 要修正
        for i in range(len(u)):
            if i < 10:
                continue
            _u = u[:i]
            u_hippo = hippo.reconstruct(hippo(_u))[-1].cpu()
            axes['B'].plot(
                vals[-len(u_hippo):], u_hippo,
                label=method + ' (scaled)', ls=':'
            )

            axes['B'].set_title('HiPPO Reconstruction')
            axes['B'].set_xlabel('Time (Normalized)')
            axes['B'].legend()
            if conf.exp.save_fig:
                fig.savefig(conf.exp.fig_path, bbox_inches='tight')
            plt.pause(.1)
            axes['B'].cla()


if __name__ == '__main__':
    # plot_LTI()
    plot_LSI()
