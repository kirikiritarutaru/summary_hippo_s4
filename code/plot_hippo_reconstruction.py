# Referenced code
# https://github.com/HazyResearch/state-spaces/blob/main/src/models/hippo/visualizations.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hippo import HiPPO, HiPPOScale

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


def plot(T, dt, N, freq, output_path, save):
    np.random.seed(0)
    vals = np.arange(0.0, T, dt)

    u = whitesignal(T, dt, freq=freq)
    u = torch.tensor(u, dtype=torch.float)
    u = u.to(device)

    plt.figure(figsize=(16, 8))
    offset = 0.0
    plt.plot(vals, u.cpu() + offset, 'k', linewidth=1.0)

    # Linear Time Invariant (LTI) methods x' = Ax + Bu
    lti_methods = [
        'legs',
        'legt',
        'fourier',
    ]

    for method in lti_methods:
        hippo = HiPPO(method=method, N=N, dt=dt, T=T).to(device)
        u_hippo = hippo.reconstruct(hippo(u))[-1].cpu()
        plt.plot(vals[-len(u_hippo):], u_hippo, label=method)

    # Original HiPPO-LegS, which uses time-varying SSM x' = 1/t [ Ax + Bu]
    # we call this "linear scale invariant"
    lsi_methods = ['legs']
    for method in lsi_methods:
        hippo = HiPPOScale(N=N, method=method, max_length=int(T / dt)).to(device)
        u_hippo = hippo.reconstruct(hippo(u))[-1].cpu()
        plt.plot(vals[-len(u_hippo):], u_hippo, label=method + ' (scaled)')

    plt.xlabel('Time (normalized)')
    plt.legend()
    if save:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    plot(
        T=3, dt=1e-3, N=64, freq=3.0,
        output_path=cwd / 'output' / 'function_approximation.pdf',
        save=False
    )
