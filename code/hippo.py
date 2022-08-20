# Referenced code
# https://github.com/HazyResearch/state-spaces/blob/main/src/models/hippo/visualizations.py

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg as la
from scipy import signal
from scipy import special as ss


# HiPPO matrices
def transition(method, N, **method_args):
    # Laguerre (translated)
    if method == 'lagt':
        b = method_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Legendre (translated)
    elif method == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # Legendre (scaled)
    elif method == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        # B = B.copy()
    elif method == 'fourier':
        freqs = np.arange(N // 2)
        d = np.stack([np.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
        A = 2 * np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2
        B[0] = 2**.5
        A = A - B[:, None] * B[None, :]
        # A = A - np.eye(N)
        B *= 2**.5
        B = B[:, None]

    return A, B


def measure(method, c=0.0):
    if method == 'legt':
        def fn(x): return np.heaviside(x, 0.0) * np.heaviside(1.0 - x, 0.0)
    elif method == 'legs':
        def fn(x): return np.heaviside(x, 1.0) * np.exp(-x)
    elif method == 'lagt':
        def fn(x): return np.heaviside(x, 1.0) * np.exp(-x)
    elif method in ['fourier']:
        def fn(x): return np.heaviside(x, 1.0) * np.heaviside(1.0 - x, 1.0)
    else:
        raise NotImplementedError

    def fn_tilted(x): return np.exp(c * x) * fn(x)
    return fn_tilted


def basis(method, N, vals, c=0.0, truncate_measure=True):
    """
    vals: list of times (forward in time)
    returns: shape (T, N) where T is length of vals
    """
    if method == 'legt':
        eval_matrix = ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1).T
        eval_matrix *= (2 * np.arange(N) + 1)**.5 * (-1)**np.arange(N)
    elif method == 'legs':
        _vals = np.exp(-vals)
        # (L, N)
        eval_matrix = ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * _vals).T
        # ルジャンドル多項式の正規化
        eval_matrix *= (2 * np.arange(N) + 1)**.5 * (-1)**np.arange(N)
    elif method == 'lagt':
        vals = vals[::-1]
        eval_matrix = ss.eval_genlaguerre(np.arange(N)[:, None], 0, vals)
        eval_matrix = eval_matrix * np.exp(-vals / 2)
        eval_matrix = eval_matrix.T
    elif method == 'fourier':
        # (N/2, T/dt)
        cos = 2**.5 * np.cos(2 * np.pi * np.arange(N // 2)[:, None] * (vals))
        # (N/2, T/dt)
        sin = 2**.5 * np.sin(2 * np.pi * np.arange(N // 2)[:, None] * (vals))
        cos[0] /= 2**.5
        # (T/dt, N)
        eval_matrix = np.stack([cos.T, sin.T], axis=-1).reshape(-1, N)
    # print("eval_matrix shape", eval_matrix.shape)

    if truncate_measure:
        eval_matrix[measure(method)(vals) == 0.0] = 0.0

    p = torch.tensor(eval_matrix)
    p *= np.exp(-c * vals)[:, None]  # [::-1, None]
    return p


class HiPPO(nn.Module):
    """
    Linear time invariant x' = Ax + Bu
    """

    def __init__(
        self, N, method='legs', dt=1.0, T=1.0,
        discretization='bilinear', c=0.0
    ):
        """
        N: the order of the HiPPO projection
        dt: discretization step size
            - should be roughly inverse to the length of the sequence
        """
        super().__init__()
        self.method = method
        self.N = N
        self.dt = dt
        self.T = T
        self.c = c

        A, B = transition(method, N)
        A = A + np.eye(N) * c
        self.A = A
        self.B = B.squeeze(-1)

        C = np.ones((1, N))
        D = np.zeros((1,))
        dA, dB, _, _, _ = signal.cont2discrete(
            (A, B, C, D), dt=dt, method=discretization
        )

        dB = dB.squeeze(-1)

        self.register_buffer('dA', torch.Tensor(dA))  # (N, N)
        self.register_buffer('dB', torch.Tensor(dB))  # (N,)

        self.vals = np.arange(0.0, T, dt)
        self.eval_matrix = basis(
            self.method, self.N, self.vals, c=self.c
        )  # (T/dt, N)

    def forward(self, inputs):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        inputs = inputs.unsqueeze(-1)
        u = inputs * self.dB  # (length, ..., N)

        c = torch.zeros(u.shape[1:]).to(inputs)
        cs = []
        for f in inputs:
            c = F.linear(c, self.dA) + self.dB * f
            cs.append(c)
        return torch.stack(cs, dim=0)

    # TODO take in a times array for reconstruction
    def reconstruct(self, c, evals=None):
        """
        c: (..., N,) HiPPO coefficients (same as x(t) in S4 notation)
        output: (..., L,)
        """
        if evals is not None:
            eval_matrix = basis(self.method, self.N, evals)
        else:
            eval_matrix = self.eval_matrix

        c = c.unsqueeze(-1)
        y = eval_matrix.to(c) @ c
        return y.squeeze(-1).flip(-1)


class HiPPOScale(nn.Module):
    """
    Vanilla HiPPO-LegS model (scale invariant instead of time invariant)
    """

    def __init__(
            self, N, method='legs', max_length=1024,
            discretization='bilinear'
    ):
        """
        max_length: maximum sequence length
        """
        super().__init__()
        self.N = N
        A, B = transition(method, N)
        B = B.squeeze(-1)
        A_stacked = np.empty((max_length, N, N), dtype=A.dtype)
        B_stacked = np.empty((max_length, N), dtype=B.dtype)
        for t in range(1, max_length + 1):
            At = A / t
            Bt = B / t
            if discretization == 'forward':
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            elif discretization == 'backward':
                A_stacked[t - 1] = la.solve_triangular(
                    np.eye(N) - At, np.eye(N), lower=True
                )
                B_stacked[t - 1] = la.solve_triangular(
                    np.eye(N) - At, Bt, lower=True
                )
            elif discretization == 'bilinear':
                A_stacked[t - 1] = la.solve_triangular(
                    np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True
                )
                B_stacked[t - 1] = la.solve_triangular(
                    np.eye(N) - At / 2, Bt, lower=True
                )
            else:  # ZOH
                A_stacked[t - 1] = la.expm(A * (math.log(t + 1) - math.log(t)))
                B_stacked[t - 1] = la.solve_triangular(
                    A, A_stacked[t - 1] @ B - B, lower=True
                )
        # (max_length, N, N)
        self.register_buffer('A_stacked', torch.Tensor(A_stacked))
        # (max_length, N)
        self.register_buffer('B_stacked', torch.Tensor(B_stacked))

        vals = np.linspace(0.0, 1.0, max_length)
        self.eval_matrix = torch.Tensor((
            B[:, None] *
            ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)
        ).T)

    def forward(self, inputs):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        L = inputs.shape[0]

        inputs = inputs.unsqueeze(-1)
        u = torch.transpose(inputs, 0, -2)
        u = u * self.B_stacked[:L]
        u = torch.transpose(u, 0, -2)  # (length, ..., N)

        c = torch.zeros(u.shape[1:]).to(inputs)
        cs = []
        for t, f in enumerate(inputs):
            c = F.linear(c, self.A_stacked[t]) + self.B_stacked[t] * f
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        return self.eval_matrix.to(c) @ c.unsqueeze(-1)
