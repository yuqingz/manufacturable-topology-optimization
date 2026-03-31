"""
Local replacements for tidy3d functions:
  - make_filter_and_project
  - make_erosion_dilation_penalty
  - value_and_grad

Extracted from tidy3d (MIT-licensed, Flexcompute Inc.) and simplified
Only depends on: numpy, autograd.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Literal, Union

import autograd.numpy as anp
import numpy as np
from autograd.builtins import tuple as atuple
from autograd.core import make_vjp
from autograd.extend import vspace
from autograd.scipy.signal import convolve as _convolve_ag
from autograd.wrap_util import unary_to_nary

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
PaddingType = Literal["constant", "edge", "reflect", "symmetric", "wrap"]
KernelType = Literal["circular", "conic"]

BETA_DEFAULT = 1.0
ETA_DEFAULT = 0.5


# ===================================================================
# value_and_grad  (simplified from tidy3d differential_operators.py)
# ===================================================================

@unary_to_nary
def value_and_grad(fun, x, *, has_aux=False):
    """Compute value and gradient of a scalar function via autograd."""
    vjp, result = make_vjp(
        lambda x: atuple(fun(x)) if has_aux else fun(x), x
    )
    if has_aux:
        ans, aux = result
        return (ans, vjp((vspace(ans).ones(), None))), aux
    ans = result
    return ans, vjp(vspace(ans).ones())


# ===================================================================
# Kernel utilities  (from tidy3d utilities.py)
# ===================================================================

def _kernel_conic(size: Iterable[int]) -> np.ndarray:
    grids = np.ogrid[tuple(slice(-1, 1, 1j * s) for s in size)]
    dists = sum(grid**2 for grid in grids)
    return np.maximum(0, 1 - np.sqrt(dists))


def _kernel_circular(size: Iterable[int]) -> np.ndarray:
    grids = np.ogrid[tuple(slice(-1, 1, 1j * s) for s in size)]
    sq = sum(grid**2 for grid in grids)
    return np.array(sq <= 1, dtype=np.float64)


@lru_cache(maxsize=4)
def _make_kernel(kernel_type: KernelType, size: tuple[int, ...], normalize: bool = True) -> np.ndarray:
    if kernel_type == "conic":
        kernel = _kernel_conic(size)
    elif kernel_type == "circular":
        kernel = _kernel_circular(size)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")
    if normalize:
        kernel = kernel / np.sum(kernel)
    return kernel


def _get_kernel_size_px(radius, dl):
    if np.isscalar(radius):
        radius = [radius] if np.isscalar(dl) else [radius] * len(dl)
    if np.isscalar(dl):
        dl = [dl] * len(radius)
    rpx = [np.ceil(r / g) for r, g in zip(radius, dl)]
    if len(rpx) > 1:
        return [int(2 * r + 1) for r in rpx]
    return int(2 * rpx[0] + 1)


# ===================================================================
# Padding + convolution  (from tidy3d functions.py)
# ===================================================================

def _pad_indices(n, pad_width, *, mode):
    total = sum(pad_width)
    if n == 0:
        return anp.zeros(total, dtype=int)
    idx = anp.arange(-pad_width[0], n + pad_width[1])
    if mode == "constant":
        return idx
    if mode == "edge":
        return anp.clip(idx, 0, n - 1)
    if mode == "reflect":
        period = 2 * n - 2 if n > 1 else 1
        idx = anp.mod(idx, period)
        return anp.where(idx >= n, period - idx, idx)
    if mode == "symmetric":
        period = 2 * n if n > 1 else 1
        idx = anp.mod(idx, period)
        return anp.where(idx >= n, period - idx - 1, idx)
    if mode == "wrap":
        return anp.mod(idx, n)
    raise ValueError(f"Unsupported padding mode: {mode}")


def _pad_axis(array, pad_width, axis, *, mode="constant", constant_value=0.0):
    if mode == "constant":
        padding = [(0, 0)] * array.ndim
        padding[axis] = pad_width
        return anp.pad(array, padding, mode="constant", constant_values=constant_value)
    idx = _pad_indices(array.shape[axis], pad_width, mode=mode)
    indexer = [slice(None)] * array.ndim
    indexer[axis] = idx
    return array[tuple(indexer)]


def _pad(array, pad_width, *, mode="constant", axis=None, constant_value=0.0):
    pw = np.atleast_1d(pad_width)
    pt = (int(pw[0]), int(pw[0])) if pw.size == 1 else (int(pw[0]), int(pw[1]))
    if all(p == 0 for p in pt):
        return array
    axes = range(array.ndim) if axis is None else ([axis] if isinstance(axis, int) else axis)
    axes = [ax + array.ndim if ax < 0 else ax for ax in axes]
    result = array
    for ax in axes:
        result = _pad_axis(result, pt, axis=ax, mode=mode, constant_value=constant_value)
    return result


def _convolve(array, kernel, *, padding="constant", mode="same"):
    if mode in ("same", "full"):
        for axis, ks in enumerate(kernel.shape):
            pw = ks // 2
            if pw > 0:
                array = _pad(array, (pw, pw), mode=padding, axis=axis)
        mode = "valid" if mode == "same" else mode
    return _convolve_ag(array, kernel, mode=mode)


# ===================================================================
# Filter application  (from tidy3d filters.py)
# ===================================================================

def _apply_filter(array, radius, dl, filter_type="conic", padding="reflect"):
    ks = _get_kernel_size_px(radius, dl)
    size_px = tuple(np.atleast_1d(ks))
    original_shape = array.shape
    sq = anp.squeeze(array)
    if len(size_px) != sq.ndim:
        size_px = size_px * sq.ndim
    kernel = _make_kernel(filter_type, size_px, normalize=True)
    out = _convolve(sq, kernel, padding=padding)
    return anp.reshape(out, original_shape)


# ===================================================================
# tanh projection  (from tidy3d projections.py)
# ===================================================================

def _tanh_projection(array, beta=BETA_DEFAULT, eta=ETA_DEFAULT):
    if beta == 0:
        return array
    num = anp.tanh(beta * eta) + anp.tanh(beta * (array - eta))
    den = anp.tanh(beta * eta) + anp.tanh(beta * (1 - eta))
    return num / den


# ===================================================================
# make_filter_and_project  (from tidy3d parametrizations.py)
# ===================================================================

def make_filter_and_project(
    radius=None,
    dl=None,
    *,
    size_px=None,
    beta=BETA_DEFAULT,
    eta=ETA_DEFAULT,
    filter_type="conic",
    padding="reflect",
):
    """Return a callable that filters then projects an array.

    The returned callable accepts ``(array, beta=None, eta=None)``
    so that beta / eta can be overridden per call.
    """
    _radius = radius
    _dl = dl
    _ft = filter_type
    _pad = padding
    _beta = beta
    _eta = eta

    def _filter_and_project(array, beta=None, eta=None):
        filtered = _apply_filter(array, _radius, _dl, filter_type=_ft, padding=_pad)
        b = beta if beta is not None else _beta
        e = eta if eta is not None else _eta
        return _tanh_projection(filtered, b, e)

    return _filter_and_project


# ===================================================================
# make_erosion_dilation_penalty  (from tidy3d penalties.py)
# ===================================================================

def make_erosion_dilation_penalty(
    radius,
    dl,
    *,
    size_px=None,
    beta=20.0,
    eta=0.5,
    delta_eta=0.01,
    padding="reflect",
):
    """Return a callable that computes the erosion–dilation penalty."""
    _filt_proj = make_filter_and_project(
        radius=radius,
        dl=dl,
        size_px=size_px,
        beta=beta,
        eta=eta,
        filter_type="conic",
        padding=padding,
    )
    _de = delta_eta

    def _penalty(array):
        eta_dilate = 0.0 + _de
        eta_eroded = 1.0 - _de

        dilated_eroded = _filt_proj(_filt_proj(array, eta=eta_eroded), eta=eta_dilate)  # open
        eroded_dilated = _filt_proj(_filt_proj(array, eta=eta_dilate), eta=eta_eroded)  # close

        diff = eroded_dilated - dilated_eroded
        return anp.linalg.norm(diff) / anp.sqrt(diff.size)

    return _penalty