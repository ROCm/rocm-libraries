# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free float64 references for the gated delta rule (GDN and KDA).

One token of the rule, with ``S`` the ``DK x DV`` recurrent state, ``q``/``k``
of length ``DK``, ``v`` of length ``DV``, a per-K-channel log decay ``g`` and a
write strength ``beta``::

    S <- Diag(exp(g)) S              decay: scales the K axis (rows) of S
    u <- beta * (v - k^T S)           delta: what the state does not yet store
    S <- S + k u^T                    write
    o  = q^T S                        read, after the write

KDA's decay is per K channel; GDN's is one scalar per head, which is the same
rule with ``g`` broadcast over the channels (pass ``g`` with a trailing axis of
length 1). The two gate functions turn the models' raw logits into ``g``:

* GDN: ``g = -exp(A_log) * softplus(a + dt_bias)`` (unbounded below);
* KDA: ``g = lower_bound * sigmoid(exp(A_log) * (a + dt_bias))``, in
  ``(lower_bound, 0)``.

:func:`gated_delta_step` is one token over any leading batch axes,
:func:`gated_delta_recurrence` walks a ``[B, H, T, ...]`` sequence token by
token, and :func:`chunkwise_tiles` builds the six state-independent per-chunk
tiles a chunkwise kernel factors the same recurrence into. The recurrence is
deliberately not chunked, so it checks a chunkwise kernel's factorization rather
than restating it; the tiles are formed from pairwise exponent differences with
no midpoint factoring, so they never build the overflowing ``Gamma_i / Gamma_j``
a kernel is designed around.

All arithmetic is float64 on the values given (pass the decoded storage values,
see :mod:`rocke.numeric.storage`); rounding a result, or an intermediate the
kernel stores narrower, is the caller's choice.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..helpers.activations import SOFTPLUS_THRESHOLD

__all__ = [
    "CHUNKWISE_TILES",
    "chunkwise_tiles",
    "gated_delta_recurrence",
    "gated_delta_step",
    "gdn_log_decay",
    "kda_log_decay",
    "l2_normalize",
    "sigmoid",
    "softplus",
]

#: The tiles :func:`chunkwise_tiles` returns, in the order a chunkwise kernel
#: writes them.
CHUNKWISE_TILES = ("a", "gk", "gq", "aqk", "kt", "dec")

#: ``torch.nn.functional.normalize``'s default floor on the norm.
_L2_NORM_FLOOR = 1e-12


def _f64(values: npt.ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def softplus(x: npt.ArrayLike) -> np.ndarray:
    """``log1p(exp(x))``, switching to ``x`` above ``SOFTPLUS_THRESHOLD``.

    The switch point is the one every emitter uses
    (:data:`rocke.helpers.activations.SOFTPLUS_THRESHOLD`): a reference that
    switched elsewhere would differ from the kernels it checks there."""
    x = _f64(x)
    below = np.log1p(np.exp(np.minimum(x, SOFTPLUS_THRESHOLD)))
    return np.where(x > SOFTPLUS_THRESHOLD, x, below)


def sigmoid(x: npt.ArrayLike) -> np.ndarray:
    """``1 / (1 + exp(-x))``, evaluated as ``exp(-log(1 + exp(-x)))`` through
    ``logaddexp`` so no input overflows an intermediate and both tails keep
    their relative precision."""
    return np.exp(-np.logaddexp(0.0, -_f64(x)))


def l2_normalize(x: npt.ArrayLike) -> np.ndarray:
    """``x`` divided by its L2 norm over the last axis, the norm floored at
    ``1e-12`` (``torch.nn.functional.normalize``'s definition)."""
    x = _f64(x)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, _L2_NORM_FLOOR)


def gdn_log_decay(
    a: npt.ArrayLike, dt_bias: npt.ArrayLike, a_log: npt.ArrayLike
) -> np.ndarray:
    """GDN's scalar log decay ``-exp(A_log) * softplus(a + dt_bias)``.

    The operands broadcast against each other (numpy rules), so the caller
    aligns the head axis of ``dt_bias`` and ``a_log`` with ``a``'s."""
    return -np.exp(_f64(a_log)) * softplus(_f64(a) + _f64(dt_bias))


def kda_log_decay(
    g: npt.ArrayLike,
    dt_bias: npt.ArrayLike,
    a_log: npt.ArrayLike,
    lower_bound: float,
) -> np.ndarray:
    """KDA's per-channel log decay
    ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``.

    ``lower_bound`` must be negative: the result then lies in
    ``(lower_bound, 0)``, a fade that never amplifies the state. The operands
    broadcast against each other (numpy rules)."""
    if not lower_bound < 0:
        raise ValueError(
            f"kda_log_decay failed - lower_bound={lower_bound!r}, expected < 0"
        )
    inner = np.exp(_f64(a_log)) * (_f64(g) + _f64(dt_bias))
    return lower_bound * sigmoid(inner)


def gated_delta_step(
    state: npt.ArrayLike,
    q: npt.ArrayLike,
    k: npt.ArrayLike,
    v: npt.ArrayLike,
    log_decay: npt.ArrayLike,
    beta: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """One token of the gated delta rule: ``(o, state_after)``.

    ``state`` is ``[..., DK, DV]``, ``q`` and ``k`` are ``[..., DK]``, ``v`` is
    ``[..., DV]``, ``beta`` is ``[...]`` and ``log_decay`` broadcasts against
    ``k`` (``[..., DK]`` per channel, ``[..., 1]`` for one scalar per row). Any
    scale on the read (``1/sqrt(DK)``) is folded into ``q`` by the caller. The
    output reads the state after this token's write.
    """
    state, q, k, v = _f64(state), _f64(q), _f64(k), _f64(v)
    if state.shape[-2:] != (k.shape[-1], v.shape[-1]):
        raise ValueError(
            f"gated_delta_step failed - state.shape={state.shape}, expected "
            f"[..., DK={k.shape[-1]}, DV={v.shape[-1]}] to match k and v"
        )
    decay = np.broadcast_to(np.exp(_f64(log_decay)), k.shape)
    faded = state * decay[..., :, None]
    stored = np.einsum("...d,...de->...e", k, faded)
    update = _f64(beta)[..., None] * (v - stored)
    after = faded + k[..., :, None] * update[..., None, :]
    return np.einsum("...d,...de->...e", q, after), after


def gated_delta_recurrence(
    q: npt.ArrayLike,
    k: npt.ArrayLike,
    v: npt.ArrayLike,
    log_decay: npt.ArrayLike,
    beta: npt.ArrayLike,
    *,
    scale: float,
    initial_state: npt.ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The gated delta rule walked token by token: ``(o, final_state)``.

    ``q``/``k`` are ``[B, H, T, DK]``, ``v`` is ``[B, H, T, DV]``, ``beta`` is
    ``[B, H, T]``, ``log_decay`` is ``[B, H, T, DK]`` (or ``[B, H, T, 1]`` for a
    scalar gate) and ``initial_state`` is ``[B, H, DK, DV]`` (zero when
    omitted). Returns ``o`` ``[B, H, T, DV]`` with ``o_t = scale * q_t^T S_t``
    and the state after the last token, ``[B, H, DK, DV]``.
    """
    q, k, v = _f64(q), _f64(k), _f64(v)
    log_decay, beta = _f64(log_decay), _f64(beta)
    batch, heads, tokens, head_k = q.shape
    head_v = v.shape[-1]
    if initial_state is None:
        state = np.zeros((batch, heads, head_k, head_v))
    else:
        state = _f64(initial_state)
    out = np.empty((batch, heads, tokens, head_v))
    for t in range(tokens):
        out[:, :, t], state = gated_delta_step(
            state,
            scale * q[:, :, t],
            k[:, :, t],
            v[:, :, t],
            log_decay[:, :, t],
            beta[:, :, t],
        )
    return out, state


def chunkwise_tiles(
    q: npt.ArrayLike,
    k: npt.ArrayLike,
    log_decay: npt.ArrayLike,
    beta: npt.ArrayLike,
    scale: float,
    *,
    chunks_per_block: int = 32,
) -> dict[str, np.ndarray]:
    """The six state-independent tiles of each chunk, keyed by
    :data:`CHUNKWISE_TILES`.

    ``q``/``k``/``log_decay`` are ``[NC, C, DK]`` and ``beta`` is ``[NC, C]``:
    ``NC`` independent chunks of ``C`` tokens. With ``G_i`` the running sum of
    ``log_decay`` over tokens ``0..i`` of a chunk and ``G_L`` its last row:

    * ``a`` ``[NC, C, C]``: ``(I + T)^-1 Diag(beta)``, ``T`` the strictly lower
      triangle of ``beta_i * sum_d k_id k_jd exp(G_id - G_jd)``;
    * ``gk`` ``[NC, C, DK]``: ``k * exp(G)``;
    * ``gq`` ``[NC, C, DK]``: ``scale * q * exp(G)``;
    * ``aqk`` ``[NC, C, C]``: the lower triangle (diagonal kept) of
      ``scale * sum_d q_id k_jd exp(G_id - G_jd)``;
    * ``kt`` ``[NC, DK, C]``: ``(k * exp(G_L - G))^T``;
    * ``dec`` ``[NC, DK]``: ``exp(G_L)``.

    From a chunk's start state ``S0`` they give ``U = a (V - gk S0)``, the
    chunk's output ``gq S0 + aqk U`` and its end state ``dec * S0 + kt U``,
    which is the token-serial recurrence over the chunk.

    The pairwise exponent is clamped at 0 before ``exp``: on the kept triangle
    it is already ``<= 0``, so the clamp is exact there and keeps the discarded
    triangle finite. ``chunks_per_block`` chunks are built at a time to bound the
    ``C x C x DK`` temporary.
    """
    q, k, log_decay, beta = _f64(q), _f64(k), _f64(log_decay), _f64(beta)
    num_chunks, chunk, _ = q.shape
    rows = np.arange(chunk)
    strictly_lower = rows[:, None] > rows[None, :]
    lower = rows[:, None] >= rows[None, :]
    identity = np.eye(chunk)

    tiles: dict[str, list[np.ndarray]] = {name: [] for name in CHUNKWISE_TILES}
    for start in range(0, num_chunks, chunks_per_block):
        block = slice(start, start + chunks_per_block)
        q_b, k_b, beta_b = q[block], k[block], beta[block]
        cumulative = np.cumsum(log_decay[block], axis=1)
        last = cumulative[:, -1:, :]
        gamma = np.exp(cumulative)
        ratio = np.exp(
            np.minimum(cumulative[:, :, None, :] - cumulative[:, None, :, :], 0.0)
        )
        akk = np.einsum("nid,njd,nijd->nij", k_b, k_b, ratio)
        aqk = scale * np.einsum("nid,njd,nijd->nij", q_b, k_b, ratio)
        triangle = np.where(strictly_lower, beta_b[:, :, None] * akk, 0.0)
        diag_beta = beta_b[:, :, None] * identity
        tiles["a"].append(np.linalg.solve(identity + triangle, diag_beta))
        tiles["gk"].append(k_b * gamma)
        tiles["gq"].append(scale * q_b * gamma)
        tiles["aqk"].append(np.where(lower, aqk, 0.0))
        tiles["kt"].append(np.swapaxes(k_b * np.exp(last - cumulative), 1, 2))
        tiles["dec"].append(np.exp(last[:, 0, :]))
    return {name: np.concatenate(parts) for name, parts in tiles.items()}
