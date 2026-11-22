# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side pieces the gfx942 and gfx950 chunkwise KDA / GDN builders share.

Torch-free: inputs are seeded numpy arrays in the kernels' storage types (bf16
as raw ``uint16`` bits, :mod:`rocke.numeric.storage`), the per-chunk tile
workspace is device memory the rocKE runtime allocates, and a parity check is
scored by :func:`score`. The references themselves live in
:mod:`rocke.numeric.gated_delta`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from rocke.numeric import storage
from rocke.numeric.compare import peak_relative_error
from rocke.numeric.gated_delta import CHUNKWISE_TILES, chunkwise_tiles, l2_normalize
from rocke.runtime.device_array import DeviceArray

__all__ = [
    "ChunkTiles",
    "check_prep",
    "dense_inputs",
    "initial_state",
    "score",
    "tile_inputs",
    "worst_relative_error",
]

#: Floor of the peak a relative error divides by, so an all-zero reference
#: yields the absolute error rather than a division by zero.
_PEAK_FLOOR = 1e-30

#: The tiles stored as bf16, in pool order; ``dec`` is a separate fp32 buffer.
_BF16_TILES = tuple(name for name in CHUNKWISE_TILES if name != "dec")


def dense_inputs(
    batch: int,
    heads: int,
    tokens: int,
    head_k: int,
    head_v: int,
    *,
    gate_low: float = -0.5,
    seed: int = 0,
) -> tuple[np.ndarray, ...]:
    """``(q, k, v, g, beta)`` for ``[B, H, T, *]`` chunkwise KDA.

    ``q``/``k`` are L2-normalised normals and ``v`` normals scaled by 0.2, all
    bf16 bits; ``g`` (the per-channel log decay) is uniform in
    ``(gate_low, 0]`` and ``beta`` uniform in ``[0, 1)``, both fp32.
    """
    rng = np.random.default_rng(seed)
    q = l2_normalize(rng.standard_normal((batch, heads, tokens, head_k)))
    k = l2_normalize(rng.standard_normal((batch, heads, tokens, head_k)))
    v = 0.2 * rng.standard_normal((batch, heads, tokens, head_v))
    g = gate_low * rng.random((batch, heads, tokens, head_k))
    beta = rng.random((batch, heads, tokens))
    return (
        storage.encode(q, "bf16"),
        storage.encode(k, "bf16"),
        storage.encode(v, "bf16"),
        storage.encode(g, "fp32"),
        storage.encode(beta, "fp32"),
    )


def tile_inputs(
    num_tiles: int, chunk: int, head_k: int, *, gate_low: float = -0.5, seed: int = 0
) -> tuple[np.ndarray, ...]:
    """``(q, k, g, beta)`` for ``num_tiles`` independent chunks: ``[NT, C, DK]``
    and ``[NT, C]``, drawn as in :func:`dense_inputs` (no ``v``)."""
    rng = np.random.default_rng(seed)
    q = l2_normalize(rng.standard_normal((num_tiles, chunk, head_k)))
    k = l2_normalize(rng.standard_normal((num_tiles, chunk, head_k)))
    g = gate_low * rng.random((num_tiles, chunk, head_k))
    beta = rng.random((num_tiles, chunk))
    return (
        storage.encode(q, "bf16"),
        storage.encode(k, "bf16"),
        storage.encode(g, "fp32"),
        storage.encode(beta, "fp32"),
    )


def initial_state(
    batch: int, heads: int, head_k: int, head_v: int, *, seed: int = 7
) -> np.ndarray:
    """A non-zero fp32 state ``[B, H, DK, DV]`` carried into prefill."""
    rng = np.random.default_rng(seed)
    return storage.encode(
        0.1 * rng.standard_normal((batch, heads, head_k, head_v)), "fp32"
    )


class ChunkTiles:
    """Device memory for the six per-chunk tiles of ``num_tiles`` chunks.

    ``a``/``aqk`` ``[NT, C, C]``, ``gk``/``gq`` ``[NT, C, DK]`` and ``kt``
    ``[NT, DK, C]`` are bf16 views of one allocation (six separate allocations
    show up on the host critical path at short sequences, and the tiles are
    always produced and consumed together); ``dec`` ``[NT, DK]`` is fp32.
    ``self[name]`` is a tile's :class:`DeviceArray`, for a launcher's values.
    Zero-filled until a prep kernel writes it.
    """

    def __init__(self, num_tiles: int, chunk: int, head_k: int) -> None:
        self.shapes = {
            "a": (num_tiles, chunk, chunk),
            "gk": (num_tiles, chunk, head_k),
            "gq": (num_tiles, chunk, head_k),
            "aqk": (num_tiles, chunk, chunk),
            "kt": (num_tiles, head_k, chunk),
            "dec": (num_tiles, head_k),
        }
        counts = [int(np.prod(self.shapes[name])) for name in _BF16_TILES]
        pool = DeviceArray.zeros((sum(counts),), "bf16")
        self._tiles = {}
        offset = 0
        for name, count in zip(_BF16_TILES, counts, strict=True):
            self._tiles[name] = pool.view(self.shapes[name], offset=offset)
            offset += count
        self._tiles["dec"] = DeviceArray.zeros(self.shapes["dec"], "fp32")

    def __getitem__(self, name: str) -> DeviceArray:
        return self._tiles[name]

    def read(self) -> dict[str, np.ndarray]:
        """Every tile's values as fp32 host arrays (blocking copies)."""
        tiles = {
            name: storage.decode(self[name].values(), "bf16") for name in _BF16_TILES
        }
        tiles["dec"] = self["dec"].values()
        return {name: tiles[name] for name in CHUNKWISE_TILES}


def worst_relative_error(
    pairs: Iterable[tuple[str, np.ndarray, np.ndarray]], *, verbose: bool = False
) -> float:
    """The worst :func:`rocke.numeric.compare.peak_relative_error` over
    ``(name, got, ref)`` pairs (the peak floored at ``1e-30``).

    NaN when any ``got`` holds a NaN, so a NaN never passes a ``<=`` test.
    With ``verbose`` it prints one line per pair.
    """
    worst = 0.0
    for name, got, ref in pairs:
        rel = peak_relative_error(got, ref, floor=_PEAK_FLOOR)
        worst = rel if np.isnan(rel) else max(worst, rel)
        if verbose:
            peak = max(float(np.abs(ref).max()), _PEAK_FLOOR)
            print(
                f"  {name:12s} max_abs={rel * peak:.3e} rel={rel:.3e} "
                f"ref_absmax={peak:.3e} finite={bool(np.isfinite(got).all())}"
            )
    return worst


def score(
    pairs: Iterable[tuple[str, np.ndarray, np.ndarray]],
    tol: float,
    *,
    verbose: bool = True,
) -> float:
    """:func:`worst_relative_error` of ``pairs``; with ``verbose`` it also
    prints the verdict against ``tol``."""
    worst = worst_relative_error(pairs, verbose=verbose)
    if verbose:
        verdict = "PASS" if worst <= tol else "FAIL"
        print(f"  -> worst rel {worst:.3e}  tol {tol:.1e}  {verdict}")
    return worst


def check_prep(
    spec, run_prep, num_tiles: int, *, gate_low: float, tol: float, verbose: bool
) -> bool:
    """Run a prep kernel (``run_prep``, the arch builder's launch) on
    :func:`tile_inputs` and score all six tiles against
    :func:`rocke.numeric.gated_delta.chunkwise_tiles`; True when every tile is
    within ``tol``."""
    chunk, head_k = spec.tile.chunk, spec.head_k
    q, k, g, beta = tile_inputs(num_tiles, chunk, head_k, gate_low=gate_low)
    scale = head_k**-0.5
    tiles = ChunkTiles(num_tiles, chunk, head_k)
    run_prep(
        spec,
        DeviceArray(q.reshape(num_tiles, -1), "bf16"),
        DeviceArray(k.reshape(num_tiles, -1), "bf16"),
        DeviceArray(g.reshape(num_tiles, -1)),
        DeviceArray(beta),
        tiles,
        scale,
    )
    got = tiles.read()
    ref = chunkwise_tiles(
        storage.decode(q, "bf16"), storage.decode(k, "bf16"), g, beta, scale
    )
    pairs = [(name, got[name], ref[name]) for name in ref]
    # A's diagonal is beta and dominates its magnitude, so scoring the whole
    # tile against its own absmax would hide the off-diagonal terms -- which are
    # the part that actually came out of the solve. Score them separately.
    pairs.append(("a_low", np.tril(got["a"], -1), np.tril(ref["a"], -1)))
    return score(pairs, tol, verbose=verbose) <= tol
