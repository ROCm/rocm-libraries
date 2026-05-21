# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Block/warp/lane decomposition for CK Tile-style kernels.

`WarpGrid` packs the boilerplate that every tile-MMA kernel re-derives
into one immutable view:

    block tile        = (tile_m, tile_n, tile_k)        # global tile size
    warp grid         = (warp_m, warp_n, warp_k)        # warps along M/N/K
    warp_tile         = (warp_tile_m, warp_tile_n,
                         warp_tile_k)                   # one MFMA atom
    block_size        = warp_m * warp_n * warp_k * wave_size

From those constants and the IR's `thread_id_x`, the grid materialises:

    tid               : i32   # block-local thread id
    lane              : i32   # within-wave lane id
    warp_id           : i32   # within-block warp id
    warp_m_idx        : i32   # warp grid coord along M
    warp_n_idx        : i32   # warp grid coord along N
    warp_k_idx        : i32   # warp grid coord along K (0 unless split-K)
    block_m_off       : i32   # block.x | block.y -> M tile base coordinate
    block_n_off       : i32
    block_k_off       : i32   # 0 unless persistent/split-K

Every value is a real `Value` in the kernel's IR, computed once and
shared across the load / MFMA / epilogue helpers. This is the same
decomposition CK Tile's `BlockGemmShape` + `BlockGemmAdaptor` produces
in C++ template code; we expose it as a single Python helper so kernel
authors do not re-derive `tid / lane / warp_id` per kernel.

`WarpGrid.bind(b)` is the constructor — it expects an `IRBuilder` and
returns a frozen instance. After that the instance is read-only.

`WarpGrid.kernel_attrs(b)` sets the kernel's `max_workgroup_size`
attribute so the AMDGPU backend bakes the correct
`amdgpu-flat-work-group-size` into the kernel descriptor.

`mfmas_per_warp_m / mfmas_per_warp_n / k_atoms_per_tile_k` answer the
"how many MFMAs of this atom shape do we issue per warp per K step?"
question. They are integers (compile-time), not IR values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.ir import IRBuilder, Value
from .atoms import MfmaAtom


@dataclass(frozen=True)
class WarpGrid:
    """Block geometry + per-thread IR values, frozen after `bind`."""

    # ---- compile-time geometry ----
    tile_m: int
    tile_n: int
    tile_k: int

    warp_m: int
    warp_n: int
    warp_k: int = 1

    warp_tile_m: int = 16
    warp_tile_n: int = 16
    warp_tile_k: int = 16

    wave_size: int = 64

    # ---- runtime IR values, populated by `bind` ----
    tid: Optional[Value] = None
    lane: Optional[Value] = None
    warp_id: Optional[Value] = None
    warp_m_idx: Optional[Value] = None
    warp_n_idx: Optional[Value] = None
    warp_k_idx: Optional[Value] = None
    block_m_off: Optional[Value] = None
    block_n_off: Optional[Value] = None
    block_k_off: Optional[Value] = None

    # ---- compile-time derived ----

    @property
    def block_size(self) -> int:
        return self.warp_m * self.warp_n * self.warp_k * self.wave_size

    @property
    def mfmas_per_warp_m(self) -> int:
        div = self.warp_m * self.warp_tile_m
        q, r = divmod(self.tile_m, div)
        if r:
            raise ValueError(
                f"tile_m {self.tile_m} not divisible by warp_m * warp_tile_m = {div}"
            )
        return q

    @property
    def mfmas_per_warp_n(self) -> int:
        div = self.warp_n * self.warp_tile_n
        q, r = divmod(self.tile_n, div)
        if r:
            raise ValueError(
                f"tile_n {self.tile_n} not divisible by warp_n * warp_tile_n = {div}"
            )
        return q

    @property
    def k_atoms_per_tile_k(self) -> int:
        q, r = divmod(self.tile_k, self.warp_tile_k)
        if r:
            raise ValueError(
                f"tile_k {self.tile_k} not divisible by warp_tile_k {self.warp_tile_k}"
            )
        return q

    # ---- factories ----

    @classmethod
    def from_atom(
        cls,
        atom: MfmaAtom,
        *,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        warp_m: int,
        warp_n: int,
        warp_k: int = 1,
        wave_size: int = 64,
    ) -> "WarpGrid":
        """Build an unbound grid from an MFMA atom + block tile + warp grid.

        Use `bind(b, ...)` to materialise the IR values.
        """
        return cls(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_k=warp_k,
            warp_tile_m=atom.m,
            warp_tile_n=atom.n,
            warp_tile_k=atom.k,
            wave_size=wave_size,
        )

    # ---- binding ----

    def bind(
        self,
        b: IRBuilder,
        *,
        block_m_axis: str = "y",
        block_n_axis: str = "x",
        block_k_axis: Optional[str] = None,
    ) -> "WarpGrid":
        """Emit the lane/warp/tid SSA into kernel `b` and return a new
        `WarpGrid` with the runtime values populated.

        `block_m_axis` / `block_n_axis` pick which `block_id_{x,y,z}` axis
        carries the M and N tile indices respectively. The default is
        the CK Tile/`gemm_universal` convention: `block.y -> M tile`,
        `block.x -> N tile`. Set `block_n_axis="y", block_m_axis="x"` to
        invert (the convolution direct-conv kernels use a different
        layout — see `conv_direct_grouped.py`).

        `block_k_axis` optionally specifies a split-K axis. If `None`,
        `block_k_off = 0` (no split-K).
        """
        if block_m_axis == block_n_axis:
            raise ValueError(
                f"block_m_axis and block_n_axis must differ (got {block_m_axis!r})"
            )
        if block_k_axis is not None and block_k_axis in (block_m_axis, block_n_axis):
            raise ValueError(f"block_k_axis {block_k_axis!r} collides with M/N axes")

        b.kernel.attrs["max_workgroup_size"] = self.block_size

        wave = b.const_i32(self.wave_size)
        c_warps_n = b.const_i32(self.warp_n)
        c_warps_n_warp_m = b.const_i32(self.warp_n * self.warp_m)
        c_tile_m = b.const_i32(self.tile_m)
        c_tile_n = b.const_i32(self.tile_n)
        c_tile_k = b.const_i32(self.tile_k)

        tid = b.thread_id_x()
        lane = b.mod(tid, wave)
        warp_id = b.div(tid, wave)

        # warp_id = warp_k_idx * (warp_m * warp_n)
        #         + warp_m_idx * warp_n + warp_n_idx
        if self.warp_k == 1:
            warp_m_idx = b.div(warp_id, c_warps_n)
            warp_n_idx = b.mod(warp_id, c_warps_n)
            warp_k_idx = b.const_i32(0)
        else:
            warp_k_idx = b.div(warp_id, c_warps_n_warp_m)
            in_plane = b.mod(warp_id, c_warps_n_warp_m)
            warp_m_idx = b.div(in_plane, c_warps_n)
            warp_n_idx = b.mod(in_plane, c_warps_n)

        block_axes = {
            "x": b.block_id_x,
            "y": b.block_id_y,
            "z": b.block_id_z,
        }
        block_m_off = b.mul(block_axes[block_m_axis](), c_tile_m)
        block_n_off = b.mul(block_axes[block_n_axis](), c_tile_n)
        if block_k_axis is None:
            block_k_off = b.const_i32(0)
        else:
            block_k_off = b.mul(block_axes[block_k_axis](), c_tile_k)

        return WarpGrid(
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            warp_m=self.warp_m,
            warp_n=self.warp_n,
            warp_k=self.warp_k,
            warp_tile_m=self.warp_tile_m,
            warp_tile_n=self.warp_tile_n,
            warp_tile_k=self.warp_tile_k,
            wave_size=self.wave_size,
            tid=tid,
            lane=lane,
            warp_id=warp_id,
            warp_m_idx=warp_m_idx,
            warp_n_idx=warp_n_idx,
            warp_k_idx=warp_k_idx,
            block_m_off=block_m_off,
            block_n_off=block_n_off,
            block_k_off=block_k_off,
        )

    # ---- convenience accessors (assertions vs unbound) ----

    @property
    def is_bound(self) -> bool:
        return self.tid is not None

    def _check_bound(self) -> None:
        if not self.is_bound:
            raise RuntimeError("WarpGrid is not bound; call .bind(b) first")

    def warp_m_off(self, b: IRBuilder) -> Value:
        """Per-warp M offset within the block tile."""
        self._check_bound()
        return b.mul(
            self.warp_m_idx,
            b.const_i32(self.mfmas_per_warp_m * self.warp_tile_m),
        )

    def warp_n_off(self, b: IRBuilder) -> Value:
        """Per-warp N offset within the block tile."""
        self._check_bound()
        return b.mul(
            self.warp_n_idx,
            b.const_i32(self.mfmas_per_warp_n * self.warp_tile_n),
        )

    def __repr__(self) -> str:
        return (
            f"WarpGrid(tile={self.tile_m}x{self.tile_n}x{self.tile_k}, "
            f"warps={self.warp_m}x{self.warp_n}x{self.warp_k}, "
            f"warp_tile={self.warp_tile_m}x{self.warp_tile_n}x"
            f"{self.warp_tile_k}, "
            f"block_size={self.block_size}, "
            f"bound={self.is_bound})"
        )
