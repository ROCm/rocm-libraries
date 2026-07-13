# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 FP8/BF8 fused-MoE forward, wired from the component kernels.

This driver composes the already-verified MoE component kernels with the
gfx1250 K=64 FP8/BF8 ``block_scaled_gemm`` expert GEMMs into an end-to-end
forward:

    sort (static-offset, host)              -> SortedTokenIds / SortedWeights
    build_moe_gather                        -> GroupedInput[rows, H]   (bf16)
    build_smoothquant (per-row dyn quant)   -> A_fp8[rows, H], a_scale[rows]
    block_scaled_gemm  x E  (gate)          -> GateOut[rows, I]        (bf16)
    block_scaled_gemm  x E  (up)            -> UpOut[rows, I]          (bf16)
    build_moe_silu_mul                      -> Hidden[rows, I]         (bf16)
    build_smoothquant (per-row dyn quant)   -> H_fp8[rows, I], h_scale[rows]
    block_scaled_gemm  x E  (down)          -> DownOut[rows, H]        (bf16)
    build_moe_topk_weighted_reduce          -> Y[tokens, H]            (f32)

``rows = experts * slot_size`` with ``slot_size = round_up(tokens*topk, 16)``;
each expert owns a fixed ``slot_size`` row band so the per-expert GEMM M is a
constant multiple of the 16x16 WMMA output tile. Padded slots carry
``SortedWeights == 0`` so they contribute nothing to the weighted reduce.

The block-scale model uses ``block_k = K`` (one group): ``a_scale`` is the
per-row dynamic activation scale (from smoothquant) and ``b_scale`` is the
per-output-channel weight scale, so the GEMM dequantises to
``a_scale[m] * b_scale[n] * (A_fp8 . W_fp8)``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from ...helpers import compile_kernel
from ...instances.common.fused_moe import (
    FusedMoeSpec,
    build_moe_gather,
    build_moe_silu_mul,
    build_moe_topk_weighted_reduce,
    moe_gather_grid,
    moe_silu_mul_grid,
    moe_topk_weighted_reduce_grid,
)
from ...instances.common.smoothquant import (
    SmoothQuantSpec,
    build_smoothquant,
    smoothquant_grid,
)
from .block_scaled_gemm import (
    BlockScaledGemmSpec,
    block_scaled_gemm_grid,
    build_block_scaled_gemm,
)


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _bs_for(dim: int) -> int:
    for bs in (256, 128, 64):
        if dim % bs == 0:
            return bs
    return 16 if dim % 16 == 0 else 1


def _bs_common(*dims: int) -> int:
    for bs in (256, 128, 64):
        if all(d % bs == 0 for d in dims):
            return bs
    return 16 if all(d % 16 == 0 for d in dims) else 1


def _vec_for(dim: int, bs: int) -> int:
    ept = dim // bs
    for v in (8, 4, 2, 1):
        if ept % v == 0:
            return v
    return 1


def _sq_bsvec(dim: int) -> tuple:
    """Pick (block_size, vec) valid for smoothquant: vec in {2,4,8},
    dim % (block_size*vec) == 0, (dim//block_size) % vec == 0."""
    for bs in (256, 128, 64):
        if dim % bs:
            continue
        ept = dim // bs
        for v in (8, 4, 2):
            if ept % v == 0 and dim % (bs * v) == 0:
                return bs, v
    raise ValueError(
        f"no smoothquant (block_size, vec) for dim={dim}; need dim >= 128 "
        f"and divisible so block_size in {{64,128,256}} yields vec in {{2,4,8}}"
    )


def _block_k_for(k: int) -> int:
    return 128 if k % 128 == 0 else 64


@dataclass(frozen=True)
class Gfx1250Fp8MoeSpec:
    tokens: int
    experts: int
    topk: int
    hidden: int
    intermediate: int
    lowbit: str = "fp8e4m3"  # fp8e4m3 | bf8e5m2
    dtype: str = "bf16"
    name: str = "rocke_gfx1250_fp8_moe"

    @property
    def slot_size(self) -> int:
        return _round_up(self.tokens * self.topk, 16)

    @property
    def rows(self) -> int:
        return self.experts * self.slot_size


class Gfx1250Fp8Moe:
    """Compiled gfx1250 FP8/BF8 MoE forward driver (component-kernel wiring)."""

    def __init__(self, spec: Gfx1250Fp8MoeSpec, *, arch: str = "gfx1250") -> None:
        self.spec = spec
        self.arch = arch
        H, I = (
            spec.hidden,
            spec.intermediate,
        )  # noqa: E741 -- I = intermediate dim (matches kernel notation)
        rows, slot = spec.rows, spec.slot_size

        # --- streaming component kernels (gather / silu / reduce) ---
        # FusedMoeSpec validates both hidden and intermediate against its single
        # block_size, so use a block size that divides both.
        bcommon = _bs_common(H, I)
        self._gather_spec = FusedMoeSpec(
            tokens=rows,
            experts=1,
            topk=1,
            hidden=H,
            intermediate=I,
            dtype=spec.dtype,
            block_size=bcommon,
            vec=8,
            name=f"{spec.name}_gather",
        )
        self._silu_spec = FusedMoeSpec(
            tokens=rows,
            experts=1,
            topk=1,
            hidden=H,
            intermediate=I,
            dtype=spec.dtype,
            block_size=bcommon,
            vec=8,
            name=f"{spec.name}_silu",
        )
        self._reduce_spec = FusedMoeSpec(
            tokens=rows,
            experts=1,
            topk=1,
            hidden=H,
            intermediate=I,
            dtype=spec.dtype,
            block_size=bcommon,
            vec=8,
            name=f"{spec.name}_reduce",
        )
        self._gather = self._compile(build_moe_gather(self._gather_spec))
        self._silu = self._compile(build_moe_silu_mul(self._silu_spec))
        self._reduce = self._compile(build_moe_topk_weighted_reduce(self._reduce_spec))

        # --- per-row dynamic activation quant (smoothquant, SmScale = 1) ---
        aq_bs, aq_vec = _sq_bsvec(H)
        hq_bs, hq_vec = _sq_bsvec(I)
        self._aq_spec = SmoothQuantSpec(
            n_per_block=H,
            dtype=spec.dtype,
            out_dtype=spec.lowbit,
            block_size=aq_bs,
            vec=aq_vec,
            save_yscale=True,
            wave_size=32,
            name=f"{spec.name}_actq",
        )
        self._hq_spec = SmoothQuantSpec(
            n_per_block=I,
            dtype=spec.dtype,
            out_dtype=spec.lowbit,
            block_size=hq_bs,
            vec=hq_vec,
            save_yscale=True,
            wave_size=32,
            name=f"{spec.name}_hidq",
        )
        self._actq = self._compile(build_smoothquant(self._aq_spec, arch=arch))
        self._hidq = self._compile(build_smoothquant(self._hq_spec, arch=arch))

        # --- per-expert FP8/BF8 block-scaled GEMMs (fixed M = slot_size) ---
        # block_k = K -> a single K-group, so the activation scale is one value
        # per row (matching smoothquant's per-row dynamic scale) and the weight
        # scale is one value per output channel (per-channel weight quant).
        self._gu_spec = BlockScaledGemmSpec(
            name=f"{spec.name}_gu",
            M=slot,
            N=I,
            K=H,
            dtype_a=spec.lowbit,
            dtype_b=spec.lowbit,
            dtype_c=spec.dtype,
            scale_dtype="fp32",
            block_k=H,
        )
        self._down_spec = BlockScaledGemmSpec(
            name=f"{spec.name}_down",
            M=slot,
            N=H,
            K=I,
            dtype_a=spec.lowbit,
            dtype_b=spec.lowbit,
            dtype_c=spec.dtype,
            scale_dtype="fp32",
            block_k=I,
        )
        self._gu = self._compile(build_block_scaled_gemm(self._gu_spec, arch=arch))
        self._down = self._compile(build_block_scaled_gemm(self._down_spec, arch=arch))

    def _compile(self, kernel):
        return compile_kernel(kernel, arch=self.arch)

    # ------------------------------------------------------------------
    # Host-side static-offset routing layout.
    # ------------------------------------------------------------------
    def build_sorted(self, topk_ids, topk_weights):
        """Return (sorted_token_ids, sorted_weights) for the static layout.

        ``topk_ids`` / ``topk_weights`` are ``(tokens, topk)`` numpy arrays.
        Padded slots get token 0 / weight 0 so they are inert in the reduce.
        """
        import numpy as np

        s = self.spec
        sids = np.zeros((s.rows,), np.int32)
        sw = np.zeros((s.rows,), np.float32)
        counter = [0] * s.experts
        for t in range(s.tokens):
            for k in range(s.topk):
                e = int(topk_ids[t, k])
                pos = e * s.slot_size + counter[e]
                sids[pos] = t
                sw[pos] = float(topk_weights[t, k])
                counter[e] += 1
        return sids, sw

    # ------------------------------------------------------------------
    # End-to-end forward (host arrays in, host Y out).
    # ------------------------------------------------------------------
    def forward_numpy(
        self, rt, *, X, topk_ids, topk_weights, Wg, Wu, Wd, return_debug=False
    ):
        """Run the full gfx1250 FP8 MoE forward.

        ``X`` is ``(tokens, hidden)`` bf16; ``Wg``/``Wu`` are
        ``(experts, intermediate, hidden)`` fp32; ``Wd`` is
        ``(experts, hidden, intermediate)`` fp32. Weights are quantised
        per output channel (offline) to the spec low-bit dtype. Returns
        ``Y`` as ``(tokens, hidden)`` float32.
        """
        import ctypes

        import ml_dtypes
        import numpy as np

        self.bind(rt)
        s = self.spec
        H, I, E = (
            s.hidden,
            s.intermediate,
            s.experts,
        )  # noqa: E741 -- I = intermediate dim (matches kernel notation)
        slot, rows = s.slot_size, s.rows
        bf16 = ml_dtypes.bfloat16
        fp8 = (
            ml_dtypes.float8_e4m3fn if s.lowbit == "fp8e4m3" else ml_dtypes.float8_e5m2
        )
        qmax = 448.0 if s.lowbit == "fp8e4m3" else 57344.0

        def u8(a):
            a = np.ascontiguousarray(a)
            return (ctypes.c_uint8 * int(a.nbytes)).from_buffer_copy(a)

        def quant_rows(W):  # per-output-row (channel) quant -> (Wq fp8, scale f32)
            amax = np.abs(W.astype(np.float32)).max(-1, keepdims=True)
            scale = np.maximum(amax, 1e-12) / qmax
            Wq = (W.astype(np.float32) / scale).astype(fp8)
            return Wq, scale[..., 0].astype(np.float32)

        Wg_q, Wg_s = quant_rows(Wg)  # (E,I,H), (E,I)
        Wu_q, Wu_s = quant_rows(Wu)
        Wd_q, Wd_s = quant_rows(Wd)  # (E,H,I), (E,H)

        sids, sw = self.build_sorted(topk_ids, topk_weights)
        Xb = X.astype(bf16)

        # Device allocations.
        ones_h = np.ones((H,), np.float32)
        ones_i = np.ones((I,), np.float32)
        dX = _up(rt, u8(Xb), Xb.nbytes)
        dSids = _up(rt, u8(sids), sids.nbytes)
        dSw = _up(rt, u8(sw), sw.nbytes)
        dWg, dWgS = _up(rt, u8(Wg_q), Wg_q.nbytes), _up(rt, u8(Wg_s), Wg_s.nbytes)
        dWu, dWuS = _up(rt, u8(Wu_q), Wu_q.nbytes), _up(rt, u8(Wu_s), Wu_s.nbytes)
        dWd, dWdS = _up(rt, u8(Wd_q), Wd_q.nbytes), _up(rt, u8(Wd_s), Wd_s.nbytes)
        dSmH, dSmI = (
            _up(rt, u8(ones_h), ones_h.nbytes),
            _up(rt, u8(ones_i), ones_i.nbytes),
        )

        dGI = _zero(rt, rows * H * 2)  # bf16 grouped input
        dAq = _zero(rt, rows * H * 1)  # fp8 act
        dAs = _zero(rt, rows * 4)  # f32 act scale
        dGate = _zero(rt, rows * I * 2)  # bf16
        dUp = _zero(rt, rows * I * 2)  # bf16
        dHid = _zero(rt, rows * I * 2)  # bf16
        dHq = _zero(rt, rows * I * 1)  # fp8 hidden
        dHs = _zero(rt, rows * 4)  # f32 hidden scale
        dDown = _zero(rt, rows * H * 2)  # bf16
        dY = _zero(rt, s.tokens * H * 4)  # f32 out

        # 1. gather grouped input rows.
        self._launch(
            self._gather,
            moe_gather_grid(self._gather_spec),
            self._gather_spec.block_size,
            struct.pack("<QQQii", dX, dSids, dGI, rows, H),
        )
        # 2. dynamic per-row activation quant.
        self._launch(
            self._actq,
            smoothquant_grid(rows, self._aq_spec),
            self._aq_spec.block_size,
            struct.pack("<QQQQiif", dGI, dSmH, dAq, dAs, rows, H, 1e-12),
        )
        # 3. per-expert gate/up GEMMs.
        gu_grid = block_scaled_gemm_grid(self._gu_spec)
        for e in range(E):
            a = dAq + e * slot * H
            asc = dAs + e * slot * 4
            c_g = dGate + e * slot * I * 2
            c_u = dUp + e * slot * I * 2
            self._launch(
                self._gu,
                gu_grid,
                self._gu_spec.block_size,
                struct.pack(
                    "<QQQQQiii",
                    a,
                    dWg + e * I * H,
                    asc,
                    dWgS + e * I * 4,
                    c_g,
                    slot,
                    I,
                    H,
                ),
            )
            self._launch(
                self._gu,
                gu_grid,
                self._gu_spec.block_size,
                struct.pack(
                    "<QQQQQiii",
                    a,
                    dWu + e * I * H,
                    asc,
                    dWuS + e * I * 4,
                    c_u,
                    slot,
                    I,
                    H,
                ),
            )
        # 4. SwiGLU activation.
        self._launch(
            self._silu,
            moe_silu_mul_grid(self._silu_spec),
            self._silu_spec.block_size,
            struct.pack("<QQQii", dGate, dUp, dHid, rows, I),
        )
        # 5. dynamic per-row hidden quant.
        self._launch(
            self._hidq,
            smoothquant_grid(rows, self._hq_spec),
            self._hq_spec.block_size,
            struct.pack("<QQQQiif", dHid, dSmI, dHq, dHs, rows, I, 1e-12),
        )
        # 6. per-expert down GEMMs.
        down_grid = block_scaled_gemm_grid(self._down_spec)
        for e in range(E):
            self._launch(
                self._down,
                down_grid,
                self._down_spec.block_size,
                struct.pack(
                    "<QQQQQiii",
                    dHq + e * slot * I,
                    dWd + e * H * I,
                    dHs + e * slot * 4,
                    dWdS + e * H * 4,
                    dDown + e * slot * H * 2,
                    slot,
                    H,
                    I,
                ),
            )
        # 7. topk-weighted scatter back to token rows.
        self._launch(
            self._reduce,
            moe_topk_weighted_reduce_grid(self._reduce_spec),
            self._reduce_spec.block_size,
            struct.pack("<QQQQiii", dDown, dSids, dSw, dY, rows, H, s.tokens),
        )
        rt.sync()

        out = (ctypes.c_uint8 * (s.tokens * H * 4))()
        rt.memcpy_d2h(out, dY, s.tokens * H * 4)
        Y = np.frombuffer(bytes(out), dtype=np.float32).reshape(s.tokens, H).copy()

        debug = None
        if return_debug:

            def _rd(dptr, n, dt):
                buf = (ctypes.c_uint8 * n)()
                rt.memcpy_d2h(buf, dptr, n)
                return np.frombuffer(bytes(buf), dtype=dt)

            aq = _rd(dAq, rows * H, fp8).astype(np.float32).reshape(rows, H)
            as_ = _rd(dAs, rows * 4, np.float32)
            hq = _rd(dHq, rows * I, fp8).astype(np.float32).reshape(rows, I)
            hs = _rd(dHs, rows * 4, np.float32)
            debug = {
                "act_dq": aq * as_[:, None],  # GPU-dequantised grouped act
                "hid_dq": hq * hs[:, None],  # GPU-dequantised SwiGLU hidden
                "sorted_ids": self.build_sorted(topk_ids, topk_weights)[0],
            }

        for d in (
            dX,
            dSids,
            dSw,
            dWg,
            dWgS,
            dWu,
            dWuS,
            dWd,
            dWdS,
            dSmH,
            dSmI,
            dGI,
            dAq,
            dAs,
            dGate,
            dUp,
            dHid,
            dHq,
            dHs,
            dDown,
            dY,
        ):
            rt.free(d)
        return (Y, debug) if return_debug else Y

    def _launch(self, art, grid, block_size, packed):
        if art.kernel_name not in self._mods:
            m = self._rt.load_module(art.hsaco)
            self._mods[art.kernel_name] = (m, m.get_function(art.kernel_name))
        _m, fn = self._mods[art.kernel_name]
        self._rt.launch(fn, grid, (block_size, 1, 1), packed)

    def bind(self, rt):
        self._rt = rt
        self._mods = {}
        return self


def _up(rt, host_u8, nbytes):
    d = rt.alloc(nbytes)
    rt.memcpy_h2d(d, host_u8, nbytes)
    return d


def _zero(rt, nbytes):
    d = rt.alloc(nbytes)
    rt.memset(d, 0, nbytes)
    return d


__all__ = ["Gfx1250Fp8MoeSpec", "Gfx1250Fp8Moe"]
