# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for the gfx950 dense flash-attn kernel.

Drives the PUBLIC entry point ``run_attention_dense_torch`` end-to-end on a real
gfx950 GPU with device arrays (no torch) and checks every element against the float64
``rocke.numeric.sdpa_reference`` oracle (sinks, bands and LSE included), computed
from exactly the values stored on the device.

Every test is marked ``gpu`` and gated on the current device's arch
(``rocke.tests.gpu.current_device_arch_in``), so it is a graceful skip on a CPU CI
box and only executes on a gfx950. Select it with ``run_all.py --gpu`` (or
``pytest -m gpu``); the default CPU lane excludes it via ``-m "not gpu"``. Run
standalone:

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_attention_dense_gfx950_numeric.py
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import replace

import numpy as np
import pytest
from rocke.numeric import storage
from rocke.numeric.compare import Tolerance, max_abs_error, within_tolerance
from rocke.runtime.device_array import DeviceArray, host_values
from rocke.tests.gpu import current_device_arch_in
from rocke.tests.host_arrays import HostArray
from rocke.tests.ieee_compare import same_bits

from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES
from kernels.gfx950.attention_dense import (
    Gfx950AttentionDenseSpec,
    run_attention_dense_torch,
    supports_attention_dense,
)

from ._attention_tolerance import LSE_TOLERANCE, attention_tolerance
from ._dense_numeric_gpu import (
    Alias,
    assert_close,
    dense_reference,
    filled,
    first_tile_max,
    fp8_kv,
    guard_intact,
    guarded,
    operand,
    paged_cache,
    run_dispatched_paged,
    same_values,
    upload,
    values,
    window64_analytic_lse,
)


@pytest.fixture
def _gfx950_gpu():
    """Skip unless the current GPU is gfx950 (the ISA target, not a board name)."""
    current_device_arch_in(("gfx950",), subject="the gfx950 dense kernel")


requires_gfx950_gpu = pytest.mark.usefixtures("_gfx950_gpu")

# (dtype, head_size, num_query_heads, num_kv_heads, persistent) -- a compact cohort
# spanning both dtypes, D64/D128, GQA + MHA, and both grid variants. Sq is fixed at
# 512 (a 256 multiple so the persistent grid-stride has >1 q-block of work).
_COHORT = [
    ("fp16", 128, 16, 4, False),  # flagship default
    ("fp16", 128, 16, 4, True),  # flagship persistent
    ("bf16", 128, 16, 4, True),  # bf16 D128 persistent
    ("bf16", 128, 16, 4, False),  # bf16 D128 default
    ("fp16", 64, 16, 16, False),  # D64 MHA default
    ("fp16", 128, 16, 16, True),  # D128 MHA persistent (auto -> qb_major)
    ("bf16", 64, 16, 4, True),  # D64 bf16 persistent
]


# (dtype, persist_decode, interleave, sliding_window, configs, tile): one
# persistent binary per row serves every (B, S, Hq, Hkv). GQA rows mix ratios 4
# and 5 and a non-pow2 Hq; hq_minor_swz needs MHA. The window-free causal D128
# rows run the wide-DMA candidate.
#
# The tile is pinned rather than left to the tuner. ``block_m`` is baked, so two
# configs on different tiles are two binaries and can never share a cache key,
# and the tuner halves the query tile for a launch too small to fill the device
# -- which is exactly the W < NP config every row opens with. Each row names the
# tile its work counts were sized against: at the 256 tile they straddle NP = 256
# (W < NP, NP < W < 2*NP, W > 2*NP, W % NP != 0), while the windowed row runs
# NP = 512, where only the 128 tile reaches past 2*NP.
_PERSIST_GQA = ((1, 512, 16, 4), (2, 1024, 40, 8), (3, 2048, 28, 4))
_PERSIST_MHA = ((1, 512, 16, 16), (2, 1024, 48, 48), (3, 2048, 32, 32))
_PERSIST_ONE_BINARY = [
    ("bf16", "qb_major", False, 0, _PERSIST_GQA, "default"),
    ("fp16", "qb_major", True, 0, _PERSIST_GQA, "default"),
    ("bf16", "hkv_major", False, 0, _PERSIST_GQA, "default"),
    ("fp16", "bt_hkv_minor", False, 0, _PERSIST_GQA, "default"),
    ("bf16", "hq_minor_swz", False, 0, _PERSIST_MHA, "default"),
    ("bf16", "qb_major", False, 128, _PERSIST_GQA, "bm128"),
]


def _spec(
    dtype,
    d,
    hq,
    hkv,
    persistent,
    *,
    batch=1,
    sq=512,
    sk=None,
    use_sinks=False,
    sliding_window=0,
    causal=True,
    kv_block_size=0,
    causal_bottom_right=False,
    num_cus=None,
    use_fp8=False,
):
    """The SHIPPED gfx950 dense spec for a cohort row, built through the dispatch
    factory rather than hand-rolled.

    Deriving the spec from the factory means a future gfx950 tuning change is picked
    up here with no edit. The candidate is pinned by ``spec_id``: the cohort
    asserts BOTH bodies at one fixed Sq, and the persistent row runs wide
    DMA on aligned causal D128 without sinks/SWA (``TestWideDmaFeatures``
    covers the other masks).
    """
    # Imported lazily: keeps module import (and hence CPU collection of this
    # gpu-marked file) independent of the dispatch package.
    from dispatch.attention import AttentionRequest, attention_tuning_spec

    wide = d == 128 and causal and not use_sinks and not sliding_window
    if not persistent:
        spec_id = "gfx950_dense_grid"
    elif wide:
        spec_id = "gfx950_dense_persist_widedma"
    else:
        spec_id = "gfx950_dense_persist"
    if causal_bottom_right:
        mask_type = 2
    elif causal:
        mask_type = 1
    else:
        mask_type = 0
    return attention_tuning_spec(
        AttentionRequest(
            batch=batch,
            nhead_q=hq,
            nhead_k=hkv,
            seqlen_q=sq,
            seqlen_k=sq if sk is None else sk,
            hdim_q=d,
            hdim_v=d,
            arch="gfx950",
            mask_type=mask_type,
            dtype=dtype,
            use_sinks=use_sinks,
            sliding_window=sliding_window,
            **(
                {"layout": "paged", "kv_block_size": kv_block_size}
                if kv_block_size
                else {}
            ),
            use_fp8=use_fp8,
            **({} if num_cus is None else {"num_cus": num_cus}),
        ),
        spec_id,
    ).kernel_spec


def _launcher_for(spec):
    """The cached ``KernelLauncher`` for ``spec``, or None if none is compiled.

    ``run_attention_dense_torch`` owns ``_DENSE_LAUNCHER_CACHE`` internally and
    exposes no accessor, so this reads the module-level dict through the very
    key function the production path uses -- a test-local reimplementation of
    the key would assert against itself rather than against the shipped one.
    """
    from kernels.common.attention_dense_spec import attention_dense_cache_key
    from kernels.gfx950.attention_dense import _DENSE_LAUNCHER_CACHE

    return _DENSE_LAUNCHER_CACHE.get(attention_dense_cache_key(spec, arch="gfx950"))


class TestDenseNumeric:
    """Standard (non-sink) dense attention numerical validation."""

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype,d,hq,hkv,persistent", _COHORT)
    def test_dense_numeric_vs_fp32_sdpa(self, dtype, d, hq, hkv, persistent):
        """Verify the standard dense kernel against the float64 reference."""
        tol = attention_tolerance(dtype)
        B, S = 1, 512
        scale = 1.0 / math.sqrt(d)
        rng = np.random.default_rng(0)

        # run_attention_dense_torch ABI: q/out [B,S,Hq,D], k/v [B,S,Hkv,D], dense.
        q = operand(rng, (B, S, hq, d), dtype)
        k = operand(rng, (B, S, hkv, d), dtype)
        v = operand(rng, (B, S, hkv, d), dtype)
        out = filled((B, S, hq, d), dtype)

        spec = _spec(dtype, d, hq, hkv, persistent, batch=B, sq=S)
        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)

        # Reference: float64 SDPA on the stored inputs
        ref = dense_reference(q, k, v, scale)[0]

        assert_close(
            out,
            ref,
            tol,
            f"{dtype} D{d} GQA{hq}/{hkv} " f"{'persist' if persistent else 'default'}",
        )

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("persistent", [False, True])
    def test_bf16_d128_non_default_scale(self, persistent):
        """BF16 D128 at softmax scale 0.5, well above the default 1/sqrt(D).

        The score error a lossy scale step introduces grows with ``scale``, so
        the default-scale cohort above cannot see it. Rounding
        ``Q * scale * log2(e)`` back to bf16 before the QK MFMA puts this shape
        past the bf16 tolerance; the kernel must apply the scale in fp32.
        """
        dtype, d, hq, hkv = "bf16", 128, 16, 4
        tol = attention_tolerance(dtype)
        B, S, scale = 1, 512, 0.5
        rng = np.random.default_rng(0)

        q = operand(rng, (B, S, hq, d), "bf16")
        k = operand(rng, (B, S, hkv, d), "bf16")
        v = operand(rng, (B, S, hkv, d), "bf16")
        out = filled((B, S, hq, d), "bf16")

        spec = _spec(dtype, d, hq, hkv, persistent, batch=B, sq=S)
        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)

        ref = dense_reference(q, k, v, scale)[0]
        assert_close(
            out,
            ref,
            tol,
            f"bf16 D128 GQA16/4 scale=0.5 " f"{'persist' if persistent else 'default'}",
        )

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("persistent", [False, True])
    @pytest.mark.parametrize("dtype", ["bf16", "fp16"])
    @pytest.mark.parametrize("scale", [2.0**-64, 2.0**4], ids=["2^-64", "2^4"])
    def test_d128_scale_range_edges(self, scale, dtype, persistent):
        """Both bounds of the supported softmax-scale range, on silicon.

        At 2**-64 every scaled score is about 0, so the softmax is close to
        uniform; at 2**4 it is close to one-hot and the ordinary kernel's fma
        residue is at its largest. Both must match the fp32 reference.
        """
        d, hq, hkv = 128, 16, 4
        tol = attention_tolerance(dtype)
        B, S = 1, 512
        rng = np.random.default_rng(0)

        q = operand(rng, (B, S, hq, d), dtype)
        k = operand(rng, (B, S, hkv, d), dtype)
        v = operand(rng, (B, S, hkv, d), dtype)
        out = filled((B, S, hq, d), dtype)

        spec = _spec(dtype, d, hq, hkv, persistent, batch=B, sq=S)
        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)

        assert np.isfinite(values(out)).all(), "non-finite output"
        ref = dense_reference(q, k, v, scale)[0]
        assert_close(
            out,
            ref,
            tol,
            f"{dtype} D128 GQA16/4 scale={scale:g} "
            f"{'persist' if persistent else 'default'}",
        )

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("persistent", [False, True])
    @pytest.mark.parametrize(
        "d,scale", [(64, 1.0 / math.sqrt(64)), (128, 0.5), (128, 1.0), (128, 2.0**4)]
    )
    def test_bf16_sliding_window_no_sinks(self, d, scale, persistent):
        """Causal sliding window without sinks, where whole query rows of the
        first visited KV tile are masked.

        Those rows start the online softmax from the mask sentinel, not from a
        real score or a sink logit. The sentinel must survive the softmax scale
        exactly, or exp2 of a huge rounding residue turns the row into inf/NaN.
        scale * log2(e) <= 1 (D64 at its default scale, D128 at 0.5) and > 1
        (D128 at 1.0, the hipDNN default when no scale is given, and at 2**4,
        the upper bound) take different exact branches in the ordinary kernel;
        both are covered.
        """
        dtype, hq, hkv, window = "bf16", 16, 4, 128
        tol = attention_tolerance(dtype)
        B, S = 1, 512
        rng = np.random.default_rng(0)

        q = operand(rng, (B, S, hq, d), "bf16")
        k = operand(rng, (B, S, hkv, d), "bf16")
        v = operand(rng, (B, S, hkv, d), "bf16")
        out = filled((B, S, hq, d), "bf16")

        spec = _spec(
            dtype, d, hq, hkv, persistent, batch=B, sq=S, sliding_window=window
        )
        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)

        assert np.isfinite(values(out)).all(), "non-finite output"
        ref = dense_reference(q, k, v, scale, window=window)[0]
        assert_close(
            out,
            ref,
            tol,
            f"bf16 D{d} SWA{window} scale={scale:g} "
            f"{'persist' if persistent else 'default'}",
        )

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", ["bf16", "fp16"])
    @pytest.mark.parametrize("d,hq,hkv", [(64, 16, 4), (128, 16, 4)])
    # (causal, sliding_window): flash-causal, SWA-128, non-causal (full), and the
    # left-only window. Each is a distinct mask path the PR adds; the IR golden
    # pins their bytes, this pins their correctness. The kernel does serve a
    # non-causal window (_supports_noncausal_window is True and the validator
    # accepts it), and that band runs its own left-only pipeline, so it needs its
    # own numeric case rather than being assumed unreachable.
    @pytest.mark.parametrize(
        "causal,sliding_window", [(True, 0), (True, 128), (False, 0), (False, 128)]
    )
    def test_fp8_kv_dequant_numeric_vs_fp32_sdpa(
        self, d, hq, hkv, dtype, causal, sliding_window
    ):
        """fp8 KV-dequant dense: the kernel reads float8_e4m3fn K/V and dequants
        with per-tensor k_scale/v_scale into the bf16 or fp16 compute/output
        path; the reference attends over the SAME dequantized K/V,
        so this isolates kernel error from quantization loss (the convention the
        live bench uses). Covered over the causal, sliding-window and non-causal
        masks the grid variant serves for fp8."""
        B, S = 1, 512
        scale = 1.0 / math.sqrt(d)
        rng = np.random.default_rng(0)
        q = operand(rng, (B, S, hq, d), dtype, scale=0.3)
        k = storage.round_to(rng.standard_normal((B, S, hkv, d)) * 0.3, dtype)
        v = storage.round_to(rng.standard_normal((B, S, hkv, d)) * 0.3, dtype)
        k_fp8, k_scale, k_deq = fp8_kv(k, dtype)
        v_fp8, v_scale, v_deq = fp8_kv(v, dtype)

        spec = _spec(
            dtype,
            d,
            hq,
            hkv,
            persistent=False,
            batch=B,
            sq=S,
            use_fp8=True,
            causal=causal,
            sliding_window=sliding_window,
        )
        out = filled((B, S, hq, d), dtype)
        run_attention_dense_torch(
            spec=spec,
            q=q,
            k=k_fp8,
            v=v_fp8,
            out=out,
            scale=scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        ref, _ = dense_reference(
            q, k_deq, v_deq, scale, causal=causal, window=sliding_window
        )
        max_abs = max_abs_error(values(out), values(ref))
        # Reference attends over the dequantized K/V, so the residual is kernel
        # compute error at the compute precision; keep a small fp8 margin over tol.
        assert max_abs < 6e-2, (
            f"fp8 {dtype} D{d} GQA{hq}/{hkv} causal={causal} "
            f"swa={sliding_window}: max_abs={max_abs:.3e}"
        )

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "d,persistent,width",
        [(128, False, 1), (128, True, 2), (64, False, 2), (64, True, 1)],
    )
    def test_bf16_narrow_output_store_is_bit_identical(self, d, persistent, width):
        """bf16 o_store_width 1/2 must write the same bits as the width-4 store
        (fp16 is rejected below 4 because it does not)."""
        hq, hkv, B, S = 16, 4, 1, 512
        scale = 1.0 / math.sqrt(d)
        rng = np.random.default_rng(0)
        q = operand(rng, (B, S, hq, d), "bf16")
        k = operand(rng, (B, S, hkv, d), "bf16")
        v = operand(rng, (B, S, hkv, d), "bf16")
        base = _spec("bf16", d, hq, hkv, persistent, batch=B, sq=S)

        outs = []
        for spec in (base, dataclasses.replace(base, o_store_width=width)):
            out = filled((B, S, hq, d), "bf16")
            run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
            outs.append(out)
        assert same_values(outs[0], outs[1]), (
            f"bf16 D{d} {'persist' if persistent else 'default'} "
            f"o_store_width={width} diverged from width 4"
        )

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("sliding_window", (0, 128))
    def test_one_binary_serves_every_shape(self, sliding_window):
        """One compiled artifact, two shapes, correct numerics at both.

        The cohort above runs many shapes, but it stopped discriminating the
        moment batch/seqlen_q/seqlen_kv became runtime kernel params: it passes
        identically whether N shapes are served by N binaries or by one. The
        property that actually needs a guard now is *artifact reuse*.

        ``is`` on the launcher covers the whole path rather than just the key
        function. A key regression would land the two shapes in different cache
        slots; a lookup regression would overwrite the one slot with a freshly
        compiled launcher. Both yield a different object, and neither is visible
        to a numeric assertion -- recompiling per shape is *correct*, merely
        wasteful, so what regresses is the AOT instance count and first-call
        latency, which no accuracy check can see.

        Paired with the numeric check at both shapes so the test cannot pass by
        reusing one binary that happens to be wrong for the second shape.
        """
        from kernels.common.attention_dense_spec import attention_dense_cache_key
        from kernels.gfx950.attention_dense import _DENSE_LAUNCHER_CACHE

        dtype, d, hq, hkv = "bf16", 128, 16, 4  # flagship default, non-persistent
        tol = attention_tolerance(dtype)
        scale = 1.0 / math.sqrt(d)

        shapes = ((1, 512), (4, 1024))
        specs = [
            _spec(
                dtype, d, hq, hkv, False, batch=b, sq=s, sliding_window=sliding_window
            )
            for b, s in shapes
        ]

        # Preconditions: genuinely different shapes, on the runtime path, and
        # sharing one key -- otherwise the reuse assertion is vacuous.
        assert specs[0].runtime_shape, "cohort row is not on the runtime-shape path"
        assert (specs[0].batch, specs[0].seqlen_q) != (
            specs[1].batch,
            specs[1].seqlen_q,
        )
        keys = [attention_dense_cache_key(s, arch="gfx950") for s in specs]
        assert keys[0] == keys[1], f"shapes {shapes} did not share a cache key"

        # Own the cache state: evicting first makes the "exactly one new entry"
        # assertion independent of which tests ran before this one.
        _DENSE_LAUNCHER_CACHE.pop(keys[0], None)
        before = set(_DENSE_LAUNCHER_CACHE)

        launchers = []
        for (B, S), spec in zip(shapes, specs):
            rng = np.random.default_rng(0)
            q = operand(rng, (B, S, hq, d), dtype)
            k = operand(rng, (B, S, hkv, d), dtype)
            v = operand(rng, (B, S, hkv, d), dtype)
            out = filled((B, S, hq, d), dtype)

            run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
            launchers.append(_launcher_for(spec))

            ref = dense_reference(q, k, v, scale, window=sliding_window)[0]
            assert_close(out, ref, tol, f"B={B} S={S}")

        assert launchers[0] is not None, (
            "no launcher cached after a successful run; _DENSE_LAUNCHER_CACHE is "
            "no longer keyed by attention_dense_cache_key and this test is blind"
        )
        assert launchers[0] is launchers[1], (
            f"shapes {shapes} share a cache key but were served by DIFFERENT "
            "launcher objects -- the runtime-shape kernel recompiled per shape, "
            "so the AOT instance count still scales with the shape space"
        )
        assert set(_DENSE_LAUNCHER_CACHE) - before == {keys[0]}, (
            "two shapes on the runtime path added more than one cache entry: "
            f"{sorted(set(_DENSE_LAUNCHER_CACHE) - before)}"
        )

    @requires_gfx950_gpu
    @pytest.mark.gpu
    def test_one_binary_serves_every_head_config(self):
        """One compiled artifact, four head configs, correct numerics at each.

        The head-count twin of :func:`test_one_binary_serves_every_shape`: on the
        runtime-shape path num_query_heads / num_kv_heads are kernel params, so the
        GQA ratio and the token strides are computed at run time. A body that still
        baked one of them would compute wrong strides for every config but the first
        one compiled -- caught here by the numeric check, while the ``is`` /
        single-entry assertions catch a key or lookup regression that recompiles per
        head config. Covers GQA 16, non-pow2 GQA 5 and 7, and MHA.
        """
        from kernels.common.attention_dense_spec import attention_dense_cache_key
        from kernels.gfx950.attention_dense import _DENSE_LAUNCHER_CACHE

        dtype, d, B, S = "bf16", 128, 1, 512  # flagship default, non-persistent
        tol = attention_tolerance(dtype)
        scale = 1.0 / math.sqrt(d)

        heads = ((128, 8), (40, 8), (28, 4), (32, 32))
        specs = [_spec(dtype, d, hq, hkv, False, batch=B, sq=S) for hq, hkv in heads]

        # Preconditions: all on the runtime path and sharing one key -- otherwise the
        # reuse assertion below is vacuous.
        assert all(
            s.runtime_shape for s in specs
        ), "a head config left the runtime path"
        keys = [attention_dense_cache_key(s, arch="gfx950") for s in specs]
        assert len(set(keys)) == 1, f"head configs {heads} did not share a cache key"

        _DENSE_LAUNCHER_CACHE.pop(keys[0], None)
        before = set(_DENSE_LAUNCHER_CACHE)

        launchers = []
        for (hq, hkv), spec in zip(heads, specs):
            rng = np.random.default_rng(0)
            q = operand(rng, (B, S, hq, d), dtype)
            k = operand(rng, (B, S, hkv, d), dtype)
            v = operand(rng, (B, S, hkv, d), dtype)
            out = filled((B, S, hq, d), dtype)

            run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
            launchers.append(_launcher_for(spec))

            ref = dense_reference(q, k, v, scale)[0]
            assert_close(out, ref, tol, f"Hq={hq} Hkv={hkv}")

        assert launchers[0] is not None, (
            "no launcher cached after a successful run; _DENSE_LAUNCHER_CACHE is no "
            "longer keyed by attention_dense_cache_key and this test is blind"
        )
        assert all(lau is launchers[0] for lau in launchers), (
            f"head configs {heads} share a cache key but were served by DIFFERENT "
            "launcher objects -- the runtime-heads kernel recompiled per head config"
        )
        assert set(_DENSE_LAUNCHER_CACHE) - before == {keys[0]}, (
            "four head configs on the runtime path added more than one cache entry: "
            f"{sorted(set(_DENSE_LAUNCHER_CACHE) - before)}"
        )

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "dtype,persist_decode,interleave,window,configs,tile", _PERSIST_ONE_BINARY
    )
    def test_one_persistent_binary_serves_every_shape_and_head_config(
        self, dtype, persist_decode, interleave, window, configs, tile
    ):
        """The persistent twin of the one-binary tests above.

        The persistent body takes batch, both seqlens and both head counts as kernel
        params; only D, dtype, mask, window, the tile geometry, ``num_persistent``
        and the decode order stay baked. Its work count is derived in the prologue
        and the grid-stride decode divides by runtime B / Hq / gqa / NQB through
        host-computed fast-division magics. A decode that kept a baked shape term,
        or a magic for the wrong divisor, would map work items to the wrong
        (qb, hq, bt) for every config but the first compiled: duplicated or skipped
        tiles only the numeric check sees. The configs straddle NP (W < NP,
        NP < W < 2*NP, W > 2*NP, W % NP != 0), so idle CTAs, partial last rounds
        and multiple trips all run.
        """
        from kernels.common.attention_dense_spec import attention_dense_cache_key
        from kernels.gfx950.attention_dense import _DENSE_LAUNCHER_CACHE

        d = 128
        tol = attention_tolerance(dtype)
        scale = 1.0 / math.sqrt(d)

        geometry = DENSE_TILE_GEOMETRIES[tile]
        specs = [
            replace(
                _spec(dtype, d, hq, hkv, True, batch=b, sq=s, sliding_window=window),
                persist_decode=persist_decode,
                interleave=interleave,
                block_m=int(geometry["block_m"]),
                block_n=int(geometry["block_n"]),
            )
            for b, s, hq, hkv in configs
        ]

        # Preconditions: persistent, on the runtime path, the pinned order, one key,
        # and the configs really straddle NP, or the grid-stride claims are vacuous.
        assert all(
            s.persistent and s.runtime_shape for s in specs
        ), "a config left the persistent runtime-shape path"
        assert {s.resolved_persist_decode for s in specs} == {persist_decode}
        assert all(
            s.fastdiv_divisors for s in specs
        ), "persistent decode has no fastdiv"
        np_ = specs[0].num_persistent
        work = [(s.seqlen_q // s.block_m) * s.num_query_heads * s.batch for s in specs]
        assert min(work) < np_ and max(work) > 2 * np_, (work, np_)
        assert any(w % np_ for w in work), (work, np_)
        keys = [attention_dense_cache_key(s, arch="gfx950") for s in specs]
        assert (
            len(set(keys)) == 1
        ), f"persistent configs {configs} did not share a cache key"

        _DENSE_LAUNCHER_CACHE.pop(keys[0], None)
        before = set(_DENSE_LAUNCHER_CACHE)

        launchers = []
        for (B, S, hq, hkv), spec in zip(configs, specs):
            rng = np.random.default_rng(0)
            q = operand(rng, (B, S, hq, d), dtype)
            k = operand(rng, (B, S, hkv, d), dtype)
            v = operand(rng, (B, S, hkv, d), dtype)
            out = filled((B, S, hq, d), dtype)

            run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
            launchers.append(_launcher_for(spec))

            ref = dense_reference(q, k, v, scale, window=window)[0]
            assert_close(
                out,
                ref,
                tol,
                f"{persist_decode} B={B} S={S} Hq={hq} Hkv={hkv}: ",
            )

        assert launchers[0] is not None, "no launcher cached; this test is blind"
        assert all(lau is launchers[0] for lau in launchers), (
            f"persistent configs {configs} share a cache key but were served by "
            "DIFFERENT launchers -- the runtime-shape kernel recompiled per config"
        )
        assert set(_DENSE_LAUNCHER_CACHE) - before == {keys[0]}, (
            "persistent configs on the runtime path added more than one cache entry: "
            f"{sorted(set(_DENSE_LAUNCHER_CACHE) - before)}"
        )


# Sink-enabled numerical tests: verify that the sink-enabled kernel matches the
# reference softmax(concat([QK scores, sink]))[..., :-1] @ V formula.
#
# Two critical cases:
# 1. Sink ABOVE first tile's QK maximum -> tests m_init handling
# 2. Sink BELOW first tile's QK maximum -> tests l_init * alpha0 correction
#
# These verify the online softmax rescale logic that IR parity alone cannot validate.
# Cohort format: (dtype, head_size, num_query_heads, num_kv_heads, persistent, sliding_window, causal)
_SINK_COHORT = [
    # Plain causal (no sliding window)
    ("bf16", 128, 32, 8, False, 0, True),  # bf16 D128 GQA default
    ("bf16", 128, 32, 8, True, 0, True),  # bf16 D128 GQA persistent
    ("bf16", 64, 32, 8, False, 0, True),  # bf16 D64 GQA default
    ("bf16", 64, 32, 8, True, 0, True),  # bf16 D64 GQA persistent
    ("fp16", 128, 32, 8, False, 0, True),  # fp16 D128 GQA default
    ("fp16", 128, 32, 8, True, 0, True),  # fp16 D128 GQA persistent
    # SWA (sinks + sliding window)
    ("bf16", 128, 32, 8, False, 128, True),  # bf16 D128 GQA SWA default
    ("bf16", 128, 32, 8, True, 128, True),  # bf16 D128 GQA SWA persistent
    ("fp16", 128, 32, 8, False, 256, True),  # fp16 D128 GQA SWA default
    ("fp16", 128, 32, 8, True, 256, True),  # fp16 D128 GQA SWA persistent
    # Non-causal (full attention + sinks)
    ("bf16", 128, 32, 8, False, 0, False),  # bf16 D128 GQA non-causal default
    ("bf16", 128, 32, 8, True, 0, False),  # bf16 D128 GQA non-causal persistent
]


_GQA_PAIR_VARIANTS = [
    ("plain", 0, False, False),
    ("interleave", 0, False, True),
    ("sliding_window", 128, False, False),
    ("sinks", 0, True, False),
    ("sinks_sliding_window", 128, True, False),
]


class TestDenseGqaPairVariants:
    """Numeric coverage for features composed with the optimized work mapping."""

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "_name,sliding_window,use_sinks,interleave", _GQA_PAIR_VARIANTS
    )
    def test_gqa_pair_variant_numeric(
        self, _name, sliding_window, use_sinks, interleave
    ):
        B, S, Hq, Hkv, D = 1, 512, 32, 8, 128
        rng = np.random.default_rng(0)
        q = operand(rng, (B, S, Hq, D), "fp16")
        k = operand(rng, (B, S, Hkv, D), "fp16")
        v = operand(rng, (B, S, Hkv, D), "fp16")
        out = filled(q.shape, q.dtype)
        sinks = filled((Hq,), "fp16", 0.0) if use_sinks else None
        scale = 1.0 / math.sqrt(D)

        # NQB=2 and Hkv=8, so the balanced mapping needs exactly 16 CTAs.
        spec = Gfx950AttentionDenseSpec(
            batch=B,
            seqlen_q=S,
            seqlen_kv=S,
            num_query_heads=Hq,
            num_kv_heads=Hkv,
            head_size=D,
            causal=True,
            dtype="fp16",
            block_n=64,
            persistent=True,
            num_persistent=16,
            persist_decode="gqa_pair",
            interleave=interleave,
            sliding_window=sliding_window,
            use_sinks=use_sinks,
            wide_lds_dma=True,
        )
        run_attention_dense_torch(
            spec=spec,
            q=q,
            k=k,
            v=v,
            out=out,
            scale=scale,
            sinks=sinks,
        )

        ref, _ = dense_reference(q, k, v, scale, window=sliding_window, sinks=sinks)
        assert_close(out, ref, attention_tolerance("fp16"), f"gqa_pair {_name}")

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("block_m", (256, 128), ids=("bm256", "bm128"))
    def test_wide_dma_mha_numeric(self, block_m):
        """Both wide-DMA line-pass geometries work without GQA reuse."""
        B, S, H, D = 1, 512, 16, 128
        rng = np.random.default_rng(0)
        q = operand(rng, (B, S, H, D), "fp16")
        k = operand(rng, (B, S, H, D), "fp16")
        v = operand(rng, (B, S, H, D), "fp16")
        out = filled(q.shape, q.dtype)
        scale = 1.0 / math.sqrt(D)
        spec = Gfx950AttentionDenseSpec(
            batch=B,
            seqlen_q=S,
            seqlen_kv=S,
            num_query_heads=H,
            num_kv_heads=H,
            head_size=D,
            causal=True,
            dtype="fp16",
            block_m=block_m,
            block_n=64,
            persistent=True,
            num_persistent=32,
            persist_decode="qb_major",
            wide_lds_dma=True,
        )
        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
        ref = dense_reference(q, k, v, scale)[0]
        assert_close(out, ref, attention_tolerance("fp16"), "wide DMA MHA")


class TestDenseSinksNumeric:
    """Sink-enabled dense attention numerical validation.

    Verifies that the sink-enabled kernel matches the reference
    softmax(concat([QK scores, sink]))[..., :-1] @ V formula.

    Two critical cases:
    1. Sink ABOVE first tile's QK maximum -> tests m_init handling
    2. Sink BELOW first tile's QK maximum -> tests l_init * alpha0 correction

    These verify the online softmax rescale logic that IR parity alone cannot validate.
    """

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype,d,hq,hkv,persistent,sw,causal", _SINK_COHORT)
    @pytest.mark.parametrize("sink_magnitude", ["above_qk_max", "below_qk_max"])
    def test_sinks_numeric(
        self, dtype, d, hq, hkv, persistent, sw, causal, sink_magnitude
    ):
        """Verify sink-enabled kernel against manual sink reference.

        Tests two critical cases:
        - sink_magnitude='above_qk_max': Sink > first tile's QK max -> tests m_init
        - sink_magnitude='below_qk_max': Sink < first tile's QK max -> tests l_init*alpha0

        These verify the online softmax rescale logic with sinks, which IR parity alone
        cannot validate (m_init and l_init depend on the runtime QK values).

        Note on sliding windows (sw > 0): With the first tile (queries [0:256], keys
        [0:64]), some query rows may see no keys from [0:64] (their sliding window
        has moved past the key range) and produce all -inf scores. However, queries
        earlier in the sequence still see keys and produce finite QK scores. The max
        operation across all query rows naturally picks the finite maximum from the
        queries that DO see keys, ensuring the above/below sink comparison remains
        meaningful. At least one WG (first tile) hits the valid path.
        """
        tol = attention_tolerance(dtype)
        B, S = 1, 512
        scale = 1.0 / math.sqrt(d)
        rng = np.random.default_rng(0)

        # Standard Q/K/V tensors
        q = operand(rng, (B, S, hq, d), dtype)
        k = operand(rng, (B, S, hkv, d), dtype)
        v = operand(rng, (B, S, hkv, d), dtype)
        out = filled((B, S, hq, d), dtype)

        # Compute first tile's QK maximum to set sink relative to it.
        # We use the FIRST tile (not global max) because sinks are constant scalars
        # available from the start, so the critical test is: "when the kernel processes
        # tile 0, does it correctly initialize m_init and l_init relative to the sink?"
        # Setting sink above/below first_tile_max directly tests this initialization.
        # First tile: queries [0:256], keys [0:64] (block_n=64 default)
        tile_max = first_tile_max(q, k, scale, 256, 64, causal=causal)  # [Hq]

        # Create sink values relative to first tile's QK max, stored in the input
        # dtype.
        if sink_magnitude == "above_qk_max":
            # Sink ABOVE first tile max -> tests m_init = max(m_tile0, sink)
            sinks = upload(tile_max + 1.0, dtype)
        else:  # below_qk_max
            # Sink BELOW first tile max -> tests l_init * alpha0 correction
            sinks = upload(tile_max - 1.0, dtype)

        # Run kernel with sinks (and optional sliding window / causal)
        spec = _spec(
            dtype,
            d,
            hq,
            hkv,
            persistent,
            batch=B,
            sq=S,
            use_sinks=True,
            sliding_window=sw,
            causal=causal,
        )
        run_attention_dense_torch(
            spec=spec, q=q, k=k, v=v, out=out, scale=scale, sinks=sinks
        )

        # Reference: softmax(concat([QK, sink]))[..., :-1] @ V
        ref = dense_reference(q, k, v, scale, causal=causal, window=sw, sinks=sinks)[0]

        # Build descriptive label
        mask_label = []
        if not causal:
            mask_label.append("full")
        else:
            mask_label.append("causal")
        if sw > 0:
            mask_label.append(f"sw{sw}")
        mask_str = "+".join(mask_label)

        assert_close(
            out,
            ref,
            tol,
            f"{dtype} D{d} GQA{hq}/{hkv} "
            f"{'persist' if persistent else 'default'} {mask_str} sink_{sink_magnitude}",
        )


# The original nine PR cases plus one aligned and one ragged BM128 case. Each
# pytest.param is a separate GPU test node: on a real gfx950 none can disappear
# behind an aggregate helper that reports success after skipping every case.
_BOTTOM_RIGHT_CASES = [
    pytest.param("default", 197, 400, 1, True, False, id="bm256-ragged-197x400"),
    pytest.param("default", 300, 1234, 1, True, False, id="bm256-ragged-300x1234"),
    pytest.param("default", 512, 4097, 1, True, False, id="bm256-ragged-512x4097"),
    pytest.param("default", 100, 8000, 1, True, False, id="bm256-ragged-100x8000"),
    # The ragged buffer spans the whole tensor: padded rows in batch 0 can read
    # live batch-1 data, so correctness depends on the qtok < Sq store guard.
    pytest.param("default", 300, 1000, 2, True, False, id="bm256-ragged-batch2"),
    pytest.param("default", 256, 512, 1, False, False, id="bm256-aligned-256x512"),
    pytest.param("default", 512, 1024, 1, False, False, id="bm256-aligned-512x1024"),
    pytest.param("default", 512, 1024, 1, False, True, id="bm256-sinks-aligned"),
    pytest.param("default", 300, 1234, 1, True, True, id="bm256-sinks-ragged"),
    pytest.param("bm128", 512, 1024, 1, False, False, id="bm128-aligned-512x1024"),
    pytest.param("bm128", 197, 400, 1, True, False, id="bm128-ragged-197x400"),
]

#: Every moving-diagonal case here is bf16.
_BOTTOM_RIGHT_TOL = attention_tolerance("bf16")


def _assert_shifted_not_top_left(out, shifted, top_left, label: str) -> None:
    """``out`` matches the bottom-right (shifted-diagonal) reference and is
    outside the bound around the top-left one, so the check cannot pass on a
    kernel that ignores the shift."""
    assert_close(out, shifted, _BOTTOM_RIGHT_TOL, f"{label}: shifted")
    assert not within_tolerance(
        values(out), values(top_left), _BOTTOM_RIGHT_TOL
    ), f"{label}: unexpectedly matches top-left"


class TestDenseBottomRightNumeric:
    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "geometry,sq,skv,batch,ragged,use_sinks", _BOTTOM_RIGHT_CASES
    )
    def test_jit_matches_shifted_diagonal_not_top_left(
        self, geometry, sq, skv, batch, ragged, use_sinks
    ):
        tile = DENSE_TILE_GEOMETRIES[geometry]
        spec = Gfx950AttentionDenseSpec(
            batch=batch,
            seqlen_q=sq,
            seqlen_kv=skv,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=128,
            causal=True,
            dtype="bf16",
            block_m=int(tile["block_m"]),
            block_n=int(tile["block_n"]),
            ragged=ragged,
            persistent=False,
            use_sinks=use_sinks,
            causal_bottom_right=True,
        )
        assert skv > sq, "every case must exercise a moving diagonal"

        rng = np.random.default_rng(0)
        q = operand(rng, (batch, sq, 4, 128), "bf16")
        k = operand(rng, (batch, skv, 1, 128), "bf16")
        v = operand(rng, (batch, skv, 1, 128), "bf16")
        out = filled(q.shape, q.dtype)
        sinks = operand(rng, (4,), "bf16") if use_sinks else None
        scale = 1.0 / math.sqrt(spec.head_size)

        run_attention_dense_torch(
            spec=spec,
            q=q,
            k=k,
            v=v,
            out=out,
            scale=scale,
            sinks=sinks,
        )

        shifted = dense_reference(q, k, v, scale, bottom_right=True, sinks=sinks)[0]
        top_left = dense_reference(q, k, v, scale, sinks=sinks)[0]
        label = (
            f"{geometry} B{batch} {sq}x{skv} "
            f"{'ragged' if ragged else 'aligned'}"
            f"{'+sinks' if use_sinks else ''}"
        )
        _assert_shifted_not_top_left(out, shifted, top_left, label)

    # fp8 KV is aligned-only and has no sink path, so only aligned, sink-free
    # moving diagonals apply. D128 and D64 cover both K layouts of the fp8 loader.
    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "d,sq,skv",
        [
            pytest.param(128, 256, 512, id="d128-256x512"),
            pytest.param(128, 512, 1024, id="d128-512x1024"),
            pytest.param(64, 512, 1024, id="d64-512x1024"),
        ],
    )
    def test_fp8_kv_matches_shifted_diagonal_not_top_left(self, d, sq, skv):
        tile = DENSE_TILE_GEOMETRIES["default"]
        spec = Gfx950AttentionDenseSpec(
            batch=1,
            seqlen_q=sq,
            seqlen_kv=skv,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=d,
            causal=True,
            dtype="bf16",
            block_m=int(tile["block_m"]),
            block_n=int(tile["block_n"]),
            persistent=False,
            causal_bottom_right=True,
            kv_storage_dtype="fp8e4m3",
        )
        assert skv > sq, "every case must exercise a moving diagonal"

        rng = np.random.default_rng(0)
        q = operand(rng, (1, sq, 4, d), "bf16")
        k = storage.round_to(rng.standard_normal((1, skv, 1, d)), "bf16")
        v = storage.round_to(rng.standard_normal((1, skv, 1, d)), "bf16")
        # The oracle attends over the same dequantized K/V the kernel reads, so
        # the check isolates kernel error from quantization loss.
        k_fp8, k_scale, k_deq = fp8_kv(k, "bf16")
        v_fp8, v_scale, v_deq = fp8_kv(v, "bf16")
        out = filled(q.shape, q.dtype)
        scale = 1.0 / math.sqrt(d)

        run_attention_dense_torch(
            spec=spec,
            q=q,
            k=k_fp8,
            v=v_fp8,
            out=out,
            scale=scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        shifted = dense_reference(q, k_deq, v_deq, scale, bottom_right=True)[0]
        top_left = dense_reference(q, k_deq, v_deq, scale)[0]
        _assert_shifted_not_top_left(out, shifted, top_left, f"fp8 D{d} {sq}x{skv}")

    # The test above pins the fp8 bottom-right MATH on a hand-built default-tile
    # spec, which takes one KV segment. Dispatch does not select that spec: for a
    # moving diagonal it splits KV across segments and runs the combine kernel, a
    # different code path. This case runs what dispatch actually ships.
    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "d,sq,skv",
        [
            pytest.param(128, 512, 1024, id="d128-512x1024"),
            pytest.param(64, 512, 1024, id="d64-512x1024"),
        ],
    )
    def test_fp8_bottom_right_dispatched_split_matches_reference(self, d, sq, skv):
        spec = _spec(
            "bf16",
            d,
            4,
            1,
            persistent=False,
            batch=1,
            sq=sq,
            sk=skv,
            use_fp8=True,
            causal=True,
            causal_bottom_right=True,
            num_cus=256,
        )
        # Assert the split, so this never silently degrades into a second copy of
        # the single-segment case if dispatch stops splitting.
        assert spec.kv_segments > 1, (
            "dispatch no longer splits KV for fp8 bottom-right "
            f"(kv_segments={spec.kv_segments}); this case must cover the "
            "segment+combine path"
        )

        rng = np.random.default_rng(0)
        q = operand(rng, (1, sq, 4, d), "bf16")
        k = storage.round_to(rng.standard_normal((1, skv, 1, d)), "bf16")
        v = storage.round_to(rng.standard_normal((1, skv, 1, d)), "bf16")
        k_fp8, k_scale, k_deq = fp8_kv(k, "bf16")
        v_fp8, v_scale, v_deq = fp8_kv(v, "bf16")
        out = filled(q.shape, q.dtype)
        scale = 1.0 / math.sqrt(d)

        run_attention_dense_torch(
            spec=spec,
            q=q,
            k=k_fp8,
            v=v_fp8,
            out=out,
            scale=scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        shifted = dense_reference(q, k_deq, v_deq, scale, bottom_right=True)[0]
        top_left = dense_reference(q, k_deq, v_deq, scale)[0]
        _assert_shifted_not_top_left(
            out, shifted, top_left, f"fp8 split D{d} {sq}x{skv} segs={spec.kv_segments}"
        )


# KV-segment split on the default grid: (geometry, mask, diag_split, segments, sq,
# skv, head_size). mask "br" is bottom-right causal, "tl" top-left causal (early
# query blocks leave trailing segments empty or fully masked), "full" no mask
# (exercises the empty-segment discard).
_KV_SEGMENT_CASES = [
    pytest.param("bm128", "br", True, 4, 512, 8192, 128, id="bm128-br-s4-512x8192"),
    pytest.param("bm128", "br", False, 3, 256, 1088, 128, id="bm128-br-s3-nosplit"),
    pytest.param("default", "br", True, 8, 512, 4096, 128, id="bm256-br-s8"),
    pytest.param("bm128", "br", True, 4, 512, 2048, 64, id="bm128-br-s4-d64"),
    pytest.param("bm128", "tl", True, 4, 1024, 1024, 128, id="bm128-tl-s4"),
    pytest.param("bm128", "full", False, 4, 256, 320, 128, id="bm128-full-s4-empty"),
]


class TestDenseKvSegmentsNumeric:
    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "geometry,mask,diag_split,segments,sq,skv,d", _KV_SEGMENT_CASES
    )
    def test_kv_segments_match_reference(
        self, geometry, mask, diag_split, segments, sq, skv, d
    ):
        tile = DENSE_TILE_GEOMETRIES[geometry]
        spec = Gfx950AttentionDenseSpec(
            batch=1,
            seqlen_q=sq,
            seqlen_kv=skv,
            num_query_heads=8,
            num_kv_heads=2,
            head_size=d,
            causal=mask != "full",
            dtype="bf16",
            block_m=int(tile["block_m"]),
            block_n=int(tile["block_n"]),
            persistent=False,
            causal_bottom_right=mask == "br",
            causal_diag_split=diag_split,
            kv_segments=segments,
        )
        ok, why = supports_attention_dense(spec, arch="gfx950")
        assert ok, why
        rng = np.random.default_rng(0)
        q = operand(rng, (1, sq, 8, d), "bf16")
        k = operand(rng, (1, skv, 2, d), "bf16")
        v = operand(rng, (1, skv, 2, d), "bf16")
        out = filled(q.shape, q.dtype)
        scale = 1.0 / math.sqrt(d)
        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
        ref, _ = dense_reference(
            q, k, v, scale, causal=mask != "full", bottom_right=mask == "br"
        )
        assert_close(
            out,
            ref,
            _BOTTOM_RIGHT_TOL,
            f"{geometry} {mask} split={diag_split} S={segments} {sq}x{skv} D{d}",
        )


# The wide-DMA candidate admits every mask the persistent body implements, not
# only the causal no-sinks no-SWA shapes the cohort above routes to it.
_WIDE_DMA_MASKS = [
    # (name, causal, sliding_window, use_sinks)
    ("non_causal", False, 0, False),
    ("causal_sinks", True, 0, True),
    ("causal_swa", True, 128, False),
    ("non_causal_sinks", False, 0, True),
    ("causal_swa_sinks", True, 128, True),
]


class TestWideDmaFeatures:
    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", ("bf16", "fp16"))
    @pytest.mark.parametrize("block_m", (256, 128))
    @pytest.mark.parametrize("_name,causal,sliding_window,use_sinks", _WIDE_DMA_MASKS)
    def test_wide_dma_numeric(
        self, _name, causal, sliding_window, use_sinks, block_m, dtype
    ):
        from dispatch.attention import AttentionRequest, tuning_spec_with_knobs

        B, S, Hq, Hkv, D = 1, 512, 32, 8, 128
        spec = tuning_spec_with_knobs(
            AttentionRequest(
                batch=B,
                nhead_q=Hq,
                nhead_k=Hkv,
                seqlen_q=S,
                seqlen_k=S,
                hdim_q=D,
                hdim_v=D,
                arch="gfx950",
                mask_type=1 if causal else 0,
                dtype=dtype,
                use_sinks=use_sinks,
                sliding_window=sliding_window,
            ),
            "gfx950_dense_persist_widedma",
            {"block_m": block_m},
        ).kernel_spec
        assert spec.wide_lds_dma
        assert spec.block_m == block_m

        rng = np.random.default_rng(0)
        q = operand(rng, (B, S, Hq, D), dtype)
        k = operand(rng, (B, S, Hkv, D), dtype)
        v = operand(rng, (B, S, Hkv, D), dtype)
        out = filled(q.shape, q.dtype)
        sinks = operand(rng, (Hq,), dtype) if use_sinks else None
        scale = 1.0 / math.sqrt(D)
        run_attention_dense_torch(
            spec=spec, q=q, k=k, v=v, out=out, scale=scale, sinks=sinks
        )

        ref, _ = dense_reference(
            q, k, v, scale, causal=causal, window=sliding_window, sinks=sinks
        )
        assert_close(
            out, ref, attention_tolerance(dtype), f"widedma bm{block_m} {dtype} {_name}"
        )


def _scattered(sequence, page_ids, block_size=16):
    """One sequence's ``[S, H, D]`` host K or V cut into pages of ``block_size`` rows,
    page ``p`` stored at physical page ``page_ids[p]``."""
    pages = sequence.reshape(len(page_ids), block_size, *sequence.shape[1:])
    cache = np.empty_like(pages)
    cache[page_ids] = pages
    return cache


@requires_gfx950_gpu
@pytest.mark.gpu
class TestDenseLseNumeric:
    """Direct-runner LSE coverage; no dispatcher or hipDNN feature is implied."""

    @pytest.mark.parametrize("dtype", ["fp16", "bf16"])
    @pytest.mark.parametrize("head_size", [64, 128])
    @pytest.mark.parametrize(
        "heads", [(4, 4), (4, 1), (8, 2)], ids=["mha", "mqa", "gqa"]
    )
    @pytest.mark.parametrize(
        "persistent", [False, True], ids=["ordinary", "persistent"]
    )
    @pytest.mark.parametrize("geometry", ["default", "bm128"])
    def test_output_and_lse(self, dtype, head_size, heads, persistent, geometry):
        hq, hkv = heads
        spec = Gfx950AttentionDenseSpec(
            batch=2,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=hq,
            num_kv_heads=hkv,
            head_size=head_size,
            dtype=dtype,
            persistent=persistent,
            block_m=int(DENSE_TILE_GEOMETRIES[geometry]["block_m"]),
            emit_lse=True,
        )
        scale = 0.5
        rng = np.random.default_rng(11)
        q = operand(rng, (2, 512, hq, head_size), dtype)
        k = operand(rng, (2, 512, hkv, head_size), dtype)
        v = operand(rng, k.shape, k.dtype)
        out = filled(q.shape, q.dtype)
        lse = filled((2, hq, 512, 1), "fp32")
        returned = run_attention_dense_torch(
            spec=spec,
            q=q,
            k=k,
            v=v,
            out=out,
            lse=lse,
            scale=scale,
        )
        ref, lse_ref = dense_reference(q, k, v, scale, causal=True)
        assert returned is out
        assert_close(out, ref, attention_tolerance(dtype))
        assert_close(lse, lse_ref, LSE_TOLERANCE)
        off_out = filled(out.shape, out.dtype)
        run_attention_dense_torch(
            spec=replace(spec, emit_lse=False),
            q=q,
            k=k,
            v=v,
            out=off_out,
            scale=scale,
        )
        # Every row has a key. Empty-row corrections are enabled-path-only.
        assert same_values(out, off_out)

    @pytest.mark.parametrize("width", [1, 2])
    @pytest.mark.parametrize(
        "persistent", [False, True], ids=["ordinary", "persistent"]
    )
    def test_narrow_o_store(self, width, persistent):
        """BF16 store widths 1/2: keyless rows are zero, keyed rows match LSE-off."""
        spec = Gfx950AttentionDenseSpec(
            batch=1,
            seqlen_q=256,
            seqlen_kv=64,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=128,
            dtype="bf16",
            persistent=persistent,
            sliding_window=64,
            o_store_width=width,
            emit_lse=True,
        )
        scale = 0.5
        rng = np.random.default_rng(17)
        q = operand(rng, (1, 256, 4, 128), "bf16")
        k = operand(rng, (1, 64, 1, 128), q.dtype)
        v = operand(rng, k.shape, k.dtype)
        out = filled(q.shape, q.dtype)
        lse = filled((1, 4, 256, 1), "fp32")
        run_attention_dense_torch(
            spec=spec, q=q, k=k, v=v, out=out, lse=lse, scale=scale
        )
        off_out = filled(q.shape, q.dtype)
        run_attention_dense_torch(
            spec=replace(spec, emit_lse=False),
            q=q,
            k=k,
            v=v,
            out=off_out,
            scale=scale,
        )
        ref, lse_ref = dense_reference(q, k, v, scale, causal=True, window=64)
        # Rows 0..126 see at least one key; rows 127.. see none.
        o, o_off = values(out), values(off_out)
        assert_close(o[:, :127], ref[:, :127], attention_tolerance("bf16"))
        assert same_values(o[:, :127], o_off[:, :127])
        assert same_values(o[:, 127:], np.zeros_like(o[:, 127:]))
        lse_got = values(lse)
        finite = np.isfinite(lse_ref)
        assert_close(lse_got[finite], lse_ref[finite], LSE_TOLERANCE)
        assert np.array_equal(np.isneginf(lse_got), np.isneginf(lse_ref))

    @pytest.mark.parametrize("right", [0, -1], ids=["causal", "left-only"])
    def test_runtime_shape_keyless_rows_match_with_and_without_lse(self, right):
        """On the runtime-shape grid both paths zero keyless rows through the O
        store's integer bit mask -- the LSE path via its row_has_key, the LSE-off
        path via the pre-loop runtime predicate -- so O must be bit-identical with
        and without LSE, keyless rows included. Sq > Skv + W - 1, batch 2 so a
        keyless q-block cannot borrow the next batch's keys unnoticed."""
        window, sq, skv = 128, 1024, 512
        tile = DENSE_TILE_GEOMETRIES["default"]
        spec = Gfx950AttentionDenseSpec(
            batch=_BAND_BATCH,
            seqlen_q=sq,
            seqlen_kv=skv,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=128,
            causal=right >= 0,
            right_bound=max(right, 0),
            dtype="bf16",
            block_m=int(tile["block_m"]),
            block_n=int(tile["block_n"]),
            sliding_window=window,
            emit_lse=True,
        )
        assert spec.runtime_shape, "test setup error: not on the runtime-shape path"
        rng = np.random.default_rng(23)
        q = operand(rng, (_BAND_BATCH, sq, 4, 128), "bf16")
        k = operand(rng, (_BAND_BATCH, skv, 1, 128), q.dtype)
        v = operand(rng, k.shape, k.dtype)
        scale = 1.0 / math.sqrt(128)
        out = filled(q.shape, q.dtype)
        lse = filled((_BAND_BATCH, 4, sq, 1), "fp32")
        run_attention_dense_torch(
            spec=spec, q=q, k=k, v=v, out=out, lse=lse, scale=scale
        )
        off_out = filled(q.shape, q.dtype)
        run_attention_dense_torch(
            spec=replace(spec, emit_lse=False),
            q=q,
            k=k,
            v=v,
            out=off_out,
            scale=scale,
        )
        ref, empty = _band_reference(q, k, v, scale, window, right)
        assert empty.any(), "test setup error: the shape has no keyless rows"
        _check_band(out, ref, empty)
        _check_band(off_out, ref, empty)
        assert same_values(out, off_out)

    @pytest.mark.parametrize(
        "persistent", [False, True], ids=["ordinary", "persistent"]
    )
    @pytest.mark.parametrize("with_sink", [False, True], ids=["no-sink", "sink"])
    def test_empty_window_rows(self, persistent, with_sink):
        spec = Gfx950AttentionDenseSpec(
            batch=1,
            seqlen_q=256,
            seqlen_kv=64,
            num_query_heads=3,
            num_kv_heads=1,
            head_size=128,
            dtype="bf16",
            persistent=persistent,
            sliding_window=64,
            use_sinks=with_sink,
            emit_lse=True,
        )
        q = filled((1, 256, 3, 128), "bf16", 0.0)
        k = filled((1, 64, 1, 128), "bf16", 0.0)
        v = filled(k.shape, "bf16", 1.0)
        out = filled(q.shape, q.dtype)
        lse = filled((1, 3, 256, 1), "fp32")
        sinks = upload([-2.0, 0.0, 2.0], "bf16") if with_sink else None
        run_attention_dense_torch(
            spec=spec,
            q=q,
            k=k,
            v=v,
            out=out,
            lse=lse,
            scale=0.5,
            sinks=sinks,
        )
        analytic_lse = window64_analytic_lse(3, sinks)
        ref, lse_ref = dense_reference(
            q,
            k,
            v,
            0.5,
            causal=True,
            window=64,
            sinks=sinks,
        )
        lse_got = values(lse)
        finite = np.isfinite(analytic_lse)
        assert max_abs_error(lse_got[finite], analytic_lse[finite]) < 1e-5
        assert max_abs_error(lse_ref[finite], analytic_lse[finite]) < 1e-5
        assert np.array_equal(np.isneginf(lse_got), np.isneginf(analytic_lse))
        assert_close(out, ref, attention_tolerance("bf16"))
        o = values(out)
        assert same_values(o[:, 127:], np.zeros_like(o[:, 127:]))
        if with_sink:
            expected = np.broadcast_to(
                values(sinks)[None, :, None, None], (1, 3, 129, 1)
            )
            assert max_abs_error(lse_got[:, :, 127:], values(expected)) < 1e-5
        else:
            assert np.isneginf(lse_got[:, :, 127:]).all()

    @pytest.mark.parametrize("dtype", ["fp16", "bf16"])
    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {"ragged": True, "seqlen_q": 273, "seqlen_kv": 273}, id="ragged"
            ),
            pytest.param(
                {"ragged": True, "seqlen_q": 273, "seqlen_kv": 273, "persistent": True},
                id="ragged-persistent",
            ),
            pytest.param({"causal": False}, id="noncausal"),
            pytest.param(
                {"causal": False, "persistent": True}, id="noncausal-persistent"
            ),
            pytest.param({"sliding_window": 128}, id="window"),
            pytest.param(
                {"sliding_window": 128, "persistent": True}, id="window-persistent"
            ),
            pytest.param(
                {
                    "causal_bottom_right": True,
                    "ragged": True,
                    "seqlen_q": 197,
                    "seqlen_kv": 400,
                },
                id="shifted-ragged",
            ),
            pytest.param(
                {"causal_bottom_right": True, "seqlen_q": 256},
                id="shifted-aligned",
            ),
            pytest.param({"use_sinks": True}, id="sinks"),
            pytest.param(
                {"use_sinks": True, "persistent": True}, id="sinks-persistent"
            ),
            pytest.param(
                {"persistent": True, "persist_decode": "qb_major", "num_persistent": 4},
                id="qb-major",
            ),
            pytest.param(
                {
                    "persistent": True,
                    "persist_decode": "hkv_major",
                    "num_persistent": 4,
                },
                id="hkv-major",
            ),
            pytest.param(
                {"persistent": True, "persist_decode": "gqa_pair", "num_persistent": 4},
                id="gqa-pair",
            ),
            pytest.param(
                {
                    "persistent": True,
                    "persist_decode": "gqa_pair_2phase",
                    "num_persistent": 8,
                },
                id="gqa-pair-2phase",
            ),
            pytest.param(
                {
                    "persistent": True,
                    "persist_decode": "qb_major",
                    "num_persistent": 4,
                    "interleave": True,
                },
                id="interleaved",
            ),
            pytest.param(
                {
                    "persistent": True,
                    "persist_decode": "qb_major",
                    "num_persistent": 4,
                    "wide_lds_dma": True,
                },
                id="wide-dma",
            ),
            pytest.param(
                {
                    "batch": 1,
                    "paged": True,
                    "block_size": 16,
                    "num_kv_blocks": 32,
                    "sliding_window": 128,
                },
                id="reverse-paged",
            ),
        ],
    )
    def test_legal_modes_and_tail_guards(self, dtype, case):
        # One launch/check body, with separate pytest nodes for each legal mode.
        options = dict(
            batch=2,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=128,
            dtype=dtype,
            emit_lse=True,
        )
        options.update(case)
        spec = Gfx950AttentionDenseSpec(**options)
        batch, sq, skv = spec.batch, spec.seqlen_q, spec.seqlen_kv
        rng = np.random.default_rng(19)
        q = operand(rng, (batch, sq, 4, 128), dtype)
        k = operand(rng, (batch, skv, 1, 128), dtype)
        v = operand(rng, k.shape, k.dtype)
        # Each output sits at the start of a buffer whose guard rows must survive.
        out = guarded(q.shape, dtype, 123.0)
        lse = guarded((batch, 4, sq, 1), "fp32", 12345.0)
        sinks = upload(np.linspace(-1, 1, 4), dtype) if spec.use_sinks else None
        launch_k, launch_v = k, v
        scale = 0.5
        metadata = {}
        if spec.paged:
            # Page p of the sequence lives at physical page 31 - p.
            page_ids = np.arange(31, -1, -1, dtype=np.int32)
            launch_k, launch_v = (
                DeviceArray(_scattered(host_values(src)[0], page_ids), dtype)
                for src in (k, v)
            )
            metadata = {
                "block_tables": DeviceArray(page_ids.reshape(1, 32)),
                "kv_lens": DeviceArray(np.array([skv], dtype=np.int32)),
            }
        returned = run_attention_dense_torch(
            spec=spec,
            q=q,
            k=launch_k,
            v=launch_v,
            out=out,
            lse=lse,
            scale=scale,
            sinks=sinks,
            **metadata,
        )
        ref, lse_ref = dense_reference(
            q,
            k,
            v,
            scale,
            causal=spec.causal,
            bottom_right=spec.causal_bottom_right,
            window=spec.sliding_window,
            sinks=sinks,
        )
        assert returned is out
        assert_close(out, ref, attention_tolerance(dtype))
        assert_close(lse, lse_ref, LSE_TOLERANCE)
        assert guard_intact(lse, 12345.0), "LSE written outside it"
        assert guard_intact(out, 123.0), "O written outside it"
        off_out = filled(out.shape, out.dtype)
        run_attention_dense_torch(
            spec=replace(spec, emit_lse=False),
            q=q,
            k=launch_k,
            v=launch_v,
            out=off_out,
            scale=scale,
            sinks=sinks,
            **metadata,
        )
        assert same_values(out, off_out)

    @pytest.mark.parametrize("dtype", ["fp16", "bf16"])
    @pytest.mark.parametrize(
        "zero_q", [False, True], ids=["different-lengths", "zero-q"]
    )
    def test_packed_dense_lse(self, dtype, zero_q):
        q_offsets = [0, 0, 512] if zero_q else [0, 256, 768]
        k_offsets = [0, 128, 384]
        spec = Gfx950AttentionDenseSpec(
            batch=2,
            seqlen_q=512,
            seqlen_kv=256,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=128,
            dtype=dtype,
            varlen=True,
            emit_lse=True,
        )
        rng = np.random.default_rng(23)
        q = operand(rng, (q_offsets[-1], 4, 128), dtype)
        k = operand(rng, (k_offsets[-1], 1, 128), dtype)
        v = operand(rng, k.shape, k.dtype)
        out = filled(q.shape, q.dtype)
        lse = guarded((q_offsets[-1], 4, 1), "fp32", 12345.0)
        scale = 0.5
        cu_q = DeviceArray(np.array(q_offsets, dtype=np.int32))
        cu_k = DeviceArray(np.array(k_offsets, dtype=np.int32))
        returned = run_attention_dense_torch(
            spec=spec,
            q=q,
            k=k,
            v=v,
            out=out,
            lse=lse,
            scale=scale,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_k,
        )
        # Top-left causal within each sequence.
        expected_out, expected_lse = dense_reference(
            q, k, v, scale, q_offsets=q_offsets, kv_offsets=k_offsets
        )
        assert returned is out
        assert_close(out, expected_out, attention_tolerance(dtype))
        assert_close(lse, expected_lse, LSE_TOLERANCE)
        assert guard_intact(lse, 12345.0), "LSE written outside it"
        # Packed varlen is the one mode where LSE changes the O store guard
        # (stores only rows that exist), so pin the bits against LSE off.
        off_out = filled(q.shape, q.dtype)
        run_attention_dense_torch(
            spec=replace(spec, emit_lse=False),
            q=q,
            k=k,
            v=v,
            out=off_out,
            scale=scale,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_k,
        )
        assert same_bits(host_values(out), host_values(off_out))

    @pytest.mark.parametrize(
        "shape",
        [("fp16", 128, 2), ("bf16", 64, 1)],
        ids=["fp16-d128-kv2", "bf16-d64-kv1"],
    )
    @pytest.mark.parametrize(
        "window,q_lens,kv_lens,emit_lse,keyless",
        [
            (0, (256, 256), (192, 64), False, False),
            (0, (256, 256), (192, 64), True, False),
            (256, (512, 256), (512, 64), False, False),
            (256, (512, 256), (512, 64), True, False),
            (64, (256, 512), (256, 64), True, True),
        ],
        ids=["plain", "plain-lse", "window", "window-lse", "window-keyless-lse"],
    )
    def test_packed_dense_short_last_sequence(
        self, shape, window, q_lens, kv_lens, emit_lse, keyless
    ):
        """Packed varlen whose last sequence has far fewer keys than ``seqlen_kv``.

        The kernel bounds each workgroup's K/V buffers by the end of its own
        sequence, so the loads past it (the last sequence's one-tile-ahead prefetch,
        and a sliding-window start tile past the sequence's keys) are dropped by the
        hardware instead of reading past the allocation. Every row that has keys must
        still match the fp32 reference: a bound too tight by a row or a tile zeroes
        consumed K/V and fails, while the over-read itself is not observable in the
        output (a too-loose bound passes unless the read faults). In
        ``window-keyless-lse`` the last sequence has a q-block whose window starts
        past its keys; its rows are 0 with LSE -inf. Without LSE the launcher rejects
        keyless rows, so only the LSE case has them.
        """
        dtype, head_size, kv_heads = shape
        q_offsets = [0, q_lens[0], q_lens[0] + q_lens[1]]
        k_offsets = [0, kv_lens[0], kv_lens[0] + kv_lens[1]]
        spec = Gfx950AttentionDenseSpec(
            batch=2,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=4,
            num_kv_heads=kv_heads,
            head_size=head_size,
            dtype=dtype,
            varlen=True,
            emit_lse=emit_lse,
            sliding_window=window,
        )
        rng = np.random.default_rng(31)
        q = operand(rng, (q_offsets[-1], 4, head_size), dtype)
        k = operand(rng, (k_offsets[-1], kv_heads, head_size), dtype)
        v = operand(rng, k.shape, k.dtype)
        out = filled(q.shape, q.dtype)
        lse_out = filled((q_offsets[-1], 4, 1), "fp32")
        scale = 0.5
        run_attention_dense_torch(
            spec=spec,
            q=q,
            k=k,
            v=v,
            out=out,
            lse=lse_out if emit_lse else None,
            scale=scale,
            cu_seqlens_q=DeviceArray(np.array(q_offsets, dtype=np.int32)),
            cu_seqlens_kv=DeviceArray(np.array(k_offsets, dtype=np.int32)),
        )
        expected_out, expected_lse = dense_reference(
            q, k, v, scale, window=window, q_offsets=q_offsets, kv_offsets=k_offsets
        )
        has_key = np.isfinite(expected_lse)
        # The case must have the rows it claims: the launcher rejects keyless rows
        # without LSE, and the keyless case is only worth running with some.
        assert bool((~has_key).any()) == keyless
        tol = attention_tolerance(dtype)
        # NaN anywhere in ``out`` (a row never stored) fails the comparison.
        assert_close(out, expected_out, tol)
        if emit_lse:
            lse_got = values(lse_out)
            assert np.array_equal(np.isfinite(lse_got), has_key)
            assert_close(lse_got[has_key], expected_lse[has_key], LSE_TOLERANCE)
            assert bool((lse_got[~has_key] == float("-inf")).all())
            keyless_out = values(out)[~has_key.squeeze(-1)]
            assert same_values(keyless_out, np.zeros_like(keyless_out))

    @pytest.mark.parametrize(
        "failure",
        [
            "q_tail",
            "kv_tail",
            "zero_kv",
            "decreasing",
            "first_offset",
            "terminal_offset",
            "q_above_max",
            "kv_above_max",
            "q_offset_dtype",
            "kv_offset_dtype",
            "q_offset_count",
            "kv_offset_count",
            "offset_layout",
            "offset_device",
            "missing_offsets",
            "kv_extent",
            "out_extent",
            "packed_shape",
            "packed_layout",
        ],
    )
    def test_packed_lse_rejections(self, failure):
        q_offsets, k_offsets = [0, 256, 768], [0, 128, 384]
        if failure == "q_tail":
            q_offsets = [0, 257, 768]
        elif failure == "kv_tail":
            k_offsets = [0, 129, 384]
        elif failure == "zero_kv":
            # The other KV length stays aligned and within the maximum.
            k_offsets = [0, 0, 256]
        elif failure == "decreasing":
            q_offsets = [0, 512, 256]
        elif failure == "first_offset":
            q_offsets = [1, 256, 768]
        elif failure == "terminal_offset":
            q_offsets = [0, 256, 512]
        elif failure == "q_above_max":
            q_offsets = [0, 0, 768]
        elif failure == "kv_above_max":
            k_offsets = [0, 64, 384]
        spec = Gfx950AttentionDenseSpec(
            batch=2,
            seqlen_q=512,
            seqlen_kv=256,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=128,
            dtype="bf16",
            varlen=True,
            emit_lse=True,
        )
        tq = 256 if failure == "decreasing" else 768
        tk = 256 if failure == "zero_kv" else 384
        q = DeviceArray.zeros((tq, 4, 128), "bf16")
        k = DeviceArray.zeros((tk, 1, 128), "bf16")
        v = DeviceArray.zeros(k.shape, "bf16")
        out = filled(q.shape, q.dtype)
        lse = DeviceArray.zeros((tq, 4, 1), "fp32")
        cu_q = DeviceArray(np.array(q_offsets, dtype=np.int32))
        cu_k = DeviceArray(np.array(k_offsets, dtype=np.int32))
        if failure == "q_offset_dtype":
            cu_q = DeviceArray(np.array(q_offsets, dtype=np.int64))
        elif failure == "kv_offset_dtype":
            cu_k = DeviceArray(np.array(k_offsets, dtype=np.int64))
        elif failure == "q_offset_count":
            cu_q = DeviceArray(np.array(q_offsets[:-1], dtype=np.int32))
        elif failure == "kv_offset_count":
            cu_k = DeviceArray(np.array(k_offsets[:-1], dtype=np.int32))
        elif failure == "offset_layout":
            # Every other element of [0, 0, 256, 0, 768, 0]: right values, strided.
            cu_q = HostArray((3,), "i32", values=q_offsets, strides=(2,))
        elif failure == "offset_device":
            cu_k = HostArray((3,), "i32", values=k_offsets, device=None)
        elif failure == "missing_offsets":
            cu_q = None
        elif failure == "kv_extent":
            v = DeviceArray.zeros((tk - 1, 1, 128), "bf16")
        elif failure == "out_extent":
            out = filled((tq - 1, 4, 128), "bf16")
        elif failure == "packed_shape":
            k = DeviceArray.zeros((tk, 2, 64), "bf16")
        elif failure == "packed_layout":
            # The first half of every row of a [tk, 1, 256] buffer: not packed.
            k = DeviceArray(np.zeros((tk, 1, 256), np.uint16), "bf16", shape=k.shape)
        with pytest.raises(ValueError):
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                lse=lse,
                scale=0.5,
                cu_seqlens_q=cu_q,
                cu_seqlens_kv=cu_k,
            )

    @pytest.mark.parametrize(
        "failure",
        [
            "missing",
            "unexpected",
            "dtype",
            "shape",
            "layout",
            "device",
            "overlap_q",
            "overlap_k",
            "overlap_v",
            "overlap_out",
            "overlap_sink",
            "overlap_metadata",
            "companion",
        ],
    )
    def test_lse_buffer_errors(self, failure):
        spec = Gfx950AttentionDenseSpec(
            batch=2,
            seqlen_q=256,
            seqlen_kv=256,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=128,
            dtype="bf16",
            emit_lse=True,
        )
        q = DeviceArray.zeros((2, 256, 4, 128), "bf16")
        k = DeviceArray.zeros((2, 256, 1, 128), "bf16")
        v = DeviceArray.zeros(k.shape, "bf16")
        out = filled(q.shape, q.dtype)
        lse = DeviceArray.zeros((2, 4, 256, 1), "fp32")
        sinks, metadata = None, {}
        if failure == "missing":
            lse = None
        elif failure == "unexpected":
            spec = replace(spec, emit_lse=False)
        elif failure == "dtype":
            lse = DeviceArray.zeros(lse.shape, "bf16")
        elif failure == "shape":
            lse = DeviceArray.zeros(lse.shape[:-1], "fp32")
        elif failure == "layout":
            # The first column of a [2, 4, 256, 2] buffer: strided, not packed.
            lse = DeviceArray(np.zeros((2, 4, 256, 2), np.float32), shape=lse.shape)
        elif failure == "device":
            lse = HostArray(lse.shape, "fp32", device=None)
        elif failure in {"overlap_q", "overlap_k", "overlap_v", "overlap_out"}:
            array = {
                "overlap_q": q,
                "overlap_k": k,
                "overlap_v": v,
                "overlap_out": out,
            }[failure]
            # A second view of the real input/output allocation, not a copy.
            lse = Alias(array, lse.shape, "fp32")
        elif failure == "overlap_sink":
            spec = replace(spec, use_sinks=True)
            sinks = Alias(lse, (4,), "bf16")
        elif failure == "overlap_metadata":
            spec = replace(spec, varlen=True)
            q, k, v, out = (
                DeviceArray.zeros((512, *array.shape[2:]), "bf16")
                for array in (q, k, v, out)
            )
            lse = DeviceArray.zeros((512, 4, 1), "fp32")
            offsets = [0, 256, 512]
            metadata = {
                "cu_seqlens_q": Alias(lse, (3,), "i32", read_values=offsets),
                "cu_seqlens_kv": DeviceArray(np.array(offsets, dtype=np.int32)),
            }
        elif failure == "companion":
            k = HostArray(k.shape, "bf16", device=None)
        with pytest.raises(ValueError):
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                lse=lse,
                scale=0.5,
                sinks=sinks,
                **metadata,
            )

    @pytest.mark.parametrize(
        "spec_id",
        ["gfx950_dense_grid", "gfx950_dense_persist", "gfx950_dense_persist_widedma"],
    )
    def test_dispatched_lse(self, spec_id):
        """Request -> dispatch -> bind -> launch fills a caller-owned LSE buffer."""
        from dispatch.attention import (
            AttentionRequest,
            attention_execution_candidates,
            dispatch_attention,
        )

        candidate = next(
            c for c in attention_execution_candidates() if c.spec_id == spec_id
        )
        result = dispatch_attention(
            AttentionRequest(
                batch=2,
                nhead_q=8,
                nhead_k=2,
                seqlen_q=512,
                seqlen_k=512,
                hdim_q=128,
                hdim_v=128,
                arch="gfx950",
                mask_type=1,
                dtype="bf16",
                algorithm=candidate.algorithm,
                spec_id=spec_id,
                emit_lse=True,
            )
        )
        scale = 0.5
        rng = np.random.default_rng(23)
        q = operand(rng, (2, 512, 8, 128), "bf16")
        k = operand(rng, (2, 512, 2, 128), q.dtype)
        v = operand(rng, k.shape, k.dtype)
        out = filled(q.shape, q.dtype)
        lse = filled((2, 8, 512, 1), "fp32")
        tensors = {"q": q, "k": k, "v": v, "out": out}
        with pytest.raises(ValueError):
            result.bind_torch(tensors, scale=scale).launch()
        result.bind_torch({**tensors, "lse": lse}, scale=scale).launch()
        ref, lse_ref = dense_reference(q, k, v, scale, causal=True)
        assert_close(out, ref, attention_tolerance("bf16"))
        assert_close(lse, lse_ref, LSE_TOLERANCE)

    def test_lse_runtime_shape_reuse(self):
        from kernels.common.attention_dense_spec import attention_dense_cache_key
        from kernels.gfx950.attention_dense import _DENSE_LAUNCHER_CACHE

        specs = [
            Gfx950AttentionDenseSpec(
                batch=batch,
                seqlen_q=sq,
                seqlen_kv=512,
                num_query_heads=4,
                num_kv_heads=1,
                head_size=128,
                dtype="bf16",
                emit_lse=True,
            )
            for batch, sq in [(1, 512), (2, 1024)]
        ]
        enabled_keys = [
            attention_dense_cache_key(spec, arch="gfx950") for spec in specs
        ]
        off_specs = [replace(spec, emit_lse=False) for spec in specs]
        disabled_keys = [
            attention_dense_cache_key(spec, arch="gfx950") for spec in off_specs
        ]
        # Own every potentially affected slot, even if a cache-key regression
        # unexpectedly splits shape identity. Always restore prior entries.
        owned_keys = set(enabled_keys + disabled_keys)
        old_entries = {key: _DENSE_LAUNCHER_CACHE.pop(key, None) for key in owned_keys}
        launchers = []
        scale = 0.5
        try:
            for index, spec in enumerate(specs):
                rng = np.random.default_rng(29 + index)
                batch, sq = spec.batch, spec.seqlen_q
                q = operand(rng, (batch, sq, 4, 128), "bf16")
                k = operand(rng, (batch, 512, 1, 128), q.dtype)
                v = operand(rng, k.shape, k.dtype)
                out = filled(q.shape, q.dtype)
                lse = filled((batch, 4, sq, 1), "fp32")
                run_attention_dense_torch(
                    spec=spec,
                    q=q,
                    k=k,
                    v=v,
                    out=out,
                    lse=lse,
                    scale=scale,
                )
                ref, lse_ref = dense_reference(q, k, v, scale, causal=True)
                assert_close(out, ref, attention_tolerance("bf16"))
                assert_close(lse, lse_ref, LSE_TOLERANCE)
                launchers.append(_launcher_for(spec))
            assert launchers[0] is not None
            assert launchers[0] is launchers[1]
            off_out = filled(out.shape, out.dtype)
            run_attention_dense_torch(
                spec=off_specs[-1],
                q=q,
                k=k,
                v=v,
                out=off_out,
                scale=scale,
            )
            assert same_values(out, off_out)
            assert _launcher_for(off_specs[-1]) is not launchers[0]
        finally:
            for key, entry in old_entries.items():
                _DENSE_LAUNCHER_CACHE.pop(key, None)
                if entry is not None:
                    _DENSE_LAUNCHER_CACHE[key] = entry


class TestDensePagedNumeric:
    """Paged K/V (block_tables indirection) on the non-persistent dense body."""

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", ["fp16", "bf16"])
    @pytest.mark.parametrize("block_size", [16, 32, 64])
    @pytest.mark.parametrize("sq,window", [(1024, 256), (2048, 1024), (1024, 0)])
    @pytest.mark.parametrize("d", [128, 64])
    def test_paged_matches_dense_bit_exact(self, dtype, block_size, sq, window, d):
        """A paged cache must give the same bits as the dense K/V it was scattered
        from, and match the fp32 causal (sliding-window) reference.

        The pages land at randomly permuted physical ids in a cache twice the
        needed size, and every unused page is NaN. A wrong page index (the
        per-wave ``tile_key // block_size`` lookup) or a wrong in-page row (the
        per-row ``tile_key % block_size``) therefore reads either a NaN page or
        the wrong keys, rather than a near miss the tolerance could absorb.
        block_size 16/32/64 covers 4, 2 and 1 pages per 64-key tile.
        """
        self._check_paged(dtype, block_size, sq, sq, window, d)

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "sq,sk,window",
        [
            pytest.param(512, 2048, 0, id="top-left-512x2048"),
            pytest.param(2048, 512, 0, id="top-left-2048x512"),
            pytest.param(1088, 1088, 0, id="ragged-q-1088"),
            pytest.param(1024, 512, 128, id="swa-empty-rows-1024x512"),
        ],
    )
    def test_paged_cross_length_matches_dense(self, sq, sk, window):
        """Top-left causal with ``seqlen_q != seqlen_kv``, a ragged query tail and
        rows whose window holds no key."""
        self._check_paged("fp16", 16, sq, sk, window, 128)

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("spec_id", ["gfx950_dense_grid", "gfx950_dense_qbminor"])
    def test_dispatched_paged(self, spec_id):
        """Request -> dispatch -> bind -> launch, with the block count taken from the
        cache at bind."""
        window = 1024
        q, k, v, out = run_dispatched_paged("gfx950", spec_id, window=window)
        assert np.isfinite(values(out)).all()
        scale = 1.0 / math.sqrt(q.shape[-1])
        ref = dense_reference(q, k, v, scale, window=window)[0]
        assert_close(out, ref, attention_tolerance("fp16"))

    @staticmethod
    def _check_paged(dtype, block_size, sq, sk, window, d):
        hq, hkv, B = 16, 4, 1
        scale = 1.0 / math.sqrt(d)
        rng = np.random.default_rng(0)

        q = operand(rng, (B, sq, hq, d), dtype)
        k = operand(rng, (B, sk, hkv, d), dtype)
        v = operand(rng, (B, sk, hkv, d), dtype)

        k_cache, v_cache, block_tables, kv_lens = paged_cache(rng, k, v, block_size)
        num_kv_blocks = k_cache.shape[0]

        # The shipped paged spec (its tile can differ from the contiguous one's),
        # and its contiguous twin on the same tile.
        shipped = _spec(
            dtype,
            d,
            hq,
            hkv,
            False,
            sq=sq,
            sk=sk,
            sliding_window=window,
            kv_block_size=block_size,
        )
        paged_spec = dataclasses.replace(shipped, num_kv_blocks=num_kv_blocks)
        dense_spec = dataclasses.replace(
            shipped, paged=False, block_size=0, num_kv_blocks=0
        )

        dense_out = filled(q.shape, q.dtype)
        run_attention_dense_torch(
            spec=dense_spec, q=q, k=k, v=v, out=dense_out, scale=scale
        )
        paged_out = filled(q.shape, q.dtype)
        run_attention_dense_torch(
            spec=paged_spec,
            q=q,
            k=k_cache,
            v=v_cache,
            out=paged_out,
            scale=scale,
            block_tables=block_tables,
            kv_lens=kv_lens,
        )

        name = f"{dtype} D{d} pgd{block_size} sq{sq} sk{sk} swa{window}"
        paged = values(paged_out)
        assert np.isfinite(paged).all(), f"{name}: read a NaN (unused) page"
        assert same_values(paged_out, dense_out), f"{name}: paged != dense bits"
        ref, lse_ref = dense_reference(q, k, v, scale, window=window)
        # A row whose window holds no key (an infinite reference LSE) must output
        # zero.
        live = np.isfinite(lse_ref[0, 0, :, 0])
        assert_close(paged[:, live], ref[:, live], attention_tolerance(dtype), name)
        if not live.all():
            worst = np.abs(paged[:, ~live]).max()
            assert worst == 0.0, f"{name}: {int((~live).sum())} keyless rows: {worst}"


# Cross-length top-left (unshifted diagonal): sq < sk and sq > sk, ragged and
# aligned. For sq > sk the ragged padded keys sit below the diagonal of every
# row with q >= Skv, so correctness depends on the key-pad mask in the causal loops.
_TOP_LEFT_CASES = [
    pytest.param(1000, 1050, 1, True, False, id="ragged-1000x1050"),
    pytest.param(1050, 1000, 1, True, False, id="ragged-1050x1000"),
    pytest.param(197, 400, 1, True, False, id="ragged-197x400"),
    pytest.param(400, 197, 1, True, False, id="ragged-400x197"),
    pytest.param(1234, 300, 1, True, False, id="ragged-1234x300"),
    pytest.param(1050, 1000, 2, True, False, id="ragged-1050x1000-batch2"),
    pytest.param(1050, 1000, 1, True, True, id="ragged-1050x1000-sinks"),
    pytest.param(512, 1024, 1, False, False, id="aligned-512x1024"),
    pytest.param(1024, 512, 1, False, False, id="aligned-1024x512"),
]


class TestDenseTopLeftCrossLengthNumeric:
    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("persistent", [False, True], ids=["default", "persist"])
    @pytest.mark.parametrize("sq,skv,batch,ragged,use_sinks", _TOP_LEFT_CASES)
    def test_jit_matches_top_left(self, sq, skv, batch, ragged, use_sinks, persistent):
        tile = DENSE_TILE_GEOMETRIES["default"]
        spec = Gfx950AttentionDenseSpec(
            batch=batch,
            seqlen_q=sq,
            seqlen_kv=skv,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=128,
            causal=True,
            dtype="bf16",
            block_m=int(tile["block_m"]),
            block_n=int(tile["block_n"]),
            ragged=ragged,
            persistent=persistent,
            use_sinks=use_sinks,
        )

        rng = np.random.default_rng(0)
        q = operand(rng, (batch, sq, 4, 128), "bf16")
        k = operand(rng, (batch, skv, 1, 128), "bf16")
        v = operand(rng, (batch, skv, 1, 128), "bf16")
        out = filled(q.shape, q.dtype)
        sinks = operand(rng, (4,), "bf16") if use_sinks else None
        scale = 1.0 / math.sqrt(spec.head_size)

        run_attention_dense_torch(
            spec=spec, q=q, k=k, v=v, out=out, scale=scale, sinks=sinks
        )

        ref = dense_reference(q, k, v, scale, sinks=sinks)[0]
        assert not np.isnan(values(out)).any(), "NaN in output"
        assert_close(
            out,
            ref,
            _BOTTOM_RIGHT_TOL,
            f"B{batch} {sq}x{skv} {'ragged' if ragged else 'aligned'} "
            f"{'persist' if persistent else 'default'}"
            f"{'+sinks' if use_sinks else ''}",
        )


# The cuDNN band ``(left = sliding_window, right = right_bound)`` on the top-left
# diagonal: key ``k`` is visible to query ``q`` iff ``q - k < left`` (when set) and
# ``k <= q + right`` (right = -1 is unbounded, 0 causal, R > 0 lookahead). A row whose
# band holds no key must output zeros (the contract shared with the unified kernels and
# PyTorch SDPA), never stale data. Before this matrix existed every dense window test
# used ``seqlen_q == seqlen_kv`` and a causal mask, so neither non-causal bands nor the
# empty-row case (``seqlen_q > seqlen_kv + window``) were exercised.
#   (sq, skv, window, right, persistent, ragged)
_BAND_CASES = [
    # causal window x cross-length, with empty rows
    pytest.param(1024, 512, 128, 0, False, False, id="causal-w128-1024x512-empty"),
    pytest.param(1024, 256, 128, 0, False, False, id="causal-w128-1024x256-empty"),
    pytest.param(
        1024, 512, 128, 0, True, False, id="persist-causal-w128-1024x512-empty"
    ),
    pytest.param(512, 1024, 128, 0, False, False, id="causal-w128-512x1024"),
    pytest.param(1024, 1024, 128, 0, False, False, id="causal-w128-1024x1024"),
    # non-causal (left-only) windows
    pytest.param(1024, 1024, 128, -1, False, False, id="left-only-w128-1024x1024"),
    pytest.param(512, 1024, 128, -1, False, False, id="left-only-w128-512x1024"),
    pytest.param(1024, 512, 128, -1, False, False, id="left-only-w128-1024x512-empty"),
    pytest.param(1024, 512, 128, -1, True, False, id="persist-left-only-w128-1024x512"),
    pytest.param(
        1024, 1024, 256, -1, True, False, id="persist-left-only-w256-1024x1024"
    ),
    # two-sided windows and lookahead
    pytest.param(1024, 1024, 128, 64, False, False, id="two-sided-w128-r64-1024x1024"),
    pytest.param(512, 1024, 192, 128, False, False, id="two-sided-w192-r128-512x1024"),
    pytest.param(
        1024, 512, 128, 64, True, False, id="persist-two-sided-w128-r64-1024x512"
    ),
    pytest.param(1024, 1024, 0, 100, False, False, id="lookahead-r100-1024x1024"),
    pytest.param(
        1024, 1024, 0, 100, True, False, id="persist-lookahead-r100-1024x1024"
    ),
    pytest.param(512, 1024, 0, 7, False, False, id="lookahead-r7-512x1024"),
    pytest.param(1024, 512, 0, 300, False, False, id="lookahead-r300-1024x512"),
    # full attention, aligned and ragged cross-length (cross-attention)
    pytest.param(512, 1024, 0, -1, False, False, id="full-512x1024"),
    pytest.param(1000, 1050, 0, -1, False, True, id="full-ragged-1000x1050"),
    pytest.param(1050, 1000, 0, -1, False, True, id="full-ragged-1050x1000"),
    pytest.param(300, 1234, 0, -1, True, True, id="persist-full-ragged-300x1234"),
    # lookahead on ragged cross-length (key padding can sit inside q + right)
    pytest.param(1000, 1050, 0, 64, False, True, id="lookahead-ragged-1000x1050"),
    pytest.param(1050, 1000, 0, 64, False, True, id="lookahead-ragged-1050x1000"),
]


# batch > 1 matters: a q-block that sits entirely past the keys must not read the next
# batch's tokens, and with batch == 1 the over-read lands in zero-filled (out-of-range)
# memory, which hides the bug.
_BAND_BATCH = 2


def _band_reference(q, k, v, scale, window, right):
    """The float64 O of the top-left band ``(window, right)``, and which query rows
    see no key."""
    ref, lse = dense_reference(
        q, k, v, scale, causal=right >= 0, right_bound=max(right, 0), window=window
    )
    return ref, ~np.isfinite(lse[0, 0, :, 0])


def _check_band(out, ref, empty):
    o = values(out)
    assert not np.isnan(o).any(), "NaN in output"
    assert_close(o[:, ~empty], ref[:, ~empty], _BOTTOM_RIGHT_TOL)
    if empty.any():
        worst = np.abs(o[:, empty]).max()
        assert worst == 0.0, f"{int(empty.sum())} empty rows not zero: {worst}"


class TestDenseBandNumeric:
    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("sq,skv,window,right,persistent,ragged", _BAND_CASES)
    def test_jit_matches_band(self, sq, skv, window, right, persistent, ragged):
        tile = DENSE_TILE_GEOMETRIES["default"]
        spec = Gfx950AttentionDenseSpec(
            batch=_BAND_BATCH,
            seqlen_q=sq,
            seqlen_kv=skv,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=128,
            causal=right >= 0,
            right_bound=max(right, 0),
            dtype="bf16",
            block_m=int(tile["block_m"]),
            block_n=int(tile["block_n"]),
            sliding_window=window,
            persistent=persistent,
            ragged=ragged,
        )
        rng = np.random.default_rng(0)
        q = operand(rng, (_BAND_BATCH, sq, 4, 128), "bf16")
        k = operand(rng, (_BAND_BATCH, skv, 1, 128), "bf16")
        v = operand(rng, (_BAND_BATCH, skv, 1, 128), "bf16")
        out = filled(q.shape, q.dtype)
        scale = 1.0 / math.sqrt(128)
        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
        _check_band(out, *_band_reference(q, k, v, scale, window, right))

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("sq,skv,window,right,persistent,ragged", _BAND_CASES)
    def test_dispatched_request_matches_band(
        self, sq, skv, window, right, persistent, ragged
    ):
        """The same bands through ``AttentionRequest`` -> dispatch -> dense spec."""
        from dispatch.attention import AttentionRequest, attention_tuning_spec

        req = AttentionRequest(
            batch=_BAND_BATCH,
            nhead_q=4,
            nhead_k=1,
            seqlen_q=sq,
            seqlen_k=skv,
            hdim_q=128,
            hdim_v=128,
            arch="gfx950",
            dtype="bf16",
            mask_type=0,  # NO_MASK: the band comes from (sliding_window, right_bound)
            sliding_window=window,
            right_bound=right,
        )
        spec = attention_tuning_spec(
            req, "gfx950_dense_persist" if persistent else "gfx950_dense_grid"
        ).kernel_spec
        assert spec.causal == (right >= 0) and spec.right_bound == max(right, 0)
        assert spec.ragged == ragged and spec.persistent == persistent
        rng = np.random.default_rng(1)
        q = operand(rng, (_BAND_BATCH, sq, 4, 128), "bf16")
        k = operand(rng, (_BAND_BATCH, skv, 1, 128), "bf16")
        v = operand(rng, (_BAND_BATCH, skv, 1, 128), "bf16")
        out = filled(q.shape, q.dtype)
        scale = 1.0 / math.sqrt(128)
        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
        _check_band(out, *_band_reference(q, k, v, scale, window, right))


_DENSE_SPEC_IDS = (
    "gfx950_dense_grid",
    "gfx950_dense_persist",
    "gfx950_dense_persist_widedma",
)
# Windows that are not multiples of every tile a variant may pick, with both bounds.
_VARIANT_BAND_CASES = [
    pytest.param(1024, 1024, 192, -1, id="left-only-w192-1024x1024"),
    pytest.param(1024, 512, 320, 64, id="two-sided-w320-r64-1024x512"),
    pytest.param(512, 1024, 64, 0, id="causal-w64-512x1024"),
    pytest.param(1024, 512, 128, 0, id="causal-w128-1024x512-empty"),
]


class TestDenseVariantsBand:
    """Every dense tuning variant (grid, persistent, wide-DMA persistent) against the
    band oracle, through the dispatcher so the variant's own tile geometry is used."""

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("spec_id", _DENSE_SPEC_IDS)
    @pytest.mark.parametrize("sq,skv,window,right", _VARIANT_BAND_CASES)
    def test_variant_matches_band(self, spec_id, sq, skv, window, right):
        from dispatch.attention import AttentionRequest, attention_tuning_spec

        req = AttentionRequest(
            batch=_BAND_BATCH,
            nhead_q=4,
            nhead_k=1,
            seqlen_q=sq,
            seqlen_k=skv,
            hdim_q=128,
            hdim_v=128,
            arch="gfx950",
            dtype="bf16",
            mask_type=0,
            sliding_window=window,
            right_bound=right,
        )
        spec = attention_tuning_spec(req, spec_id).kernel_spec
        rng = np.random.default_rng(1)
        q = operand(rng, (_BAND_BATCH, sq, 4, 128), "bf16")
        k = operand(rng, (_BAND_BATCH, skv, 1, 128), "bf16")
        v = operand(rng, (_BAND_BATCH, skv, 1, 128), "bf16")
        out = filled(q.shape, q.dtype)
        scale = 1.0 / math.sqrt(128)
        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
        _check_band(out, *_band_reference(q, k, v, scale, window, right))


_ONE_HOT_DENSE = [
    pytest.param(64, 0, id="causal_w64"),
    pytest.param(64, -1, id="left_only_w64"),
    pytest.param(64, 4, id="two_sided_w64_r4"),
    pytest.param(0, 4, id="lookahead_r4"),
    pytest.param(0, -1, id="full"),
    pytest.param(0, 0, id="causal"),
]


class TestDenseBandIndexTruthTable:
    """Index-level check of the dense band inequalities (top-left diagonal).

    ``q = 0`` makes the visible scores equal (uniform softmax) and ``V[k] = e_k``
    turns each output row into the indicator of its visible keys over ``1/n``, so
    ``left = W`` seeing W keys counting the diagonal (``d - k < W``) and
    ``k <= d + right`` are checked per key index. Rows with no visible key are zero.
    """

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("persistent", [False, True], ids=["grid", "persist"])
    @pytest.mark.parametrize("window,right", _ONE_HOT_DENSE)
    def test_visible_key_set_is_exact(self, window, right, persistent):
        tile = DENSE_TILE_GEOMETRIES["default"]
        sq, skv, batch, head = int(tile["block_m"]), 128, 2, 128
        spec = Gfx950AttentionDenseSpec(
            batch=batch,
            seqlen_q=sq,
            seqlen_kv=skv,
            num_query_heads=2,
            num_kv_heads=1,
            head_size=head,
            causal=right >= 0,
            right_bound=max(right, 0),
            dtype="bf16",
            block_m=int(tile["block_m"]),
            block_n=int(tile["block_n"]),
            sliding_window=window,
            persistent=persistent,
        )
        q = filled((batch, sq, 2, head), "bf16", 0.0)
        k = operand(np.random.default_rng(0), (batch, skv, 1, head), "bf16")
        one_hot = np.zeros((batch, skv, 1, head))
        one_hot[:, np.arange(skv), 0, np.arange(skv)] = 1.0
        v = upload(one_hot, "bf16")
        out = filled(q.shape, q.dtype)
        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=1.0)
        expected = np.zeros((sq, 2, head))
        for p in range(sq):
            visible = [
                kk
                for kk in range(skv)
                if (window == 0 or p - kk < window) and (right < 0 or kk <= p + right)
            ]
            if visible:
                expected[p, :, visible] = 1.0 / len(visible)
        # Each visible weight is exactly 1, so a row is 1/n rounded to bf16 (half
        # an ulp, 2**-9 relative) and an invisible key's column exactly 0: a
        # dropped or extra key misses by a whole 1/n.
        assert_close(
            out, np.broadcast_to(expected, out.shape), Tolerance(atol=0.0, rtol=2**-7)
        )


class TestDenseRowIndependence:
    """A row's output depends only on the keys that row sees.

    Causal (top-left), Sq = Skv = S: key S-1 is visible to row S-1 only, which shares
    a 32-row wave with rows S-32..S-2. K at that key is 1e4 * sign(Q of row S-1), so
    row S-1's running max jumps far past the lazy-rescale threshold, and V there is
    +-1e4. Every other row masks the key, so its output must not change by a single
    bit. A wave-wide lazy-rescale decision breaks this: the jump makes every row of
    the wave move its max and round differently.
    """

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("window", [0, 128], ids=["causal", "causal_w128"])
    @pytest.mark.parametrize("persistent", [False, True], ids=["grid", "persist"])
    @pytest.mark.parametrize("d", [64, 128])
    @pytest.mark.parametrize("dtype", ["bf16", "fp16"])
    def test_rows_independent_of_wave_neighbours(self, dtype, d, persistent, window):
        B, S, H = 1, 512, 16
        scale = 1.0 / math.sqrt(d)
        rng = np.random.default_rng(0)
        q = operand(rng, (B, S, H, d), dtype)
        k = operand(rng, (B, S, H, d), dtype)
        v = operand(rng, (B, S, H, d), dtype)
        k_hot, v_hot = values(k), values(v)
        k_hot[:, S - 1] = 1e4 * np.sign(values(q)[:, S - 1])
        v_hot[:, S - 1] = 1e4 * np.sign(rng.standard_normal(v_hot[:, S - 1].shape))
        k_hot, v_hot = upload(k_hot, dtype), upload(v_hot, dtype)

        spec = _spec(dtype, d, H, H, persistent, sq=S, sliding_window=window)
        outs = []
        for kk, vv in ((k, v), (k_hot, v_hot)):
            out = filled(q.shape, q.dtype)
            run_attention_dense_torch(spec=spec, q=q, k=kk, v=vv, out=out, scale=scale)
            outs.append(out)
        base, hot = (values(o) for o in outs)
        assert np.isfinite(hot).all(), "non-finite output"
        assert not same_values(base[:, S - 1], hot[:, S - 1]), "key S-1 had no effect"
        changed = np.argwhere((base[:, : S - 1] != hot[:, : S - 1]).any(axis=-1))
        assert changed.size == 0, (
            f"{len(changed)} (row, head) pairs that mask key {S - 1} changed, rows "
            f"{sorted(set(changed[:, 1].tolist()))}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
