# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for the gfx942 dense flash-attn kernel, without Torch.

Drives the PUBLIC entry point ``run_attention_dense_torch`` end-to-end on a real
gfx942 GPU and checks max_abs against a float64 NumPy attention oracle
(``rocke.numeric.sdpa_reference``; the sink cohorts use a masked softmax with an
appended sink column), for both the default and the P4 persistent grid. Inputs and
outputs are ``rocke.runtime`` device buffers, so the lane runs where Torch is not
installed.

Specs come from the gfx942 DISPATCH factory, never hand-rolled (see :func:`_spec`),
so every row compiles the binary that actually ships rather than one that differs
from it by an untracked tuning default.

Every test is marked ``gpu`` and gated with a device skipif, so it is a graceful
skip on a CPU CI box and only executes on a gfx942 (MI300X) ROCm runner. Select
it with ``run_all.py --gpu`` (or ``pytest -m gpu``); the default CPU lane excludes
it via ``-m "not gpu"``. Run standalone:

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_attention_dense_gfx942_numeric.py
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from kernels.gfx942.attention_dense import (
    _as_gfx942_spec,
    run_attention_dense_torch,
    supports_attention_dense,
)
from rocke.numeric.sdpa_reference import sdpa_reference
from rocke.runtime.hip_module import get_device_arch

from . import _attention_numpy as an
from ._attention_device_array import DeviceArray
from ._ci_subset import ci_subset
from ._dense_paged_gpu import paged_cache, run_dispatched_paged

# Gate on the base ISA target, not the marketing name: the whole MI300 family is
# gfx942, but the marketing string varies (MI300X/MI300A/MI308X).
requires_gfx942_gpu = pytest.mark.skipif(
    get_device_arch() != "gfx942", reason="needs a gfx942 (MI300X) GPU"
)
pytestmark = [pytest.mark.gpu, requires_gfx942_gpu]

_RT = None


def _rt():
    global _RT
    if _RT is None:
        _RT = an.runtime()
    return _RT


# (dtype, head_size, num_query_heads, num_kv_heads, persistent, causal) -- a compact
# cohort spanning both dtypes, D64/D128, GQA + MHA, causal + non-causal, and both grid
# variants. Sq is fixed at 512 (a 256 multiple so the persistent grid-stride has >1
# q-block of work). The last two rows cover the fp16-D128 swizzle default grid on the
# paths dispatch actually ships but the base rows miss: MHA, and non-causal.
_COHORT = [
    ("fp16", 128, 16, 4, False, True),  # flagship default (causal)
    ("fp16", 128, 16, 4, True, True),  # flagship persistent
    ("bf16", 128, 16, 4, True, True),  # bf16 D128 persistent (the VGPR-starved config)
    # bf16 D128 default: the plain-exp2 arm -- the one config default_use_exp2_fast
    # turns off, and it does so on the grid, at every seqlen.
    ("bf16", 128, 16, 4, False, True),
    ("fp16", 64, 16, 16, False, True),  # D64 MHA default
    ("bf16", 64, 16, 4, True, True),  # D64 bf16 persistent (the wpe=4 config)
    ("fp16", 128, 16, 16, False, True),  # fp16 D128 MHA default -- swizzle path, MHA
    ("fp16", 128, 16, 4, False, False),  # fp16 D128 non-causal default -- swizzle path
]

# (dtype, head_size, num_query_heads, num_kv_heads, persistent, sliding_window, scale)
# -- STANDALONE sliding-window (no sinks; SWA-sink is _SWA_SINK_COHORT below). All
# rows are causal (sliding_window > 0 requires causal). Window is a multiple of the
# shipped block_n (64): 128, 256. Covers both dtypes, D64/D128, and BOTH grid
# variants -- the persistent rows exercise the per-work-item start_tile prune that the
# default grid does not. ``scale`` None is the default 1/sqrt(D); the two scale-1.0
# rows run the window mask on fp32 scores scaled after the QK MFMA, one per grid.
_SWA_COHORT = [
    ("bf16", 128, 16, 4, False, 128, None),
    ("bf16", 128, 16, 4, True, 128, None),
    ("fp16", 128, 16, 4, False, 256, None),
    ("fp16", 128, 16, 4, True, 256, None),
    ("bf16", 64, 16, 4, False, 128, None),
    ("bf16", 64, 16, 4, True, 128, None),
    ("bf16", 128, 16, 4, False, 128, 1.0),
    ("bf16", 64, 16, 4, True, 128, 1.0),
    # W well past block_m: non-empty mask-free middle phase of the band loop.
    ("bf16", 128, 16, 4, False, 512, None),
    ("fp16", 128, 16, 4, True, 512, None),
]

# (dtype, head_size, num_query_heads, num_kv_heads, persistent, causal): full-sink
# cohort. Both dtypes, D64/D128, GQA + MHA, both grids, and non-causal on both grids.
_SINK_COHORT = [
    ("fp16", 128, 16, 4, False, True),
    ("fp16", 128, 16, 4, True, True),
    ("bf16", 128, 16, 4, False, True),
    ("bf16", 128, 16, 4, True, True),
    ("fp16", 64, 16, 16, False, True),
    ("bf16", 64, 16, 4, True, True),
    ("fp16", 128, 16, 4, False, False),
    ("bf16", 128, 16, 4, False, False),
    ("bf16", 128, 16, 4, True, False),
]

# SWA-sink: the standalone SWA rows with sinks on, scale column included.
_SWA_SINK_COHORT = _SWA_COHORT

# (head_size, dtype, persistent, scale) for test_dense_non_default_scale: every D128
# combination, plus bf16 D64 at the larger scale on both grids. D64 has its own K
# layout and, for bf16, its own waves-per-eu; the scaling line is shared.
_NON_DEFAULT_SCALE_ROWS = [
    (128, dtype, persistent, scale)
    for scale in (0.5, 1.0)
    for dtype in ("bf16", "fp16")
    for persistent in (False, True)
] + [(64, "bf16", False, 1.0), (64, "bf16", True, 1.0)]


def _spec(
    dtype,
    d,
    hq,
    hkv,
    persistent,
    *,
    causal=True,
    batch=1,
    sq=512,
    sk=None,
    sliding_window=0,
    use_sinks=False,
    persist_decode="auto",
    bottom_right=False,
    kv_block_size=0,
    block_m=None,
):
    """The SHIPPED gfx942 dense spec for a cohort row, built through the dispatch
    candidate (``gfx942_dense`` via ``dispatch.attention``) rather than hand-rolled.

    Hand-rolling the spec silently pins every tuned lever to the shared (gfx950)
    dataclass default, so the lane would assert on configs that do not ship. The
    concrete one this cohort hit: ``waves_per_eu``. Dispatch resolves it from the
    kernel's own policy (``_tuned_waves_per_eu``), which returns 4 for bf16/D64 --
    the row ``_COHORT`` above labels "the wpe=4 config" -- while the dataclass default
    is 2. That is not a cosmetic difference: ``waves_per_eu`` is emitted as the
    ``amdgpu-waves-per-eu`` attribute, changes register allocation, and is tagged into
    ``gfx942_kernel_name`` as ``wpe{N}``, so wpe2 and wpe4 are DIFFERENT binaries.
    ``num_persistent`` was likewise hard-coded to 304 beside a dispatch constant that
    already resolves to 304 -- left at the candidate default here so dispatch
    supplies the gfx942 CU count itself and the two cannot drift apart.

    Deriving the spec from the factory (the pattern
    ``test_attention_dense_gfx942_golden.py::mk_dispatch`` uses for its D64 cases)
    also means a future gfx942 tuning change is picked up here with no edit.

    Only the ``persistent`` knob is set rather than left at the default: the cohort
    asserts BOTH grid variants at one fixed Sq, where the default picks one. Every
    other lever -- block_n, the D64 K row-group pad, ragged -- is whatever the
    shipped path folds in. ``persist_decode`` stays "auto" unless a test pins one
    order to hold it fixed across shapes "auto" would resolve differently; a pinned
    order is set on the spec, since the knob refuses one that restates "auto".

    Pinning the persistent grid also pins ``kv_segments=1``: the KV-segment split
    runs on the default grid only, and dispatch picks one (dropping persistent) for
    short bottom-right chunks, so the persistent pin alone would ask for a
    combination dispatch never produces.

    ``block_m`` pins the query-tile height knob; ``None`` leaves the tile (and with it
    ``block_n``) to dispatch, which picks it from the shape and the window.
    """
    # Imported lazily, mirroring the golden sibling: keeps module import (and hence
    # CPU collection of this gpu-marked file) independent of the dispatch package.
    from dispatch.attention import AttentionRequest, tuning_spec_with_knobs

    knobs = (
        {"persistent": True, "kv_segments": 1} if persistent else {"persistent": False}
    )
    if block_m is not None:
        knobs["block_m"] = block_m
    spec = tuning_spec_with_knobs(
        AttentionRequest(
            batch=batch,
            nhead_q=hq,
            nhead_k=hkv,
            seqlen_q=sq,
            seqlen_k=sq if sk is None else sk,
            hdim_q=d,
            hdim_v=d,
            arch="gfx942",
            mask_type=(2 if bottom_right else 1) if causal else 0,
            dtype=dtype,
            sliding_window=sliding_window,
            use_sinks=use_sinks,
            **(
                {"layout": "paged", "kv_block_size": kv_block_size}
                if kv_block_size
                else {}
            ),
        ),
        "gfx942_dense",
        knobs,
    ).kernel_spec
    if persist_decode == "auto":
        return spec
    return dataclasses.replace(spec, persist_decode=persist_decode)


class _Run:
    """One launch: the output as fp32, its raw 16-bit pattern, and the LSE if any."""

    def __init__(self, out_dev, dtype, lse_dev=None):
        host = out_dev.read()
        self.out = an.decode(host, dtype)
        self.bits = host.view(np.uint16)
        self.lse = None if lse_dev is None else lse_dev.read()


def _qkv(seed, dtype, hq, hkv, d, *, sq, sk=None, batch=1):
    """Seeded unit-normal q, k, v as fp32 arrays rounded to ``dtype``."""
    rng = np.random.default_rng(seed)
    sk = sq if sk is None else sk
    q = an.randn(rng, (batch, sq, hq, d), dtype)
    k = an.randn(rng, (batch, sk, hkv, d), dtype)
    v = an.randn(rng, (batch, sk, hkv, d), dtype)
    return q, k, v


def _run(spec, q, k, v, scale, *, sinks=None):
    """Launch ``spec`` once. The output starts NaN-filled so an unwritten element
    cannot pass as a value; the LSE buffer is made iff ``spec.emit_lse``."""
    rt, dtype = _rt(), spec.dtype
    out = an.empty_device(rt, q.shape, dtype)
    kwargs = {}
    lse = None
    if sinks is not None:
        kwargs["sinks"] = an.to_device(rt, sinks, dtype)
    if spec.emit_lse:
        lse = an.lse_device(rt, (spec.batch, spec.num_query_heads, spec.seqlen_q, 1))
        kwargs["lse"] = lse
    returned = run_attention_dense_torch(
        spec=spec,
        q=an.to_device(rt, q, dtype),
        k=an.to_device(rt, k, dtype),
        v=an.to_device(rt, v, dtype),
        out=out,
        scale=scale,
        **kwargs,
    )
    assert returned is out
    return _Run(out, dtype, lse)


def _max_abs(got, want):
    return float(np.max(np.abs(np.asarray(got, np.float64) - want)))


def _assert_allclose(got, want, *, atol, rtol, what=""):
    """``|got - want| <= atol + rtol * |want|`` everywhere, with ``got`` finite."""
    got = np.asarray(got, np.float64)
    ok = np.isfinite(got) & (np.abs(got - want) <= atol + rtol * np.abs(want))
    assert ok.all(), (
        f"{what}: {int((~ok).sum())} of {ok.size} elements outside "
        f"atol={atol} rtol={rtol}; max_abs={_max_abs(got, want):.3e}"
    )


def _sdpa_reference(q, k, v, scale, *, causal=True, window=0, bottom_right=False):
    """float64 SDPA oracle for ``[B,S,H,D]`` arrays, returned as ``[B,Sq,Hq,D]``.

    ``causal`` is top-left aligned unless ``bottom_right``; ``window`` keeps the
    ``window`` newest keys at each query (the diagonal included). A row with no
    visible key is 0."""
    out, _ = sdpa_reference(
        np.swapaxes(q, 1, 2),
        np.swapaxes(k, 1, 2),
        np.swapaxes(v, 1, 2),
        scale=scale,
        left_bound=window - 1 if window else None,
        right_bound=0 if causal else None,
        top_left=not bottom_right,
    )
    return np.swapaxes(out, 1, 2)


@pytest.mark.parametrize("dtype,d,hq,hkv,persistent,causal", _COHORT)
def test_dense_numeric_vs_fp32_sdpa(dtype, d, hq, hkv, persistent, causal):
    # Element-wise atol + rtol*|ref| rather than a raw max-abs bound: rounding the
    # reference to the output dtype alone consumes most of a fixed absolute
    # budget on randn inputs, which would leave the check seed-sensitive.
    tol = 2e-3 if dtype == "fp16" else 1e-2
    B, S = 1, 512
    scale = 1.0 / math.sqrt(d)

    # run_attention_dense_torch ABI: q/out [B,S,Hq,D], k/v [B,S,Hkv,D], dense.
    q, k, v = _qkv(0, dtype, hq, hkv, d, sq=S, batch=B)
    spec = _spec(dtype, d, hq, hkv, persistent, causal=causal, batch=B, sq=S)
    run = _run(spec, q, k, v, scale)

    ref = _sdpa_reference(q, k, v, scale, causal=causal)
    _assert_allclose(
        run.out,
        ref,
        atol=tol,
        rtol=tol,
        what=(
            f"{dtype} D{d} GQA{hq}/{hkv} {'causal' if causal else 'full'} "
            f"{'persist' if persistent else 'default'}"
        ),
    )


@pytest.mark.parametrize("d,dtype,persistent,scale", _NON_DEFAULT_SCALE_ROWS)
def test_dense_non_default_scale(d, dtype, persistent, scale):
    """Softmax scales well above the default 1/sqrt(D).

    The score error a lossy scale step introduces grows with ``scale``, so the
    default-scale cohort above cannot see it. Rounding ``Q * scale * log2(e)``
    back to bf16/fp16 before the QK MFMA puts bf16 past its tolerance on both grids:
    D128 at both scales and D64 at 1.0. The kernel must apply the scale to the fp32
    scores. fp16 has three more mantissa bits and stays inside its tolerance either
    way, so the fp16 rows guard against regressions rather than catch this defect.
    The sliding-window paths are covered by the scale-1.0 rows of ``_SWA_COHORT``.
    Not covered: scales other than 0.5 and 1.0.
    """
    hq, hkv = 16, 4
    tol = 2e-2 if dtype == "fp16" else 4e-2
    B, S = 1, 512

    q, k, v = _qkv(0, dtype, hq, hkv, d, sq=S, batch=B)
    spec = _spec(dtype, d, hq, hkv, persistent, batch=B, sq=S)
    run = _run(spec, q, k, v, scale)

    max_abs = _max_abs(run.out, _sdpa_reference(q, k, v, scale))
    assert max_abs < tol, (
        f"{dtype} D{d} GQA16/4 scale={scale:g} "
        f"{'persist' if persistent else 'default'}: max_abs={max_abs:.3e} >= {tol}"
    )


@pytest.mark.parametrize("persistent", [False, True])
def test_dense_fp16_d128_large_equal_scores(persistent):
    """Every score in a row is equal and large (raw 128 * 427**2 = 23338112, at
    scale 16), so the softmax is uniform and the output is the causal running
    mean of V. Scaling the fp32 scores keeps the row max's exp2 argument exactly
    0 at any magnitude; it must stay finite and correct.

    A regression guard, not a reproducer: the pre-fix gfx942 kernel also passes
    it. It fails a kernel that takes the row max on unscaled scores and folds the
    scale into the exp2 argument (fp16 P overflows there), which is the form the
    gfx950 ordinary grid uses. Only fp16, D128 and this one magnitude are covered.
    """
    d, hq, hkv, scale = 128, 16, 4, 16.0
    B, S = 1, 512

    q = np.full((B, S, hq, d), 427.0, np.float32)
    k = np.full((B, S, hkv, d), 427.0, np.float32)
    _, _, v = _qkv(0, "fp16", hq, hkv, d, sq=S, batch=B)

    spec = _spec("fp16", d, hq, hkv, persistent, batch=B, sq=S)
    run = _run(spec, q, k, v, scale)

    assert np.isfinite(run.out).all(), "non-finite output"
    max_abs = _max_abs(run.out, _sdpa_reference(q, k, v, scale))
    assert max_abs < 2e-2, (
        f"fp16 D128 equal scores scale=16 "
        f"{'persist' if persistent else 'default'}: max_abs={max_abs:.3e}"
    )


# (dtype, head_size, num_query_heads, num_kv_heads, persistent, causal_diag_split,
# seqlen_q, seqlen_kv): bottom-right causal, seqlen_q < seqlen_kv (chunked prefill).
# The diagonal offset seqlen_kv - seqlen_q is a runtime value: block-aligned (512,
# 3584) and not (64, 576), both grids, with and without the unmasked-body split.
_BOTTOM_RIGHT_COHORT = [
    ("bf16", 128, 16, 4, False, False, 512, 1024),
    ("bf16", 128, 16, 4, False, True, 512, 1088),
    ("bf16", 128, 16, 4, True, False, 512, 576),
    ("bf16", 128, 16, 4, True, True, 256, 3840),
    ("fp16", 128, 16, 4, False, True, 512, 1024),
    ("fp16", 128, 16, 4, True, True, 512, 1088),
    ("bf16", 64, 16, 4, False, True, 512, 576),
    ("bf16", 64, 16, 4, True, False, 256, 1024),
]


@pytest.mark.parametrize(
    "dtype,d,hq,hkv,persistent,diag_split,sq,sk", _BOTTOM_RIGHT_COHORT
)
def test_dense_bottom_right_numeric_vs_fp32_sdpa(
    torch_free_kv_split, dtype, d, hq, hkv, persistent, diag_split, sq, sk
):
    tol = 2e-2 if dtype == "fp16" else 4e-2
    B = 1
    scale = 1.0 / math.sqrt(d)
    q, k, v = _qkv(0, dtype, hq, hkv, d, sq=sq, sk=sk, batch=B)

    spec = _spec(dtype, d, hq, hkv, persistent, sq=sq, sk=sk, bottom_right=True)
    assert spec.causal_bottom_right
    spec = dataclasses.replace(spec, causal_diag_split=diag_split)
    run = _run(spec, q, k, v, scale)

    ref = _sdpa_reference(q, k, v, scale, bottom_right=True)
    max_abs = _max_abs(run.out, ref)
    assert max_abs < tol, (
        f"{dtype} D{d} GQA{hq}/{hkv} bottom-right sq={sq} sk={sk} "
        f"{'persist' if persistent else 'default'} split={diag_split}: "
        f"max_abs={max_abs:.3e} >= {tol}"
    )


# (dtype, head_size, num_query_heads, num_kv_heads, mask, causal_diag_split,
# kv_segments, seqlen_q, seqlen_kv): the KV-segment split on the default grid. mask
# is "br" (bottom-right causal), "tl" (top-left causal, where early query blocks
# leave trailing segments empty) or "full".
_KV_SEGMENT_COHORT = [
    ("bf16", 128, 16, 4, "br", True, 4, 512, 4096),
    ("bf16", 128, 16, 4, "br", False, 3, 256, 1088),
    ("fp16", 128, 16, 4, "br", True, 8, 512, 2048),
    ("bf16", 64, 16, 4, "br", True, 4, 512, 2048),
    ("bf16", 128, 16, 4, "tl", True, 4, 1024, 1024),
    ("fp16", 128, 16, 16, "full", False, 2, 1024, 1024),
]


@pytest.fixture
def torch_free_kv_split(monkeypatch):
    """Run the KV-segment split on ``rocke.runtime`` buffers.

    The product workspace allocation and combine launch build their buffers with
    Torch, so this swaps in the same launch on device arrays: the same combine
    kernel, ABI and launcher cache, with only the buffers' owner changed."""
    import kernels.gfx942.attention_dense as dense
    from kernels.common import attention_dense_kv_split as split
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig

    def allocate(spec, device):
        shape = (spec.batch * spec.seqlen_q, spec.num_query_heads, spec.kv_segments)

        def zeros(s):
            return DeviceArray(_rt(), np.zeros(s, np.float32))

        return {
            "segm_output_ptr": zeros(shape + (spec.head_size,)),
            "segm_max_ptr": zeros(shape),
            "segm_expsum_ptr": zeros(shape),
        }

    def combine(spec, arch, out, workspace, stream):
        red, signature, scalars = split.combine_launch_abi(spec, arch)
        _, build = split._combine_impl(arch)
        key = (arch, red)
        launcher = split._COMBINE_LAUNCHER_CACHE.get(key)
        if launcher is None:
            art = compile_kernel(
                build(red, arch=arch),
                arch=arch,
                backend="python",
                capture_ir_text=False,
            )
            launcher = KernelLauncher(
                hsaco=art.hsaco, kernel_name=art.kernel_name, signature=signature
            )
            split._COMBINE_LAUNCHER_CACHE[key] = launcher
        shape = workspace["segm_max_ptr"].shape
        vals = dict(workspace)
        vals.update(scalars)
        vals["output_ptr"] = out
        # Unused by the combine body; any int32 buffer satisfies the ABI.
        vals["seq_lens_ptr"] = DeviceArray(_rt(), np.zeros(shape, np.int32))
        launcher(
            vals,
            config=LaunchConfig(
                grid=(spec.batch * spec.seqlen_q, spec.num_query_heads, 1),
                block=(64, 1, 1),
                stream=int(stream),
            ),
        )

    monkeypatch.setattr(dense, "allocate_segment_workspace", allocate)
    monkeypatch.setattr(dense, "launch_segment_combine", combine)


@pytest.mark.parametrize(
    "dtype,d,hq,hkv,mask,diag_split,segments,sq,sk", _KV_SEGMENT_COHORT
)
def test_dense_kv_segments_numeric_vs_fp32_sdpa(
    torch_free_kv_split, dtype, d, hq, hkv, mask, diag_split, segments, sq, sk
):
    tol = 2e-2 if dtype == "fp16" else 4e-2
    scale = 1.0 / math.sqrt(d)
    q, k, v = _qkv(0, dtype, hq, hkv, d, sq=sq, sk=sk)

    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        False,
        causal=mask != "full",
        sq=sq,
        sk=sk,
        bottom_right=mask == "br",
    )
    spec = dataclasses.replace(
        spec,
        causal_diag_split=diag_split,
        kv_segments=segments,
        block_m=128,
    )
    run = _run(spec, q, k, v, scale)

    ref = _sdpa_reference(
        q, k, v, scale, causal=mask != "full", bottom_right=mask == "br"
    )
    max_abs = _max_abs(run.out, ref)
    assert max_abs < tol, (
        f"{dtype} D{d} {mask} split={diag_split} segments={segments} sq={sq} "
        f"sk={sk}: max_abs={max_abs:.3e} >= {tol}"
    )


def test_one_bottom_right_binary_serves_every_chunk_length(torch_free_kv_split):
    """The diagonal offset is read from the shape params, so one compiled binary
    must serve chunks with different seqlen_kv - seqlen_q, each correctly."""
    from kernels.common.attention_dense_spec import attention_dense_cache_key

    dtype, d, hq, hkv = "bf16", 128, 16, 4
    tol = 4e-2
    scale = 1.0 / math.sqrt(d)
    shapes = ((512, 1024), (256, 4096), (512, 576))
    specs = [
        _as_gfx942_spec(_spec(dtype, d, hq, hkv, False, sq=a, sk=c, bottom_right=True))
        for a, c in shapes
    ]
    keys = {attention_dense_cache_key(s, arch="gfx942") for s in specs}
    assert len(keys) == 1, f"chunk lengths {shapes} did not share a cache key"
    launchers = []
    for (sq, sk), spec in zip(shapes, specs):
        q, k, v = _qkv(0, dtype, hq, hkv, d, sq=sq, sk=sk)
        run = _run(spec, q, k, v, scale)
        launchers.append(_launcher_for(spec))
        ref = _sdpa_reference(q, k, v, scale, bottom_right=True)
        max_abs = _max_abs(run.out, ref)
        assert max_abs < tol, f"sq={sq} sk={sk}: max_abs={max_abs:.3e} >= {tol}"
    assert launchers[0] is not None
    assert all(x is launchers[0] for x in launchers)


def _launcher_for(spec):
    """The cached ``KernelLauncher`` for ``spec``, or None if none is compiled.

    ``run_attention_dense_torch`` owns ``_DENSE_LAUNCHER_CACHE`` internally and
    exposes no accessor, so this reads the module-level dict through the very key
    function the production path uses -- a test-local reimplementation of the key
    would assert against itself rather than against the shipped one.
    """
    from kernels.common.attention_dense_spec import attention_dense_cache_key
    from kernels.gfx942.attention_dense import _DENSE_LAUNCHER_CACHE

    return _DENSE_LAUNCHER_CACHE.get(attention_dense_cache_key(spec, arch="gfx942"))


def test_one_binary_serves_every_shape():
    """One compiled artifact, two shapes, correct numerics at both.

    The cohort above runs many shapes, but it stopped discriminating the moment
    batch/seqlen_q/seqlen_kv became runtime kernel params: it passes identically
    whether N shapes are served by N binaries or by one. The property that
    actually needs a guard now is *artifact reuse*.

    ``is`` on the launcher covers the whole path rather than just the key
    function. A key regression would land the two shapes in different cache
    slots; a lookup regression would overwrite the one slot with a freshly
    compiled launcher. Both yield a different object, and neither is visible to a
    numeric assertion -- recompiling per shape is *correct*, merely wasteful, so
    what regresses is the AOT instance count and first-call latency, which no
    accuracy check can see.

    Paired with the numeric check at both shapes so the test cannot pass by
    reusing one binary that happens to be wrong for the second shape.

    fp16 is the flagship default config; bf16 D128 would serve equally well now
    that no dtype forks the body on shape, but this is the path dispatch ships
    most of.
    """
    from kernels.common.attention_dense_spec import attention_dense_cache_key
    from kernels.gfx942.attention_dense import _DENSE_LAUNCHER_CACHE

    dtype, d, hq, hkv = "fp16", 128, 16, 4  # flagship default, non-persistent
    tol = 2e-2
    scale = 1.0 / math.sqrt(d)

    shapes = ((1, 512), (4, 1024))
    specs = [
        _as_gfx942_spec(_spec(dtype, d, hq, hkv, False, batch=b, sq=s))
        for b, s in shapes
    ]

    # Preconditions: genuinely different shapes, on the runtime path, and sharing
    # one key -- otherwise the reuse assertion below is vacuous.
    assert specs[0].runtime_shape, "cohort row is not on the runtime-shape path"
    assert (specs[0].batch, specs[0].seqlen_q) != (specs[1].batch, specs[1].seqlen_q)
    keys = [attention_dense_cache_key(s, arch="gfx942") for s in specs]
    assert keys[0] == keys[1], f"shapes {shapes} did not share a cache key"

    # Own the cache state: evicting first makes the "exactly one new entry"
    # assertion independent of which tests ran before this one.
    _DENSE_LAUNCHER_CACHE.pop(keys[0], None)
    before = set(_DENSE_LAUNCHER_CACHE)

    launchers = []
    for (B, S), spec in zip(shapes, specs):
        q, k, v = _qkv(0, dtype, hq, hkv, d, sq=S, batch=B)
        run = _run(spec, q, k, v, scale)
        launchers.append(_launcher_for(spec))

        max_abs = _max_abs(run.out, _sdpa_reference(q, k, v, scale))
        assert max_abs < tol, f"B={B} S={S}: max_abs={max_abs:.3e} >= {tol}"

    assert launchers[0] is not None, (
        "no launcher cached after a successful run; _DENSE_LAUNCHER_CACHE is no "
        "longer keyed by attention_dense_cache_key and this test is blind"
    )
    assert launchers[0] is launchers[1], (
        f"shapes {shapes} share a cache key but were served by DIFFERENT launcher "
        "objects -- the runtime-shape kernel recompiled per shape, so the AOT "
        "instance count still scales with the shape space"
    )
    assert set(_DENSE_LAUNCHER_CACHE) - before == {keys[0]}, (
        "two shapes on the runtime path added more than one cache entry: "
        f"{sorted(set(_DENSE_LAUNCHER_CACHE) - before)}"
    )


def test_one_binary_serves_every_head_config():
    """One compiled artifact, four head configs, correct numerics at each.

    The head-count twin of :func:`test_one_binary_serves_every_shape`: on the
    runtime-shape path ``num_query_heads`` / ``num_kv_heads`` are kernel params,
    so the GQA ratio and the token strides are computed at run time. A body that
    still baked one of them would compute wrong strides for every config but the
    first one compiled -- caught here by the numeric check, while the ``is`` /
    single-entry assertions catch a key or lookup regression that recompiles per
    head config. Covers GQA 16, non-pow2 GQA 5 and 7, and MHA.
    """
    from kernels.common.attention_dense_spec import attention_dense_cache_key
    from kernels.gfx942.attention_dense import _DENSE_LAUNCHER_CACHE

    dtype, d, B, S = "fp16", 128, 1, 512  # flagship default, non-persistent
    tol = 2e-2
    scale = 1.0 / math.sqrt(d)

    heads = ((128, 8), (40, 8), (28, 4), (32, 32))
    specs = [
        _as_gfx942_spec(_spec(dtype, d, hq, hkv, False, batch=B, sq=S))
        for hq, hkv in heads
    ]

    # Preconditions: all on the runtime path and sharing one key -- otherwise the
    # reuse assertion below is vacuous.
    assert all(s.runtime_shape for s in specs), "a head config left the runtime path"
    keys = [attention_dense_cache_key(s, arch="gfx942") for s in specs]
    assert len(set(keys)) == 1, f"head configs {heads} did not share a cache key"

    _DENSE_LAUNCHER_CACHE.pop(keys[0], None)
    before = set(_DENSE_LAUNCHER_CACHE)

    launchers = []
    for (hq, hkv), spec in zip(heads, specs):
        q, k, v = _qkv(0, dtype, hq, hkv, d, sq=S, batch=B)
        run = _run(spec, q, k, v, scale)
        launchers.append(_launcher_for(spec))

        max_abs = _max_abs(run.out, _sdpa_reference(q, k, v, scale))
        assert max_abs < tol, f"Hq={hq} Hkv={hkv}: max_abs={max_abs:.3e} >= {tol}"

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


@pytest.mark.parametrize("dtype,d,hq,hkv,persistent,sliding_window,scale", _SWA_COHORT)
def test_dense_swa_numeric_vs_fp32_sdpa(
    dtype, d, hq, hkv, persistent, sliding_window, scale
):
    """Sliding-window (SWA) numeric parity, standalone (no sinks), both grids.

    The band is the same one the gfx950 sibling masks (``_sink_reference``: causal
    ``ki > qi`` plus window ``ki <= qi - W`` masked out) and the kernel applies
    (``cmp_le(ktok, q)`` + ``cmp_gt(ktok, q - SW)``): keep key k for query q iff
    ``q - W < k <= q``. The sink cohorts use the manual masked-softmax in
    :func:`_sink_reference` instead, since the oracle cannot append a sink column.
    The diagonal k==q is always kept (W>0), so no row is fully masked."""
    tol = 2e-2 if dtype == "fp16" else 4e-2
    B, S = 1, 512
    scale = 1.0 / math.sqrt(d) if scale is None else scale

    q, k, v = _qkv(0, dtype, hq, hkv, d, sq=S, batch=B)
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        persistent,
        causal=True,
        batch=B,
        sq=S,
        sliding_window=sliding_window,
    )
    run = _run(spec, q, k, v, scale)

    ref = _sdpa_reference(q, k, v, scale, window=sliding_window)
    max_abs = _max_abs(run.out, ref)
    assert max_abs < tol, (
        f"{dtype} D{d} GQA{hq}/{hkv} swa{sliding_window} scale={scale:g} "
        f"{'persist' if persistent else 'default'}: max_abs={max_abs:.3e} >= {tol}"
    )


def _sink_reference(q, k, v, sinks, scale, *, causal=True, sliding_window=0):
    """float64 ``softmax(concat([QK*scale, sink]))[..., :-1] @ V`` and the LSE of
    the concatenated row, the gfx950 sibling's oracle. The sink is one extra logit
    per query head with no value row, so it only grows the softmax denominator.
    Causal is top-left aligned (key ki visible to query qi iff ki <= qi), so a row
    whose window holds no key softmaxes over the sink alone and outputs 0. A row
    with neither a key nor a finite sink outputs 0 with LSE ``-inf``. Returns
    ``(O [B,Sq,Hq,D], LSE [B,Hq,Sq,1])``."""
    B, Sq, Hq, D = q.shape
    Skv = k.shape[1]
    rep = Hq // k.shape[2]
    qi = np.arange(Sq)[:, None]
    ki = np.arange(Skv)[None, :]
    masked = np.zeros((Sq, Skv), dtype=bool)
    if causal:
        masked |= ki > qi
    if sliding_window > 0:
        masked |= ki <= qi - sliding_window
    out = np.zeros((B, Sq, Hq, D))
    lse = np.empty((B, Hq, Sq))
    for b in range(B):
        for h in range(Hq):
            scores = (
                np.asarray(q[b, :, h], np.float64)
                @ np.asarray(k[b, :, h // rep], np.float64).T
                * scale
            )
            scores = np.where(masked, -np.inf, scores)
            sink = float(sinks[h])
            top = np.maximum(scores.max(axis=1), sink)
            top = np.where(np.isfinite(top), top, 0.0)
            weights = np.exp(scores - top[:, None])
            denom = weights.sum(axis=1) + np.exp(sink - top)
            with np.errstate(divide="ignore"):
                lse[b, h] = top + np.log(denom)
            out[b, :, h] = (weights / np.where(denom > 0, denom, 1.0)[:, None]) @ (
                np.asarray(v[b, :, h // rep], np.float64)
            )
    return out, lse[..., None]


_SINK_OFFSET = {"above_qk_max": 1.0, "below_qk_max": -1.0, "dominates_qk": 8.0}


def _check_sinks(
    dtype,
    d,
    hq,
    hkv,
    persistent,
    causal,
    sliding_window,
    magnitude,
    *,
    sq=512,
    sk=None,
    scale=None,
):
    """Run the shipped sink spec and compare against :func:`_sink_reference`.

    The sink is set relative to each head's max over the first (block_m x block_n)
    tile. One logit above: the seeded m0 stays the running max through tile 0, so
    alpha is 1 and the tile's p values carry the scaling. One below: tile 0 raises
    the max and the seeded l0 = 1 must be rescaled by alpha on the first tile. Both
    halves of the seed are exercised, which IR goldens cannot do. Eight above
    (``dominates_qk``): the sink takes nearly all of every row's softmax, so the
    non-vacuity check holds even with no causal mask."""
    tol = 2e-2 if dtype == "fp16" else 4e-2
    B, Sq, Skv = 1, sq, sq if sk is None else sk
    scale = 1.0 / math.sqrt(d) if scale is None else scale

    q, k, v = _qkv(0, dtype, hq, hkv, d, sq=Sq, sk=Skv, batch=B)
    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        persistent,
        causal=causal,
        batch=B,
        sq=Sq,
        sk=Skv,
        sliding_window=sliding_window,
        use_sinks=True,
    )

    bm, bn = spec.block_m, spec.block_n
    qt = np.swapaxes(q[:, :bm], 1, 2).astype(np.float64)
    kt = np.repeat(np.swapaxes(k[:, :bn], 1, 2).astype(np.float64), hq // hkv, axis=1)
    qk = np.einsum("bhqd,bhkd->bhqk", qt, kt) * scale
    if causal:
        qi = np.arange(bm)[:, None]
        ki = np.arange(bn)[None, :]
        qk = np.where((ki > qi)[None, None], -np.inf, qk)
    tile_max = qk.max(axis=(-1, -2))[0]  # [Hq]
    sinks = an.round_to(tile_max + _SINK_OFFSET[magnitude], dtype)

    run = _run(spec, q, k, v, scale, sinks=sinks)

    ref, _ = _sink_reference(
        q, k, v, sinks, scale, causal=causal, sliding_window=sliding_window
    )
    label = (
        f"{dtype} D{d} GQA{hq}/{hkv} {'causal' if causal else 'full'} "
        f"sq{Sq} sk{Skv} swa{sliding_window} scale={scale:g} "
        f"{'persist' if persistent else 'default'} {magnitude}"
    )
    # max() propagates NaN and NaN < tol is False, so a NaN row fails here too.
    max_abs = _max_abs(run.out, ref)
    assert max_abs < tol, f"{label}: max_abs={max_abs:.3e} >= {tol}"

    # Rows whose window holds no key (qi >= Skv + W - 1) see no key: the default grid runs a
    # zero-trip KV loop for them, the persistent grid one fully masked tile. With sinks they
    # softmax over the sink alone, so the output is exactly 0.
    empty_rows = 0
    if causal and sliding_window > 0 and Sq >= Skv + sliding_window:
        first_empty = Skv + sliding_window - 1
        tail = run.out[:, first_empty:]
        empty_rows = tail.shape[1]
        assert (
            np.count_nonzero(tail) == 0
        ), f"{label}: rows >= {first_empty} have no visible key but are not all 0"

    if causal or magnitude == "dominates_qk":
        # Non-vacuity: under causal, query row 0 sees only key 0, so the sink takes a
        # large share of its softmax; with dominates_qk it takes nearly all of every
        # row. Either way the sink reference must move well past tol, so a kernel
        # that ignored the sink would fail the parity assert above. Not asserted for
        # non-causal at +/-1: with 512 visible keys the sink's share can sit under
        # tol. Empty-window rows are 0 without a sink; they are checked above.
        ref_nosink, _ = _sink_reference(
            q,
            k,
            v,
            np.full_like(sinks, -np.inf),
            scale,
            causal=causal,
            sliding_window=sliding_window,
        )
        delta = _max_abs(ref, ref_nosink)
        assert delta > tol, f"{label}: sinks moved the reference only {delta:.3e}"

    return empty_rows


@pytest.mark.parametrize("sink_magnitude", ["above_qk_max", "below_qk_max"])
@pytest.mark.parametrize("dtype,d,hq,hkv,persistent,causal", _SINK_COHORT)
def test_dense_sinks_numeric_vs_fp32_reference(
    dtype, d, hq, hkv, persistent, causal, sink_magnitude
):
    """Full-sink numeric parity (no sliding window), both grids."""
    _check_sinks(dtype, d, hq, hkv, persistent, causal, 0, sink_magnitude)


@pytest.mark.parametrize(
    "dtype,d,hq,hkv,persistent,causal", [r for r in _SINK_COHORT if not r[-1]]
)
def test_dense_sinks_non_causal_sink_dominates(dtype, d, hq, hkv, persistent, causal):
    """Non-causal sink parity with a non-vacuity check: the sink sits eight logits
    above the QK max, so it takes nearly all of every row and a kernel that dropped
    it would miss the reference by far more than tol."""
    _check_sinks(dtype, d, hq, hkv, persistent, causal, 0, "dominates_qk")


# (dtype, head_size, num_query_heads, num_kv_heads, persistent, sq, sk,
# sliding_window): SWA-sink with Sq > Skv so the trailing query blocks' windows
# start past seqlen_kv. Those blocks have a zero-trip KV loop on the default grid;
# the sink seeds l = 1, so their rows are 0 (without sinks the epilogue masks them,
# see _SWA_KEYLESS_COHORT). The first empty row is Skv + W - 1 = 383; which blocks
# straddle it or lie fully past it follows from the dispatch tile.
_SWA_SINK_EMPTY_WINDOW_COHORT = [
    ("bf16", 128, 16, 4, False, 1024, 256, 128),
    ("bf16", 128, 16, 4, True, 1024, 256, 128),
    ("fp16", 128, 16, 4, False, 1024, 256, 128),
    ("fp16", 128, 16, 4, True, 1024, 256, 128),
    ("bf16", 64, 16, 4, False, 1024, 256, 128),
    ("bf16", 64, 16, 4, True, 1024, 256, 128),
]


@pytest.mark.parametrize("sink_magnitude", ["above_qk_max", "below_qk_max"])
@pytest.mark.parametrize(
    "dtype,d,hq,hkv,persistent,sq,sk,sliding_window", _SWA_SINK_EMPTY_WINDOW_COHORT
)
def test_dense_swa_sinks_empty_window_outputs_zero(
    dtype, d, hq, hkv, persistent, sq, sk, sliding_window, sink_magnitude
):
    """Empty-window rows output exactly 0 (softmax over the sink alone) and every
    other row keeps parity, on both grids."""
    empty_rows = _check_sinks(
        dtype,
        d,
        hq,
        hkv,
        persistent,
        True,
        sliding_window,
        sink_magnitude,
        sq=sq,
        sk=sk,
    )
    assert empty_rows > 0, "cohort row does not reach an empty window"


# (dtype, head_size, num_query_heads, num_kv_heads, persistent, sq, sk,
# sliding_window, block_m): sliding window WITHOUT sinks and WITHOUT LSE, with
# Sq > Skv so the trailing query rows see no key. The first keyless row is
# Skv + W - 1 (383 for Skv 256, W 128). ``block_m`` None leaves the tile to dispatch;
# 256 pins the wider 256-row tile, so both window loop shapes run. Which query blocks
# straddle the first keyless row, or lie fully past the KV extent (a zero-trip KV loop
# on the default grid), follows from the tile and Sq. The packaged catalog's window
# sizes are 128, 512, 1024 and 2048; the rows below cover those, 256, and 64 (whose
# first keyless row, Skv + 63, falls mid-block instead of at a multiple of 128).
_SWA_KEYLESS_COHORT = [
    ("bf16", 128, 16, 4, False, 512, 256, 128, None),
    ("bf16", 128, 16, 4, True, 512, 256, 128, None),
    ("bf16", 128, 16, 4, False, 1024, 256, 128, None),
    ("bf16", 128, 16, 4, True, 1024, 256, 128, None),
    ("fp16", 64, 16, 4, False, 1024, 256, 128, None),
    ("fp16", 64, 16, 4, True, 1024, 256, 128, None),
    ("bf16", 64, 16, 4, False, 1024, 256, 128, None),
    ("fp16", 128, 16, 4, True, 1024, 256, 128, None),
    ("fp16", 128, 16, 4, False, 1024, 256, 256, None),
    ("bf16", 128, 16, 4, True, 1024, 256, 256, None),
    ("bf16", 128, 16, 4, False, 2048, 512, 512, None),
    ("bf16", 128, 16, 4, True, 2048, 512, 512, None),
    ("bf16", 128, 16, 4, False, 4096, 1024, 1024, None),
    ("bf16", 128, 16, 4, True, 4096, 1024, 1024, None),
    ("bf16", 128, 16, 4, False, 6144, 2048, 2048, None),
    ("bf16", 128, 16, 4, True, 6144, 2048, 2048, None),
    ("bf16", 128, 16, 4, False, 512, 256, 64, None),
    ("bf16", 128, 16, 4, True, 512, 256, 64, None),
    ("fp16", 64, 16, 4, False, 512, 256, 64, None),
    ("fp16", 64, 16, 4, True, 512, 256, 64, None),
    ("bf16", 128, 16, 4, False, 512, 256, 128, 256),
    ("bf16", 128, 16, 4, True, 512, 256, 128, 256),
    ("bf16", 128, 16, 4, False, 1024, 256, 128, 256),
    ("bf16", 128, 16, 4, True, 1024, 256, 128, 256),
]


@pytest.mark.parametrize(
    "dtype,d,hq,hkv,persistent,sq,sk,sliding_window,block_m", _SWA_KEYLESS_COHORT
)
def test_dense_swa_keyless_rows_output_zero_without_lse(
    dtype, d, hq, hkv, persistent, sq, sk, sliding_window, block_m
):
    """A row with no visible key outputs exactly 0 whether or not the LSE side output
    is requested. Without the guard, such a row is the mean of V (every score equals
    the mask sentinel) or NaN (zero-trip KV loop: 0 * rcp(0)). Rows that do see keys
    keep parity, and the LSE-off output is bit-identical to LSE-on."""
    tol = _tolerance(dtype)
    scale = 1.0 / math.sqrt(d)
    q, k, v = _qkv(0, dtype, hq, hkv, d, sq=sq, sk=sk)

    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        persistent,
        sq=sq,
        sk=sk,
        sliding_window=sliding_window,
        block_m=block_m,
    )
    assert not spec.emit_lse and not spec.use_sinks
    if block_m is not None:
        assert spec.block_m == block_m, "the block_m pin did not change what is built"
    run = _run(spec, q, k, v, scale)

    ref, _ = _lse_reference(q, k, v, scale, causal=True, window=sliding_window)
    label = (
        f"{dtype} D{d} sq{sq} sk{sk} swa{sliding_window} bm{block_m or 'auto'} "
        f"{'persist' if persistent else 'default'}"
    )
    first_keyless = sk + sliding_window - 1
    tail = run.out[:, first_keyless:]
    assert tail.shape[1] > 0, f"{label}: cohort row has no keyless rows"
    assert (
        np.count_nonzero(tail) == 0
    ), f"{label}: rows >= {first_keyless} see no key but are not all 0"
    # max() propagates NaN and NaN < tol is False, so a NaN row fails here too.
    max_abs = _max_abs(run.out, ref)
    assert max_abs < tol, f"{label}: max_abs={max_abs:.3e} >= {tol}"

    on_spec = _lse_spec(
        dtype,
        d,
        hq,
        hkv,
        persistent,
        sq=sq,
        sk=sk,
        sliding_window=sliding_window,
        block_m=block_m,
    )
    on, off = _run_lse_and_off(on_spec, q, k, v, scale)
    assert np.array_equal(off.bits, on.bits), f"{label}: emit_lse changed the O output"


@pytest.mark.parametrize("persistent", [False, True], ids=["default", "persistent"])
def test_one_sliding_window_binary_serves_keyless_and_keyed_shapes(persistent):
    """One runtime-shape binary serves shapes with and without keyless rows, at batch 1
    and 2, on both grids. The row predicate reads the runtime seqlen_kv, so a shape with
    keyless rows must not change what a keyed shape writes (or the reverse) through the
    same launcher."""
    from kernels.common.attention_dense_spec import attention_dense_cache_key

    dtype, d, hq, hkv, window = "bf16", 128, 16, 4, 128
    tol = _tolerance(dtype)
    scale = 1.0 / math.sqrt(d)
    base = _as_gfx942_spec(
        _spec(dtype, d, hq, hkv, persistent, sq=1024, sliding_window=window)
    )
    # (batch, sq, sk): keyed, keyless, keyless at batch 2, keyless with a longer KV,
    # keyed again.
    shapes = (
        (1, 1024, 1024),
        (1, 1024, 256),
        (2, 512, 256),
        (2, 2048, 512),
        (1, 1024, 1024),
    )
    specs = [
        dataclasses.replace(base, batch=b, seqlen_q=sq, seqlen_kv=sk)
        for b, sq, sk in shapes
    ]
    assert all(s.runtime_shape for s in specs), "a window config left the runtime path"
    keys = {attention_dense_cache_key(s, arch="gfx942") for s in specs}
    assert len(keys) == 1, f"shapes {shapes} did not share a cache key"

    launchers = []
    for (b, sq, sk), spec in zip(shapes, specs):
        q, k, v = _qkv(0, dtype, hq, hkv, d, sq=sq, sk=sk, batch=b)
        run = _run(spec, q, k, v, scale)
        launchers.append(_launcher_for(spec))

        label = f"B{b} sq{sq} sk{sk}"
        ref, _ = _lse_reference(q, k, v, scale, causal=True, window=window)
        max_abs = _max_abs(run.out, ref)
        assert max_abs < tol, f"{label}: max_abs={max_abs:.3e} >= {tol}"
        first_keyless = sk + window - 1
        if first_keyless < sq:
            tail = run.out[:, first_keyless:]
            assert np.count_nonzero(tail) == 0, f"{label}: keyless rows != 0"
    assert launchers[0] is not None, (
        "no launcher cached after a successful run; _DENSE_LAUNCHER_CACHE is no "
        "longer keyed by attention_dense_cache_key and this test is blind"
    )
    assert all(
        lch is launchers[0] for lch in launchers
    ), "shapes did not share a binary"


@pytest.mark.parametrize("sink_magnitude", ["above_qk_max", "below_qk_max"])
@pytest.mark.parametrize(
    "dtype,d,hq,hkv,persistent,sliding_window,scale", _SWA_SINK_COHORT
)
def test_dense_swa_sinks_numeric_vs_fp32_reference(
    dtype, d, hq, hkv, persistent, sliding_window, scale, sink_magnitude
):
    """SWA-sink numeric parity: the window prune and the sink seed together, on the
    same rows as the standalone SWA cohort. Always causal (SWA requires it)."""
    _check_sinks(
        dtype,
        d,
        hq,
        hkv,
        persistent,
        True,
        sliding_window,
        sink_magnitude,
        scale=scale,
    )


def test_one_sliding_window_binary_serves_every_shape_and_head_config():
    """The sliding-window twin of :func:`test_one_binary_serves_every_head_config`.

    Non-persistent SWA is on the runtime-shape path: the window stays baked, while
    batch, both seqlens and both head counts are kernel params. Varying all of them
    at once under one window must reuse one launcher, and the band must still be
    right at every config -- a prune or mask that kept a baked shape term would pass
    the first config compiled and fail the rest.
    """
    from kernels.common.attention_dense_spec import attention_dense_cache_key
    from kernels.gfx942.attention_dense import _DENSE_LAUNCHER_CACHE

    dtype, d, window = "bf16", 128, 128
    tol = 4e-2
    scale = 1.0 / math.sqrt(d)

    # (batch, seqlen, Hq, Hkv): GQA 16, non-pow2 GQA 5 and 7, MHA.
    configs = ((1, 512, 128, 8), (2, 1024, 40, 8), (1, 2048, 28, 4), (4, 512, 32, 32))
    specs = [
        _as_gfx942_spec(
            _spec(dtype, d, hq, hkv, False, batch=b, sq=s, sliding_window=window)
        )
        for b, s, hq, hkv in configs
    ]

    assert all(s.runtime_shape for s in specs), "an SWA config left the runtime path"
    keys = [attention_dense_cache_key(s, arch="gfx942") for s in specs]
    assert len(set(keys)) == 1, f"SWA configs {configs} did not share a cache key"

    _DENSE_LAUNCHER_CACHE.pop(keys[0], None)
    before = set(_DENSE_LAUNCHER_CACHE)

    launchers = []
    for (B, S, hq, hkv), spec in zip(configs, specs):
        q, k, v = _qkv(0, dtype, hq, hkv, d, sq=S, batch=B)
        run = _run(spec, q, k, v, scale)
        launchers.append(_launcher_for(spec))

        ref = _sdpa_reference(q, k, v, scale, window=window)
        max_abs = _max_abs(run.out, ref)
        assert max_abs < tol, (
            f"swa{window} B={B} S={S} Hq={hq} Hkv={hkv}: "
            f"max_abs={max_abs:.3e} >= {tol}"
        )

    assert launchers[0] is not None, (
        "no launcher cached after a successful run; _DENSE_LAUNCHER_CACHE is no "
        "longer keyed by attention_dense_cache_key and this test is blind"
    )
    assert all(lau is launchers[0] for lau in launchers), (
        f"SWA configs {configs} share a cache key but were served by DIFFERENT "
        "launcher objects -- the runtime-shape SWA kernel recompiled per config"
    )
    assert set(_DENSE_LAUNCHER_CACHE) - before == {keys[0]}, (
        "four SWA configs on the runtime path added more than one cache entry: "
        f"{sorted(set(_DENSE_LAUNCHER_CACHE) - before)}"
    )


# (batch, seqlen, Hq, Hkv) served by ONE persistent binary. W = NQB*Hq*B is the
# grid-stride work-item count (NQB = seqlen // 256); NP is the shipped 304:
#   W=40   < NP, NQB=1 (most CTAs idle; the interleave flip degenerates to qb=0)
#   W=320  in (NP, 2*NP), W % NP = 16
#   W=672  > 2*NP, W % NP = 64, odd B (non-pow2 batch magic)
#   W=384  MHA (gqa=1), W % NP = 80
#   W=608  == 2*NP exactly -- the tail-free control
#   W=120  odd Hq, even NQB=4: an interleave flip keyed on rem = qb0*Hq + hq
#          (instead of hq) maps qb0 {0,1,2,3} -> {0,2,2,0}, skipping tiles
# Non-pow2 GQA 5 and 7 put a real magic on the gqa divide.
_PERSIST_CONFIGS = (
    (1, 256, 40, 8),
    (2, 1024, 40, 8),
    (3, 2048, 28, 4),
    (2, 1536, 32, 32),
    (19, 512, 16, 4),
    (2, 1024, 15, 5),
)

# hq_minor_swz needs Hq divisible by the XCD count and an even NQB under causal
# masking, so it gets its own set, straddling NP the same way:
#   W=80 < NP; W=320 and W=608 (pair fold, even NQB); MHA W=384; W=1536, W % NP = 16
_SWZ_CONFIGS = (
    (1, 512, 40, 8),
    (2, 1024, 40, 8),
    (2, 1536, 32, 32),
    (19, 512, 16, 4),
    (3, 2048, 64, 8),
)

# (dtype, persist_decode, interleave, sliding_window, configs). Every decode order
# pinned explicitly: "auto" resolves per shape, which would split the set across
# keys. interleave only changes the qb_major body, so it rides that order alone. A
# window of 1024 is at least as long as the keys for some configs and shorter for
# others, so one binary balances causal work for some launches and not for others.
_PERSIST_ONE_BINARY = [
    pytest.param("fp16", "hkv_major", False, 0, _PERSIST_CONFIGS, id="hkv_major"),
    pytest.param("fp16", "qb_major", False, 0, _PERSIST_CONFIGS, id="qb_major"),
    pytest.param(
        "fp16", "qb_major", True, 0, _PERSIST_CONFIGS, id="qb_major-interleave"
    ),
    pytest.param("fp16", "bt_hkv_minor", False, 0, _PERSIST_CONFIGS, id="bt_hkv_minor"),
    pytest.param("fp16", "hq_minor_swz", False, 0, _SWZ_CONFIGS, id="hq_minor_swz"),
    pytest.param(
        "bf16", "hkv_major", False, 128, _PERSIST_CONFIGS, id="hkv_major-swa128"
    ),
    pytest.param(
        "bf16", "qb_major", False, 128, _PERSIST_CONFIGS, id="qb_major-swa128"
    ),
    *(
        pytest.param(
            "bf16",
            name,
            False,
            1024,
            _SWZ_CONFIGS if name == "hq_minor_swz" else _PERSIST_CONFIGS,
            id=f"{name}-swa1024",
        )
        for name in ("qb_major", "hkv_major", "bt_hkv_minor", "hq_minor_swz")
    ),
]


@pytest.mark.parametrize(
    "dtype,persist_decode,interleave,window,configs", _PERSIST_ONE_BINARY
)
def test_one_persistent_binary_serves_every_shape_and_head_config(
    dtype, persist_decode, interleave, window, configs
):
    """The persistent-grid twin of the one-binary tests above.

    The persistent grid takes batch, both seqlens and both head counts as kernel
    params too; only D, dtype, mask, window, ``num_persistent`` and the decode order
    stay baked. Its work-item count W is derived in the prologue and the grid-stride
    decode divides by runtime B / Hq / gqa / NQB through host-computed fast-division
    magics. A decode that kept a baked shape term, or a magic computed for the wrong
    divisor, would map work items to the wrong (qb, hq, bt) for every config but the
    first compiled -- duplicated or skipped tiles that only the numeric check sees.
    The configs straddle NP (W < NP, NP < W < 2*NP, W > 2*NP, W % NP != 0) so the
    grid-stride bound is exercised with idle CTAs, ragged tails and multiple trips.
    """
    from kernels.common.attention_dense_spec import attention_dense_cache_key
    from kernels.gfx942.attention_dense import _DENSE_LAUNCHER_CACHE

    d = 128
    tol = 2e-2 if dtype == "fp16" else 4e-2
    scale = 1.0 / math.sqrt(d)

    specs = [
        dataclasses.replace(
            _as_gfx942_spec(
                _spec(
                    dtype,
                    d,
                    hq,
                    hkv,
                    True,
                    batch=b,
                    sq=s,
                    sliding_window=window,
                    persist_decode=persist_decode,
                )
            ),
            interleave=interleave,
            # One query tile for every config: the default picks a 128-row tile
            # for under-filled shapes, which is a different binary.
            block_m=256,
        )
        for b, s, hq, hkv in configs
    ]

    # Preconditions: persistent, on the runtime path, the pinned order, one key --
    # and the configs really straddle NP, or the grid-stride claims are vacuous.
    assert all(
        s.persistent and s.runtime_shape for s in specs
    ), "a config left the persistent runtime-shape path"
    assert {s.resolved_persist_decode for s in specs} == {persist_decode}
    assert all(s.fastdiv_divisors for s in specs), "persistent decode has no fastdiv"
    np_ = specs[0].num_persistent
    work = [(s.seqlen_q // s.block_m) * s.num_query_heads * s.batch for s in specs]
    assert min(work) < np_ and max(work) > 2 * np_, (work, np_)
    assert any(w % np_ for w in work), (work, np_)
    keys = [attention_dense_cache_key(s, arch="gfx942") for s in specs]
    assert (
        len(set(keys)) == 1
    ), f"persistent configs {configs} did not share a cache key"

    _DENSE_LAUNCHER_CACHE.pop(keys[0], None)
    before = set(_DENSE_LAUNCHER_CACHE)

    launchers = []
    for (B, S, hq, hkv), spec in zip(configs, specs):
        q, k, v = _qkv(0, dtype, hq, hkv, d, sq=S, batch=B)
        # The output starts NaN so a work item the decode skips cannot pass as
        # stale zeros.
        run = _run(spec, q, k, v, scale)
        launchers.append(_launcher_for(spec))

        ref = _sdpa_reference(q, k, v, scale, window=window)
        max_abs = _max_abs(run.out, ref)
        assert max_abs < tol, (
            f"persist {persist_decode}{'+intl' if interleave else ''} "
            f"swa{window} B={B} S={S} Hq={hq} Hkv={hkv}: "
            f"max_abs={max_abs:.3e} >= {tol}"
        )

    assert launchers[0] is not None, (
        "no launcher cached after a successful run; _DENSE_LAUNCHER_CACHE is no "
        "longer keyed by attention_dense_cache_key and this test is blind"
    )
    assert all(lau is launchers[0] for lau in launchers), (
        f"persistent configs {configs} share a cache key but were served "
        "by DIFFERENT launcher objects -- the persistent kernel recompiled per config"
    )
    assert set(_DENSE_LAUNCHER_CACHE) - before == {keys[0]}, (
        "persistent configs on the runtime path added more than one cache entry: "
        f"{sorted(set(_DENSE_LAUNCHER_CACHE) - before)}"
    )


@pytest.mark.parametrize("block_n", [32])
def test_dense_fp16_d128_numeric_correct_at_non_shipped_tile_width(block_n):
    """fp16-D128 stays numerically correct at a tile width dispatch never emits
    (``_DENSE_BLOCK_N = 64`` is a hard constant; block_n=32 is the small-tile double-K
    sweep direction). This does NOT prove the swizzle is engaged -- store and read
    apply the same permutation, so an off build is bit-identical here; the IR guard
    ``test_cfvst_swizzle_is_emitted_in_ir_with_matching_store_read_mask`` (CPU lane)
    covers that. This is the on-silicon correctness guard for the tile-width axis."""
    d, hq, hkv, tol = 128, 16, 4, 2e-2
    B, S, scale = 1, 512, 1.0 / math.sqrt(128)
    q, k, v = _qkv(0, "fp16", hq, hkv, d, sq=S, batch=B)
    spec = dataclasses.replace(
        _spec("fp16", d, hq, hkv, False, batch=B, sq=S), block_n=block_n
    )
    run = _run(spec, q, k, v, scale)
    max_abs = _max_abs(run.out, _sdpa_reference(q, k, v, scale))
    assert max_abs < tol, f"fp16 D128 block_n={block_n}: max_abs={max_abs:.3e} >= {tol}"


# bf16 D128 is the config this PR flips to exp2_fast, on BOTH grids. The oracle
# test above (tol=4e-2 vs fp32 SDPA) is far too loose to tell the two exp2 arms
# apart, so it cannot back the "results unchanged" claim. This does: a forced-off
# vs forced-on A/B on one identical spec must agree for every reachable softmax
# argument (both softmax args are always <= 0, exactly exp2_fast's precondition).
_EXP2_AB_COHORT = [
    ("bf16", 128, 16, 4, False),  # default grid (newly enabled here)
    ("bf16", 128, 16, 4, True),  # persistent grid
]


@pytest.mark.parametrize("dtype,d,hq,hkv,persistent", _EXP2_AB_COHORT)
def test_exp2_fast_matches_plain_exp2(dtype, d, hq, hkv, persistent):
    B, S = 1, 512
    scale = 1.0 / math.sqrt(d)
    q, k, v = _qkv(0, dtype, hq, hkv, d, sq=S, batch=B)
    spec = _spec(dtype, d, hq, hkv, persistent, batch=B, sq=S)

    # Forced-off vs forced-on compile to distinct binaries (gfx942_kernel_name
    # tags the non-policy arm `e2f0`), so this really exercises both code paths.
    runs = {}
    for use_fast in (False, True):
        # The dispatch factory hands back the SHARED spec; promote it to the gfx942
        # subclass (at shipped defaults for every other private knob) so only
        # use_exp2_fast moves between the two arms.
        runs[use_fast] = _run(
            dataclasses.replace(_as_gfx942_spec(spec), use_exp2_fast=use_fast),
            q,
            k,
            v,
            scale,
        )

    max_abs = _max_abs(runs[False].out, runs[True].out)
    assert np.array_equal(runs[False].bits, runs[True].bits), (
        f"{dtype} D{d} {'persist' if persistent else 'default'}: exp2_fast "
        f"diverged from plain exp2 (max_abs={max_abs:.3e})"
    )


# Performance-only codegen knobs. None of them changes arithmetic (PV order keeps
# each output tile's key-step order, a width only splits the same stores, the
# diagonal split skips selects whose condition is always true), so every override
# must be BIT-identical to the shipped default. Sq=512 gives a second query block,
# whose below-diagonal tiles are what the split leaves unmasked.
_KNOB_AB_COHORT = [
    ("fp16", 128, False, dict(causal_diag_split=True)),
    ("fp16", 128, True, dict(causal_diag_split=True)),
    ("bf16", 64, False, dict(causal_diag_split=True)),
    ("fp16", 128, False, dict(pv_loop_order="k_major")),
    ("bf16", 128, True, dict(pv_loop_order="k_major")),
    ("bf16", 128, False, dict(o_store_width=1)),
    ("bf16", 64, True, dict(o_store_width=2)),
    ("fp16", 128, False, dict(pv_priority=2, pv_sched_fence_mask=0)),
    ("fp16", 128, False, dict(iglp=True, iglp_mode=1)),
]


def _tolerance(dtype):
    return 2e-2 if dtype == "fp16" else 4e-2


def _lse_reference(q, k, v, scale, *, causal, window=0, sinks=None):
    """Independent float64 O and natural-log LSE, including truly empty rows.

    Scores use the rounded Q/K with the softmax scale applied in float64. ``sinks``
    adds one logit per query head with no value row, so a row with no key has the
    sink as its LSE. Returns ``(O [B,Sq,Hq,D], LSE [B,Hq,Sq,1])``.
    """
    if sinks is not None:
        return _sink_reference(
            q, k, v, sinks, scale, causal=causal, sliding_window=window
        )
    out, lse = sdpa_reference(
        np.swapaxes(q, 1, 2),
        np.swapaxes(k, 1, 2),
        np.swapaxes(v, 1, 2),
        scale=scale,
        left_bound=window - 1 if window else None,
        right_bound=0 if causal else None,
    )
    return np.swapaxes(out, 1, 2), lse[..., None]


def _lse_spec(dtype, d, hq, hkv, persistent, **kwargs):
    """The shipped dispatch spec for the row, with the LSE output enabled."""
    return dataclasses.replace(
        _as_gfx942_spec(_spec(dtype, d, hq, hkv, persistent, **kwargs)), emit_lse=True
    )


def _run_lse_and_off(spec, q, k, v, scale):
    """Launch ``spec`` with LSE and its LSE-off twin on the same inputs."""
    on = _run(spec, q, k, v, scale)
    off = _run(dataclasses.replace(spec, emit_lse=False), q, k, v, scale)
    return on, off


@pytest.mark.parametrize("dtype,d,persistent,overrides", _KNOB_AB_COHORT)
def test_codegen_knob_is_bit_identical_to_default(dtype, d, persistent, overrides):
    hq, hkv = 16, 4
    B, S = 1, 512
    scale = 1.0 / math.sqrt(d)
    q, k, v = _qkv(0, dtype, hq, hkv, d, sq=S, batch=B)
    base = _as_gfx942_spec(_spec(dtype, d, hq, hkv, persistent, batch=B, sq=S))

    runs = [
        _run(spec, q, k, v, scale)
        for spec in (base, dataclasses.replace(base, **overrides))
    ]

    max_abs = _max_abs(runs[0].out, runs[1].out)
    assert np.array_equal(runs[0].bits, runs[1].bits), (
        f"{dtype} D{d} {'persist' if persistent else 'default'} {overrides}: "
        f"diverged from the shipped default (max_abs={max_abs:.3e})"
    )


# Persistent grid size for independent CFVST, KDB and VPF tests. At the shipped
# NP (304) every
# shape below has fewer work items than workgroups, so no workgroup ever runs a
# second item and the LDS hand-off between items (the next item's prologue writing
# K slot 0 and V_lds after the previous item's epilogue) is never exercised.
_MULTI_ITEM_NP = 3


def _multi_item_persistent(spec):
    """``spec`` on a small persistent grid: more than two work items per workgroup
    and a ragged last round, the same preconditions as
    ``test_one_persistent_binary_serves_every_shape_and_head_config``."""
    spec = dataclasses.replace(spec, num_persistent=_MULTI_ITEM_NP)
    work = (spec.seqlen_q // spec.block_m) * spec.num_query_heads * spec.batch
    assert work > 2 * spec.num_persistent, (work, spec.num_persistent)
    assert work % spec.num_persistent, (work, spec.num_persistent)
    return spec


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize(
    "sq,skv",
    [
        (256, 64),  # n_up=1: zero-trip pipeline main loop
        (256, 192),  # n_up=3: the loop runs, then the last tile is in K slot 0
        (512, 512),  # Multiple KV tiles (even n_up: last tile in K slot 1)
    ],
)
def test_cfvst_pipeline_strict_numeric(dtype, persistent, sq, skv):
    B, Hq, Hkv, D = 1, 16, 4, 128
    tol = 1e-2 if dtype == "bf16" else 2e-3
    scale = 1.0 / math.sqrt(D)

    q, k, v = _qkv(42, dtype, Hq, Hkv, D, sq=sq, sk=skv, batch=B)

    # Global top-left causal mask, including non-square Sq/Skv.
    ref = _sdpa_reference(q, k, v, scale)

    base = _as_gfx942_spec(_spec(dtype, D, Hq, Hkv, persistent, batch=B, sq=sq))

    # BF16 CFVST alone can run on either grid; KDB/VPF require persistent.
    if dtype == "bf16":
        variants = [
            {"use_cfvst": False, "lds_num_buffers": 1, "v_reg_prefetch": False},
            {"use_cfvst": True, "lds_num_buffers": 1, "v_reg_prefetch": False},
        ]
        if persistent:
            variants.extend(
                [
                    {"use_cfvst": True, "lds_num_buffers": 2, "v_reg_prefetch": False},
                    {"use_cfvst": True, "lds_num_buffers": 1, "v_reg_prefetch": True},
                    {"use_cfvst": True, "lds_num_buffers": 2, "v_reg_prefetch": True},
                ]
            )
    else:
        variants = [{}]  # Preserve the existing FP16 path.

    for knobs in variants:
        spec = dataclasses.replace(base, seqlen_kv=skv, **knobs)
        if persistent:
            spec = _multi_item_persistent(spec)

        run = _run(spec, q, k, v, scale)

        assert np.isfinite(run.out).all(), (
            f"Non-finite output: {dtype}, {knobs}, "
            f"persistent={persistent}, Sq={sq}, Skv={skv}"
        )

        _assert_allclose(
            run.out,
            ref,
            atol=tol,
            rtol=tol,
            what=f"{dtype} {knobs} persistent={persistent} Sq={sq} Skv={skv}",
        )


# Representative BF16 D128 CFVST numeric tests covering tile, scheduling,
# output-store, and persistent-grid configurations with independently enabled
# K double buffering and V register prefetch. Not an exhaustive sweep.
# Each row is (id, (Hq, Hkv), Skv, persistent, knobs).
_CFVST_KNOB_ROWS = [
    ("bn32", (16, 4), 512, False, {"block_n": 32}),
    ("bm128", (16, 4), 512, False, {"block_m": 128}),
    # block_m=64 -> n_up = 1..8 across query blocks, so odd n_up > 1 is covered.
    ("bm64", (16, 4), 512, False, {"block_m": 64}),
    ("iglp0", (16, 4), 512, False, {"iglp": True}),
    ("iglp1", (16, 4), 512, False, {"iglp": True, "iglp_mode": 1}),
    ("kmajor", (16, 4), 512, False, {"pv_loop_order": "k_major"}),
    ("ostore1", (16, 4), 512, False, {"o_store_width": 1}),
    ("ostore2", (16, 4), 512, False, {"o_store_width": 2}),
    ("pvprio", (16, 4), 512, False, {"pv_priority": 1}),
    ("pvfence", (16, 4), 512, False, {"pv_sched_fence_mask": 0}),
    ("persist_intl", (16, 4), 512, True, {"interleave": True}),
    ("persist_qbmaj", (16, 4), 512, True, {"persist_decode": "qb_major"}),
    ("persist_hkvmaj", (16, 4), 512, True, {"persist_decode": "hkv_major"}),
    ("skv_gt_sq", (16, 4), 1024, False, {}),
    ("mha", (16, 16), 512, False, {}),
    ("gqa8", (32, 4), 512, False, {}),
]


# KDB/VPF require BOTH persistent=True and block_n=64. Keep the CFVST-only
# coverage on either grid; never generate an illegal positive pipeline test.
_CFVST_PIPELINE_MODES = (
    (1, False, "cfvst_only"),
    (2, False, "kdb_only"),
    (1, True, "vpf_only"),
    (2, True, "kdb_vpf"),
)
_CFVST_KNOB_CASES = [
    pytest.param(*row[1:], nbuf, vpf, id=f"{row[0]}-{mode}")
    for row in _CFVST_KNOB_ROWS
    for nbuf, vpf, mode in _CFVST_PIPELINE_MODES
    if (nbuf == 1 and not vpf) or (row[3] and row[4].get("block_n", 64) == 64)
]


@pytest.mark.parametrize(
    "heads,skv,persistent,knobs,lds_num_buffers,v_reg_prefetch",
    _CFVST_KNOB_CASES,
)
def test_cfvst_knob_matrix_numeric(
    heads, skv, persistent, knobs, lds_num_buffers, v_reg_prefetch
):
    hq, hkv = heads
    sq, d = 512, 128
    scale = 1.0 / math.sqrt(d)
    q, k, v = _qkv(5, "bf16", hq, hkv, d, sq=sq, sk=skv)
    ref = _sdpa_reference(q, k, v, scale)

    base = _as_gfx942_spec(_spec("bf16", d, hq, hkv, persistent, batch=1, sq=sq))
    spec = dataclasses.replace(
        base,
        seqlen_kv=skv,
        use_cfvst=True,
        lds_num_buffers=lds_num_buffers,
        v_reg_prefetch=v_reg_prefetch,
        **knobs,
    )
    if persistent:
        spec = _multi_item_persistent(spec)
    ok, why = supports_attention_dense(spec, arch="gfx942")
    assert ok, f"gate no longer accepts a row it used to open: {why}"
    run = _run(spec, q, k, v, scale)
    _assert_allclose(run.out, ref, atol=1e-2, rtol=1e-2, what=f"{knobs}")
    # KDB and VPF are performance-only changes. Compare their outputs
    # against the same CFVST kernel with both optimizations disabled.
    if lds_num_buffers == 2 or v_reg_prefetch:
        baseline_spec = dataclasses.replace(
            spec,
            lds_num_buffers=1,
            v_reg_prefetch=False,
        )

        baseline = _run(baseline_spec, q, k, v, scale)

        assert np.array_equal(run.bits, baseline.bits), (
            f"KDB/VPF changed BF16 output bits: "
            f"{knobs}, nbuf={lds_num_buffers}, "
            f"vpf={v_reg_prefetch}"
        )


class _HostTensor:
    """A tensor that is not on the GPU: the launch must refuse it."""

    is_cuda = False
    device = "cpu"

    def __init__(self, shape, torch_dtype):
        self.shape = tuple(shape)
        self.dtype = torch_dtype

    def data_ptr(self):
        return 0

    def stride(self, dim=None):
        strides = tuple(
            int(np.prod(self.shape[i + 1 :])) for i in range(len(self.shape))
        )
        return strides if dim is None else strides[dim]

    def numel(self):
        return int(np.prod(self.shape))

    def element_size(self):
        return 2

    def is_contiguous(self):
        return True


class _Alias:
    """A float32 view of another device buffer's memory (same address, new shape)."""

    is_cuda = True
    device = None
    dtype = "torch.float32"

    def __init__(self, base, shape):
        self._base = base
        self.shape = tuple(shape)

    def data_ptr(self):
        return self._base.data_ptr()

    def stride(self, dim=None):
        strides = tuple(
            int(np.prod(self.shape[i + 1 :])) for i in range(len(self.shape))
        )
        return strides if dim is None else strides[dim]

    def numel(self):
        return int(np.prod(self.shape))

    def element_size(self):
        return 4

    def is_contiguous(self):
        return True


class TestDenseLseNumericGfx942:
    """Direct-runner LSE coverage on gfx942; no dispatcher or hipDNN feature implied.

    Every random-input row also requires O to be bitwise identical to the LSE-off
    kernel: LSE must not perturb O, keyless rows included (test_narrow_o_store runs
    that comparison on a keyless shape).
    """

    @ci_subset(cover=("heads", "persistent", "causal"), rotate=("dtype", "d"))
    @pytest.mark.parametrize("dtype", ["fp16", "bf16"])
    @pytest.mark.parametrize("d", [64, 128])
    @pytest.mark.parametrize("heads", [(8, 8), (8, 2)], ids=["mha", "gqa"])
    @pytest.mark.parametrize(
        "persistent", [False, True], ids=["ordinary", "persistent"]
    )
    @pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
    def test_output_and_lse(self, dtype, d, heads, persistent, causal):
        hq, hkv = heads
        spec = _lse_spec(dtype, d, hq, hkv, persistent, causal=causal, batch=2)
        scale = 0.5
        q, k, v = _qkv(11, dtype, hq, hkv, d, sq=512, batch=2)
        on, off = _run_lse_and_off(spec, q, k, v, scale)
        ref, lse_ref = _lse_reference(q, k, v, scale, causal=causal)
        assert _max_abs(on.out, ref) < _tolerance(dtype)
        assert _max_abs(on.lse, lse_ref) < _tolerance(dtype)
        assert np.array_equal(on.bits, off.bits)

    @pytest.mark.parametrize(
        "sq,skv", [(256, 64), (512, 512)], ids=["zero_trip", "multi_tile"]
    )
    @pytest.mark.parametrize(
        "persistent,lds_num_buffers,v_reg_prefetch",
        [
            (False, 1, False),
            (True, 1, False),
            (True, 2, False),
            (True, 1, True),
            (True, 2, True),
        ],
        ids=[
            "ordinary_cfvst",
            "persistent_cfvst",
            "persistent_kdb",
            "persistent_vpf",
            "persistent_kdb_vpf",
        ],
    )
    def test_cfvst_pipeline_lse_modes(
        self, persistent, sq, skv, lds_num_buffers, v_reg_prefetch
    ):
        """LSE uses the final tile's row max in all independent pipeline modes.

        A stale max shifts LSE by (m_new - m_old) * ln2; at n_up=1
        (zero_trip) the pipeline loop does not run at all. Both grids are
        exercised, including the persistent multi-work-item handoff.
        """
        spec = dataclasses.replace(
            _lse_spec("bf16", 128, 8, 2, persistent, batch=1, sq=512),
            seqlen_q=sq,
            seqlen_kv=skv,
            use_cfvst=True,
            lds_num_buffers=lds_num_buffers,
            v_reg_prefetch=v_reg_prefetch,
        )
        if persistent:
            spec = _multi_item_persistent(spec)
        scale = 1.0 / math.sqrt(128)
        q, k, v = _qkv(7, "bf16", 8, 2, 128, sq=sq, sk=skv)
        on, off = _run_lse_and_off(spec, q, k, v, scale)
        ref, lse_ref = _lse_reference(q, k, v, scale, causal=True)
        assert _max_abs(on.out, ref) < _tolerance("bf16")
        assert _max_abs(on.lse, lse_ref) < 1e-3
        assert np.array_equal(on.bits, off.bits)

    @pytest.mark.parametrize("dtype", ["fp16", "bf16"])
    @pytest.mark.parametrize("d", [64, 128])
    @pytest.mark.parametrize(
        "persistent", [False, True], ids=["ordinary", "persistent"]
    )
    def test_sliding_window_lse(self, dtype, d, persistent):
        spec = _lse_spec(dtype, d, 8, 2, persistent, sliding_window=128)
        scale = 0.5
        q, k, v = _qkv(13, dtype, 8, 2, d, sq=512)
        on, off = _run_lse_and_off(spec, q, k, v, scale)
        ref, lse_ref = _lse_reference(q, k, v, scale, causal=True, window=128)
        assert _max_abs(on.out, ref) < _tolerance(dtype)
        assert _max_abs(on.lse, lse_ref) < _tolerance(dtype)
        assert np.array_equal(on.bits, off.bits)

    @pytest.mark.parametrize("width", [1, 2])
    @pytest.mark.parametrize(
        "persistent", [False, True], ids=["ordinary", "persistent"]
    )
    def test_narrow_o_store(self, width, persistent):
        """BF16 store widths 1/2: keyless rows are zero, keyed rows match the
        reference, and O is bitwise the LSE-off kernel's at the same width on the
        same keyless shape."""
        from dispatch.attention import AttentionRequest, tuning_spec_with_knobs

        def lse_spec(seqlen_k):
            return dataclasses.replace(
                tuning_spec_with_knobs(
                    AttentionRequest(
                        batch=1,
                        nhead_q=4,
                        nhead_k=1,
                        seqlen_q=256,
                        seqlen_k=seqlen_k,
                        hdim_q=128,
                        hdim_v=128,
                        arch="gfx942",
                        mask_type=1,
                        dtype="bf16",
                        sliding_window=64,
                        emit_lse=True,
                    ),
                    "gfx942_dense",
                    {"persistent": bool(persistent), "o_store_width": width},
                ).kernel_spec,
                emit_lse=True,
            )

        spec = lse_spec(64)
        assert spec.o_store_width == width
        scale = 0.5
        q, k, v = _qkv(17, "bf16", 4, 1, 128, sq=256, sk=64)
        run = _run(spec, q, k, v, scale)
        out, lse = run.out, run.lse
        ref, lse_ref = _lse_reference(q, k, v, scale, causal=True, window=64)
        # Rows 0..126 see at least one key; rows 127.. see none.
        assert _max_abs(out[:, :127], ref[:, :127]) < _tolerance("bf16")
        assert not out[:, 127:].any()
        finite = np.isfinite(lse_ref)
        assert _max_abs(lse[finite], lse_ref[finite]) < _tolerance("bf16")
        assert np.array_equal(np.isneginf(lse), np.isneginf(lse_ref))

        # LSE off on the same keyless shape: the width-1/2 store must write O exactly
        # as the LSE-on kernel does, keyless rows included.
        on, off = _run_lse_and_off(spec, q, k, v, scale)
        assert np.array_equal(on.bits, run.bits)
        assert np.array_equal(off.bits, run.bits)

    @pytest.mark.parametrize("dtype", ["fp16", "bf16"])
    @pytest.mark.parametrize(
        "persistent", [False, True], ids=["ordinary", "persistent"]
    )
    @pytest.mark.parametrize("with_sink", [False, True], ids=["no-sink", "sink"])
    def test_empty_window_rows(self, dtype, persistent, with_sink):
        """Sq256/Skv64/window64: rows 63/126/127 see 64/1/0 keys (analytic LSE).

        With sinks a row with no key has the sink as its only mass, so its LSE is
        the sink logit rather than -inf."""
        from dispatch.attention import AttentionRequest, tuning_spec_with_knobs

        spec = dataclasses.replace(
            tuning_spec_with_knobs(
                AttentionRequest(
                    batch=1,
                    nhead_q=4,
                    nhead_k=1,
                    seqlen_q=256,
                    seqlen_k=64,
                    hdim_q=128,
                    hdim_v=128,
                    arch="gfx942",
                    mask_type=1,
                    dtype=dtype,
                    sliding_window=64,
                    use_sinks=with_sink,
                    emit_lse=True,
                ),
                "gfx942_dense",
                {"persistent": bool(persistent)},
            ).kernel_spec,
            emit_lse=True,
        )
        q = np.zeros((1, 256, 4, 128), np.float32)
        k = np.zeros((1, 64, 1, 128), np.float32)
        v = np.ones_like(k)
        sinks = np.array([-2.0, 0.0, 1.0, 2.0], np.float32) if with_sink else None
        run = _run(spec, q, k, v, 0.5, sinks=sinks)
        out, lse = run.out, run.lse
        rows = np.arange(256)
        counts = np.clip(
            np.minimum(rows + 1, 64) - np.clip(rows - 63, 0, None), 0, None
        )
        counts = counts.astype(np.float32)
        assert counts[63] == 64
        assert counts[126] == 1
        assert counts[127] == 0
        mass = np.broadcast_to(counts.reshape(1, 1, 256, 1), (1, 4, 256, 1))
        if with_sink:
            mass = mass + np.exp(sinks.astype(np.float32)).reshape(1, 4, 1, 1)
        with np.errstate(divide="ignore"):
            analytic_lse = np.log(mass)
        ref, lse_ref = _lse_reference(q, k, v, 0.5, causal=True, window=64, sinks=sinks)
        finite = np.isfinite(analytic_lse)
        assert _max_abs(lse[finite], analytic_lse[finite]) < 1e-5
        assert _max_abs(lse_ref[finite], analytic_lse[finite]) < 1e-5
        assert np.array_equal(np.isneginf(lse), np.isneginf(analytic_lse))
        assert _max_abs(out, ref) < _tolerance(dtype)
        assert not out[:, 127:].any()
        if with_sink:
            expected = np.broadcast_to(sinks.reshape(1, 4, 1, 1), (1, 4, 129, 1))
            assert _max_abs(lse[:, :, 127:], expected) < 1e-5
        else:
            assert np.isneginf(lse[:, :, 127:]).all()

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
            "companion",
        ],
    )
    def test_lse_buffer_errors(self, failure):
        rt = _rt()
        spec = _lse_spec("bf16", 128, 4, 1, False, batch=2, sq=256)
        q = an.to_device(rt, np.zeros((2, 256, 4, 128), np.float32), "bf16")
        k = an.to_device(rt, np.zeros((2, 256, 1, 128), np.float32), "bf16")
        v = an.to_device(rt, np.zeros((2, 256, 1, 128), np.float32), "bf16")
        out = an.empty_device(rt, (2, 256, 4, 128), "bf16")
        lse = DeviceArray(rt, np.zeros((2, 4, 256, 1), np.float32))
        if failure == "missing":
            lse = None
        elif failure == "unexpected":
            spec = dataclasses.replace(spec, emit_lse=False)
        elif failure == "dtype":
            lse = an.to_device(rt, np.zeros((2, 4, 256, 1), np.float32), "bf16")
        elif failure == "shape":
            lse = DeviceArray(rt, np.zeros((2, 4, 256), np.float32))
        elif failure == "layout":
            lse = DeviceArray(
                rt, np.zeros((2, 4, 256, 2), np.float32), shape=(2, 4, 256, 1)
            )
        elif failure == "device":
            lse = _HostTensor((2, 4, 256, 1), "torch.float32")
        elif failure in {"overlap_q", "overlap_k", "overlap_v", "overlap_out"}:
            tensor = {
                "overlap_q": q,
                "overlap_k": k,
                "overlap_v": v,
                "overlap_out": out,
            }[failure]
            # The same allocation as the real input/output, not a copied tensor.
            lse = _Alias(tensor, (2, 4, 256, 1))
        elif failure == "companion":
            k = _HostTensor((2, 256, 1, 128), "torch.bfloat16")
        with pytest.raises(ValueError):
            run_attention_dense_torch(
                spec=spec, q=q, k=k, v=v, out=out, lse=lse, scale=0.5
            )

    def test_dispatched_lse(self):
        """Request -> dispatch -> bind -> launch fills a caller-owned LSE buffer."""
        from dispatch.attention import (
            AttentionRequest,
            attention_execution_candidates,
            dispatch_attention,
        )

        rt = _rt()
        candidate = next(
            c for c in attention_execution_candidates() if c.spec_id == "gfx942_dense"
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
                arch="gfx942",
                mask_type=1,
                dtype="bf16",
                algorithm=candidate.algorithm,
                spec_id="gfx942_dense",
                emit_lse=True,
            )
        )
        scale = 0.5
        q, k, v = _qkv(23, "bf16", 8, 2, 128, sq=512, batch=2)
        out = an.empty_device(rt, q.shape, "bf16")
        lse = an.lse_device(rt, (2, 8, 512, 1))
        tensors = {
            "q": an.to_device(rt, q, "bf16"),
            "k": an.to_device(rt, k, "bf16"),
            "v": an.to_device(rt, v, "bf16"),
            "out": out,
        }
        with pytest.raises(ValueError):
            result.bind_torch(tensors, scale=scale).launch()
        result.bind_torch({**tensors, "lse": lse}, scale=scale).launch()
        ref, lse_ref = _lse_reference(q, k, v, scale, causal=True)
        assert _max_abs(an.read(out, "bf16"), ref) < _tolerance("bf16")
        assert _max_abs(lse.read(), lse_ref) < _tolerance("bf16")

    @pytest.mark.parametrize(
        "persistent,decode",
        [
            (False, "qb_major"),
            (True, "qb_major"),
            (True, "hkv_major"),
        ],
        ids=["ordinary", "persistent_qb", "persistent_hkv"],
    )
    def test_lse_runtime_shape_reuse(self, persistent, decode):
        """One LSE binary serves different B/Sq/Hq/Hkv values and writes each row.

        This directly guards the runtime LSE index ``((b*Hq+h)*Sq+q)`` and the two
        persistent work-item decoders introduced by #12859.
        """
        from kernels.common.attention_dense_spec import attention_dense_cache_key
        from kernels.gfx942.attention_dense import _DENSE_LAUNCHER_CACHE

        configs = [(1, 512, 8, 2), (2, 1024, 16, 4)]
        specs = [
            _lse_spec(
                "bf16",
                128,
                hq,
                hkv,
                persistent,
                batch=batch,
                sq=sq,
                persist_decode=decode,
            )
            for batch, sq, hq, hkv in configs
        ]
        assert all(s.runtime_shape for s in specs)
        off_specs = [dataclasses.replace(spec, emit_lse=False) for spec in specs]
        keys = {
            attention_dense_cache_key(spec, arch="gfx942") for spec in specs + off_specs
        }
        assert len(keys) == 2, "LSE-on/off should each reuse one runtime-shape binary"
        old_entries = {key: _DENSE_LAUNCHER_CACHE.pop(key, None) for key in keys}
        scale = 0.5
        launchers = []
        try:
            for index, spec in enumerate(specs):
                q, k, v = _qkv(
                    29 + index,
                    "bf16",
                    spec.num_query_heads,
                    spec.num_kv_heads,
                    128,
                    sq=spec.seqlen_q,
                    sk=spec.seqlen_kv,
                    batch=spec.batch,
                )
                on, off = _run_lse_and_off(spec, q, k, v, scale)
                ref, lse_ref = _lse_reference(q, k, v, scale, causal=True)
                assert _max_abs(on.out, ref) < _tolerance("bf16")
                assert _max_abs(on.lse, lse_ref) < _tolerance("bf16")
                assert np.array_equal(on.bits, off.bits)
                launchers.append(_launcher_for(spec))
            assert launchers[0] is not None
            assert launchers[0] is launchers[1]
            assert _launcher_for(off_specs[-1]) is not launchers[0]
        finally:
            for key, entry in old_entries.items():
                _DENSE_LAUNCHER_CACHE.pop(key, None)
                if entry is not None:
                    _DENSE_LAUNCHER_CACHE[key] = entry


class TestDensePagedNumeric:
    """Paged K/V (block_tables indirection) on the default grid."""

    @ci_subset(cover=("block_size", "sq,window"), rotate=("dtype", "d"))
    @pytest.mark.parametrize("dtype", ["fp16", "bf16"])
    @pytest.mark.parametrize("block_size", [16, 32, 64])
    @pytest.mark.parametrize("sq,window", [(1024, 256), (2048, 1024), (1024, 0)])
    @pytest.mark.parametrize("d", [128, 64])
    def test_paged_matches_dense_bit_exact(self, dtype, block_size, sq, window, d):
        """A paged cache must give the same bits as the dense K/V it was scattered
        from, and match the fp32 causal (sliding-window) reference.

        The pages land at randomly permuted physical ids in a cache twice the
        needed size, and every unused page is NaN, so a wrong page or in-page row
        reads a NaN page or the wrong keys rather than a near miss. block_size
        16/32/64 covers 4, 2 and 1 pages per 64-key tile. Paged V never takes the
        cfvst path, so the dense run turns it off too, for identical arithmetic.
        """
        self._check_paged(dtype, block_size, sq, sq, window, d)

    @pytest.mark.parametrize(
        "sq,sk,window",
        [
            pytest.param(512, 2048, 0, id="top-left-512x2048"),
            pytest.param(2048, 512, 0, id="top-left-2048x512"),
            pytest.param(1024, 512, 128, id="swa-empty-rows-1024x512"),
        ],
    )
    def test_paged_cross_length_matches_dense(self, sq, sk, window):
        """Top-left causal with ``seqlen_q != seqlen_kv``, and rows whose window holds
        no key."""
        self._check_paged("fp16", 16, sq, sk, window, 128)

    def test_dispatched_paged(self):
        """Request -> dispatch -> bind -> launch, with the block count taken from the
        cache at bind."""
        window = 1024
        q, k, v, out = run_dispatched_paged(
            _rt(), "gfx942", "gfx942_dense", window=window
        )
        assert np.isfinite(out).all()
        scale = 1.0 / math.sqrt(q.shape[-1])
        ref = _sdpa_reference(q, k, v, scale, window=window)
        assert _max_abs(out, ref) < _tolerance("fp16")

    @staticmethod
    def _check_paged(dtype, block_size, sq, sk, window, d):
        rt = _rt()
        hq, hkv, B = 16, 4, 1
        scale = 1.0 / math.sqrt(d)
        q, k, v = _qkv(0, dtype, hq, hkv, d, sq=sq, sk=sk, batch=B)

        k_cache, v_cache, block_tables, kv_lens = paged_cache(
            rt, k, v, block_size, dtype
        )
        num_kv_blocks = k_cache.shape[0]

        # The shipped paged spec (its tile can differ from the contiguous one's),
        # and its contiguous twin on the same tile and V path.
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
            shipped, paged=False, block_size=0, num_kv_blocks=0, use_cfvst=False
        )

        dense = _run(dense_spec, q, k, v, scale)
        paged_out = an.empty_device(rt, q.shape, dtype)
        run_attention_dense_torch(
            spec=paged_spec,
            q=an.to_device(rt, q, dtype),
            k=k_cache,
            v=v_cache,
            out=paged_out,
            scale=scale,
            block_tables=block_tables,
            kv_lens=kv_lens,
        )
        paged = _Run(paged_out, dtype)

        name = f"{dtype} D{d} pgd{block_size} sq{sq} sk{sk} swa{window}"
        assert np.isfinite(paged.out).all(), f"{name}: read a NaN (unused) page"
        assert np.array_equal(paged.bits, dense.bits), f"{name}: paged != dense bits"
        qi = np.arange(sq)[:, None]
        ki = np.arange(sk)[None, :]
        keep = ki <= qi  # top-left diagonal
        if window:
            keep &= ki > qi - window
        ref = _sdpa_reference(q, k, v, scale, window=window)
        # A row whose window holds no key has no softmax; its output must be zero.
        live = keep.any(axis=1)
        max_abs = _max_abs(paged.out[:, live], ref[:, live])
        assert max_abs < _tolerance(dtype), f"{name}: max_abs={max_abs:.3e}"
        if not live.all():
            worst = float(np.abs(paged.out[:, ~live]).max())
            assert worst == 0.0, f"{name}: {int((~live).sum())} keyless rows: {worst}"


# All following cases are within the BF16 D128 causal BN64 persistent gate.
# Unlike the unsplit pipeline, diagonal split has TWO processing loops; K/V
# publication and the running softmax state cross that loop boundary.
@pytest.mark.parametrize(
    "lds_num_buffers,v_reg_prefetch,lds_row_pad",
    [
        (1, False, 8),  # CFVST alone: no pipeline, split loops only.
        (2, False, 8),
        (1, True, 8),
        (2, True, 8),
        (2, True, 4),  # Measured full optimization (pad4 + dsplit).
    ],
    ids=["cfvst_only", "kdb", "vpf", "kdb_vpf", "winner_pad4"],
)
@pytest.mark.parametrize(
    "sq,skv",
    [(256, 64), (512, 192), (512, 512)],
    ids=["single_tile", "odd_k_tiles", "split_loop_boundary"],
)
def test_cfvst_pipeline_diag_split_numeric(
    sq, skv, lds_num_buffers, v_reg_prefetch, lds_row_pad
):
    """Validate CFVST alone, each pipeline and the pad4/dsplit winner on GPU.

    NP=3 forces work-item LDS reuse and a ragged grid; the same Q/K/V inputs
    exercise split and unsplit code paths against a true float64 causal oracle.
    """
    hq, hkv, d = 16, 4, 128
    scale = 1.0 / math.sqrt(d)
    q, k, v = _qkv(47, "bf16", hq, hkv, d, sq=sq, sk=skv)
    ref = _sdpa_reference(q, k, v, scale)

    base = _as_gfx942_spec(_spec("bf16", d, hq, hkv, True, sq=sq))
    split = _multi_item_persistent(
        dataclasses.replace(
            base,
            seqlen_kv=skv,
            use_cfvst=True,
            lds_num_buffers=lds_num_buffers,
            v_reg_prefetch=v_reg_prefetch,
            lds_row_pad=lds_row_pad,
            causal_diag_split=True,
        )
    )
    ok, why = supports_attention_dense(split, arch="gfx942")
    assert ok, why
    unsplit = dataclasses.replace(split, causal_diag_split=False)
    out = _run(split, q, k, v, scale)
    control = _run(unsplit, q, k, v, scale)
    assert np.isfinite(out.out).all()
    _assert_allclose(out.out, ref, atol=1e-2, rtol=1e-2, what="split")
    _assert_allclose(control.out, ref, atol=1e-2, rtol=1e-2, what="unsplit")
    assert np.array_equal(out.bits, control.bits), (
        "causal_diag_split changed BF16 output bits in the pipeline: "
        f"Sq={sq} Skv={skv} KDB={lds_num_buffers == 2} "
        f"VPF={v_reg_prefetch} pad={lds_row_pad}"
    )


@pytest.mark.parametrize("sq,skv", [(256, 64), (512, 192), (512, 512)])
def test_cfvst_pipeline_pad4_dsplit_lse_numeric(sq, skv):
    """The full split pipeline must compute LSE from the FINAL tile's row max."""
    hq, hkv, d = 16, 4, 128
    scale = 1.0 / math.sqrt(d)
    q, k, v = _qkv(53, "bf16", hq, hkv, d, sq=sq, sk=skv)
    spec = _multi_item_persistent(
        dataclasses.replace(
            _lse_spec("bf16", d, hq, hkv, True, sq=sq),
            seqlen_kv=skv,
            use_cfvst=True,
            lds_num_buffers=2,
            v_reg_prefetch=True,
            lds_row_pad=4,
            causal_diag_split=True,
        )
    )
    ok, why = supports_attention_dense(spec, arch="gfx942")
    assert ok, why
    on, off = _run_lse_and_off(spec, q, k, v, scale)
    ref, lse_ref = _lse_reference(q, k, v, scale, causal=True)
    _assert_allclose(on.out, ref, atol=1e-2, rtol=1e-2, what="O")
    _assert_allclose(on.lse, lse_ref, atol=1e-3, rtol=1e-3, what="LSE")
    assert np.array_equal(on.bits, off.bits)


@pytest.mark.parametrize("block_m", [64, 512])
def test_cfvst_pipeline_extreme_legal_block_m_numeric(block_m):
    """Keep the BN64 pipeline valid at both ends of the implemented BM tuning axis."""
    hq, hkv, d, sq = 16, 4, 128, 512
    scale = 1.0 / math.sqrt(d)
    q, k, v = _qkv(59, "bf16", hq, hkv, d, sq=sq)
    spec = _multi_item_persistent(
        dataclasses.replace(
            _as_gfx942_spec(_spec("bf16", d, hq, hkv, True, sq=sq)),
            block_m=block_m,
            use_cfvst=True,
            lds_num_buffers=2,
            v_reg_prefetch=True,
        )
    )
    ok, why = supports_attention_dense(spec, arch="gfx942")
    assert ok, why
    run = _run(spec, q, k, v, scale)
    _assert_allclose(
        run.out, _sdpa_reference(q, k, v, scale), atol=1e-2, rtol=1e-2, what="O"
    )


def test_cfvst_kdb_vpf_pipeline_reuses_runtime_shape_binary():
    """Same combined pipeline launcher, different B/S/Hq/Hkv and distinct inputs."""
    from kernels.common.attention_dense_spec import attention_dense_cache_key
    from kernels.gfx942.attention_dense import _DENSE_LAUNCHER_CACHE

    shape_rows = [(1, 512, 16, 4), (2, 1024, 32, 8)]
    specs = []
    for batch, sq, hq, hkv in shape_rows:
        base = _as_gfx942_spec(
            _spec(
                "bf16",
                128,
                hq,
                hkv,
                True,
                batch=batch,
                sq=sq,
                persist_decode="qb_major",
            )
        )
        spec = dataclasses.replace(
            base,
            num_persistent=3,
            use_cfvst=True,
            lds_num_buffers=2,
            v_reg_prefetch=True,
        )
        assert supports_attention_dense(spec, arch="gfx942")[0]
        assert spec.runtime_shape
        specs.append(spec)
    keys = [attention_dense_cache_key(s, arch="gfx942") for s in specs]
    assert keys[0] == keys[1], "same runtime-shape kernel must have one cache key"
    old_entry = _DENSE_LAUNCHER_CACHE.pop(keys[0], None)
    try:
        launchers = []
        for i, spec in enumerate(specs):
            q, k, v = _qkv(
                61 + i,
                "bf16",
                spec.num_query_heads,
                spec.num_kv_heads,
                128,
                sq=spec.seqlen_q,
                sk=spec.seqlen_kv,
                batch=spec.batch,
            )
            scale = 1.0 / math.sqrt(128)
            run = _run(spec, q, k, v, scale)
            _assert_allclose(
                run.out, _sdpa_reference(q, k, v, scale), atol=1e-2, rtol=1e-2, what="O"
            )
            launchers.append(_launcher_for(spec))
        assert launchers[0] is not None
        assert launchers[0] is launchers[1], "pipeline recompiled across runtime shapes"
    finally:
        _DENSE_LAUNCHER_CACHE.pop(keys[0], None)
        if old_entry is not None:
            _DENSE_LAUNCHER_CACHE[keys[0]] = old_entry


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
