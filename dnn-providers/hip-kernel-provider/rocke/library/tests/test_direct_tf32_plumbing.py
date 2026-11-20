# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the fp32 / tf32 plumbing of the direct-conv benchmark and cache.

``problem.dtype`` takes "fp32" (fp32 tensors, exact fp32 math) and "tf32"
(fp32 tensors, TF32 operands at the MMA). These pin what the sweep layer does
with them: MIOpen command parsing (``conv ... -M 0``), the storage and identity
dtype maps, that every existing fp16/bf16 cache identity keeps its hash, the
verify tolerance table, the per-dtype knob grids, and that a variant without a
kernel for a dtype is skipped rather than crashing the enumeration. No GPU.
"""

from __future__ import annotations

import json
import struct

import pytest

from benchmarks.common import direct_kernel_sweep as dks
from kernels.common.conv_abi import conv_direct_transform_signature
from kernels.conv import direct_aot_jobs as daj
from benchmarks.common.kernel_cache import KernelCache, KernelIdentity
from kernels.common.conv_direct_grouped import DirectConvProblem
from rocke.core.arch import ArchTarget
from rocke.runtime.packing import compile_packer

_ARCH = "gfx950"
_WAVE = 64


def _problem(**kw) -> DirectConvProblem:
    shape = dict(
        N=2, H=28, W=28, groups=16, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
    )
    shape.update(kw)
    return DirectConvProblem(**shape)


# ---------------------------------------------------------------------------
# MIOpen command parsing
# ---------------------------------------------------------------------------

_CMD = "MIOpenDriver {kw} -n 2 -c 64 -H 28 -W 28 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 16 -F 1 -in_layout NHWC {extra}"


@pytest.fixture(scope="module")
def parse():
    pytest.importorskip("torch")  # the benchmark module imports the reference
    from benchmarks.common.benchmark_direct_conv import parse_miopen_cmd_direct

    return parse_miopen_cmd_direct


@pytest.mark.parametrize(
    "kw, extra, dtype",
    [
        ("convfp16", "", "fp16"),
        ("convbfp16", "", "bf16"),
        ("conv", "", "fp32"),
        ("convfp32", "", "fp32"),
        ("conv", "-M 1", "fp32"),
        ("conv", "-M 0", "tf32"),
        ("conv", "--math_type 0", "tf32"),
        ("./bin/MIOpenDriver conv", "-M 0 -t 1 -V 0 -i 10", "tf32"),
        # MIOpen applies TF32 only to fp32 tensors: -M is inert on 16-bit.
        ("convfp16", "-M 0", "fp16"),
        ("convbfp16", "-M 0", "bf16"),
    ],
)
def test_miopen_dtype(parse, kw, extra, dtype):
    problem, got, forw = parse(_CMD.format(kw=kw, extra=extra))
    assert got == dtype
    # The problem carries the token the kernels read; nothing is coerced.
    assert problem.dtype == dtype
    assert forw == 1
    assert (problem.groups, problem.cpg, problem.kpg) == (16, 4, 4)


def test_miopen_short_layout_flag(parse):
    problem, dtype, _ = parse(
        _CMD.format(kw="conv", extra="-M 0").replace("-in_layout NHWC", "-I NHWC")
    )
    assert dtype == "tf32"
    with pytest.raises(ValueError, match="NHWC"):
        parse(_CMD.format(kw="conv", extra="").replace("-in_layout NHWC", "-I NCHW"))


@pytest.mark.parametrize(
    "kw, extra, match",
    [
        ("convint8", "", "convint8"),
        # Direct conv is 2-D only: a 3-D command must not parse as 2-D.
        ("conv", "-M 0 -_ 3 -! 16 -@ 3", "2-D"),
    ],
)
def test_miopen_rejects(parse, kw, extra, match):
    with pytest.raises(ValueError, match=match):
        parse(_CMD.format(kw=kw, extra=extra))


# ---------------------------------------------------------------------------
# dtype maps
# ---------------------------------------------------------------------------


def test_dtype_lists():
    assert dks.DIRECT_DTYPES == ("fp16", "bf16", "fp32", "tf32")
    # The cache stores what the kernels' own storage map says.
    from kernels.common.conv_direct_grouped import _DIRECT_STORAGE

    assert tuple(_DIRECT_STORAGE) == dks.DIRECT_DTYPES
    # --compile-all without --dtype builds what it always did.
    assert dks.DIRECT_COMPILE_DTYPES == ("fp16", "bf16")


@pytest.mark.parametrize(
    "dtype, storage, triple",
    [
        ("fp16", "fp16", ("fp16", "fp16", "fp16")),
        ("bf16", "bf16", ("bf16", "bf16", "bf16")),
        ("fp32", "fp32", ("fp32", "fp32", "fp32")),
        ("tf32", "fp32", ("tf32", "tf32", "fp32")),
    ],
)
def test_storage_and_identity_dtypes(dtype, storage, triple):
    assert dks.direct_storage_dtype(dtype) == storage
    assert dks.direct_dtype_triple(dtype) == triple


@pytest.mark.parametrize("dtype", ["int8", "f32", "xf32", ""])
def test_unknown_dtype_is_refused(dtype):
    with pytest.raises(ValueError):
        dks.direct_storage_dtype(dtype)
    with pytest.raises(ValueError):
        dks.direct_dtype_triple(dtype)
    with pytest.raises(ValueError):
        dks.direct_verify_tol(dtype, _ARCH)


def test_torch_dtype_map():
    torch = pytest.importorskip("torch")
    from benchmarks.common.benchmark_direct_conv import _abi_dtype, _torch_dtype_of

    assert _torch_dtype_of(torch, "fp16") is torch.float16
    assert _torch_dtype_of(torch, "bf16") is torch.bfloat16
    assert _torch_dtype_of(torch, "fp32") is torch.float32
    assert _torch_dtype_of(torch, "tf32") is torch.float32
    assert _abi_dtype("tf32") == "fp32"


def test_transform_signature_elements():
    for dtype, elem in (("fp16", "f16"), ("bf16", "bf16"), ("fp32", "f32")):
        sig = conv_direct_transform_signature(dtype)
        assert sig[0]["type"] == f"ptr<{elem}, global>"
    assert conv_direct_transform_signature("tf32") == conv_direct_transform_signature(
        "fp32"
    )
    assert conv_direct_transform_signature(
        "tf32", True
    ) == conv_direct_transform_signature("fp32", True)


# ---------------------------------------------------------------------------
# Cache identities
# ---------------------------------------------------------------------------

_GROUPED_CAPS = dks.DirectCaps(KH=3, PAD=1, stride=1, cpg=16, kpg=16)
_GROUPED_KNOBS = dict(block_q=16, block_groups=2, double_buffer=True)
_WGRAD_CAPS = dks.DirectCaps(KH=3, PAD=1, stride=1, cpg=0, kpg=0)
_WGRAD_KNOBS = dict(waves_k=1, waves_c=1, waves_q=1, ho_per_block=30, mfma_k=32)


def _pinned_hash(identity) -> str:
    from dataclasses import replace

    return replace(identity, llvm_flavor="llvm22").stable_hash()


def test_existing_identity_hashes_are_unchanged():
    """Hashes recorded before fp32/tf32 existed: a changed one would orphan
    every cached fp16/bf16 binary. The LLVM flavor is pinned: it comes from the
    host's comgr, which the test environment may override."""
    assert (
        _pinned_hash(
            daj._identity(
                _ARCH,
                _WAVE,
                "direct_grouped",
                "direct_fwd",
                _GROUPED_CAPS,
                _GROUPED_KNOBS,
                "fp16",
            )
        )
        == "c0fcc18a0ef37b8a722d9a46efccab0d0145199b"
    )
    assert (
        _pinned_hash(
            daj._identity(
                _ARCH,
                _WAVE,
                "direct_wgrad",
                "direct_wgrad",
                _WGRAD_CAPS,
                _WGRAD_KNOBS,
                "bf16",
            )
        )
        == "95560d3abed95b1ffe94fda1d76f6c85ffe53e4f"
    )
    assert (
        _pinned_hash(
            dks._helper_identity(
                _ARCH, _WAVE, dks._REORGANIZE, _GROUPED_CAPS, {"fold_k32": True}, "fp16"
            )
        )
        == "be84649f23aa6023a6468aee8ab66a05759dd6b9"
    )


def test_16bit_knob_grids_are_unchanged():
    """The fold knob is fp32-storage only: no fp16/bf16 knob dict gains it."""
    for dtype in ("fp16", "bf16"):
        knobs = list(daj._knob_grid("direct_grouped", _ARCH, dtype))
        assert all(
            set(k) == {"block_q", "block_groups", "double_buffer"} for k in knobs
        )
        assert len(knobs) == len(dks.BLOCK_Q) * len(dks.BLOCK_GROUPS) * len(
            dks.DOUBLE_BUFFER
        )


@pytest.mark.parametrize("dtype", ["fp32", "tf32"])
def test_fp32_storage_grouped_grid_sweeps_the_fold(dtype):
    knobs = list(daj._knob_grid("direct_grouped", _ARCH, dtype))
    assert {k["fold_k32"] for k in knobs} == {False, True}
    assert len(knobs) == 2 * len(dks.BLOCK_Q) * len(dks.BLOCK_GROUPS) * len(
        dks.DOUBLE_BUFFER
    )


def test_dgrad_waves_grid():
    assert dks.dgrad_waves("fp16") == dks.DGRAD_WAVES
    assert dks.dgrad_waves("bf16") == dks.DGRAD_WAVES
    for dtype in ("fp32", "tf32"):
        waves = dks.dgrad_waves(dtype)
        assert set(dks.DGRAD_WAVES) <= set(waves)
        assert len(set(waves)) == len(waves)
        # Every unfolded entry has its K-32 twin.
        for wq, wk, rk, pg, k32 in waves:
            assert (wq, wk, rk, pg, True) in waves


def test_tf32_identity_is_distinct():
    ids = {
        dtype: daj._identity(
            _ARCH,
            _WAVE,
            "direct_wgrad",
            "direct_wgrad",
            _WGRAD_CAPS,
            _WGRAD_KNOBS,
            dtype,
        )
        for dtype in dks.DIRECT_DTYPES
    }
    assert (ids["tf32"].dtype_a, ids["tf32"].dtype_b, ids["tf32"].dtype_d) == (
        "tf32",
        "tf32",
        "fp32",
    )
    assert len({i.stable_hash() for i in ids.values()}) == len(ids)
    assert len({i.label() for i in ids.values()}) == len(ids)
    for ident in ids.values():
        assert KernelIdentity.from_label(ident.label()) == ident


def test_job_for_identity_round_trips_tf32():
    ident = daj._identity(
        _ARCH, _WAVE, "direct_wgrad", "direct_wgrad", _WGRAD_CAPS, _WGRAD_KNOBS, "tf32"
    )
    try:
        job = dks.direct_job_for_identity(ident)
    except LookupError as e:
        pytest.skip(f"no tf32 direct wgrad kernel in this tree yet: {e}")
    assert job.identity == ident


def test_job_for_identity_rejects_a_tf32_output():
    """tf32 is a math mode, never a storage type: dtype_d must be fp32."""
    from dataclasses import replace

    ident = daj._identity(
        _ARCH, _WAVE, "direct_wgrad", "direct_wgrad", _WGRAD_CAPS, _WGRAD_KNOBS, "tf32"
    )
    with pytest.raises(LookupError):
        dks.direct_job_for_identity(replace(ident, dtype_d="tf32"))


# ---------------------------------------------------------------------------
# Verify tolerance
# ---------------------------------------------------------------------------


def test_tf32_emulation_by_arch():
    assert dks.tf32_is_emulated("gfx950")
    assert not dks.tf32_is_emulated("gfx942")
    assert not dks.tf32_is_emulated("gfx_not_an_arch")


@pytest.mark.parametrize(
    "dtype, arch, tol",
    [
        ("fp16", "gfx950", 5e-2),
        ("bf16", "gfx942", 5e-2),
        ("fp32", "gfx950", 1e-4),
        ("fp32", "gfx942", 1e-4),
        # BF16x3 emulation is about fp32-accurate: a dropped cross term or a
        # single bf16 pass (~2e-3) must fail.
        ("tf32", "gfx950", 1e-4),
        # Native XF32 truncates the operands to a 10-bit mantissa.
        ("tf32", "gfx942", 2**-8),
    ],
)
def test_verify_tolerance(dtype, arch, tol):
    assert dks.direct_verify_tol(dtype, arch) == tol


def test_native_tf32_bound_clears_operand_truncation():
    """A correct native XF32 result against the IEEE reference must pass.

    Four same-sign products of 0x3f001fff: truncation to a 10-bit mantissa
    gives exactly 1.0, the reference about 1.00195, a 2^-9 loss that 1e-3
    rejected. The bound must also stay well below a wrong answer's error.
    """
    x = struct.unpack("<f", struct.pack("<I", 0x3F001FFF))[0]
    x_tf32 = struct.unpack("<f", struct.pack("<I", 0x3F001FFF & 0xFFFFE000))[0]
    ref = 4 * x * x
    out = 4 * x_tf32 * x_tf32
    rel_err = abs(out - ref) / max(abs(ref), 1.0)
    assert out == 1.0
    assert rel_err > 1e-3
    tol = dks.direct_verify_tol("tf32", "gfx942")
    assert rel_err < tol
    assert tol < dks.direct_verify_tol("fp16", "gfx942") / 10


@pytest.mark.parametrize("dtype", ["fp32", "tf32"])
def test_batch_split_counts_fp32_storage_bytes(dtype):
    """The LDS-halo batch split sizes each launch by the stored bytes: an
    fp32 / tf32 image is twice an fp16 one, so the same batch needs more
    launches, and a chunk's buffer views advance by 4-byte elements."""
    from kernels.common.conv_direct_nongrouped import _OOB_BASE

    shape = dict(N=16, H=768, W=768, groups=1, cpg=256, kpg=256, KH=3, KW=3, PAD=1)
    half = DirectConvProblem(**shape, dtype="fp16")
    full = DirectConvProblem(**shape, dtype=dtype)
    assert dks._image_bytes(full) == tuple(2 * n for n in dks._image_bytes(half))
    chunks_half = dks._batch_chunks(half)
    chunks_full = dks._batch_chunks(full)
    assert sum(chunks_full) == full.N
    assert len(chunks_full) > len(chunks_half)
    in_img, out_img = dks._image_bytes(full)
    assert max(chunks_full) * max(in_img, out_img) <= _OOB_BASE


@pytest.mark.parametrize("dtype", ["fp32", "tf32"])
@pytest.mark.parametrize("direction", ["fwd", "dgrad"])
def test_jit_batch_split_counts_fp32_storage_bytes(dtype, direction):
    """The JIT sweep's batch split keeps every activation under 2 GiB in the
    storage type: fp32 / tf32 split where the same fp16 batch does not."""
    bdc = pytest.importorskip("benchmarks.common.benchmark_direct_conv")

    # 3 fp16 images of 512x512x1024 (1.5 GiB) fit one launch; fp32 does not.
    shape = dict(N=3, H=512, W=512, groups=1, cpg=1024, kpg=1024)
    assert bdc._batch_chunks(DirectConvProblem(**shape, dtype="fp16"), direction) == [3]
    full = DirectConvProblem(**shape, dtype=dtype)
    chunks = bdc._batch_chunks(full, direction)
    assert sum(chunks) == 3 and len(chunks) > 1
    assert max(chunks) * max(dks._image_bytes(full)) < bdc._TENSOR_LIMIT


# ---------------------------------------------------------------------------
# Narrow (zero-padded) TF32 atoms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant, problem, knobs, arch, dropped",
    [
        # Grouped forward: the fold covers cpg % 32 == 0 only.
        (
            "direct_grouped",
            _problem(dtype="tf32", cpg=32, kpg=32),
            {"fold_k32": False},
            "gfx950",
            True,
        ),
        (
            "direct_grouped",
            _problem(dtype="tf32", cpg=32, kpg=32),
            {"fold_k32": True},
            "gfx950",
            False,
        ),
        (
            "direct_grouped",
            _problem(dtype="tf32", cpg=16, kpg=16),
            {"fold_k32": False},
            "gfx950",
            False,
        ),
        # Native XF32 splits a narrow fragment exactly: nothing to drop.
        (
            "direct_grouped",
            _problem(dtype="tf32", cpg=32, kpg=32),
            {"fold_k32": False},
            "gfx942",
            False,
        ),
        # Exact fp32 and 16-bit problems are never touched.
        (
            "direct_grouped",
            _problem(dtype="fp32", cpg=32, kpg=32),
            {"fold_k32": False},
            "gfx950",
            False,
        ),
        (
            "direct_grouped",
            _problem(dtype="fp16", cpg=32, kpg=32),
            {"fold_k32": False},
            "gfx950",
            False,
        ),
        # Non-grouped: narrow atoms only where C fits a K-32 chunk.
        (
            "direct_nongrouped",
            _problem(dtype="tf32", groups=1, cpg=64, kpg=64),
            {"atom": "32x32x8"},
            "gfx950",
            True,
        ),
        (
            "direct_nongrouped",
            _problem(dtype="tf32", groups=1, cpg=64, kpg=64),
            {"atom": "16x16x16"},
            "gfx950",
            True,
        ),
        (
            "direct_nongrouped",
            _problem(dtype="tf32", groups=1, cpg=64, kpg=64),
            {"atom": "32x32x16"},
            "gfx950",
            False,
        ),
        (
            "direct_nongrouped",
            _problem(dtype="tf32", groups=1, cpg=48, kpg=64),
            {"atom": "32x32x8"},
            "gfx950",
            False,
        ),
        ("direct_wgrad", _problem(dtype="tf32"), {"mfma_k": 16}, "gfx950", True),
        ("direct_wgrad", _problem(dtype="tf32"), {"mfma_k": 32}, "gfx950", False),
    ],
)
def test_tf32_narrow_reason(variant, problem, knobs, arch, dropped):
    assert bool(dks.tf32_narrow_reason(variant, problem, knobs, arch)) == dropped


def _mfma_knobs(**kw):
    knobs = dict(
        block_q=16,
        block_groups=1,
        block_h=8,
        waves_q=1,
        waves_k=1,
        runtime_k_loop=False,
        persistent_grid=False,
        fold_k32=False,
    )
    knobs.update(kw)
    return knobs


def test_tf32_narrow_reason_mfma_dgrad():
    p = _problem(dtype="tf32", cpg=32, kpg=64)
    reason = dks.tf32_narrow_reason
    assert reason("direct_grouped_dgrad_mfma", p, _mfma_knobs(), "gfx950")
    assert not reason(
        "direct_grouped_dgrad_mfma", p, _mfma_knobs(fold_k32=True), "gfx950"
    )
    # kpg 48: no fold covers it, so the padded config is the kernel.
    p48 = _problem(dtype="tf32", cpg=32, kpg=48)
    assert not reason(
        "direct_grouped_dgrad_mfma", p48, _mfma_knobs(waves_k=1), "gfx950"
    )
    # A 2-way K split of kpg=96: 6 narrow atoms split, 3 wide ones do not.
    p96 = _problem(dtype="tf32", cpg=32, kpg=96)
    assert not reason(
        "direct_grouped_dgrad_mfma", p96, _mfma_knobs(waves_k=2), "gfx950"
    )
    # Runtime channels: the fold binary must have the problem's atom count.
    rtc = _mfma_knobs(k_atoms=4, m_tiles=2)
    assert reason("direct_dgrad_mfma_rtc", p, rtc, "gfx950")
    assert not reason("direct_dgrad_mfma_rtc", p, rtc, "gfx942")


def test_tf32_narrow_reason_keeps_the_rtc_config_whose_fold_spills():
    """A runtime-channel narrow config is dropped only where its folded twin
    fits its registers. Unsplit (waves_k 1), the preload budget never looks at
    the twin; a 7x7 filter over 3 M-tiles and 2 Q subtiles holds 168
    accumulators, which spill in a 16-wave block (budget 0.875 * 512 / 4) but
    not in an 8-wave one."""
    p = _problem(dtype="tf32", groups=1, cpg=48, kpg=64, KH=7, KW=7, PAD=3)
    for block_groups, dropped in ((8, True), (16, False)):
        knobs = _mfma_knobs(block_q=32, block_groups=block_groups, k_atoms=4, m_tiles=3)
        wide = dict(knobs, fold_k32=True, k_atoms=2)
        assert dks.dgrad_rtc_spills(wide, 7, 7, "gfx950", "tf32") != dropped
        why = dks.tf32_narrow_reason("direct_dgrad_mfma_rtc", p, knobs, "gfx950")
        assert bool(why) == dropped, block_groups


# ---------------------------------------------------------------------------
# Enumeration and plans
# ---------------------------------------------------------------------------


def test_a_spec_construction_error_is_not_swallowed(monkeypatch):
    """Every direct variant builds fp32/tf32 specs, so the grid never needs to
    skip one that raises on construction -- and must not: a knob bug in any
    dtype has to stop the enumeration, not silently drop a cache entry."""

    def broken(variant, problem, knobs, *_):
        raise ValueError(f"{variant}: a knob bug")

    # _caps_jobs lives in (and calls make_spec through) direct_aot_jobs.
    monkeypatch.setattr(daj, "make_spec", broken)
    caps = dks.DIRECT_CAPABILITIES["direct_grouped_dgrad"][1][0]
    for dtype in ("fp16", "tf32"):
        with pytest.raises(ValueError, match="a knob bug"):
            dks._caps_jobs(_ARCH, _WAVE, "direct_grouped_dgrad", "dgrad", caps, dtype)


@pytest.fixture(scope="module")
def tf32_jobs():
    jobs = dks.enumerate_direct_jobs(
        arch=_ARCH,
        target=ArchTarget.from_gfx(_ARCH),
        directions=dks.DIRECT_DIRECTIONS,
        dtype="tf32",
        jobs=4,
    )
    if not jobs:
        pytest.skip("no direct kernel builds tf32 in this tree yet")
    return jobs


def test_tf32_jobs_carry_the_tf32_triple(tf32_jobs):
    for job in tf32_jobs:
        ident = job.identity
        assert (ident.dtype_a, ident.dtype_b, ident.dtype_d) == ("tf32", "tf32", "fp32")
    # No baked-channel grouped binary zero-pads where the fold covers cpg.
    for job in tf32_jobs:
        if job.identity.algorithm == "direct_grouped" and job.caps["cpg"] % 32 == 0:
            assert job.spec_kwargs["fold_k32"]


_PLAN_SHAPES = [
    ("fwd", dict(groups=64, cpg=1, kpg=1)),
    ("fwd", dict(groups=1, cpg=64, kpg=64)),
    ("fwd", dict(groups=8, cpg=32, kpg=32)),
    ("dgrad", dict(groups=64, cpg=1, kpg=1)),
    ("dgrad", dict()),
    ("dgrad", dict(groups=1, cpg=48, kpg=192)),
    ("wgrad", dict()),
]
_PLAN_DTYPES = ("fp16", "tf32")


def _serves_a_planned_problem(identity) -> bool:
    """Would ``cache.compatible`` offer ``identity`` to any planned problem?

    Helper kernels (weight transforms, split reductions) are looked up by
    identity when a plan is made, so every one is kept.
    """
    if identity.direction.endswith("_helper"):
        return True
    return any(
        identity.direction == f"direct_{direction}"
        and KernelCache.supports_problem(identity, _problem(dtype=dtype, **shape))[0]
        for direction, shape in _PLAN_SHAPES
        for dtype in _PLAN_DTYPES
    )


@pytest.fixture(scope="module")
def mixed_cache(tmp_path_factory, tf32_jobs):
    """fp16 and tf32 binaries side by side, placeholder bytes.

    The cache holds every enumerated binary that ``cache.compatible`` would
    offer to a planned problem, of both dtypes, plus every helper; the plans
    are the ones the whole grid's cache gives. The rest of the grid (about
    three quarters of it, no planned problem's) is not written: one file per
    entry made this fixture the slowest step of the installed TF32 entry on
    slow-filesystem CI runners.
    """
    c = KernelCache(tmp_path_factory.mktemp("direct_tf32_cache"), _ARCH)
    fp16_jobs = dks.enumerate_direct_jobs(
        arch=_ARCH,
        target=ArchTarget.from_gfx(_ARCH),
        directions=dks.DIRECT_DIRECTIONS,
        dtype="fp16",
        jobs=4,
    )
    kept = [
        job
        for job in list(fp16_jobs) + list(tf32_jobs)
        if _serves_a_planned_problem(job.identity)
    ]
    # Both dtypes stay in the cache side by side for every planned shape.
    for dtype in _PLAN_DTYPES:
        for direction in dks.DIRECT_DIRECTIONS:
            assert any(
                j.identity.dtype_a == dtype
                and j.identity.direction == f"direct_{direction}"
                for j in kept
            ), (dtype, direction)
    for job in kept:
        c.put(job.identity, b"placeholder", {"kernel_name": job.identity.label()})
    return c


@pytest.mark.parametrize("direction, shape", _PLAN_SHAPES)
@pytest.mark.parametrize("dtype", _PLAN_DTYPES)
def test_plans_serve_only_their_dtype(mixed_cache, direction, shape, dtype):
    p = _problem(dtype=dtype, **shape)
    plans, _ = dks.direct_plans(mixed_cache, p, direction, _ARCH)
    names = ("x", "w", "y", "dy", "dx", "dw", "ws_t", "ws_coa", "ws_split")
    ptrs = {n: 0x1000 * (i + 1) for i, n in enumerate(names)}
    sizes = {n: 1 << 20 for n in names}
    for plan in plans:
        assert plan.identity.dtype_a == dtype
        for step in plan.steps:
            compile_packer(step.signature)(dks.launch_values(step, ptrs, sizes))
            if dtype == "tf32":
                # The ABI is spelled in the storage type.
                assert "tf32" not in str(step.signature)
                assert any("ptr<f32" in a["type"] for a in step.signature)
        if dtype == "tf32":
            knobs = json.loads(plan.identity.knobs)
            assert not dks.tf32_narrow_reason(plan.identity.algorithm, p, knobs, _ARCH)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_best_ranks_only_verified_kernels(capsys):
    pytest.importorskip("torch")
    from benchmarks.common.benchmark_direct_conv import _rank_verified

    results = [
        {"tflops": 9.0, "passed": False},
        {"tflops": 5.0, "passed": True},
        {"tflops": 7.0, "passed": None},
        {"tflops": 3.0, "passed": True},
    ]
    ranked = _rank_verified(results, lambda r: r["tflops"], lambda r: r["passed"])
    assert [r["tflops"] for r in ranked] == [7.0, 5.0, 3.0]
    assert "1 kernel(s) failed verify" in capsys.readouterr().out
    assert (
        _rank_verified(
            [{"tflops": 1.0, "passed": False}],
            lambda r: r["tflops"],
            lambda r: r["passed"],
        )
        == []
    )
