# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx950 dgrad dispatcher only ships binaries the AOT caches hold.

CPU-only (no GPU, nothing compiled). For a synthetic grid of dgrad requests
every dispatched selection is mapped to its cache identity
(``benchmarks.common.grouped_conv.dgrad_dispatch_cache``) and checked against
the grids the caches are built from:

* implicit GEMM: the identity is one the dgrad generator produces
  (``kernel_sweep.job_for_identity``), its spec passes the cache build's
  validator, and ``KernelCache.supports_problem`` offers it for the request;
* direct: the identity's knobs are in the cache cell of its variant and
  capabilities (``direct_kernel_sweep._caps_jobs``), and ``plan_for`` gives a
  launch plan for the request whose single step has the signature, grid and
  block dispatch returns.

Also the launch plumbing: real signatures on the direct candidates (and in
``dispatch_conv_grouped_all``), ``launch_values_for`` on a direct pick, and
the spec-hash stability of the new ``ConvGroupedSpec`` fields.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict, replace

import pytest
from benchmarks.common import direct_kernel_sweep as dks
from benchmarks.common import kernel_sweep as ks
from benchmarks.common.grouped_conv.dgrad_dispatch_cache import (
    dgrad_cache_identity,
    dgrad_cache_problem,
)
from benchmarks.common.kernel_cache import KernelCache
from dispatch.grouped_convolution import (
    _DGRAD_SPEC_FIELDS,
    ConvDirectDgradSpec,
    ConvGroupedRequest,
    ConvGroupedSpec,
    _kernel_id,
    _signature_for,
    conv_grouped_candidates,
    dispatch_conv_grouped,
    dispatch_conv_grouped_all,
    launch_values_for,
)
from kernels.common.conv_abi import conv_direct_args_signature
from rocke.dispatch.core import stable_json_hash

_ARCH = "gfx950"
_WAVE = 64


def _req(N, C, K, H, W, Y, G, *, pad=None, stride=1, dtype="bf16", **kw):
    pad = Y // 2 if pad is None else pad
    return ConvGroupedRequest(
        N=N,
        C=C,
        K=K,
        Hi=H,
        Wi=W,
        Y=Y,
        X=Y,
        G=G,
        pad_h=pad,
        pad_w=pad,
        stride_h=stride,
        stride_w=stride,
        dtype=dtype,
        arch=_ARCH,
        direction="dgrad",
        **kw,
    )


def _grid(seed: int, n: int) -> list[ConvGroupedRequest]:
    """A seeded spread of dgrad requests over every gfx950 candidate's region:
    dense and grouped implicit GEMM (strided, 'valid' padding, the halo
    filters and widths), small-channel grouped / dense direct, depthwise."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        kind = rng.choice(("dense", "grouped", "small", "depthwise"))
        Y = rng.choice((1, 3, 5, 7))
        H = rng.choice((7, 9, 12, 15, 28, 33, 50, 62, 63, 100, 130))
        W = rng.choice((H, rng.choice((5, 8, 12, 17, 32, 48, 96))))
        N = rng.choice((1, 2, 8, 32, 96))
        dtype = rng.choice(("bf16", "fp16"))
        stride = rng.choice((1, 1, 1, 2))
        pad = rng.choice((Y // 2, Y // 2, 0))
        if kind == "dense":
            C = rng.choice((24, 48, 64, 96, 256, 320, 576, 1152))
            K = rng.choice((32, 64, 128, 256, 448, 1152))
            G = 1
        elif kind == "grouped":
            G = rng.choice((2, 4, 12, 24))
            C = G * rng.choice((8, 16, 48, 64, 128))
            K = G * rng.choice((16, 64, 96, 128))
        elif kind == "small":
            G = rng.choice((1, 3, 8, 16, 32, 64))
            C = G * rng.choice((4, 8, 12, 16, 20, 24, 28, 32))
            K = G * rng.choice((4, 8, 12, 16, 20, 24, 28, 32))
        else:
            Y = rng.choice((3, 5, 7, 9))
            pad = rng.choice((Y // 2, Y // 2, 0))
            C = K = G = rng.choice((8, 32, 64, 128, 384, 768))
        req = _req(N, C, K, H, W, Y, G, pad=pad, stride=stride, dtype=dtype)
        p = req.dims()
        if p.get("Ho", 0) <= 0 or p.get("Wo", 0) <= 0:
            continue
        out.append(req)
    return out


_DIRECT_CELLS: dict = {}


def _direct_cell(variant: str, ident) -> list[dict]:
    """Knob dicts of the direct cache cell of ``ident``'s variant and caps."""
    key = (variant, dks._caps_of(ident), ident.dtype_a)
    if key not in _DIRECT_CELLS:
        _DIRECT_CELLS[key] = [
            json.loads(job.identity.knobs)
            for _, job in dks._caps_jobs(
                _ARCH, _WAVE, variant, "dgrad", key[1], ident.dtype_a
            )[1]
        ]
    return _DIRECT_CELLS[key]


def _check_igemm(req, r, cache) -> None:
    ident = dgrad_cache_identity(req, r.spec)
    job = ks.job_for_identity(ident)  # raises if the grid has no such binary
    assert ks._spec_is_valid(job, _ARCH, ident.dtype_a), ident.label()
    ok, why = cache.supports_problem(ident, dgrad_cache_problem(req, r.spec))
    assert ok, (ident.label(), why)
    # The instance dispatch builds is the binary's configuration.
    inst = r.spec.to_dgrad_spec(dgrad_cache_problem(req, r.spec))
    for f in ("vector_size_a", "vector_size_b", "vector_size_c"):
        assert getattr(inst, f) == getattr(ident, f)
    assert (inst.tap_outer_k, inst.dy_halo, inst.halo_max_wo) == (
        ident.tap_outer_k,
        ident.dy_halo,
        ident.halo_max_wo,
    )


def _check_direct(req, r, cache) -> None:
    spec = r.spec
    ident = dgrad_cache_identity(req, spec)
    assert spec.knob_dict() in _direct_cell(spec.variant, ident), ident.label()
    problem = dgrad_cache_problem(req, spec)
    ok, why = cache.supports_problem(ident, problem)
    assert ok, (ident.label(), why)
    cache.put(ident, b"placeholder", {"kernel_name": ident.label()})
    plan, why = dks.plan_for(
        cache, ident, cache.hsaco_path(ident), cache.meta(ident), problem, _ARCH
    )
    assert plan is not None, (ident.label(), why)
    (step,) = plan.steps
    assert plan.workspaces == {}
    assert step.signature == list(r.signature)
    assert step.grid == tuple(r.grid)
    assert step.block == tuple(r.block)


@pytest.mark.parametrize("seed", range(4))
def test_dgrad_dispatch_outputs_are_cached_identities(tmp_path, seed):
    """Every dgrad selection on gfx950 is a binary the AOT caches build and
    offer for the request (D13: dispatch can only pick what was built)."""
    cache = KernelCache(tmp_path, _ARCH)
    seen = set()
    for req in _grid(seed, 60):
        try:
            r = dispatch_conv_grouped(req)
        except ValueError:
            continue  # no gfx950 dgrad path (e.g. a non-cached depthwise filter)
        seen.add(r.candidate.name)
        if isinstance(r.spec, ConvDirectDgradSpec):
            _check_direct(req, r, cache)
        else:
            _check_igemm(req, r, cache)
    assert "implicit_gemm_conv_dgrad" in seen


def test_grid_reaches_every_candidate():
    names = set()
    for seed in range(4):
        for req in _grid(seed, 60):
            try:
                names.add(dispatch_conv_grouped(req).candidate.name)
            except ValueError:
                pass
    assert names == {c.name for c in conv_grouped_candidates("dgrad")}


@pytest.mark.parametrize(
    "req",
    [
        _req(32, 128, 128, 28, 28, 3, 32),  # 4c, row-staged
        _req(64, 256, 256, 12, 12, 1, 64, dtype="fp16"),  # 4c 1x1
        _req(8, 128, 128, 12, 12, 3, 8),  # fused + row-stream stack
        _req(8, 256, 256, 12, 12, 3, 8),  # fused, fold_k32
        _req(4, 8, 8, 32, 32, 5, 1),  # fused, groups == 1
        _req(8, 384, 384, 13, 13, 7, 384),  # windowed MFMA
        _req(8, 96, 96, 20, 20, 3, 96, dtype="fp16"),  # windowed VALU
        _req(8, 64, 64, 28, 28, 5, 64, stride=2),  # streaming depthwise
    ],
)
def test_direct_signatures_are_the_cached_plans(tmp_path, req):
    """A direct pick's signature (dispatch result, candidate callable,
    _signature_for and dispatch_conv_grouped_all) is the step signature
    plan_for gives its cached binary, and launch_values_for binds the step's
    kernargs."""
    r = dispatch_conv_grouped(req)
    assert isinstance(r.spec, ConvDirectDgradSpec)
    cache = KernelCache(tmp_path, _ARCH)
    _check_direct(req, r, cache)
    sig = list(r.signature)
    assert list(r.candidate.signature(r.spec)) == sig
    assert list(_signature_for(req, r.spec)) == sig
    assert sig == conv_direct_args_signature(req.dtype, direction=r.spec.abi_direction)
    (top,) = [x for x in dispatch_conv_grouped_all(req) if x.candidate is r.candidate]
    assert list(top.signature) == sig

    ident = dgrad_cache_identity(req, r.spec)
    plan, _ = dks.plan_for(
        cache,
        ident,
        cache.hsaco_path(ident),
        cache.meta(ident),
        dgrad_cache_problem(req, r.spec),
        _ARCH,
    )
    (step,) = plan.steps
    ptrs = {"dy": 11, "w": 22, "dx": 33}
    sizes = {"dy": 1000, "w": 2000, "dx": 3000}
    values = launch_values_for(
        req,
        r.spec,
        A_ptr=11,
        B_ptr=22,
        D_ptr=33,
        A_bytes=1000,
        B_bytes=2000,
        D_bytes=3000,
    )
    assert values == dks.launch_values(step, ptrs, sizes)
    assert [e["name"] for e in sig] == list(values)
    with pytest.raises(ValueError, match="no workspace"):
        launch_values_for(
            req,
            r.spec,
            A_ptr=1,
            B_ptr=2,
            D_ptr=3,
            A_bytes=4,
            B_bytes=5,
            D_bytes=6,
            sub_gemm_buf=7,
            num_sub_gemms=1,
        )


def test_igemm_launch_values_still_need_the_record_buffer():
    req = _req(2, 384, 384, 30, 30, 3, 1)
    r = dispatch_conv_grouped(req)
    assert isinstance(r.spec, ConvGroupedSpec)
    with pytest.raises(ValueError, match="sub_gemm_buf"):
        launch_values_for(
            req, r.spec, A_ptr=1, B_ptr=2, D_ptr=3, A_bytes=4, B_bytes=5, D_bytes=6
        )
    values = launch_values_for(
        req,
        r.spec,
        A_ptr=1,
        B_ptr=2,
        D_ptr=3,
        A_bytes=4,
        B_bytes=5,
        D_bytes=6,
        sub_gemm_buf=7,
        num_sub_gemms=1,
    )
    assert [e["name"] for e in r.signature] == list(values)


@pytest.mark.parametrize("direction", ["fwd", "wgrad"])
def test_new_spec_fields_leave_existing_spec_hashes_alone(direction):
    """The dgrad binary fields are hashed only when set: a fwd / wgrad spec
    (and a dgrad spec without them) hashes as it did before they existed."""
    req = replace(_req(2, 64, 64, 12, 12, 3, 1), direction=direction)
    if direction == "wgrad":
        req = replace(req, arch="gfx942", vec_size_c=1)
    r = dispatch_conv_grouped(req)
    legacy = {k: v for k, v in asdict(r.spec).items() if k not in _DGRAD_SPEC_FIELDS}
    assert r.kernel_id.spec_hash == stable_json_hash(legacy, n=16)


def test_dgrad_fields_rejected_outside_dgrad():
    base = {
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": 64,
        "warp_m": 2,
        "warp_n": 2,
        "warp_tile_mn": 32,
        "warp_tile_k": 16,
        "pipeline": "mem",
        "epilogue": "default",
        "dtype": "bf16",
        "arch": _ARCH,
    }
    for field, value in itertools.product(_DGRAD_SPEC_FIELDS, (1,)):
        with pytest.raises(ValueError, match="only apply to direction='dgrad'"):
            ConvGroupedSpec(direction="fwd", **base, **{field: value})
    spec = ConvGroupedSpec(direction="dgrad", **base, dy_halo=2, halo_max_wo=64)
    assert "halo2" in spec.kernel_name() and "hwo64" in spec.kernel_name()


def test_dgrad_spec_hash_tracks_the_binary_fields():
    """Two dgrad picks that differ only in a binary field hash differently."""
    req = _req(8, 896, 384, 15, 15, 3, 1)
    r = dispatch_conv_grouped(req)
    assert r.spec.dy_halo
    other = replace(r.spec, dy_halo_kouter_pad=0 if r.spec.dy_halo_kouter_pad else 32)
    assert _kernel_id(req, r.candidate, other).spec_hash != r.kernel_id.spec_hash
    assert other.kernel_name() != r.spec.kernel_name()
