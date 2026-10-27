# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device numerics of the backward block GEMM and the C-to-A relabel vs numpy.

One GEMM per warp grid, atom, operand source and dtype on the visible device,
plus the relabel chain (``S = Q K^T`` on ``(1, W)``, then ``dV = P^T dO`` on
``(W, 1)`` with ``P^T`` taken from the ``S`` registers; on gfx950 also with the
K=32 consumer atom and the K-permuted ``dO`` load). Negative controls break the
relabel on purpose and require the device result to be wrong, so a pass is not
an artefact of an insensitive check.

Skipped unless the device is gfx942, gfx950, gfx1151 or gfx1201. Uses only the
torch-free runtime (numpy + HIP + comgr).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kernels.common import _attention_bwd_gemm as G
from kernels.common._attention_bwd_caps import BWD_ARCHES, bwd_arch_caps

_ARCH = None
try:
    from rocke.runtime.hip_module import get_device_arch

    _ARCH = get_device_arch(0)
except Exception:  # noqa: BLE001 - no runtime / no device
    _ARCH = None

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        _ARCH not in BWD_ARCHES, reason=f"needs one of {BWD_ARCHES}; device is {_ARCH}"
    ),
]

from ._attention_bwd_gemm_probe import run_chain_probe, run_gemm_probe  # noqa: E402

DTYPES = ("fp16", "bf16")
TOL = {"rtol": 2e-3, "atol": 2e-3}


def _grids():
    if _ARCH is None or _ARCH not in BWD_ARCHES:
        return [(1, 1)]
    out = [(1, 4), (4, 1), (2, 2), (1, 1)]
    if 8 in bwd_arch_caps(_ARCH).legal_waves:
        out += [(1, 8), (8, 1), (2, 4)]
    return out


def _atom_ks():
    return (16, 32) if _ARCH == "gfx950" else (16,)


_SOURCES = [
    # (a_src, b_src, a_storage, b_storage)
    ("lds", "lds", "k_inner", "k_inner"),
    ("reg", "reg", "k_inner", "k_outer"),
    ("reg", "lds", "k_outer", "k_inner"),
    ("lds", "reg", "k_outer", "k_outer"),
]


@pytest.mark.parametrize("src", _SOURCES, ids=lambda s: "-".join(x[:3] for x in s))
@pytest.mark.parametrize("atom_k", _atom_ks())
@pytest.mark.parametrize("grid", _grids(), ids=lambda g: f"{g[0]}x{g[1]}")
@pytest.mark.parametrize("dtype", DTYPES)
def test_block_gemm_matches_numpy(dtype, grid, atom_k, src):
    a_src, b_src, a_st, b_st = src
    got, ref = run_gemm_probe(
        _ARCH, dtype, m=16 * grid[0] * 2, n=16 * grid[1] * 2, k=64, grid=grid,
        atom_k=atom_k, a_src=a_src, b_src=b_src, a_storage=a_st, b_storage=b_st,
    )  # fmt: skip
    np.testing.assert_allclose(got, ref, **TOL)


def _chain_cases():
    ks = _atom_ks()
    cases = []
    for k_m0, k_n0, d, w in ((16, 64, 64, 4), (32, 128, 64, 4), (16, 16, 32, 1),
                             (32, 64, 128, 2)):  # fmt: skip
        for k0 in ks:
            for k1 in ks:
                if k1 == 32 and k_m0 < 32:
                    continue
                cases.append((k_m0, k_n0, d, w, k0, k1))
    return cases


@pytest.mark.parametrize("do_storage", ["k_outer", "k_inner"])
@pytest.mark.parametrize(
    "case", _chain_cases(), ids=lambda c: "m{}n{}d{}w{}k{}k{}".format(*c)
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_relabel_chain_matches_numpy(dtype, case, do_storage):
    k_m0, k_n0, d, w, k0, k1 = case
    s, s_ref, dv, dv_ref = run_chain_probe(
        _ARCH, dtype, k_m0=k_m0, k_n0=k_n0, d=d, waves=w, atom_k0=k0, atom_k1=k1,
        do_storage=do_storage,
    )  # fmt: skip
    np.testing.assert_allclose(s, s_ref, **TOL)
    np.testing.assert_allclose(dv, dv_ref, **TOL)


def _slice_cases():
    out = []
    for k_slice_steps in (1, 2, 4):
        for buffers, first in ((1, False), (2, False), (2, True)):
            out.append((k_slice_steps, buffers, first))
    return out


@pytest.mark.skipif(
    _ARCH not in ("gfx942", "gfx950"), reason="G4 key slicing is MFMA-only"
)
@pytest.mark.parametrize(
    "case", _slice_cases(), ids=lambda c: "s{}b{}{}".format(c[0], c[1], "f" * c[2])
)
@pytest.mark.parametrize("b_src", ["lds", "reg"])
@pytest.mark.parametrize("atom_k", _atom_ks())
@pytest.mark.parametrize("dtype", DTYPES)
def test_block_gemm_k_sliced_matches_mma(dtype, atom_k, b_src, case):
    """``mma_k_sliced`` (the G4 dS key slicing) is bit for bit :meth:`mma` on
    the device: a G4-shaped block (kM0 32 x D 64 over 8 atom-K steps, grid
    (1, 4)) with slices of 1, 2 and 4 steps, one or two buffers, slice 0 read
    ahead or not. Several slices exercise every slice's base offset."""
    steps, buffers, first = case
    kw = dict(m=32, n=64, k=8 * atom_k, grid=(1, 4), atom_k=atom_k, b_src=b_src)
    kw.update(a_storage="k_inner", b_storage="k_outer")
    ref, exact = run_gemm_probe(_ARCH, dtype, **kw)
    np.testing.assert_allclose(ref, exact, **TOL)
    got, _ = run_gemm_probe(
        _ARCH, dtype, k_slice=steps * atom_k, slice_buffers=buffers,
        slice_first=first, **kw,
    )  # fmt: skip
    assert np.array_equal(got, ref), f"k-sliced GEMM differs ({case})"


def _broken(plan):
    """A plan that still emits but places values wrongly."""
    if plan.kind == "xlane16":
        return dataclasses.replace(
            plan, xlane=tuple((j, s1, s0) for j, s0, s1 in plan.xlane)
        )
    if plan.needs_k_perm:  # drop the matching permutation of the B load
        return dataclasses.replace(plan, k_perm=tuple(range(len(plan.k_perm))))
    n = len(plan.slot_src)
    return dataclasses.replace(plan, slot_src=plan.slot_src[1:] + plan.slot_src[:1][:n])


def _readers():
    return ("plain", "tr16") if _ARCH == "gfx950" else ("plain",)


@pytest.mark.parametrize("reader", _readers())
@pytest.mark.parametrize("k1", _atom_ks())
def test_broken_relabel_is_detected(monkeypatch, k1, reader):
    real = G.plan_c_to_a
    monkeypatch.setattr(G, "plan_c_to_a", lambda c, a: _broken(real(c, a)))
    s, s_ref, dv, dv_ref = run_chain_probe(
        _ARCH, "fp16", k_m0=32, k_n0=64, d=64, waves=4, atom_k0=16, atom_k1=k1,
        do_reader=reader,
    )  # fmt: skip
    np.testing.assert_allclose(s, s_ref, **TOL)
    assert not np.allclose(dv, dv_ref, **TOL), "a broken relabel went unnoticed"


_on_gfx950 = pytest.mark.skipif(
    _ARCH != "gfx950", reason=f"the transpose read is gfx950-only; device is {_ARCH}"
)


@_on_gfx950
@pytest.mark.parametrize(
    "case", _chain_cases(), ids=lambda c: "m{}n{}d{}w{}k{}k{}".format(*c)
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_relabel_chain_through_the_transpose_read(dtype, case):
    """The chain with ``dO`` read by ``ds_read_tr16_b64``: the catalog plan for
    a 16x16x16 consumer, the stacked plan for the K-permuted B of a 16x16x32
    consumer (the backward's G1 / G3 path on the transpose read)."""
    k_m0, k_n0, d, w, k0, k1 = case
    s, s_ref, dv, dv_ref = run_chain_probe(
        _ARCH, dtype, k_m0=k_m0, k_n0=k_n0, d=d, waves=w, atom_k0=k0, atom_k1=k1,
        do_storage="k_outer", do_reader="tr16",
    )  # fmt: skip
    np.testing.assert_allclose(s, s_ref, **TOL)
    np.testing.assert_allclose(dv, dv_ref, **TOL)


@_on_gfx950
@pytest.mark.parametrize("offsets", [(16, 0), (0, 8), (0, 4)])
def test_wrong_stacked_offsets_are_detected_on_the_device(monkeypatch, offsets):
    """Negative control of the stacked plan: an emitter whose second read
    lands at another K offset (the proof bypassed) gives a wrong dV."""
    from kernels.common import _attention_bwd_frag as F

    real = F.tr16_read_plan

    def wrong(coords):
        plan = real(coords)
        if plan.kind != "stacked":
            return plan
        return dataclasses.replace(plan, k_offsets=offsets)

    monkeypatch.setattr(F, "tr16_read_plan", wrong)
    s, s_ref, dv, dv_ref = run_chain_probe(
        _ARCH, "fp16", k_m0=32, k_n0=64, d=64, waves=4, atom_k0=16, atom_k1=32,
        do_storage="k_outer", do_reader="tr16",
    )  # fmt: skip
    np.testing.assert_allclose(s, s_ref, **TOL)
    assert not np.allclose(dv, dv_ref, **TOL), "a wrong stacked read went unnoticed"
