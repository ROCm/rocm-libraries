# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device evidence for the ``lds_swizzle`` LDS write-granularity rule.

``lds_swizzle`` XORs whole ``2^L``-half groups of an LDS row; the LDS write
chunk (the DTL chunk, or the ``load_vec`` store of the global-load path) must
not be wider than a group, or one chunk is scattered over several groups and
the fragment reads fetch the wrong elements. The geometry rule
(:func:`lds_swizzle_params`) therefore widens the group to the LDS store vector
(:func:`_lds_swizzle_vec`), so the narrow-fragment atoms (16x16x16, 32x32x8:
``L = 2`` from the fragment) get ``L = 3`` against 8-half chunks; the gate
(``is_valid_spec``) still refuses any spec whose group stays narrower than its
chunk.

* ``test_narrow_atoms_get_a_chunk_wide_group``: every spec of :data:`_NARROW`
  is admitted and its group holds the LDS write chunk (CPU).
* ``test_fragment_group_computes_wrong``: the same specs built with the old
  fragment-sized group (``CK_SWZ_*`` debug overrides) compute wrong outputs on
  the device where the chunk is wider than the fragment, which pins the rule to
  observed wrong results, not to a guess.
* ``test_admitted_specs_compute_right``: the narrow-atom specs and their
  neighbours (the wide atoms, the narrow atoms without the swizzle, the
  ``vgpr_prefetch`` group-sized LDS writes) are exact against a float64
  reference.

The device tests run on a gfx942 / gfx950 device only.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from rocke.instances.common import gemm_universal as GU
from rocke.instances.common.gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
    mono_data_spec,
)

_ARCH = None
try:
    from rocke.runtime.hip_module import get_device_arch

    _ARCH = get_device_arch(0)
except Exception:  # noqa: BLE001 - no runtime / no device
    _ARCH = None

_DEVICE = _ARCH in ("gfx942", "gfx950")
_device = pytest.mark.skipif(
    not _DEVICE, reason=f"needs a gfx942 / gfx950 device; device is {_ARCH}"
)

_M = _N = 512
_K = 256

_VGPR = TraitSpec(pipeline="compv4", epilogue="default", lds_swizzle=True)
_VGPR_DB = replace(_VGPR, ab_double_buffer=True)
_DTL = TraitSpec(
    pipeline="compv4",
    epilogue="default",
    direct_to_lds=True,
    dtl_prefetch=True,
    lds_swizzle=True,
    dtl_cache_a=0,
    dtl_cache_b=0,
)

# (label, tile, trait, arches): specs whose fragment-sized (4-half) group is
# narrower than the LDS write chunk on ``arches``. The global-load path stores
# 8-half vectors on every arch; the DTL chunk is 8 halves on gfx950 (16-byte
# LDS DMA) but 2 halves on gfx942 (4-byte LDS DMA), where a 4-half group is
# wide enough.
_NARROW = [
    (
        f"{path}_{'x'.join(map(str, atom))}",
        TileSpec(128, 128, 32, 2, 2, 1, *atom),
        tr,
        arches,
    )
    for atom in ((16, 16, 16), (32, 32, 8))
    for path, tr, arches in (
        ("vgpr", _VGPR, ("gfx942", "gfx950")),
        ("vgpr_db", _VGPR_DB, ("gfx942", "gfx950")),
        ("dtl", _DTL, ("gfx950",)),
    )
]

# the admitted neighbours of the refused specs
_ADMITTED = [
    ("vgpr_16x16x32", TileSpec(128, 128, 32, 2, 2, 1, 16, 16, 32), _VGPR),
    ("vgpr_db_16x16x32", TileSpec(128, 128, 32, 2, 2, 1, 16, 16, 32), _VGPR_DB),
    ("vgpr_32x32x16", TileSpec(128, 128, 32, 2, 2, 1, 32, 32, 16), _VGPR),
    ("dtl_16x16x32", TileSpec(128, 128, 32, 2, 2, 1, 16, 16, 32), _DTL),
    (
        "noswz_16x16x16",
        TileSpec(128, 128, 32, 2, 2, 1, 16, 16, 16),
        replace(_VGPR, lds_swizzle=False),
    ),
    (
        "noswz_32x32x8",
        TileSpec(128, 128, 32, 2, 2, 1, 32, 32, 8),
        replace(_VGPR, lds_swizzle=False),
    ),
]


def _spec(label, tile, trait, dtype="fp16"):
    return UniversalGemmSpec(
        name=f"swz_{label}",
        tile=tile,
        trait=trait,
        data=mono_data_spec(dtype),
        wave_size=64,
    )


def _arch():
    return _ARCH if _DEVICE else "gfx950"


def _write_chunk(spec, arch):
    """The LDS write chunk in halves: the DTL chunk, else the load vector."""
    from rocke.core.arch import ArchTarget

    if spec.trait.direct_to_lds:
        return min(4, ArchTarget.from_gfx(arch).async_lds_max_dwords) * 2
    return GU._choose_load_vec(spec)


@pytest.mark.parametrize(
    "label,tile,trait,arches", _NARROW, ids=[r[0] for r in _NARROW]
)
def test_narrow_atoms_get_a_chunk_wide_group(label, tile, trait, arches):
    for arch in ("gfx942", "gfx950"):
        spec = _spec(label, tile, trait)
        ok, why = is_valid_spec(spec, arch=arch)
        assert ok, (arch, label, why)
        op = GU._resolve_mma_op(spec, arch)
        _r, _w, l_ = GU._spec_swizzle_params(spec, op.a_frag_len, op.b_frag_len)
        assert (1 << l_) % _write_chunk(spec, arch) == 0, (arch, label, l_)
        if arch in arches:
            # the fragment alone would give a narrower group than the chunk
            assert op.a_frag_len < _write_chunk(spec, arch), (arch, label)


def _run(spec, monkeypatch=None, swz=None):
    """Build (with the debug ``CK_SWZ_R/W/L`` overrides ``swz`` when given),
    run once on random data and return the worst error relative to the f64
    bound."""
    import torch

    from rocke.core.ir import PtrType
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    if swz is not None:
        for k, v in zip(("CK_SWZ_R", "CK_SWZ_W", "CK_SWZ_L"), swz):
            monkeypatch.setenv(k, str(v))
    kd = build_universal_gemm(spec, arch=_arch())
    if swz is not None:
        monkeypatch.undo()
    art = compile_kernel(kd, arch=_arch(), capture_ir_text=False, backend="python")
    sig = []
    for prm in kd.params:
        nm = getattr(prm.type, "name", "") or ""
        if isinstance(prm.type, PtrType):
            sig.append(
                {
                    "name": prm.name,
                    "type": nm if nm.startswith("ptr<") else f"ptr<{nm}>",
                }
            )
        else:
            sig.append({"name": prm.name, "type": str(nm)})
    launch = KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)
    g = torch.Generator(device="cuda")
    g.manual_seed(7)
    A = (torch.rand(_M, _K, device="cuda", generator=g) * 2 - 1).half()
    B = (torch.rand(_N, _K, device="cuda", generator=g) * 2 - 1).half()
    C = torch.full((_M, _N), float("nan"), dtype=torch.float16, device="cuda")
    t = spec.tile
    launch(
        {"A": A, "B": B, "C": C, "M": _M, "N": _N, "K": _K},
        config=LaunchConfig(
            grid=(-(-_N // t.tile_n), -(-_M // t.tile_m), 1),
            block=(spec.block_size, 1, 1),
        ),
    )
    torch.cuda.synchronize()
    ref = A.double() @ B.double().t()
    bound = (
        ref.abs() * 2.0**-10
        + (A.double().abs() @ B.double().abs().t()) * (_K * 2.0**-23)
        + 2.0**-24
    )
    err = ((C.double() - ref).abs() / bound).nan_to_num(nan=float("inf"))
    return float(err.max())


@_device
@pytest.mark.parametrize(
    "label,tile,trait,arches", _NARROW, ids=[r[0] for r in _NARROW]
)
def test_fragment_group_computes_wrong(label, tile, trait, arches, monkeypatch):
    if _ARCH not in arches:
        pytest.skip(f"the fragment group holds the chunk on {_ARCH}")
    spec = _spec(label, tile, trait)
    op = GU._resolve_mma_op(spec, _ARCH)
    if op is None:
        pytest.skip(f"no {label} atom on {_ARCH}")
    # the old geometry rule: the group of one fragment (lds_swizzle_params
    # without the store vector)
    swz = GU.lds_swizzle_params(op.a_frag_len, op.b_frag_len, tile.tile_k)
    worst = _run(spec, monkeypatch, swz=swz)
    assert worst > 1.0, f"{label}: fragment-group build was exact (worst {worst})"


@_device
@pytest.mark.parametrize(
    "label,tile,trait",
    [r[:3] for r in _NARROW] + _ADMITTED,
    ids=[r[0] for r in _NARROW] + [r[0] for r in _ADMITTED],
)
def test_admitted_specs_compute_right(label, tile, trait):
    spec = _spec(label, tile, trait)
    ok, why = is_valid_spec(spec, arch=_arch())
    if not ok and "unsupported" in why:
        pytest.skip(why)  # the atom does not exist on this arch
    assert ok, why
    assert _run(spec) <= 1.0, label
