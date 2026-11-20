# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A conv kernel label rebuilds its kernel from the repo alone.

``rebuild_job_for_identity`` runs the generators first; an identity their
grid does not walk -- the opt-in knob sets of the shipped pin and route
tables, or a grid narrowed by the ``ROCKE_AOT_*`` scoping variables -- is read
off its own fields by ``job_from_identity``. These tests hold that inverse to
the generators (same spec, same capability axes for every job they build),
check that every label of the shipped tables resolves and builds its kernel
IR in a process with no ``ROCKE_AOT_*`` variable set, and that
``job_for_identity`` stays the strict inverse of the grid (it refuses what the
sweep does not build).

CPU only: kernel IR is built, nothing is compiled.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rocke.core.arch import ArchTarget

from kernels.conv import aot_jobs as aj
from kernels.conv.kernel_identity import KernelIdentity

_LIBRARY = Path(__file__).resolve().parents[1]
_TABLE_DIRS = ("conv_pin_tables", "conv_route_tables")
_FLAVOR = "llvm22"


def _shipped_labels() -> dict:
    """Every kernel label of the shipped pin and route tables -> one use."""
    labels: dict = {}
    for sub in _TABLE_DIRS:
        for path in sorted((_LIBRARY / "dispatch" / sub).glob("*.json")):
            doc = json.loads(path.read_text())
            for mode, body in doc["modes"].items():
                for case, row in body["cases"].items():
                    if row.get("kernel"):
                        labels.setdefault(
                            row["kernel"].split()[0], f"{path.stem}/{sub}#{mode}:{case}"
                        )
    return labels


_LABELS = _shipped_labels()


def test_the_shipped_tables_name_kernels():
    assert len(_LABELS) > 50
    assert {KernelIdentity.from_label(lab).direction for lab in _LABELS} == {
        "fwd",
        "dgrad",
        "wgrad",
    }


@pytest.mark.parametrize("label", sorted(_LABELS))
def test_every_shipped_label_rebuilds_its_kernel(label):
    ident = KernelIdentity.from_label(label)
    job = aj.rebuild_job_for_identity(ident)
    assert job.identity == ident
    assert aj.job_from_identity(ident) == job
    kernel, meta = aj.kernel_for_job(job, ident.arch, ident.dtype_a)
    assert kernel is not None
    assert meta["block_size"] % ident.wave_size == 0


def test_shipped_labels_resolve_without_the_aot_scope_environment():
    """In a fresh process with every ``ROCKE_AOT_*`` variable cleared (and in
    one narrowed to 3-D, 8-wide caches), every shipped label still resolves."""
    script = (
        "import json, sys\n"
        "from kernels.conv.aot_jobs import rebuild_job_for_identity\n"
        "from kernels.conv.kernel_identity import KernelIdentity\n"
        "bad = []\n"
        "for lab in json.loads(sys.argv[1]):\n"
        "    try:\n"
        "        rebuild_job_for_identity(KernelIdentity.from_label(lab))\n"
        "    except LookupError as e:\n"
        "        bad.append(str(e))\n"
        "print(len(bad)); print('\\n'.join(bad[:3]))\n"
    )
    base = {k: v for k, v in os.environ.items() if not k.startswith("ROCKE_AOT_")}
    for extra in ({}, {"ROCKE_AOT_SPATIAL": "3d", "ROCKE_AOT_VECS": "8"}):
        out = subprocess.run(
            [sys.executable, "-c", script, json.dumps(sorted(_LABELS))],
            env={**base, **extra},
            cwd=_LIBRARY,
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        ).stdout
        assert out.splitlines()[0] == "0", (extra, out)


def test_job_for_identity_stays_the_strict_inverse_of_the_grid():
    """Packaging asks ``job_for_identity`` whether the sweep builds an
    identity: an identity off the grid is refused there even though its
    fields spell a valid job, which only ``rebuild_job_for_identity`` reads."""
    off_grid = []
    for label in sorted(_LABELS):
        ident = KernelIdentity.from_label(label)
        try:
            aj.job_for_identity(ident)
        except LookupError:
            off_grid.append(ident)
    assert off_grid, "every shipped label is on the default grid"
    from benchmarks.common.reproduce_kernel import job_and_builder

    for ident in off_grid[:3]:
        assert aj.rebuild_job_for_identity(ident).identity == ident
        # reproduce_kernel.py rebuilds a label off the grid too.
        assert job_and_builder(ident)[0] == aj.rebuild_job_for_identity(ident)
    on_grid = next(iter(_grid_jobs("gfx950", "fwd", "bf16", 1)))
    basic = dataclasses.replace(on_grid.identity, pipeline="basic")
    with pytest.raises(LookupError):
        aj.job_for_identity(basic)


def test_the_dispatcher_imports_under_narrowed_aot_scoping():
    """The grouped-convolution registry rebuilds its listed instances' jobs at
    import; a cache-scoping variable that leaves their rank or width out of
    the grid must not make the dispatcher unimportable."""
    base = {k: v for k, v in os.environ.items() if not k.startswith("ROCKE_AOT_")}
    for extra in ({"ROCKE_AOT_SPATIAL": "3d"}, {"ROCKE_AOT_VECS": "8"}):
        subprocess.run(
            [sys.executable, "-c", "import dispatch.grouped_convolution"],
            env={**base, **extra},
            cwd=_LIBRARY,
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )


def _effective(job: aj.BuildJob) -> dict:
    """The spec fields a job builds with: its kwargs over the spec defaults."""
    out = {}
    for f in dataclasses.fields(aj._spec_class(job.direction)):
        if f.name in ("problem", "data"):
            continue
        if f.name in job.spec_kwargs:
            out[f.name] = job.spec_kwargs[f.name]
        elif f.default is not dataclasses.MISSING:
            out[f.name] = f.default
        else:
            out[f.name] = "<factory>"
    return out


def _grid_jobs(arch: str, direction: str, dtype: str, n_geometries: int):
    target = ArchTarget.from_gfx(arch)
    geos = aj._geometry_list(target, dtype, direction)
    step = max(1, len(geos) // n_geometries)
    return aj._direction_jobs(
        direction,
        arch,
        dtype,
        target,
        aj.CACHE_WGRAD_SPLIT_KS,
        aj.CACHE_DGRAD_MAX_SUB_GEMMS,
        geometries=geos[::step],
        llvm_flavor=_FLAVOR,
        **(
            {"streamk_modes": aj.CACHE_WGRAD_STREAMK_MODES}
            if direction == "wgrad"
            else {}
        ),
    )


@pytest.mark.parametrize(
    "arch,direction,dtype",
    [
        (arch, direction, dtype)
        for arch in ("gfx942", "gfx950", "gfx1250")
        for direction in ("fwd", "wgrad", "dgrad")
        for dtype in ("bf16", "fp32")
    ],
)
def test_the_inverse_is_the_generators_job(arch, direction, dtype):
    """For every job the grid builds, valid or not, the job read off its
    identity builds the same spec on the same capability axes."""
    n = 0
    for job in itertools.islice(_grid_jobs(arch, direction, dtype, 4), 0, None, 3):
        inv = aj.job_from_identity(job.identity, validate=False)
        assert inv.identity == job.identity
        assert inv.caps == job.caps, job.identity.label()
        assert _effective(inv) == _effective(job), job.identity.label()
        n += 1
    assert n


def test_the_inverse_refuses_a_field_no_spec_has():
    job = next(iter(_grid_jobs("gfx942", "fwd", "bf16", 1)))
    for bad in (
        dict(dtype_d="fp16"),
        dict(tapk_pingpong=4),  # a dgrad knob on a forward identity
        dict(warp_tile_n=job.identity.warp_tile_m * 2),
        dict(wave_size=32),
    ):
        with pytest.raises(LookupError):
            aj.job_from_identity(dataclasses.replace(job.identity, **bad))


def test_wave32_forward_k_loops_carry_no_wave64_drivers():
    """The register-prefetch, per-tile decode, swizzle and tap-order K loops
    are wave64 MFMA drivers: a WMMA target's forward grid has none of them
    (they would only be rejected by the validator, one by one)."""
    wave64_only = {
        "prefetch_regs",
        "reg_prefetch",
        "uniform_k_decode",
        "lds_xor",
        "uk_tap_fast",
        "xcd_remap",
        "a_direct",
    }
    for arch in ("gfx1201", "gfx1250"):
        assert ArchTarget.from_gfx(arch).wave_size == 32
        for pipeline in ("mem", "compv3", "compv4", "wavelet"):
            for kloop in aj._fwd_k_loops(pipeline, arch):
                assert not (wave64_only & {k for k, v in kloop.items() if v}), (
                    arch,
                    pipeline,
                    kloop,
                )


def test_forward_k_loop_variants_are_built_for_3d_only():
    """The gfx942 forward K-loop variants (register prefetch, per-tile decode,
    swizzle, XCD and tap orders) are built on the 3-D capabilities only: a
    2-D capability gets the plain, unroll_k and (on "mem") async_dma loops, as
    every other arch does."""
    variant_keys = {
        "prefetch_regs",
        "reg_prefetch",
        "uniform_k_decode",
        "lds_xor",
        "uk_tap_fast",
        "xcd_remap",
    }
    for pipeline in ("mem", "compv3", "compv4"):
        flat = aj._fwd_k_loops(pipeline, "gfx942", is_3d=False)
        assert flat == aj._fwd_k_loops(pipeline, "gfx950", is_3d=True)
        assert not any(variant_keys & {k for k, v in kl.items() if v} for kl in flat)
        assert len(aj._fwd_k_loops(pipeline, "gfx942")) > len(flat)
        pairs = aj._fwd_kloop_caps(pipeline, "gfx942")
        assert all(kl in flat for kl, caps in pairs if not caps["is_3d"])


def test_identity_init_and_hash_are_the_dataclass_ones():
    """KernelIdentity's __dict__ __init__ and its field reader in stable_hash
    stand in for the dataclass ones: same parameters and defaults, a frozen
    instance with every field set, and the hash of asdict()."""
    import inspect

    from kernels.conv import kernel_identity as ki

    params = list(inspect.signature(KernelIdentity.__init__).parameters.values())[1:]
    fields = [f for f in dataclasses.fields(KernelIdentity) if f.init]
    assert [p.name for p in params] == [f.name for f in fields]
    for p, f in zip(params, fields):
        if f.default is not dataclasses.MISSING:
            assert p.default is f.default, f.name
        elif f.default_factory is dataclasses.MISSING:
            assert p.default is inspect.Parameter.empty, f.name
    assert ki._FIELD_NAMES == tuple(f.name for f in dataclasses.fields(KernelIdentity))

    job = next(iter(_grid_jobs("gfx942", "fwd", "bf16", 1)))
    ident = job.identity
    assert set(vars(ident)) == set(ki._FIELD_NAMES)
    assert KernelIdentity(**dataclasses.asdict(ident)) == ident
    assert KernelIdentity(*ki._field_values(ident)) == ident
    llvm_free = {
        k: v for k, v in dataclasses.asdict(ident).items() if k != "llvm_flavor"
    }
    (flavor,) = [f for f in fields if f.name == "llvm_flavor"]
    assert KernelIdentity(**llvm_free).llvm_flavor == flavor.default_factory()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ident.tile_m = 1
    with pytest.raises(TypeError):
        KernelIdentity()

    # stable_hash reads the fields as asdict() returns them.
    for label in sorted(_LABELS) + [ident.label()]:
        i = KernelIdentity.from_label(label)
        assert dict(zip(ki._FIELD_NAMES, ki._field_values(i))) == dataclasses.asdict(i)
