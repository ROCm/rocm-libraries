# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The kernel names benchmarks print are lossless and lead back to the kernel.

``KernelIdentity.label()`` has to name every field, so that a name copied
out of a log is enough to rebuild the binary (``reproduce_kernel.py``). Three
things make that true, and each has a test here:

* ``from_label(label(i)) == i`` over every identity the AOT grids produce,
  and over identities with any field set to an awkward value -- including
  fields the label format has no tag for;
* the job rebuilt from an identity is the one ``--compile-all`` builds;
* the tags cannot be mistaken for one another.

CPU only: nothing is compiled.
"""

from __future__ import annotations

import dataclasses
import itertools
import json

import pytest

from benchmarks.common import direct_kernel_sweep as dks
from benchmarks.common import kernel_label
from benchmarks.common import kernel_sweep as ks
from benchmarks.common.kernel_cache import KernelIdentity
from rocke.core.arch import ArchTarget

_ARCH = "gfx950"
_FLAVOR = "llvm22"


def _round_trips(identity: KernelIdentity) -> None:
    name = identity.label()
    assert KernelIdentity.from_label(name) == identity, name


def _implicit_jobs(direction: str, n_geometries: int):
    target = ArchTarget.from_gfx(_ARCH)
    geos = ks._geometry_list(target, "bf16")
    # A spread of geometries, every pipeline and epilogue among them.
    step = max(1, len(geos) // n_geometries)
    return ks._direction_jobs(
        direction,
        _ARCH,
        "bf16",
        target,
        ks.CACHE_WGRAD_SPLIT_KS,
        64,
        geometries=geos[::step],
        llvm_flavor=_FLAVOR,
    )


def _direct_cells():
    target = ArchTarget.from_gfx(_ARCH)
    for variant, (direction, caps_list) in dks.DIRECT_CAPABILITIES.items():
        for caps in caps_list:
            yield (_ARCH, target.wave_size, variant, direction, caps, "fp16")


@pytest.mark.parametrize("direction", ["fwd", "wgrad", "dgrad"])
def test_implicit_grid_round_trips(direction):
    names = set()
    n = 0
    for job in _implicit_jobs(direction, n_geometries=40):
        _round_trips(job.identity)
        names.add(job.identity.label())
        n += 1
    assert n > 0
    # Distinct identities, distinct names (the round trip implies it; this
    # makes a failure point at the collision).
    assert len(names) == len({j.identity for j in _implicit_jobs(direction, 40)})


def test_direct_grid_round_trips():
    """Every direct identity, knobs included, before spec validation."""
    target = ArchTarget.from_gfx(_ARCH)
    n = 0
    for variant, (direction, caps_list) in dks.DIRECT_CAPABILITIES.items():
        for caps in caps_list:
            for dtype in dks.DIRECT_DTYPES:
                for knobs in dks._knob_grid(variant, _ARCH, dtype):
                    ident = dks._identity(
                        _ARCH,
                        target.wave_size,
                        variant,
                        f"direct_{direction}",
                        caps,
                        knobs,
                        dtype,
                    )
                    _round_trips(ident)
                    n += 1
            for helper, knobs in (
                (dks._TRANSPOSE, {}),
                (dks._REORGANIZE, {"fold_k32": True}),
            ):
                _round_trips(
                    dks._helper_identity(
                        _ARCH, target.wave_size, helper, caps, knobs, "bf16"
                    )
                )
    assert n > 0


def _base_identity(direct: bool) -> KernelIdentity:
    if direct:
        return dks._identity(
            _ARCH,
            64,
            "direct_grouped",
            "direct_fwd",
            dks.DirectCaps(KH=3, PAD=1, stride=1, cpg=16, kpg=16),
            dict(block_q=16, block_groups=2, double_buffer=True),
            "fp16",
        )
    return next(iter(_implicit_jobs("fwd", n_geometries=1))).identity


# Awkward values for every field type: separators, escapes, values that look
# like another type, negatives, and types a field does not declare.
_STRANGE = (
    0,
    1,
    7,
    -3,
    True,
    False,
    None,
    "",
    "x-y",
    "a.b",
    "k=v",
    "50%",
    "@x",
    "'q",
    "true",
    "null",
    "8",
    "-8",
    "sp ace",
    "ünï",
)


@pytest.mark.parametrize("direct", [False, True], ids=["implicit", "direct"])
def test_any_field_value_round_trips(direct):
    """A value no tag can spell falls back to ``@field=value`` -- including
    for fields the label format does not know about."""
    base = _base_identity(direct)
    for f in dataclasses.fields(KernelIdentity):
        for value in _STRANGE:
            _round_trips(dataclasses.replace(base, **{f.name: value}))


def test_knobs_round_trip_byte_for_byte():
    base = _base_identity(direct=True)
    for knobs in (
        json.dumps({"atom": "16x16x32", "waves_per_eu": None, "flag": False}),
        json.dumps({"tile": 8, "name": "8"}, sort_keys=True),
        # Not what json.dumps(sort_keys=True) writes: must survive verbatim.
        '{"b": 1, "a": 2}',
        '{"a":1}',
        json.dumps({"nested": [1, 2]}),
        "not json",
        "",
        "{}",
    ):
        _round_trips(dataclasses.replace(base, knobs=knobs))


def test_empty_knob_dict_is_not_the_implicit_baseline():
    # The implicit-GEMM baseline is "" -- an empty dict must not encode to
    # the same name.
    base = _base_identity(direct=False)
    for knobs in ("", "{}"):
        _round_trips(dataclasses.replace(base, knobs=knobs))
    assert dataclasses.replace(base, knobs="{}").label() != base.label()


def test_tags_do_not_overlap():
    samples = {
        tag: (
            tag.encode([True] * len(tag.fields))
            if tag.is_flag
            else tag.encode(list(range(2, 2 + len(tag.fields))))
        )
        for tag in kernel_label._TAGS
    }
    for tag, token in samples.items():
        matching = [t for t in kernel_label._TAGS if t.regex.fullmatch(token)]
        assert matching == [tag], token


def test_name_is_readable():
    ident = _base_identity(direct=False)
    ident = dataclasses.replace(ident, is_3d=True, unroll_k=True, grouped=True)
    name = ident.label()
    assert name.startswith(f"fwd-implicit_gemm-bf16-3d-t{ident.tile_m}x")
    assert "-grp-" in name and "-unroll-" in name
    assert name.endswith(f"-wave64-{_ARCH}-{_FLAVOR}")
    direct = _base_identity(direct=True).label()
    assert direct.startswith("fwd-direct_grouped-fp16-2d-f3x3-p1x1-s1x1-d1x1-c16k16-")
    assert "-block_q=16-" in direct and "-double_buffer=true-" in direct


def test_artifact_stem_fits_a_file_name():
    from benchmarks.common.reproduce_kernel import artifact_stem

    short = _base_identity(direct=True).label()
    assert artifact_stem(short) == short
    # Knobs that leave the baseline lengthen a direct name past NAME_MAX.
    long = dataclasses.replace(
        _base_identity(direct=True),
        knobs=json.dumps({f"knob_{i}": i for i in range(40)}, sort_keys=True),
    ).label()
    assert len(long.encode()) > 255
    stem = artifact_stem(long)
    assert len(f"{stem}.hsaco".encode()) <= 255
    assert stem.startswith(long[:64])
    assert stem == artifact_stem(long)
    other = artifact_stem(long.replace("knob_39=39", "knob_39=38"))
    assert other != stem


def test_launch_suffix_is_ignored():
    ident = _base_identity(direct=False)
    assert KernelIdentity.from_label(f"  {ident.label()} @split_k=4\n") == ident


@pytest.mark.parametrize(
    "name",
    [
        "",
        "fwd-implicit_gemm",
        "fwd-implicit_gemm-bf16-4d-t1x1x1-w1x1-a1x1x1-v1x1x1-mem-default-wave64-gfx950-llvm22",
        "fwd-implicit_gemm-bf16-2d-t1x1-w1x1-a1x1x1-v1x1x1-mem-default-wave64-gfx950-llvm22",
        "fwd-implicit_gemm-bf16-2d-t1x1x1-w1x1-a1x1x1-v1x1x1-mem-default-bogus-wave64-gfx950-llvm22",
        "fwd-implicit_gemm-bf16-2d-t1x1x1-w1x1-a1x1x1-v1x1x1-mem-default-@nope=1-wave64-gfx950-llvm22",
        "fwd-implicit_gemm-bf16-2d-t1x1x1-w1x1-a1x1x1-v1x1x1-mem-default-grp-grp-wave64-gfx950-llvm22",
    ],
)
def test_malformed_names_are_rejected(name):
    with pytest.raises(ValueError):
        KernelIdentity.from_label(name)


@pytest.mark.parametrize("direction", ["fwd", "wgrad", "dgrad"])
def test_implicit_job_for_identity_is_the_compile_all_job(direction):
    for job in itertools.islice(_implicit_jobs(direction, n_geometries=6), 0, None, 7):
        assert ks.job_for_identity(job.identity) == job


def test_implicit_job_for_identity_rejects_what_the_grid_cannot_build():
    ident = next(iter(_implicit_jobs("fwd", n_geometries=1))).identity
    for bad in (
        dict(vector_size_a=16),
        dict(dtype_d="fp16"),
        dict(warp_tile_n=ident.warp_tile_m * 2),
        dict(lds_k_pad=4),
    ):
        with pytest.raises(LookupError):
            ks.job_for_identity(dataclasses.replace(ident, **bad))


def test_direct_job_for_identity_is_the_compile_all_job():
    """Every valid direct job of a cell per variant, helpers included."""
    seen_variants = set()
    for payload in _direct_cells():
        variant = payload[2]
        if variant in seen_variants:
            continue
        seen_variants.add(variant)
        _, cell = dks._caps_jobs(*payload)
        assert cell, variant
        for _, job in cell:
            assert dks.direct_job_for_identity(job.identity) == job
    assert dks._TRANSPOSE in {
        j.identity.algorithm
        for p in _direct_cells()
        if p[2] == "direct_grouped_dgrad_mfma"
        for _, j in dks._caps_jobs(*p)[1]
    }


def test_direct_helper_job_builds_with_its_pipeline_caps():
    """The weight-reorganize helper (no current grid point pulls it in, so it
    is checked by hand): its job takes the MFMA dgrad row's capabilities and
    builds."""
    _, rows = dks.DIRECT_CAPABILITIES["direct_grouped_dgrad_mfma"]
    row = rows[0]
    ident = dks._helper_identity(
        _ARCH, 64, dks._REORGANIZE, row, {"fold_k32": False}, "fp16"
    )
    assert (ident.pad_h, ident.stride_h) == (0, 0)
    job = dks.direct_job_for_identity(ident)
    assert job.caps == dict(
        KH=row.KH,
        PAD=row.PAD,
        stride=row.stride,
        cpg=row.cpg,
        kpg=row.kpg,
        dil_h=row.dil_h,
        dil_w=row.dil_w,
    )
    assert dks.build_direct_job(job, _ARCH, "fp16")[0] is not None


def test_direct_job_for_identity_rejects_what_the_grid_cannot_build():
    ident = _base_identity(direct=True)
    for bad in (dict(filter_w=5), dict(stride_w=2), dict(dilation_h=2)):
        with pytest.raises(LookupError):
            dks.direct_job_for_identity(dataclasses.replace(ident, **bad))
