# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for the gfx942 AOT forward instances (hipkernel:Gfx942ConvFwdAot).

The gfx942 engine packages the same rocKE ``KernelIdentity`` instances as the gfx950
one, built for gfx942, and its catalog adds group-merged depthwise instances. These
tests hold the adapter to rocKE on the shipped gfx942 catalog
(``IngestorGenerator/configs/gfx942_conv_fwd_aot_<dtype>.instances.json``): the same
kernel IR as ``kernel_sweep.build_kernel`` / ``direct_kernel_sweep.build_direct_job``
on gfx942, the same served-problem predicate as ``KernelCache.supports_problem``, the
merged launch rocKE's benchmark makes, and rocKE's ``--run-from-cache`` selection,
``_widest_vector_sizes``. ``test_convolution_forward_aot_packaging.py`` covers what the
two archs share on the gfx950 sweep.

The catalog's identities were lowered for LLVM flavor llvm22. Their ``instance_id`` is
the hash of the identity as shipped; where rocKE checks provenance
(``supports_problem``), the identity is restated for this process's flavor, as the
gfx950 module does, so nothing depends on ``ROCKE_LLVM_FLAVOR``.
"""

from __future__ import annotations

import json
import random
from dataclasses import fields, replace
from pathlib import Path

import pytest
from benchmarks.common import direct_kernel_sweep, kernel_sweep
from benchmarks.common.kernel_cache import KernelCache, KernelIdentity
from builders.common.convolution_forward import (
    Gfx942ConvFwdAotSpec,
    Gfx950ConvFwdAotSpec,
    Gfx950ConvFwdSpec,
    aot_direct_instance_fields,
    build_gfx942_conv_fwd_aot,
    build_gfx950_conv_fwd_aot,
    gfx942_conv_fwd_aot_problem_for_request,
    gfx942_conv_fwd_aot_runtime_candidates,
    gfx950_conv_fwd_aot_problem_for_request,
    gfx950_conv_fwd_aot_serves,
    supports_gfx942_conv_fwd_aot,
    supports_gfx942_conv_fwd_aot_problem,
    supports_gfx950_conv_fwd_aot,
)
from dispatch.grouped_convolution import ConvGroupedRequest
from kernels.common.conv_args import ConvArgs
from kernels.common.conv_implicit_gemm import implicit_gemm_conv_grid
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

_SEED = 942
_ARCH = "gfx942"
_DTYPES = ("fp16", "bf16")

#: tests -> library -> rocke -> hip-kernel-provider -> dnn-providers -> repository.
_CATALOG_DIR = (
    Path(__file__).resolve().parents[5]
    / "projects"
    / "hipdnn"
    / "tools"
    / "IngestorGenerator"
    / "configs"
)

#: Identities per dtype in the shipped catalog, and of those the group-merged ones
#: per merge degree.
_CATALOG_SIZE = 553
_MERGED_COUNTS = {4: 7, 16: 11, 32: 12}
#: Unmerged identities per dtype the IR test samples, spread evenly over the catalog's
#: order; every merged identity is checked.
_IR_SAMPLE = 40


def _catalog_path(dtype: str) -> Path:
    return _CATALOG_DIR / f"gfx942_conv_fwd_aot_{dtype}.instances.json"


pytestmark = pytest.mark.skipif(
    not all(_catalog_path(dtype).is_file() for dtype in _DTYPES),
    reason="the gfx942 catalog is not in this tree (an installed rocKE copy)",
)


def _spec(identity: KernelIdentity) -> Gfx942ConvFwdAotSpec:
    """The instance make_conv_aot_config.py packages for ``identity``."""
    spec = Gfx942ConvFwdAotSpec(
        dtype=identity.dtype_a,
        tile_m=identity.tile_m,
        tile_n=identity.tile_n,
        tile_k=identity.tile_k,
        warp_m=identity.warp_m,
        warp_n=identity.warp_n,
        warp_tile_m=identity.warp_tile_m,
        warp_tile_n=identity.warp_tile_n,
        warp_tile_k=identity.warp_tile_k,
        wave_size=identity.wave_size,
        pipeline=identity.pipeline,
        epilogue=identity.epilogue,
        vector_size_a=identity.vector_size_a,
        vector_size_b=identity.vector_size_b,
        vector_size_c=identity.vector_size_c,
        unroll_k=int(identity.unroll_k),
        async_dma=int(identity.async_dma),
        async_chunk_a=identity.async_chunk_a,
        async_chunk_b=identity.async_chunk_b,
        grouped=int(identity.grouped),
        instance_id=identity.stable_hash(),
    )
    if identity.is_direct:
        return replace(spec, **aot_direct_instance_fields(identity))
    return replace(spec, group_merge=identity.group_merge)


def _as_gfx950(spec: Gfx942ConvFwdAotSpec) -> Gfx950ConvFwdAotSpec:
    """The same fields as a gfx950 instance."""
    return Gfx950ConvFwdAotSpec(
        **{field.name: getattr(spec, field.name) for field in fields(spec)}
    )


def _current(identity: KernelIdentity) -> KernelIdentity:
    """``identity`` as this process's cache would hold it (see the module doc)."""
    return replace(identity, llvm_flavor=kernel_sweep.current_llvm_flavor())


@pytest.fixture(scope="module")
def catalog():
    """The shipped gfx942 catalog: per dtype, ``(identity, spec)`` in its order."""
    return {
        dtype: [
            (identity, _spec(identity))
            for identity in (
                KernelIdentity.from_dict(entry)
                for entry in json.loads(_catalog_path(dtype).read_text())
            )
        ]
        for dtype in _DTYPES
    }


@pytest.fixture(scope="module")
def merged(catalog):
    """Every group-merged ``(dtype, identity, spec)`` of the catalog."""
    return [
        (dtype, identity, spec)
        for dtype, entries in catalog.items()
        for identity, spec in entries
        if spec.is_merged
    ]


@pytest.fixture(scope="module")
def implicit_sample(catalog):
    """Every merged implicit-GEMM instance and an even spread of the unmerged ones,
    per dtype."""
    sample = []
    for dtype, entries in catalog.items():
        implicit = [(i, s) for i, s in entries if not s.is_direct]
        unmerged = [(i, s) for i, s in implicit if not s.is_merged]
        step = len(unmerged) / _IR_SAMPLE
        picked = [unmerged[int(k * step)] for k in range(_IR_SAMPLE)]
        picked += [(i, s) for i, s in implicit if s.is_merged]
        sample += [(dtype, i, s) for i, s in picked]
    return sample


# ---- The catalog packages for gfx942, and only for gfx942 ----


def test_every_catalog_instance_is_packageable(catalog):
    for dtype, entries in catalog.items():
        assert len(entries) == _CATALOG_SIZE, dtype
        merges: dict = {}
        for identity, spec in entries:
            assert identity.arch == _ARCH and spec.ARCH == _ARCH
            assert supports_gfx942_conv_fwd_aot(spec) == (True, "ok"), identity.label()
            if spec.is_merged:
                merges[spec.group_merge] = merges.get(spec.group_merge, 0) + 1
                assert (spec.vector_size_a, spec.vector_size_b, spec.vector_size_c) == (
                    0,
                    0,
                    0,
                )
            else:
                assert spec.group_merge == 1
        assert merges == _MERGED_COUNTS, dtype


def test_instances_are_refused_for_the_other_arch(catalog):
    for entries in catalog.values():
        for spec in (entries[0][1], next(s for _, s in entries if s.is_merged)):
            gfx950 = _as_gfx950(spec)
            assert supports_gfx950_conv_fwd_aot(gfx950) == (True, "ok")
            ok, why = supports_gfx942_conv_fwd_aot(gfx950)
            assert not ok and "expected Gfx942ConvFwdAotSpec" in why, why
            ok, why = supports_gfx942_conv_fwd_aot(gfx950, arch=_ARCH)
            assert not ok and "expected Gfx942ConvFwdAotSpec" in why, why
            ok, why = supports_gfx950_conv_fwd_aot(gfx950, arch=_ARCH)
            assert not ok and "packaged for arch=gfx950" in why, why
            ok, why = supports_gfx942_conv_fwd_aot(spec, arch="gfx950")
            assert not ok and "packaged for arch=gfx942" in why, why
            # The gfx950 predicate's default arch is gfx950.
            ok, why = supports_gfx950_conv_fwd_aot(spec)
            assert not ok and "packaged for arch=gfx942" in why, why
            with pytest.raises(ValueError, match="expected Gfx942ConvFwdAotSpec"):
                build_gfx942_conv_fwd_aot(gfx950)
            with pytest.raises(ValueError, match="packaged for arch=gfx942"):
                build_gfx950_conv_fwd_aot(spec)


def test_kernel_symbols_name_the_arch(catalog):
    for entries in catalog.values():
        for spec in (entries[0][1], next(s for _, s in entries if s.is_direct)):
            assert f"_aot_{_ARCH}_" in spec.kernel_symbol_prefix()
            assert "_aot_gfx950_" in _as_gfx950(spec).kernel_symbol_prefix()


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"vector_size_a": 1}, "derived vector widths (0, 0, 0)"),
        ({"vector_size_c": 4}, "derived vector widths (0, 0, 0)"),
        ({"vector_size_a": 2, "vector_size_b": 1, "vector_size_c": 2}, "(0, 0, 0)"),
        ({"grouped": 0}, "grouped code path"),
        ({"group_merge": 0}, "positive integer"),
        ({"group_merge": True}, "positive integer"),
        ({"group_merge": 16.0}, "positive integer"),
    ],
)
def test_out_of_contract_merged_instances_are_declined(merged, change, reason):
    _, _, spec = merged[0]
    ok, why = supports_gfx942_conv_fwd_aot(replace(spec, **change))
    assert not ok and reason in why, why


def test_merge_degrees_rocke_does_not_build_are_declined(merged):
    """rocKE's own gate decides the merge degree: 3 is none of its degrees, and 64
    merges more groups than a 16- or 32-wide tile holds."""
    _, _, spec = merged[0]
    for group_merge in (3, 2 * spec.tile_n):
        ok, why = supports_gfx942_conv_fwd_aot(replace(spec, group_merge=group_merge))
        assert not ok, (group_merge, why)


def test_an_unmerged_instance_with_derived_widths_is_declined(catalog):
    _, spec = next(e for e in catalog["fp16"] if not (e[1].is_direct or e[1].is_merged))
    ok, why = supports_gfx942_conv_fwd_aot(
        replace(spec, vector_size_a=0, vector_size_b=0, vector_size_c=0)
    )
    assert not ok and "vector_size_a" in why, why


def test_a_direct_instance_has_no_group_merge(catalog):
    for entries in catalog.values():
        direct = [s for _, s in entries if s.is_direct]
        assert {s.direct_variant for s in direct} == {
            "std",
            "spatial",
            "tiled",
            "col",
            "grouped",
            "nongrouped",
        }
        for spec in direct:
            assert not spec.is_merged
            ok, why = supports_gfx942_conv_fwd_aot(replace(spec, group_merge=2))
            assert not ok and "no group merge" in why, why


# ---- The packaged kernel is rocKE's ----


def _assert_same_ir(ours, theirs, label: str) -> None:
    ours_ir = _lower_kernel_to_llvm_python(ours, arch=_ARCH, llvm_flavor="llvm22")
    theirs_ir = _lower_kernel_to_llvm_python(theirs, arch=_ARCH, llvm_flavor="llvm22")
    assert ours_ir.replace(ours.name, "@K") == theirs_ir.replace(
        theirs.name, "@K"
    ), label


def test_packaged_ir_is_the_sweep_kernel_up_to_its_symbol(implicit_sample):
    """Every merged instance and a spread of the others: the kernel the gfx942 sweep
    builds for the identity (``job_for_identity`` recovers the sweep's own job)."""
    assert sum(spec.is_merged for _, _, spec in implicit_sample) == 2 * sum(
        _MERGED_COUNTS.values()
    )
    for dtype, identity, spec in implicit_sample:
        job = kernel_sweep.job_for_identity(identity)
        assert job.identity == identity
        ours = build_gfx942_conv_fwd_aot(spec)
        theirs, _ = kernel_sweep.build_kernel(job, _ARCH, dtype)
        assert ours.name.startswith(spec.kernel_symbol_prefix())
        assert ours.name == spec.kernel_name()
        _assert_same_ir(ours, theirs, identity.label())


def test_packaged_direct_ir_is_the_sweep_kernel_up_to_its_symbol(catalog):
    """Every direct algorithm of the catalog, both dtypes."""
    seen = set()
    for dtype, entries in catalog.items():
        for identity, spec in entries:
            if not spec.is_direct or (dtype, identity.algorithm) in seen:
                continue
            seen.add((dtype, identity.algorithm))
            job = direct_kernel_sweep.direct_job_for_identity(identity)
            ours = build_gfx942_conv_fwd_aot(spec)
            theirs, _ = direct_kernel_sweep.build_direct_job(job, _ARCH, dtype)
            assert ours.name.startswith(spec.kernel_symbol_prefix())
            _assert_same_ir(ours, theirs, identity.label())
    assert len(seen) == 2 * 6


def test_a_merged_instance_is_not_its_unmerged_twin(merged):
    """The merge is compiled in: the same geometry unmerged is another kernel."""
    _, identity, spec = merged[0]
    twin = replace(
        spec,
        group_merge=1,
        vector_size_a=1,
        vector_size_b=1,
        vector_size_c=1,
        instance_id=replace(
            identity, group_merge=1, vector_size_a=1, vector_size_b=1, vector_size_c=1
        ).stable_hash(),
    )
    ours = _lower_kernel_to_llvm_python(
        build_gfx942_conv_fwd_aot(spec), arch=_ARCH, llvm_flavor="llvm22"
    )
    unmerged = build_gfx942_conv_fwd_aot(twin)
    assert ours.replace(spec.kernel_name(), "@K") != _lower_kernel_to_llvm_python(
        unmerged, arch=_ARCH, llvm_flavor="llvm22"
    ).replace(unmerged.name, "@K")


# ---- Served problems ----


def _random_problems(count: int, dtype: str, seed: int):
    """Depthwise-heavy problems (group counts divisible by every merge degree, by
    some and by none; channel multipliers the merged kernels must decline), grouped
    and dense ones, pointwise and strided/dilated filters, and output extents past
    65535 small M tiles."""
    rng = random.Random(seed)
    problems = []
    while len(problems) < count:
        kind = rng.choice(["depthwise"] * 5 + ["grouped", "dense"])
        if kind == "depthwise":
            groups = rng.choice(
                [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 256, 512]
            )
            cpg, kpg = rng.choice([(1, 1)] * 6 + [(1, 2), (2, 2), (2, 1)])
        elif kind == "grouped":
            groups = rng.choice([2, 3, 4, 8, 16, 32])
            cpg = rng.choice([1, 2, 3, 4, 8, 16, 32])
            kpg = rng.choice([1, 2, 4, 5, 8, 16, 32])
        else:
            groups = 1
            cpg = rng.choice([3, 4, 8, 16, 24, 64, 256])
            kpg = rng.choice([4, 6, 8, 16, 32, 64, 128])
        filt = rng.choice([1, 3, 3, 5, 7])
        dilation = rng.choice([1] * 5 + [2])
        big = rng.random() < 0.05
        problem = Gfx950ConvFwdSpec(
            N=rng.choice([64, 128]) if big else rng.choice([1, 2, 4, 8]),
            Hi=rng.choice([224, 512]) if big else rng.choice([7, 14, 17, 28, 56]),
            Wi=rng.choice([224, 512]) if big else rng.choice([7, 14, 17, 28, 56]),
            C=groups * cpg,
            K=groups * kpg,
            Y=filt,
            X=filt,
            sH=rng.choice([1, 2]),
            sW=rng.choice([1, 2]),
            pH=filt // 2,
            pW=filt // 2,
            dH=dilation,
            dW=dilation,
            groups=groups,
            dtype=dtype,
        )
        if supports_gfx942_conv_fwd_aot_problem(problem)[0]:
            problems.append(problem)
    return problems


def test_served_problems_agree_with_kernel_cache_supports_problem(implicit_sample):
    """Every (instance, problem) verdict matches the AOT cache's own predicate,
    merged instances included.

    As on gfx950, supports_problem leaves out the bounds ConvArgs checks at launch
    (the 24-bit reduction extent, 32-bit offsets), so those problems are excluded
    rather than compared.
    """
    cache = KernelCache.__new__(KernelCache)
    compared: dict = {}
    served: dict = {}
    for dtype, identity, spec in implicit_sample:
        current = _current(identity)
        problems = _random_problems(40 if spec.is_merged else 25, dtype, _SEED)
        for problem in problems:
            ours, reason = gfx950_conv_fwd_aot_serves(spec, problem)
            if not ours and ("2**23" in reason or "32-bit" in reason):
                continue
            theirs, why = cache.supports_problem(current, problem.to_problem())
            assert ours == theirs, (identity.label(), problem, reason, why)
            compared[spec.is_merged] = compared.get(spec.is_merged, 0) + 1
            served[spec.is_merged] = served.get(spec.is_merged, 0) + ours
    # Both verdicts occur for merged and unmerged instances alike.
    for is_merged, count in compared.items():
        assert count > 1000 and 0 < served[is_merged] < count, (is_merged, count)
    assert set(compared) == {False, True}


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"C": 64, "K": 128}, "depthwise"),
        ({"C": 128, "K": 128, "groups": 64}, "depthwise"),
        ({"C": 48, "K": 48, "groups": 48}, "does not divide groups"),
        ({"C": 1, "K": 1, "groups": 1}, "does not divide groups"),
    ],
)
def test_a_merged_instance_serves_depthwise_its_degree_divides(merged, change, reason):
    _, _, spec = next(m for m in merged if m[2].group_merge == 32)
    problem = Gfx950ConvFwdSpec(
        N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=64,
        dtype=spec.dtype,
    )  # fmt: skip
    assert gfx950_conv_fwd_aot_serves(spec, problem) == (True, "ok")
    ok, why = gfx950_conv_fwd_aot_serves(spec, replace(problem, **change))
    assert not ok and reason in why, why


def test_merged_grid_bounds(merged):
    """A merged instance is bounded by its merged grid along y, as in rocKE. Along
    z (groups / Gm) the packaged graph contract is stricter: it caps the group
    count itself at 65535 for every instance, so the group counts past it that a
    merged grid would still launch (and supports_problem offers) are declined."""
    _, identity, spec = next(m for m in merged if m[2].group_merge == 32)
    cache = KernelCache.__new__(KernelCache)
    groups = 65504  # the largest multiple of 32 within 65535
    problem = Gfx950ConvFwdSpec(
        N=1, Hi=4, Wi=4, C=groups, K=groups, Y=3, X=3, pH=1, pW=1, groups=groups,
        dtype=spec.dtype,
    )  # fmt: skip
    assert gfx950_conv_fwd_aot_serves(spec, problem) == (True, "ok")
    assert spec.grid(problem) == (1, 1, groups // 32)
    # y: N*Ho*Wo tiles past 65535, on four channels.
    _, narrow_identity, narrow = next(m for m in merged if m[2].group_merge == 4)
    tall = Gfx950ConvFwdSpec(
        N=128, Hi=512, Wi=512, C=4, K=4, Y=3, X=3, pH=1, pW=1, groups=4,
        dtype=narrow.dtype,
    )  # fmt: skip
    assert narrow.tile_m * 65535 < 128 * 512 * 512
    ok, why = gfx950_conv_fwd_aot_serves(narrow, tall)
    assert not ok and "grid y" in why, why
    assert not cache.supports_problem(_current(narrow_identity), tall.to_problem())[0]
    # z: the contract's own bound, not the merged grid's.
    wide = replace(problem, C=32 * 2048, K=32 * 2048, groups=32 * 2048)
    ok, why = gfx950_conv_fwd_aot_serves(spec, wide)
    assert not ok and "(grid z)" in why, why
    assert not supports_gfx942_conv_fwd_aot_problem(wide)[0]
    assert cache.supports_problem(_current(identity), wide.to_problem()) == (
        True,
        "ok",
    )


# ---- The merged launch ----


def test_merged_grid_and_launch_values_are_rockes_for_the_runtime_problem(merged):
    """The grid is rocKE's merged grid -- the builder's ``implicit_gemm_conv_grid``
    and the benchmark's ``_grid_for_spec`` on the instance spec for the problem,
    and ``--run-from-cache``'s ``_grid_for`` on the identity -- and the kernel
    arguments are the TRUE problem's ``ConvArgs``, as the benchmark launches."""
    from benchmarks.common.benchmark_implicit_gemm_conv import (
        _grid_for,
        _grid_for_spec,
    )

    launched = 0
    for dtype, identity, spec in merged:
        for problem in _random_problems(30, dtype, _SEED + 1):
            if not gfx950_conv_fwd_aot_serves(spec, problem)[0]:
                continue
            p = problem.to_problem()
            rocke_spec = replace(spec.to_instance_spec(), problem=p)
            assert rocke_spec.group_merge == spec.group_merge
            grid = spec.grid(problem)
            assert grid == implicit_gemm_conv_grid(rocke_spec)
            assert grid == _grid_for_spec(rocke_spec, p)
            assert grid == _grid_for("fwd", p, identity)
            assert grid[2] * spec.group_merge == problem.groups
            args = ConvArgs.from_problem(p, tile_m=spec.tile_m, tile_n=spec.tile_n)
            assert spec.launch_values(
                problem, 0, 0, 0, 1, 2, 3
            ) == args.to_launch_values(0, 0, 0, 1, 2, 3)
            assert spec.block() == (rocke_spec.launch_block_size, 1, 1)
            launched += 1
    assert launched > 100


# ---- Runtime selection: which serving instances run ----

#: (N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW, groups). Dense with even, odd and
#: 2-mod-4 channel counts, grouped (odd group counts too) and grouped pointwise, and
#: depthwise with group counts divisible by 32, by 16 not 32, by 4 not 16, and by
#: none of the merge degrees; depthwise pointwise, strided and dilated, and a
#: depthwise channel multiplier no merged instance serves.
_SELECTION_GRID = (
    (2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    (2, 3, 32, 32, 32, 7, 7, 2, 2, 3, 3, 1, 1, 1),
    (2, 32, 5, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    (2, 6, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    (2, 64, 128, 14, 14, 1, 1, 1, 1, 0, 0, 1, 1, 1),
    (2, 256, 64, 7, 7, 3, 3, 2, 2, 1, 1, 1, 1, 1),
    (2, 32, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 4),
    (1, 9, 48, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, 3),
    (2, 24, 24, 14, 14, 1, 1, 1, 1, 0, 0, 1, 1, 3),
    (2, 128, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32),
    (2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64),
    (2, 96, 96, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 96),
    (2, 512, 512, 7, 7, 5, 5, 1, 1, 2, 2, 1, 1, 512),
    (2, 48, 48, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 48),
    (1, 80, 80, 17, 17, 3, 3, 2, 2, 1, 1, 1, 1, 80),
    (2, 12, 12, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 12),
    (2, 20, 20, 9, 9, 7, 7, 1, 1, 3, 3, 1, 1, 20),
    (2, 6, 6, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 6),
    (2, 7, 7, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 7),
    (2, 64, 64, 14, 14, 1, 1, 1, 1, 0, 0, 1, 1, 64),
    (2, 32, 32, 28, 28, 3, 3, 1, 1, 2, 2, 2, 2, 32),
    (2, 32, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32),
)


def test_runtime_candidates_are_rockes_runtime_selection(catalog):
    """For every problem the instances selected are exactly the ones rocKE's
    --run-from-cache benchmark runs by default over the catalog's identities:
    KernelCache.compatible, then _widest_vector_sizes. Both are rocKE's own,
    called on the identities. Direct instances are not in that selection (another
    benchmark runs them) and all run."""
    from benchmarks.common.benchmark_implicit_gemm_conv import _widest_vector_sizes

    kinds_selected = merged_selected = 0
    for dtype, entries in catalog.items():
        implicit = {s.instance_id: _current(i) for i, s in entries if not s.is_direct}
        specs = [s for _, s in entries]
        entries_of = [(i, None, {}) for i in implicit.values()]
        cache = KernelCache.__new__(KernelCache)
        cache._entries = lambda direction, cached=entries_of: cached
        restated = {i.stable_hash(): sid for sid, i in implicit.items()}
        for values in _SELECTION_GRID:
            n, c, k, hi, wi, y, x, sh, sw, ph, pw, dh, dw, groups = values
            problem = Gfx950ConvFwdSpec(
                N=n, C=c, K=k, Hi=hi, Wi=wi, Y=y, X=x, sH=sh, sW=sw, pH=ph, pW=pw,
                dH=dh, dW=dw, groups=groups, dtype=dtype,
            )  # fmt: skip
            assert supports_gfx942_conv_fwd_aot_problem(problem) == (True, "ok")
            compatible = cache.compatible(problem.to_problem(), direction="fwd")
            theirs = {
                restated[ident.stable_hash()]
                for ident, *_ in _widest_vector_sizes(compatible)
            }
            ours = gfx942_conv_fwd_aot_runtime_candidates(problem, specs)
            assert [s.instance_id for s in ours if not s.is_direct] == [
                s.instance_id
                for s in specs
                if not s.is_direct and s.instance_id in theirs
            ], (dtype, values)
            assert [s for s in ours if s.is_direct] == [
                s
                for s in specs
                if s.is_direct and gfx950_conv_fwd_aot_serves(s, problem)[0]
            ], (dtype, values)
            kinds_selected += len({(s.epilogue, s.group_merge) for s in ours}) > 1
            merged_selected += any(s.is_merged for s in ours)
    # Merged and unmerged kinds are selected side by side, and merged ones run
    # wherever a merge degree divides a depthwise group count.
    assert kinds_selected > 0 and merged_selected >= 2 * 9


def test_runtime_candidates_select_per_merge_degree(catalog):
    """A depthwise problem every degree divides runs every merge degree's serving
    instances beside the widest unmerged ones; one no degree divides runs none."""
    specs = [s for _, s in catalog["fp16"]]
    problem = Gfx950ConvFwdSpec(
        N=2, C=64, K=64, Hi=14, Wi=14, Y=3, X=3, pH=1, pW=1, groups=64, dtype="fp16"
    )
    ours = gfx942_conv_fwd_aot_runtime_candidates(problem, specs)
    assert {s.group_merge for s in ours if not s.is_direct} == {1, 4, 16, 32}
    assert [s for s in ours if s.is_merged] == [
        s for s in specs if s.is_merged and gfx950_conv_fwd_aot_serves(s, problem)[0]
    ]
    odd = replace(problem, C=63, K=63, groups=63)
    assert not any(
        s.is_merged for s in gfx942_conv_fwd_aot_runtime_candidates(odd, specs)
    )


# ---- The profile ----


def test_profile_dispatch_resolves_gfx942_requests_including_grouped_pointwise():
    for request in (
        ConvGroupedRequest(
            N=2, Hi=8, Wi=8, C=32, K=32, Y=1, X=1, G=4, arch=_ARCH, dtype="fp16"
        ),
        ConvGroupedRequest(
            N=2,
            Hi=14,
            Wi=14,
            C=64,
            K=64,
            Y=3,
            X=3,
            G=64,
            pad_h=1,
            pad_w=1,
            arch=_ARCH,
            dtype="bf16",
        ),
    ):
        problem = gfx942_conv_fwd_aot_problem_for_request(request)
        assert (problem.groups, problem.Y, problem.X, problem.dtype) == (
            request.G,
            request.Y,
            request.X,
            request.dtype,
        )
        assert supports_gfx942_conv_fwd_aot_problem(problem) == (True, "ok")
        ok, why = supports_gfx942_conv_fwd_aot_problem(problem, arch="gfx950")
        assert not ok and "arch=gfx942" in why, why
        with pytest.raises(ValueError, match="arch=gfx942"):
            gfx942_conv_fwd_aot_problem_for_request(replace(request, arch="gfx950"))
        with pytest.raises(ValueError, match="arch=gfx950"):
            gfx950_conv_fwd_aot_problem_for_request(request)
        with pytest.raises(ValueError, match="2D"):
            gfx942_conv_fwd_aot_problem_for_request(replace(request, Di=4, Z=1))
