# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for the gfx942 AOT forward instances (hipkernel:Gfx942ConvFwdAot).

The gfx942 engine packages rocKE ``KernelIdentity`` instances of its forward sweeps
built for gfx942, through the same adapter as the gfx950 one (``TARGETS["gfx942"]``).
These tests hold the adapter to rocKE on the whole gfx942 catalog, the dispatcher's
registered list (``dispatch/grouped_convolution_instances/gfx942_fwd.json``): every
instance converts and round-trips, its packaged IR is the sweep's kernel up to its
symbol, it serves what ``KernelCache.supports_problem`` and the dispatcher admit,
group-merged instances launch over rocKE's merged grid, and the runtime selection is
rocKE's ``--run-from-cache`` selection, ``_widest_vector_sizes``.
``test_convolution_forward_aot_packaging.py`` covers the shared adapter on the gfx950
sweep.

The catalog's identities were lowered for LLVM flavor llvm22. Their ``instance_id`` is
the hash of the identity as listed; where rocKE checks provenance
(``supports_problem``, ``compatible``), the identity is restated for this process's
flavor, as the gfx950 module does.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import fields, replace

import pytest
from benchmarks.common import direct_kernel_sweep, kernel_sweep
from benchmarks.common.kernel_cache import KernelCache, KernelIdentity
from builders.common.convolution_forward import (
    AOT_DIRECT_ALGORITHMS,
    SPEC_CLASSES,
    TARGETS,
    Gfx942ConvFwdAotSpec,
    Gfx950ConvFwdAotSpec,
    Gfx950ConvFwdSpec,
    build_gfx942_conv_fwd_aot,
    build_gfx942_conv_fwd_aot_split_reduce,
    build_gfx950_conv_fwd_aot,
    conv_fwd_aot_serves,
    gfx942_conv_fwd_aot_capabilities,
    gfx942_conv_fwd_aot_problem_for_request,
    gfx942_conv_fwd_aot_runtime_candidates,
    gfx942_conv_fwd_aot_serves,
    gfx942_conv_fwd_aot_spec_for_identity,
    gfx950_conv_fwd_aot_problem_for_request,
    gfx950_conv_fwd_aot_spec_for_identity,
    supports_gfx942_conv_fwd_aot,
    supports_gfx942_conv_fwd_aot_problem,
    supports_gfx942_conv_fwd_aot_split_reduce,
    supports_gfx950_conv_fwd_aot,
)
from dispatch.grouped_convolution import (
    DEFAULT_LLVM_FLAVORS,
    ConvGroupedRequest,
    _admits,
    registered_conv_grouped_instances,
)
from kernels.common.conv_args import ConvArgs
from kernels.common.conv_implicit_gemm import implicit_gemm_conv_grid
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

_SEED = 942
_ARCH = "gfx942"
_DTYPES = ("fp16", "bf16")

#: Instances per dtype in the gfx942 list; of these the group-merged implicit-GEMM
#: ones per merge degree, the direct ones per algorithm, and the split-C direct ones
#: per algorithm.
_CATALOG_SIZE = {"fp16": 239, "bf16": 220}
_MERGED_COUNTS = {4: 2, 16: 1}
_DIRECT_COUNTS = {
    "fp16": {
        "direct_nongrouped": 85,
        "direct_depthwise_tiled": 17,
        "direct_grouped_halo": 17,
        "direct_grouped": 4,
        "direct_depthwise": 4,
        "direct_depthwise_col": 2,
        "direct_depthwise_spatial": 2,
    },
    "bf16": {
        "direct_nongrouped": 79,
        "direct_depthwise_tiled": 18,
        "direct_depthwise": 4,
        "direct_depthwise_col": 2,
        "direct_depthwise_spatial": 1,
    },
}
_SPLIT_C_COUNTS = {"direct_nongrouped": 30}
#: Unmerged implicit-GEMM instances per dtype the served-problem test samples, spread
#: evenly over the catalog; every merged instance is checked.
_SERVES_SAMPLE = 40


def _current(identity: KernelIdentity) -> KernelIdentity:
    """``identity`` as this process's cache would hold it (see the module doc)."""
    return replace(identity, llvm_flavor=kernel_sweep.current_llvm_flavor())


def _as_gfx950(spec: Gfx942ConvFwdAotSpec) -> Gfx950ConvFwdAotSpec:
    """The same fields as a gfx950 instance."""
    return Gfx950ConvFwdAotSpec(
        **{field.name: getattr(spec, field.name) for field in fields(spec)}
    )


@pytest.fixture(scope="module")
def catalog():
    """The registered gfx942 forward list: per dtype, ``(identity, spec)`` pairs."""
    by_dtype: dict = {dtype: [] for dtype in _DTYPES}
    for instance in registered_conv_grouped_instances(_ARCH, direction="fwd"):
        spec, reason = gfx942_conv_fwd_aot_spec_for_identity(instance.identity)
        assert reason == "", (instance.identity.label(), reason)
        by_dtype[spec.dtype].append((instance.identity, spec))
    return by_dtype


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
        step = len(unmerged) / _SERVES_SAMPLE
        picked = [unmerged[int(k * step)] for k in range(_SERVES_SAMPLE)]
        picked += [(i, s) for i, s in implicit if s.is_merged]
        sample += [(dtype, i, s) for i, s in picked]
    return sample


# ---- The target ----


def test_the_gfx942_target_is_gfx950s_but_the_windowed_depthwise():
    target = TARGETS[_ARCH]
    assert target.engine == "hipkernel:Gfx942ConvFwdAot"
    assert target.wave_size == 64
    assert target.llvm_flavor == DEFAULT_LLVM_FLAVORS[_ARCH] == "llvm22"
    assert set(TARGETS["gfx950"].direct_algorithms) - set(target.direct_algorithms) == {
        "direct_depthwise_win"
    }
    assert SPEC_CLASSES[_ARCH] is Gfx942ConvFwdAotSpec
    assert Gfx942ConvFwdAotSpec.target() is target
    # The spec adds no field: the KMD is the gfx950 engine's.
    assert [f.name for f in fields(Gfx942ConvFwdAotSpec)] == [
        f.name for f in fields(Gfx950ConvFwdAotSpec)
    ]


# ---- The catalog packages for gfx942, and only for gfx942 ----


def test_every_catalog_instance_is_packageable(catalog):
    for dtype, entries in catalog.items():
        assert len(entries) == _CATALOG_SIZE[dtype], dtype
        merges: dict = {}
        directs: dict = {}
        split: dict = {}
        for identity, spec in entries:
            assert identity.arch == _ARCH and spec.ARCH == _ARCH
            assert identity.llvm_flavor == TARGETS[_ARCH].llvm_flavor
            assert type(spec) is Gfx942ConvFwdAotSpec
            assert spec.identity() == identity
            assert spec.instance_id == identity.stable_hash()
            assert supports_gfx942_conv_fwd_aot(spec) == (True, "ok"), identity.label()
            if spec.is_split_c:
                split[spec.algorithm] = split.get(spec.algorithm, 0) + 1
            if spec.is_direct:
                directs[spec.algorithm] = directs.get(spec.algorithm, 0) + 1
                assert spec.direct_variant in TARGETS[_ARCH].direct_variants
            elif spec.is_merged:
                merges[spec.group_merge] = merges.get(spec.group_merge, 0) + 1
                assert (spec.vector_size_a, spec.vector_size_b, spec.vector_size_c) == (
                    0,
                    0,
                    0,
                )
            else:
                assert spec.group_merge == 1
        assert merges == _MERGED_COUNTS, dtype
        assert directs == _DIRECT_COUNTS[dtype], dtype
        assert split == _SPLIT_C_COUNTS, dtype


def test_instances_are_refused_for_the_other_arch(catalog):
    for entries in catalog.values():
        for spec in (
            entries[0][1],
            next(s for _, s in entries if s.is_merged),
            next(s for _, s in entries if s.is_direct),
        ):
            # The same fields as a gfx950 instance spell a gfx950 identity: one the
            # gfx950 sweep does not build (no 16x16x16 or 32x32x8 MMA atom there), or
            # one whose hash is not this instance_id.
            ok, why = supports_gfx950_conv_fwd_aot(_as_gfx950(spec))
            assert not ok and (
                "is not the stable_hash()" in why
                or "not an identity of rocKE's gfx950 forward AOT sweep" in why
            ), why
            ok, why = supports_gfx942_conv_fwd_aot(_as_gfx950(spec))
            assert not ok and "requires arch=gfx950" in why, why
            ok, why = supports_gfx942_conv_fwd_aot(spec, arch="gfx950")
            assert not ok and "requires arch=gfx942" in why, why
            # The gfx950 predicate's default arch is gfx950.
            ok, why = supports_gfx950_conv_fwd_aot(spec)
            assert not ok and "requires arch=gfx942" in why, why
            with pytest.raises(ValueError, match="requires arch=gfx942"):
                build_gfx950_conv_fwd_aot(spec)
            with pytest.raises(ValueError, match="requires arch=gfx950"):
                build_gfx942_conv_fwd_aot(_as_gfx950(spec))
            ok, why = supports_gfx942_conv_fwd_aot_split_reduce(spec)
            assert not ok and "only a split-C instance" in why, why
            with pytest.raises(ValueError, match="only a split-C instance"):
                build_gfx942_conv_fwd_aot_split_reduce(spec)


def test_the_gfx942_wrappers_refuse_another_archs_instance(catalog):
    """A gfx950 instance passed to the gfx942 serve/capability/selection wrappers is
    not served, has no capabilities and is never a candidate, although the shared
    rule alone would accept it."""
    spec = next(s for _, s in catalog["fp16"] if s.is_direct)
    twin = _as_gfx950(spec)
    problem = next(
        p for p in _direct_problems(spec, 64, _SEED) if conv_fwd_aot_serves(spec, p)[0]
    )
    assert conv_fwd_aot_serves(twin, problem)[0]
    assert gfx942_conv_fwd_aot_serves(spec, problem) == (True, "ok")
    ok, why = gfx942_conv_fwd_aot_serves(twin, problem)
    assert not ok and "not a gfx942 instance" in why, why
    gfx942_conv_fwd_aot_capabilities(spec)
    with pytest.raises(TypeError, match="not a gfx942 instance"):
        gfx942_conv_fwd_aot_capabilities(twin)
    assert gfx942_conv_fwd_aot_runtime_candidates(problem, [spec])
    assert gfx942_conv_fwd_aot_runtime_candidates(problem, [twin]) == []


def test_the_identity_converters_refuse_the_other_arch(catalog):
    identity, _ = catalog["fp16"][0]
    spec, why = gfx950_conv_fwd_aot_spec_for_identity(identity)
    assert spec is None and "not a gfx950 forward" in why, why
    gfx950 = next(
        s.identity for s in registered_conv_grouped_instances("gfx950", direction="fwd")
    )
    spec, why = gfx942_conv_fwd_aot_spec_for_identity(gfx950)
    assert spec is None and "not a gfx942 forward" in why, why


def _relabelled(identity: KernelIdentity) -> KernelIdentity:
    return replace(identity, arch=_ARCH)


def test_gfx950_only_instances_are_refused_on_gfx942():
    """A gfx950 identity relabelled for gfx942 is refused where it needs a gfx950
    instruction: the 16x16x32 and 32x32x16 MFMA atoms of implicit GEMM, and the
    windowed depthwise's Toeplitz MFMA form, which has no packaged launch here."""
    gfx950 = [
        s.identity for s in registered_conv_grouped_instances("gfx950", direction="fwd")
    ]
    atoms = [
        i
        for i in gfx950
        if i.algorithm == "implicit_gemm" and (i.warp_tile_m, i.warp_tile_k) == (16, 32)
    ]
    assert atoms
    spec, why = gfx942_conv_fwd_aot_spec_for_identity(_relabelled(atoms[0]))
    assert spec is None and "(16, 16, 32)" in why and _ARCH in why, why
    windowed = [i for i in gfx950 if i.algorithm == "direct_depthwise_win"]
    assert windowed
    spec, why = gfx942_conv_fwd_aot_spec_for_identity(_relabelled(windowed[0]))
    assert spec is None and "mfma needs one of ('gfx950',)" in why, why


def test_the_windowed_depthwise_has_no_gfx942_launch():
    """The target refuses the variant before rocKE's validators do."""
    identity = next(
        s.identity
        for s in registered_conv_grouped_instances("gfx950", direction="fwd")
        if s.identity.algorithm == "direct_depthwise_win"
    )
    gfx950, _ = gfx950_conv_fwd_aot_spec_for_identity(identity)
    twin = Gfx942ConvFwdAotSpec(
        **{field.name: getattr(gfx950, field.name) for field in fields(gfx950)}
    )
    ok, why = supports_gfx942_conv_fwd_aot(twin)
    assert not ok and "no packaged forward launch of the 'win'" in why, why


def test_kernel_symbols_name_the_arch(catalog):
    for entries in catalog.values():
        for spec in (
            entries[0][1],
            next(s for _, s in entries if s.is_direct),
            next(s for _, s in entries if s.is_merged),
        ):
            prefix = spec.kernel_symbol_prefix()
            assert f"_aot_{_ARCH}_{spec.instance_id[:24]}" in prefix, prefix
            assert "_aot_gfx950_" in _as_gfx950(spec).kernel_symbol_prefix()


# ---- The packaged kernel is rocKE's ----


@pytest.mark.parametrize("dtype", _DTYPES)
def test_packaged_ir_is_the_sweep_kernel_up_to_its_symbol(catalog, dtype):
    """Every catalog instance: the kernel the gfx942 sweep builds for the identity
    (``job_for_identity`` / ``direct_job_for_identity`` recover the sweep's own job),
    byte for byte up to its symbol."""
    flavor = kernel_sweep.current_llvm_flavor()
    for identity, spec in catalog[dtype]:
        if spec.is_direct:
            job = direct_kernel_sweep.direct_job_for_identity(identity)
            theirs, _ = direct_kernel_sweep.direct_kernel_for_job(job, _ARCH, dtype)
        else:
            job = kernel_sweep.job_for_identity(identity)
            theirs, _ = kernel_sweep.kernel_for_job(job, _ARCH, dtype)
        assert job.identity == identity
        ours = build_gfx942_conv_fwd_aot(spec)
        assert ours.name.startswith(spec.kernel_symbol_prefix())
        assert ours.name == spec.kernel_name()
        ours_ir = _lower_kernel_to_llvm_python(ours, arch=_ARCH, llvm_flavor=flavor)
        theirs_ir = _lower_kernel_to_llvm_python(theirs, arch=_ARCH, llvm_flavor=flavor)
        assert ours_ir.replace(ours.name, "@K") == theirs_ir.replace(
            theirs.name, "@K"
        ), identity.label()


def test_a_merged_instance_is_not_its_unmerged_twin(merged):
    """The merge is compiled in: the same geometry unmerged is another kernel."""
    _, identity, spec = merged[0]
    twin_identity = replace(
        identity, group_merge=1, vector_size_a=1, vector_size_b=1, vector_size_c=1
    )
    twin = replace(
        spec,
        group_merge=1,
        vector_size_a=1,
        vector_size_b=1,
        vector_size_c=1,
        instance_id=twin_identity.stable_hash(),
    )
    flavor = kernel_sweep.current_llvm_flavor()
    ours = build_gfx942_conv_fwd_aot(spec)
    unmerged = build_gfx942_conv_fwd_aot(twin)
    assert _lower_kernel_to_llvm_python(ours, arch=_ARCH, llvm_flavor=flavor).replace(
        ours.name, "@K"
    ) != _lower_kernel_to_llvm_python(unmerged, arch=_ARCH, llvm_flavor=flavor).replace(
        unmerged.name, "@K"
    )


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
    """Every (implicit-GEMM instance, problem) verdict matches the AOT cache's own
    predicate, merged instances included.

    As on gfx950, supports_problem leaves out the bounds ConvArgs checks at launch
    (the 24-bit reduction extent, 32-bit offsets), so those problems are excluded
    rather than compared.
    """
    cache = KernelCache.__new__(KernelCache)
    compared: dict = {}
    served: dict = {}
    for dtype, identity, spec in implicit_sample:
        current = _current(identity)
        problems = _random_problems(400 if spec.is_merged else 25, dtype, _SEED)
        for problem in problems:
            ours, reason = gfx942_conv_fwd_aot_serves(spec, problem)
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


def _direct_problems(spec, count: int, seed: int):
    """Problems around a direct instance's compiled filter, stride, padding and
    dilation (and some off them), over the group and channel counts its variants
    tell apart."""
    rng = random.Random(seed)
    problems = []
    while len(problems) < count:
        kind = rng.choice(["depthwise"] * 3 + ["grouped", "dense"])
        if kind == "depthwise":
            groups = rng.choice([1, 2, 8, 16, 32, 48, 63, 64, 96, 256])
            cpg, kpg = rng.choice([(1, 1)] * 6 + [(1, 2), (2, 2)])
        elif kind == "grouped":
            groups = rng.choice([2, 3, 8, 16, 32])
            cpg = kpg = rng.choice([4, 8, 16, 32])
            if rng.random() < 0.2:
                kpg *= 2
        else:
            groups = 1
            cpg = rng.choice([3, 4, 8, 16, 64, 256])
            kpg = rng.choice([4, 8, 16, 64, 128])
        own = rng.random() < 0.8
        filt = spec.filter_h if own else rng.choice([3, 5, 7])
        stride = spec.stride_h if own else rng.choice([1, 2])
        dilation = spec.dilation_h if own else 1
        pad = spec.pad_h if own else (filt // 2) * dilation
        problem = Gfx950ConvFwdSpec(
            N=rng.choice([1, 2, 4]),
            Hi=rng.choice([7, 14, 17, 28, 56]),
            Wi=rng.choice([7, 14, 17, 28, 56]),
            C=groups * cpg,
            K=groups * kpg,
            Y=filt,
            X=filt,
            sH=stride,
            sW=stride,
            pH=pad,
            pW=pad,
            dH=dilation,
            dW=dilation,
            groups=groups,
            dtype=spec.dtype,
        )
        if supports_gfx942_conv_fwd_aot_problem(problem)[0]:
            problems.append(problem)
    return problems


def test_direct_serving_does_not_depend_on_the_arch(catalog):
    """A direct instance serves on gfx942 exactly what the same fields serve on
    gfx950, so the native direct serve rule is shared; and where the rule is
    rocKE's dispatcher admission (every variant but std and spatial) the verdict is
    that admission for this process's flavor."""
    compared: dict = {}
    served: dict = {}
    for dtype in _DTYPES:
        for index, (identity, spec) in enumerate(catalog[dtype]):
            if not spec.is_direct:
                continue
            twin = _as_gfx950(spec)
            current = _current(identity)
            for problem in _direct_problems(spec, 12, _SEED + index):
                ours, reason = conv_fwd_aot_serves(spec, problem)
                assert (ours, reason) == conv_fwd_aot_serves(twin, problem), (
                    identity.label(),
                    problem,
                )
                if ours and spec.direct_variant not in ("std", "spatial"):
                    request = replace(
                        spec.request(problem), llvm_flavor=current.llvm_flavor
                    )
                    assert _admits(current, request)[0], (identity.label(), problem)
                compared[spec.algorithm] = compared.get(spec.algorithm, 0) + 1
                served[spec.algorithm] = served.get(spec.algorithm, 0) + ours
    assert set(compared) == {a for counts in _DIRECT_COUNTS.values() for a in counts}
    # Both verdicts occur for every algorithm, so the agreement is not vacuous.
    for algorithm, count in compared.items():
        assert 0 < served[algorithm] < count, (algorithm, served[algorithm], count)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"C": 64, "K": 128}, "depthwise"),
        ({"C": 128, "K": 128, "groups": 64}, "depthwise"),
        ({"C": 40, "K": 40, "groups": 40}, "does not divide groups"),
        ({"C": 1, "K": 1, "groups": 1}, "does not divide groups"),
    ],
)
def test_a_merged_instance_serves_depthwise_its_degree_divides(merged, change, reason):
    _, _, spec = next(m for m in merged if m[2].group_merge == 16)
    problem = Gfx950ConvFwdSpec(
        N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=64,
        dtype=spec.dtype,
    )  # fmt: skip
    assert gfx942_conv_fwd_aot_serves(spec, problem) == (True, "ok")
    ok, why = gfx942_conv_fwd_aot_serves(spec, replace(problem, **change))
    assert not ok and reason in why, why


# ---- The merged launch ----


def test_merged_grid_and_launch_values_are_rockes_for_the_runtime_problem(merged):
    """The grid is rocKE's merged grid -- the builder's ``implicit_gemm_conv_grid``
    on the instance spec for the problem -- and the kernel arguments are the TRUE
    problem's ``ConvArgs``, as the benchmark launches."""
    launched = 0
    for dtype, _, spec in merged:
        for problem in _random_problems(100, dtype, _SEED + 1):
            if not gfx942_conv_fwd_aot_serves(spec, problem)[0]:
                continue
            p = problem.to_problem()
            rocke_spec = replace(spec.to_instance_spec(), problem=p)
            assert rocke_spec.group_merge == spec.group_merge
            grid = spec.grid(problem)
            assert grid == implicit_gemm_conv_grid(rocke_spec)
            assert grid[2] * spec.group_merge == problem.groups
            args = ConvArgs.from_problem(p, tile_m=spec.tile_m, tile_n=spec.tile_n)
            assert spec.launch_values(
                problem, 0, 0, 0, 1, 2, 3
            ) == args.to_launch_values(0, 0, 0, 1, 2, 3)
            assert spec.block() == (rocke_spec.launch_block_size, 1, 1)
            launched += 1
    assert launched > 100


# ---- Runtime selection: which serving instances run ----

#: (N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW, groups). The gfx950 module's grid
#: (odd and even channel runs, grouped and grouped pointwise, strided, dilated and
#: large filters, output extents past 65535 small M tiles), then depthwise with group
#: counts divisible by 32, by 16 not 32, by 4 not 16, and by none of the merge
#: degrees, and a depthwise channel multiplier no merged instance serves.
_SELECTION_GRID = (
    (2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    (2, 3, 32, 32, 32, 7, 7, 2, 2, 3, 3, 1, 1, 1),
    (2, 32, 5, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    (1, 3, 5, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    (2, 6, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    (2, 10, 6, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    (2, 4, 12, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    (2, 32, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 4),
    (2, 12, 20, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 4),
    (1, 9, 48, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, 3),
    (2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32),
    (2, 64, 128, 14, 14, 1, 1, 1, 1, 0, 0, 1, 1, 1),
    (2, 24, 24, 14, 14, 1, 1, 1, 1, 0, 0, 1, 1, 3),
    (2, 16, 24, 17, 19, 3, 3, 2, 2, 2, 2, 2, 2, 1),
    (1, 4, 12, 31, 31, 11, 11, 4, 4, 5, 5, 1, 1, 1),
    (32, 8, 8, 192, 192, 1, 1, 1, 1, 0, 0, 1, 1, 1),
    (36, 2, 2, 500, 500, 1, 1, 1, 1, 0, 0, 1, 1, 1),
    (2, 128, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32),
    (2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64),
    (2, 96, 96, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 96),
    (2, 48, 48, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 48),
    (2, 20, 20, 9, 9, 7, 7, 1, 1, 3, 3, 1, 1, 20),
    (2, 7, 7, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 7),
    (2, 32, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32),
)


def test_runtime_candidates_are_rockes_runtime_selection(catalog):
    """For every problem the implicit-GEMM instances selected are exactly the ones
    rocKE's --run-from-cache benchmark runs by default over the catalog's identities:
    KernelCache.compatible, then _widest_vector_sizes, both rocKE's own, called on the
    identities. Direct instances are not in that selection (another benchmark runs
    them) and every serving one runs. The gfx950 engine's policy, unchanged."""
    from benchmarks.common.benchmark_implicit_gemm_conv import _widest_vector_sizes

    narrowed = merged_selected = 0
    for dtype, entries in catalog.items():
        specs = [s for _, s in entries]
        implicit = {s.instance_id: _current(i) for i, s in entries if not s.is_direct}
        restated = {i.stable_hash(): sid for sid, i in implicit.items()}
        cache = KernelCache.__new__(KernelCache)
        # compatible() needs the in-memory state __init__ sets up; _entries is the
        # only thing it reads through, so no disk-backed state is needed here.
        cache._groups = {}
        cache._compatible = OrderedDict()
        cache._compatible_rows = 0
        cached = [(i, None, {}) for i in implicit.values()]
        cache._entries = lambda direction, cached=cached: cached
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
                if s.is_direct and gfx942_conv_fwd_aot_serves(s, problem)[0]
            ], (dtype, values)
            narrowed += len([s for s in ours if not s.is_direct]) < len(compatible)
            merged_selected += any(s.is_merged for s in ours)
    # The policy narrows, and merged instances run wherever a merge degree divides a
    # depthwise group count. How often it narrows tracks the catalog's vector-width
    # spread (12 of 48 grid problems with the 459-instance list).
    assert narrowed > 0 and merged_selected >= 2 * 5


# ---- The profile ----


def test_profile_dispatch_resolves_gfx942_requests_including_grouped_pointwise():
    for request in (
        ConvGroupedRequest(
            N=2, Hi=8, Wi=8, C=32, K=32, Y=1, X=1, G=4, arch=_ARCH, dtype="fp16"
        ),
        ConvGroupedRequest(
            N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, G=64, pad_h=1, pad_w=1,
            arch=_ARCH, dtype="bf16",
        ),
    ):  # fmt: skip
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


def test_every_direct_algorithm_of_the_catalog_has_a_packaged_launch():
    assert {a for counts in _DIRECT_COUNTS.values() for a in counts} <= set(
        TARGETS[_ARCH].direct_algorithms
    )
    assert set(TARGETS[_ARCH].direct_algorithms) <= set(AOT_DIRECT_ALGORITHMS)
