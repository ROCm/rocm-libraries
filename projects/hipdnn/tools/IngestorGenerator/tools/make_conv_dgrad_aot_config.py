#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Turn rocKE AOT backward-data conv instances into a dgrad AOT engine config.

One tool for every dgrad engine of the rocKE adapter's ``TARGETS``, chosen by
``--arch`` (:func:`arches`): ``hipkernel:Gfx950ConvDgradAot`` (the default) and
``hipkernel:Gfx1250ConvDgradAot``.
The input is JSON lists of rocKE ``KernelIdentity`` dicts (or objects carrying one
under ``"identity"``); the output is a packaged-dialect generator config, followed by
a ``--stabilize-ids`` step. The YAML dumper, variant layout and id derivation are the
conv AOT tools' shared ones (``conv_aot_common.py``), with each engine as its own
product engine: kernel ids are ``uuid5(INSTANCE_UUID_NAMESPACE, instance_id)`` and the
bundle names its own native symbols (no ``native_symbols_from``).

Every entry must be a dgrad identity of the engine's architecture in fp16 or bf16
that the packaged launch runs: an implicit-GEMM identity of the dgrad AOT sweep
(``benchmarks/common/kernel_sweep.py``; it carries ``lds_k_outer`` and
``max_sub_gemms``), or a dgrad identity of the direct sweep
(``benchmarks/common/direct_kernel_sweep.py``) of a packaged algorithm
(:func:`packaged_direct_algorithms`), including each gfx950 MFMA pipeline's
weight-transform helpers (``direct_dgrad_helper``). The helpers are packaged as kernels of their own
family (``kernel_family`` 2) that the kernel matcher never admits; the MFMA dispatch
loads them from the same pack. Anything else is listed with its reason and the tool
exits non-zero, unless ``--skip-unsupported`` is given.

An engine's shipped catalog is ``configs/<arch>_conv_dgrad_aot_<dtype>.instances.json``,
written by ``tools/enumerate_conv_dgrad_aot_instances.py``::

    python tools/make_conv_dgrad_aot_config.py --arch "$ARCH" --rocke-root "$ROCKE" \\
        --instances configs/"$ARCH"_conv_dgrad_aot_{fp16,bf16}.instances.json \\
        --output "$WORK"/dgrad_aot.yaml
    python generate.py --config "$WORK"/dgrad_aot.yaml --output-dir "$GENERATED" --force
    python tools/make_conv_dgrad_aot_config.py --arch "$ARCH" \\
        --stabilize-ids "$GENERATED"/<...>/*.kdp.json

The catalog is split into packs (:data:`SHARDINGS`): by default one per dtype and
implicit-GEMM pipeline, plus one per dtype for the direct and helper kernels.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import conv_aot_common as _common  # noqa: E402
import launch_surface  # noqa: E402
from codegen.models import EngineSpec  # noqa: E402

#: The default architecture. Its engine, like every other's, is the adapter's
#: ``TARGETS[arch].engine`` (:func:`product_engine`), read once rocKE is bound.
DEFAULT_ARCH = "gfx950"
#: The rocKE adapter every kernel of every dgrad bundle is built by.
ADAPTER_SOURCE = "builders/common/convolution_backward_data.py"
#: The in-repo rocKE tree, the default ``--rocke-root``.
DEFAULT_ROCKE_ROOT = (
    launch_surface.find_repo_root(_HERE) / "dnn-providers/hip-kernel-provider/rocke"
)

#: Identity fields the packaged dgrad launch takes from an implicit-GEMM identity.
_CARRIED = (
    "tile_m",
    "tile_n",
    "tile_k",
    "warp_m",
    "warp_n",
    "warp_tile_m",
    "warp_tile_n",
    "warp_tile_k",
    "wave_size",
    "pipeline",
    "epilogue",
    "vector_size_a",
    "vector_size_b",
    "vector_size_c",
    "lds_k_outer",
    "max_sub_gemms",
    "grouped",
    # dY-halo and K-decode tuning: each selects emission, so an instance that sets
    # one is a different binary and the catalog has to say so.
    "dy_halo",
    "dy_halo_setprio",
    "dy_halo_kouter_pad",
    "halo_max_wo",
    "patch",
    "tap_outer_k",
    "uniform_k_decode",
)
_BOOL_CARRIED = (
    "lds_k_outer",
    "grouped",
    "patch",
    "tap_outer_k",
    "uniform_k_decode",
)
#: Geometry an implicit-GEMM identity may bake instead of reading from the tilde
#: records at run time. 0 means runtime, as before; a non-zero value is a kernel
#: specialized for exactly that geometry, and the matcher gates on it (_BAKEABLE of
#: the adapter's _implicit_serves).
_BAKEABLE = ("filter_h", "filter_w", "stride_h", "stride_w", "dilation_h", "dilation_w")
#: Identity fields an implicit-GEMM identity must hold at their KernelIdentity
#: default (they select emission the packaged launch or matcher does not model).
#: Padding is 0 (runtime, read from the tilde records).
_CONSTANT = (
    "chiplet_swizzle",
    "unroll_k",
    "async_dma",
    "async_chunk_a",
    "async_chunk_b",
    "lds_layout",
    "lds_k_pad",
    "waves_per_eu",
    "acc_epilogue",
    "split_k",
    "two_stage",
    "ws_replicas",
    "group_merge",
    "num_load_waves",
    "cshuffle_no_alias",
    "is_3d",
    "is_pointwise",
    "filter_d",
    "pad_h",
    "pad_w",
    "cpg",
    "kpg",
    "block_h",
    "knobs",
)
#: Spec field order of an implicit-GEMM ConvDgradAotSpec; also its descriptor
#: spec order (the KMD's required fields).
SPEC_ORDER = ("dtype",) + _CARRIED + ("instance_id", "layout")
#: The fields every direct and helper instance states after SPEC_ORDER, with their
#: KMD builder_default (an implicit-GEMM descriptor omits them).
DIRECT_FIELDS = {
    "kernel_family": 0,
    "direct_variant": "none",
    "filter_h": 0,
    "filter_w": 0,
    "stride_h": 0,
    "stride_w": 0,
    "pad_h": 0,
    "pad_w": 0,
    "dilation_h": 0,
    "dilation_w": 0,
    "cpg": 0,
    "kpg": 0,
}


def arches() -> tuple:
    """The architectures with a packaged dgrad engine: the adapter's ``TARGETS``, in
    its order. rocKE must already be bound (:func:`_bind_rocke`)."""
    from builders.common.convolution_backward_data import TARGETS

    return tuple(TARGETS)


def _target(arch: str):
    """The adapter's ``TARGETS[arch]``; a ValueError for an arch it does not package."""
    from builders.common.convolution_backward_data import SPEC_CLASSES, TARGETS

    if arch not in TARGETS or arch not in SPEC_CLASSES:
        raise ValueError(f"no packaged dgrad engine for arch {arch!r}")
    return TARGETS[arch]


def product_engine(arch: str = DEFAULT_ARCH) -> str:
    """``arch``'s shipped dgrad engine, its own product engine (ids are uuid5 of
    instance_id): the adapter's ``TARGETS[arch].engine``."""
    return _target(arch).engine


def product_native_namespace(arch: str = DEFAULT_ARCH) -> str:
    """The native-symbol namespace :func:`product_engine` registers, which a borrowing
    engine states verbatim as its ``native_symbols_from``."""
    return EngineSpec(name=product_engine(arch)).own_native_symbol_namespace


def adapter_builder(arch: str = DEFAULT_ARCH) -> str:
    """The adapter's typed builder ``arch``'s kernels are built by,
    ``build_<arch>_conv_dgrad_aot``; a ValueError when the adapter lacks it."""
    import builders.common.convolution_backward_data as adapter

    _target(arch)
    name = f"build_{arch}_conv_dgrad_aot"
    if not callable(getattr(adapter, name, None)):
        raise ValueError(f"the adapter has no builder {name} for arch {arch!r}")
    return name


def _spec_class(arch: str):
    """The adapter's spec class of ``arch``'s engine (``SPEC_CLASSES``)."""
    from builders.common.convolution_backward_data import SPEC_CLASSES

    _target(arch)
    return SPEC_CLASSES[arch]


#: The variant knobs after DIRECT_FIELDS, with their builder_default. A direct or
#: helper instance states those of its variant only (_direct_variant_fields()).
def _direct_knob_fields(arch: str = DEFAULT_ARCH) -> dict:
    """Every direct/helper knob of ``arch``'s engine with its KMD default: the
    adapter's ``KNOB_DEFAULTS`` of its spec class, not a second list that could
    drift from it."""
    return dict(_spec_class(arch).KNOB_DEFAULTS)


def _direct_variant_fields(arch: str = DEFAULT_ARCH) -> dict:
    """Per variant, the knobs it states, in KMD order. A variant states every knob it
    may take (the spec class's ``DIRECT_KNOBS``).

    A transpose states fold_k32 as well, always 0: an MFMA dispatch looks its helpers
    up by (dtype, family, variant, capabilities, cpg/kpg, fold_k32), so both helpers
    must carry a stated value rather than leaving the transpose on the KMD default.
    """
    from builders.common.convolution_backward_data import (
        HELPER_VARIANT_TRANSPOSE,
        HELPER_VARIANT_TRANSPOSE_RTC,
    )

    order = list(_direct_knob_fields(arch))
    stated = {
        variant: set(mapping.values())
        for variant, mapping in _spec_class(arch).DIRECT_KNOBS.items()
    }
    for variant in (HELPER_VARIANT_TRANSPOSE, HELPER_VARIANT_TRANSPOSE_RTC):
        if variant in stated:
            stated[variant].add("fold_k32")
    return {
        variant: tuple(f for f in order if f in fields)
        for variant, fields in stated.items()
    }


DIRECT_SPEC_ORDER = SPEC_ORDER + tuple(DIRECT_FIELDS)
_STRING_FIELDS = (
    "dtype",
    "pipeline",
    "epilogue",
    "instance_id",
    "layout",
    "direct_variant",
    # gfx1250's WMMA schedule knob.
    "sched",
)


def _int_fields(arch: str = DEFAULT_ARCH) -> tuple:
    """Every KMD field that is an int. Resolved on first use: the knob set comes from
    the adapter, which is bound at run time."""
    return tuple(
        f
        for f in DIRECT_SPEC_ORDER + tuple(_direct_knob_fields(arch))
        if f not in _STRING_FIELDS
    )


#: The UED knobs: every KMD field that tunes a kernel, so the ranking model can tell
#: apart the instances one problem admits; together they make every per-graph
#: candidate tuple unique. Left out are the fields that only restate the problem the
#: instance was matched to (dtype, layout, the direct filter, stride, padding,
#: dilation, cpg/kpg), the constants (wave_size, max_sub_gemms) and
#: instance_id. The MFMA block_h is its row tile (the knob token), not a constant.
def engine_knobs(arch: str = DEFAULT_ARCH) -> tuple:
    """The UED knobs of ``arch``'s engine: every KMD field that tunes a kernel, so the
    ranking model can tell apart the instances one problem admits. The per-variant
    knobs come from the adapter, so a pipeline this engine gains is rankable without
    a change here."""
    fixed = (
        "kernel_family",
        "direct_variant",
        "tile_m",
        "tile_n",
        "tile_k",
        "warp_m",
        "warp_n",
        "warp_tile_m",
        "warp_tile_n",
        "warp_tile_k",
        "pipeline",
        "epilogue",
        "vector_size_a",
        "vector_size_b",
        "vector_size_c",
        "lds_k_outer",
        # Not a constant: a grouped binary also serves groups == 1, so for such a
        # problem it and its ungrouped twin both match and the model has to tell
        # them apart.
        "grouped",
        "dy_halo",
        "dy_halo_setprio",
        "dy_halo_kouter_pad",
        "halo_max_wo",
        "patch",
        "tap_outer_k",
        "uniform_k_decode",
    )
    return fixed + tuple(_direct_knob_fields(arch))


def packaged_direct_algorithms(arch: str = DEFAULT_ARCH) -> dict:
    """Each dgrad direct-sweep algorithm with a packaged launch on ``arch``, and its
    ``direct_variant``: the adapter's ``TARGETS[arch].direct_algorithms``. Derived,
    not restated: the adapter decides what has a packaged launch, and a second list
    here could only drift from it."""
    from builders.common.convolution_backward_data import AOT_DGRAD_DIRECT_ALGORITHMS

    return {
        algorithm: AOT_DGRAD_DIRECT_ALGORITHMS[algorithm][1]
        for algorithm in _spec_class(arch).target().direct_algorithms
    }


def direct_directions() -> dict:
    """The ``KernelIdentity.direction`` each packaged direct algorithm carries."""
    from builders.common.convolution_backward_data import (
        AOT_DGRAD_DIRECT_ALGORITHMS,
        KERNEL_FAMILY_DGRAD_HELPER,
    )

    return {
        algorithm: (
            "direct_dgrad_helper"
            if family == KERNEL_FAMILY_DGRAD_HELPER
            else "direct_dgrad"
        )
        for algorithm, (family, _) in AOT_DGRAD_DIRECT_ALGORITHMS.items()
    }


#: The generator variant (kernel name) of the implicit-GEMM instances and of each
#: direct and helper variant, after the engine's ``{arch}``.
IMPLICIT_VARIANT_NAME = "{arch}_conv_dgrad_aot_{dtype}_{instance_id}"
_DIRECT_VARIANT_NAMES = {
    "depthwise_dgrad": "{arch}_conv_dgrad_aot_dw_{dtype}_{instance_id}",
    "depthwise_dgrad_win": "{arch}_conv_dgrad_aot_dw_win_{dtype}_{instance_id}",
    "grouped_dgrad_mfma": "{arch}_conv_dgrad_aot_mfma_{dtype}_{instance_id}",
    "dgrad_mfma_rtc": "{arch}_conv_dgrad_aot_mfma_rtc_{dtype}_{instance_id}",
    "grouped_dgrad_fused": "{arch}_conv_dgrad_aot_fused_{dtype}_{instance_id}",
    "grouped_dgrad_4c": "{arch}_conv_dgrad_aot_4c_{dtype}_{instance_id}",
    "dgrad_wmma": "{arch}_conv_dgrad_aot_wmma_{dtype}_{instance_id}",
    "dgrad_smallc_wmma": "{arch}_conv_dgrad_aot_smallc_wmma_{dtype}_{instance_id}",
    "dgrad_patch_wmma": "{arch}_conv_dgrad_aot_patch_wmma_{dtype}_{instance_id}",
    "transpose": "{arch}_conv_dgrad_aot_transpose_{dtype}_{instance_id}",
    "reorganize": "{arch}_conv_dgrad_aot_reorganize_{dtype}_{instance_id}",
    "transpose_rtc": "{arch}_conv_dgrad_aot_transpose_rtc_{dtype}_{instance_id}",
    "reorganize_rtc": "{arch}_conv_dgrad_aot_reorganize_rtc_{dtype}_{instance_id}",
}


def _variant_name(template: str, arch: str) -> str:
    """``template`` for ``arch``, its ``{dtype}`` and ``{instance_id}`` left for the
    generator."""
    return template.replace("{arch}", arch)


_bind_rocke = _common.bind_rocke
engine_slug = _common.engine_slug
dump_config = _common.dump_config
INSTANCE_UUID_NAMESPACE = _common.INSTANCE_UUID_NAMESPACE


def kernel_uuid(
    instance_id: str, engine_name: str | None = None, arch: str = DEFAULT_ARCH
) -> str:
    """:func:`conv_aot_common.kernel_uuid` with ``arch``'s engine as the
    product (and, by default, as ``engine_name``)."""
    product = product_engine(arch)
    return _common.kernel_uuid(instance_id, engine_name or product, product)


def stabilize_ids(
    kdp_path: Path, engine_name: str | None = None, arch: str = DEFAULT_ARCH
) -> int:
    """:func:`conv_aot_common.stabilize_ids` with ``arch``'s engine as the
    product (and, by default, as ``engine_name``)."""
    product = product_engine(arch)
    return _common.stabilize_ids(kdp_path, engine_name or product, product)


def spec_order(spec) -> tuple:
    """The descriptor spec order of ``spec``'s family and variant.

    An implicit-GEMM instance states its geometry, 0 where it reads the extent from
    its sub-GEMM records. All six, always: one order serves the whole variant group,
    and the fields have to be there anyway -- the native matcher gates on them, and
    the UED knob tuple has to tell two tiles apart that differ only in what they were
    compiled for.
    """
    if not spec.is_direct:
        return SPEC_ORDER + _BAKEABLE
    return DIRECT_SPEC_ORDER + _direct_variant_fields(spec.ARCH)[spec.direct_variant]


def instance_spec(identity_dict: dict):
    """``(spec, "")`` for a supported identity -- an instance of the spec class of the
    identity's architecture (``Gfx950ConvDgradAotSpec``, ``Gfx1250ConvDgradAotSpec``)
    -- else ``(None, reason)``."""
    from benchmarks.common.kernel_cache import KernelIdentity

    identity = KernelIdentity.from_dict(identity_dict)
    if identity.arch not in arches():
        return None, (
            f"no packaged dgrad engine for arch {identity.arch!r} "
            f"(packaged: {sorted(arches())})"
        )
    if identity.is_direct:
        return _direct_instance_spec(identity)
    return _implicit_instance_spec(identity)


def instance_specs(identities: list, jobs: int = 1) -> list:
    """:func:`instance_spec` of each identity, in order, over ``jobs`` forked workers."""
    if (
        jobs <= 1
        or len(identities) < 200
        or "fork" not in multiprocessing.get_all_start_methods()
    ):
        return [instance_spec(identity) for identity in identities]
    chunk = max(1, len(identities) // (jobs * 8))
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        return pool.map(instance_spec, identities, chunksize=chunk)


def _checked(spec):
    from builders.common.convolution_backward_data import (
        supports_conv_dgrad_aot,
    )

    ok, reason = supports_conv_dgrad_aot(spec)
    return (spec, "") if ok else (None, reason)


def _implicit_instance_spec(identity):
    from benchmarks.common.kernel_cache import KernelIdentity

    if (identity.direction, identity.algorithm) != ("dgrad", "implicit_gemm"):
        return None, (
            f"not a dgrad implicit-GEMM identity "
            f"({identity.arch}, {identity.direction}, {identity.algorithm})"
        )
    if not identity.dtype_a == identity.dtype_b == identity.dtype_d:
        return None, "mixed operand dtypes are not packaged"
    defaults = KernelIdentity(
        arch=identity.arch,
        direction=identity.direction,
        algorithm=identity.algorithm,
        dtype_a=identity.dtype_a,
        dtype_b=identity.dtype_b,
        dtype_d=identity.dtype_d,
        tile_m=0,
        tile_n=0,
        tile_k=0,
        warp_m=0,
        warp_n=0,
        warp_tile_m=0,
        warp_tile_n=0,
        warp_tile_k=0,
        pipeline="",
        epilogue="",
        wave_size=0,
        vector_size_a=0,
        vector_size_b=0,
        vector_size_c=0,
        llvm_flavor=identity.llvm_flavor,
    )
    for name in _CONSTANT:
        if getattr(identity, name) != getattr(defaults, name):
            return None, (
                f"{name}={getattr(identity, name)!r} is not packaged "
                f"(only {getattr(defaults, name)!r})"
            )
    # Every packaged identity must be one rocKE's dgrad generator produces.
    from benchmarks.common.kernel_sweep import job_for_identity

    try:
        job_for_identity(identity)
    except LookupError as exc:
        return None, str(exc)
    values = {name: getattr(identity, name) for name in _CARRIED}
    for name in _BOOL_CARRIED:
        values[name] = int(values[name])
    # Omitted (left at the KMD builder default 0) when the kernel reads its geometry
    # from the tilde records; stated when it was compiled for one geometry.
    values.update({name: getattr(identity, name) for name in _BAKEABLE})
    spec = _spec_class(identity.arch)(
        dtype=identity.dtype_a, instance_id=identity.stable_hash(), **values
    )
    return _checked(spec)


def _direct_instance_spec(identity):
    """A dgrad direct or helper identity: the direct sweep must be able to build it,
    and the packaged launch must run (or load) its algorithm."""
    from benchmarks.common.direct_kernel_sweep import direct_job_for_identity
    from builders.common.convolution_backward_data import (
        aot_dgrad_direct_instance_fields,
    )

    packaged = packaged_direct_algorithms(identity.arch)
    if identity.algorithm not in packaged:
        return None, (
            f"direct algorithm {identity.algorithm!r} has no packaged dgrad launch on "
            f"{identity.arch} (supported: {sorted(packaged)})"
        )
    if identity.direction != direct_directions()[identity.algorithm]:
        return None, (
            f"{identity.algorithm} is a {direct_directions()[identity.algorithm]} "
            f"algorithm, the identity says {identity.direction}"
        )
    if not identity.dtype_a == identity.dtype_b == identity.dtype_d:
        return None, "mixed operand dtypes are not packaged"
    try:
        direct_job_for_identity(identity)
        fields = aot_dgrad_direct_instance_fields(identity)
    except (LookupError, TypeError, ValueError) as exc:
        return None, str(exc)
    spec = _spec_class(identity.arch)(
        dtype=identity.dtype_a,
        instance_id=identity.stable_hash(),
        wave_size=identity.wave_size,
        tile_m=0,
        tile_n=0,
        tile_k=0,
        warp_m=0,
        warp_n=0,
        warp_tile_m=0,
        warp_tile_n=0,
        warp_tile_k=0,
        pipeline=identity.pipeline,
        epilogue=identity.epilogue,
        vector_size_a=0,
        vector_size_b=0,
        vector_size_c=0,
        lds_k_outer=0,
        max_sub_gemms=0,
        grouped=0,
        **fields,
    )
    return _checked(spec)


#: How :func:`build_config` splits a catalog into packs, one KDP each.
#: ``dtype``: one pack per dtype. ``pipeline``: one per dtype and implicit-GEMM
#: pipeline, plus one per dtype for the direct and helper kernels.
SHARDINGS = ("dtype", "pipeline")


def shard_name(spec, sharding: str = "pipeline") -> str:
    """The pack ``spec`` belongs to under ``sharding`` (see :data:`SHARDINGS`)."""
    if sharding == "dtype":
        return spec.dtype
    if sharding != "pipeline":
        raise ValueError(f"unknown sharding {sharding!r}; expected one of {SHARDINGS}")
    if spec.is_direct:
        return f"{spec.dtype}_direct"
    return f"{spec.dtype}_{spec.pipeline}"


def kmd_fields(arch: str = DEFAULT_ARCH) -> list:
    """The KMD of ``arch``'s engine: SPEC_ORDER required, then the direct fields and
    knobs with defaults."""
    int_fields = _int_fields(arch)
    return [
        {"name": name, "type": "int" if name in int_fields else "string"}
        for name in SPEC_ORDER
    ] + [
        {
            "name": name,
            "type": "int" if name in int_fields else "string",
            "default_value": default,
            "builder_default": True,
        }
        for name, default in {**DIRECT_FIELDS, **_direct_knob_fields(arch)}.items()
    ]


def build_config(
    specs: list,
    engine_name: str | None = None,
    authored_subpath: str = "",
    sharding: str = "pipeline",
) -> dict:
    """The generator config for ``specs`` (already validated, de-duplicated, all of
    one architecture's engine).

    The packs are shards of one catalog with no discriminator. Within a pack the
    implicit-GEMM instances are one variant, and each direct or helper variant its
    own, stating DIRECT_FIELDS and its knobs and leaving the others to their
    builder_default. ``engine_name`` (by default the architecture's product engine,
    :func:`product_engine`) other than that product engine borrows its native symbols
    (``native_symbols_from`` states :func:`product_native_namespace`).
    """
    spec_arches = {spec.ARCH for spec in specs}
    if len(spec_arches) != 1:
        raise ValueError(
            f"one engine's catalog is one architecture's, got {spec_arches}"
        )
    (arch,) = spec_arches
    product, builder = product_engine(arch), adapter_builder(arch)
    engine_name = engine_name or product
    metadata = list(DIRECT_SPEC_ORDER + tuple(_direct_knob_fields(arch)))
    shards = collections.defaultdict(list)
    for spec in specs:
        shards[shard_name(spec, sharding)].append(spec)
    packs = []
    for shard in sorted(shards):
        variants = []
        implicit = [s for s in shards[shard] if not s.is_direct]
        if implicit:
            variants.append(
                _common.variant(
                    _variant_name(IMPLICIT_VARIANT_NAME, arch),
                    spec_order(implicit[0]),
                    implicit,
                )
            )
        for variant_name in _DIRECT_VARIANT_NAMES:
            members = [
                s
                for s in shards[shard]
                if s.is_direct and s.direct_variant == variant_name
            ]
            if members:
                variants.append(
                    _common.variant(
                        _variant_name(_DIRECT_VARIANT_NAMES[variant_name], arch),
                        spec_order(members[0]),
                        members,
                    )
                )
        packs.append(
            {
                "name": shard,
                "arch": [arch],
                "kernel_defaults": {
                    "kind": "rocke",
                    "source": ADAPTER_SOURCE,
                    "builder": builder,
                },
                "variants": variants,
            }
        )
    engine = {
        "name": engine_name,
        # 1.3.0: the graph matcher gates tensor alignment (multiples of 16 bytes) and
        # declines ragged tensors, so graphs declaring either reach it.
        "sdk_version": "1.3.0",
        "knobs": list(engine_knobs(arch)),
        "heuristic": "native",
    }
    if engine_name != product:
        engine["native_symbols_from"] = product_native_namespace(arch)
    return {
        "dialect": "packaged",
        "authored_subpath": authored_subpath or f"rocKE/{engine_slug(engine_name)}",
        "engine": engine,
        "kmd_fields": kmd_fields(arch),
        "kernel_source_kind": "rocke",
        # The implicit-GEMM sub-GEMM record table and the MFMA weight scratch are
        # sized per problem by the native handler's workspaceBytes.
        "workspace_policy": "derived",
        "graph_match": {"shape": "shared_shape", "discriminator": "none"},
        "specialization": {
            "metadata_fields": metadata,
            "matcher_only_fields": [],
            "bindings": {name: {"field": name} for name in metadata},
        },
        "packs": packs,
    }


def check_arch(parser: argparse.ArgumentParser, arch: str) -> None:
    """Refuse ``--arch`` through ``parser`` unless ``arch`` is one of :func:`arches`;
    rocKE must already be bound. Every dgrad tool's command line checks with it."""
    if arch not in arches():
        parser.error(
            f"argument --arch: invalid choice: {arch!r} "
            f"(choose from {', '.join(sorted(arches()))})"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--rocke-root",
        type=Path,
        default=DEFAULT_ROCKE_ROOT,
        help="rocKE tree (platform/, library/; default: the in-repo tree).",
    )
    parser.add_argument(
        "--instances",
        type=Path,
        nargs="+",
        help="JSON lists of KernelIdentity dicts, read as one catalog.",
    )
    parser.add_argument("--output", type=Path, help="Generator config to write.")
    parser.add_argument(
        "--arch",
        default=DEFAULT_ARCH,
        help="The engine's architecture, one of the adapter's TARGETS; every "
        "instance must be one of its (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-unsupported",
        action="store_true",
        help="Drop unsupported identities (each is still reported) instead of failing.",
    )
    parser.add_argument(
        "--engine-name",
        default=None,
        help="Engine the bundle declares, and whose kernel ids --stabilize-ids "
        "derives (default: the --arch engine).",
    )
    parser.add_argument(
        "--authored-subpath",
        default="",
        help="The bundle's authored subpath (default: rocKE/<engine slug>).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Worker processes that convert the identities (default: %(default)s).",
    )
    parser.add_argument(
        "--sharding",
        choices=SHARDINGS,
        default="pipeline",
        help="How the catalog is split into packs, one KDP each (default: %(default)s).",
    )
    parser.add_argument(
        "--stabilize-ids",
        type=Path,
        nargs="+",
        metavar="KDP",
        help="Rewrite each generated .kdp.json's kernel UUIDs from instance_id, then exit.",
    )
    args = parser.parse_args(argv)
    # The engines and arches are the adapter's TARGETS, so rocKE is bound first,
    # --stabilize-ids included.
    _bind_rocke(args.rocke_root)
    check_arch(parser, args.arch)
    args.engine_name = args.engine_name or product_engine(args.arch)

    if args.stabilize_ids:
        for kdp in args.stabilize_ids:
            count = stabilize_ids(kdp, args.engine_name, args.arch)
            print(f"{kdp}: stabilized {count} kernel id(s)")
        return 0
    if not (args.instances and args.output):
        parser.error("--instances and --output are required")

    entries = []
    for path in args.instances:
        listed = json.loads(path.read_text())
        if not isinstance(listed, list):
            parser.error(f"--instances {path} must hold a JSON list")
        entries += listed
    specs, seen, rejected = [], set(), []
    identities = {}
    for index, entry in enumerate(entries):
        identity = entry.get("identity", entry) if isinstance(entry, dict) else None
        if isinstance(identity, dict):
            identities[index] = identity
        else:
            rejected.append((index, "entry is not an identity dict"))
    converted = instance_specs(list(identities.values()), args.jobs)
    for index, (spec, reason) in zip(identities, converted):
        if spec is None:
            rejected.append((index, reason))
        elif spec.ARCH != args.arch:
            rejected.append(
                (index, f"a {spec.ARCH} instance in the {args.arch} engine's catalog")
            )
        elif spec.instance_id not in seen:
            seen.add(spec.instance_id)
            specs.append(spec)
    specs.sort(key=lambda s: (s.dtype, s.instance_id))
    rejected.sort()
    for index, reason in rejected:
        print(f"unsupported instance [{index}]: {reason}", file=sys.stderr)
    if rejected and not args.skip_unsupported:
        print(
            f"{len(rejected)} unsupported instance(s); nothing written", file=sys.stderr
        )
        return 1
    if not specs:
        print("no supported instances; nothing written", file=sys.stderr)
        return 1
    sources = ", ".join(path.name for path in args.instances)
    header = (
        f"# Generated by tools/make_conv_dgrad_aot_config.py from {sources}; "
        "do not edit by hand.\n"
    )
    config = build_config(specs, args.engine_name, args.authored_subpath, args.sharding)
    args.output.write_text(header + dump_config(config), newline="")
    print(
        f"{len(specs)} instance(s) written to {args.output} "
        f"({len(entries) - len(rejected) - len(specs)} duplicate(s), {len(rejected)} unsupported)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
