#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Turn rocKE AOT forward-conv instances into a hipkernel conv-fwd AOT engine config.

One tool for both forward engines of the rocKE adapter's ``TARGETS``, chosen by
``--arch``: ``hipkernel:Gfx950ConvFwdAot`` (the default) and
``hipkernel:Gfx1250ConvFwdAot``. What differs per engine is read from the adapter
(its spec class and the gfx1250 knob tables) or kept in :data:`ARCHES`; the rest of
this docstring describes gfx950, and gfx1250 where it says so.

An instance is one rocKE ``KernelIdentity`` (``kernels/conv/kernel_identity.py``):
a compiled configuration with no problem extents, which serves every problem its
capabilities admit. The input is a JSON list whose entries are either an identity
dict (``KernelIdentity.to_dict()``) or an object carrying one under ``"identity"``;
other keys of such an object are ignored here.

Every entry must be a forward identity of ``--arch`` in fp16 or bf16 that the
packaged launch supports: an implicit-GEMM identity of the AOT sweep
(``kernels/conv/aot_jobs.py``), or a forward identity of the direct sweep
(``kernels/conv/direct_aot_jobs.py``) of an algorithm the packaged launch runs
(``PACKAGED_DIRECT_ALGORITHMS``; on gfx1250 the adapter's
``GFX1250_DIRECT_ALGORITHMS``). The rocKE adapter decides
(``conv_fwd_aot_spec_for_identity`` in
``builders/common/convolution_forward.py``); it refuses, with a reason, every identity
feature the packaged launch does not model, such as a direct algorithm it does
not run or any other field off its default.
Anything refused is listed with its reason and the tool exits non-zero, unless
``--skip-unsupported`` is given. Entries
are de-duplicated by ``instance_id`` (``KernelIdentity.stable_hash()``), which is the
key that ties a packaged kernel back to the identity it was built from.

The shipped catalog is a curated subset of rocKE's forward sweeps, one list per dtype
(``configs/gfx950_conv_fwd_aot_<dtype>.instances.json``), written by
``tools/enumerate_conv_aot_instances.py``; ``--instances`` takes them all as one
catalog. ``--engine-name`` emits the same instances as another engine over the
shipped engine's native code: the test-only seed is
``configs/gfx950_conv_fwd_aot_seed.instances.json`` as
``hipkernel:Gfx950ConvFwdAotSeed``. The config is an intermediate, regenerated from
the lists on every run; only the seed's small one is kept, as a worked example.

Then run ``generate.py --config <output>`` and ``--stabilize-ids`` on the emitted
bundle: the generator mints random kernel UUIDs, and a recorded autotune winner is
keyed by kernel UUID, so this tool re-derives each kernel's UUID from its
``instance_id`` (and engine) to keep it stable across regenerations::

    python tools/make_conv_aot_config.py --rocke-root "$ROCKE" \\
        --instances configs/gfx950_conv_fwd_aot_{fp16,bf16}.instances.json \\
        --output "$WORK"/aot.yaml
    python generate.py --config "$WORK"/aot.yaml --output-dir "$GENERATED" --force
    python tools/make_conv_aot_config.py --stabilize-ids "$GENERATED"/<...>/*.kdp.json

The config splits the catalog into packs, so KDPs, by ``--sharding``
(:data:`SHARDINGS`): by default one per dtype, pipeline and K loop, plus one per
dtype for the direct instances. Each is a shard of the one catalog, which keeps
every emitted file well below the size a source host accepts.

The gfx1250 catalog is ordered: its lists keep rocKE's registration order (the
gfx1250 forward list is fitted to measured timings), and within a dtype a kernel's
``priority`` falls with its position in them. That is the order the engine tries the
kernels a graph admits when no ranker decides; gfx950 ranks by its native score
instead and sorts its lists by hash::

    python tools/make_conv_aot_config.py --arch gfx1250 --rocke-root "$ROCKE" \\
        --instances configs/gfx1250_conv_fwd_aot_{fp16,bf16}.instances.json \\
        --output "$WORK"/aot.yaml
    python tools/make_conv_aot_config.py --arch gfx1250 \\
        --stabilize-ids "$GENERATED"/<...>/*.kdp.json
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import functools
import json
import multiprocessing
import os
import sys
from pathlib import Path

import conv_aot_common as _common

INSTANCE_UUID_NAMESPACE = _common.INSTANCE_UUID_NAMESPACE

#: The shipped gfx950 engine (the default ``--arch``). Its native symbols are the ones
#: every gfx950 bundle this tool emits names; another engine name (a test-only subset
#: of the catalog) borrows them.
PRODUCT_ENGINE = "hipkernel:Gfx950ConvFwdAot"
DEFAULT_ARCH = "gfx950"

_UUID = _common._UUID
_kernels = _common._kernels

#: Identity fields the packaged forward launch takes from the identity
#: (``AOT_IMPLICIT_GEMM_CARRIED`` of the rocKE adapter). Every other field must hold
#: its KernelIdentity default; the adapter refuses the identity otherwise.
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
    "unroll_k",
    "async_dma",
    "async_chunk_a",
    "async_chunk_b",
    "grouped",
)
#: Spec field order of an implicit-GEMM Gfx950ConvFwdAotSpec; also its descriptor
#: spec order.
SPEC_ORDER = ("dtype",) + _CARRIED + ("instance_id", "layout")
#: The direct fields Gfx950ConvFwdAotSpec appends, which every direct instance
#: states. Their KMD default_value is the dataclass default (builder_default), so
#: implicit-GEMM descriptors omit them.
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
    "block_w": 0,
    "block_waves": 0,
}
#: The block knobs of the tiled, col, grouped and non-grouped direct variants, after
#: DIRECT_FIELDS, with their builder_default. A direct instance states those of its
#: variant only (DIRECT_VARIANT_FIELDS), so std and spatial descriptors are as they
#: were before these existed.
DIRECT_KNOB_FIELDS = {
    "block_h": 0,
    "unroll_rows": 0,
    "block_q": 0,
    "block_groups": 0,
    "double_buffer": 0,
    "cpg": 0,
    "kpg": 0,
    "tile_h": 0,
    "tile_w": 0,
    "tile_k_out": 0,
    "ck": 0,
    "waves_m": 0,
    "waves_n": 0,
    "atom": "none",
    "swizzle_wgm": 0,
    "iglp": -1,
    "waves_per_eu": 0,
    "group_merge": 1,
    "split_c": 1,
    "prefetch_rows": 0,
    "w_fold": 0,
}
#: Spec field order of a std or spatial direct instance.
DIRECT_SPEC_ORDER = SPEC_ORDER + tuple(DIRECT_FIELDS)
#: The DIRECT_KNOB_FIELDS each direct variant states, in their KMD order.
DIRECT_VARIANT_FIELDS = {
    "std": (),
    "spatial": (),
    "tiled": ("block_h", "unroll_rows"),
    "col": ("block_h",),
    "grouped": ("block_h", "block_q", "block_groups", "double_buffer", "cpg", "kpg"),
    "nongrouped": (
        "tile_h",
        "tile_w",
        "tile_k_out",
        "ck",
        "waves_m",
        "waves_n",
        "atom",
        "swizzle_wgm",
        "iglp",
        "waves_per_eu",
    ),
    "grouped_halo": (
        "tile_h",
        "tile_w",
        "tile_k_out",
        "ck",
        "waves_m",
        "waves_n",
        "atom",
        "swizzle_wgm",
        "iglp",
        "waves_per_eu",
        "group_merge",
    ),
    "win": ("block_h", "prefetch_rows", "w_fold", "cpg", "kpg"),
}
#: Spec field order of a group-merged implicit-GEMM instance: the merge degree
#: after the implicit-GEMM fields, every direct field at its builder_default.
MERGED_SPEC_ORDER = SPEC_ORDER + ("group_merge",)
_STRING_FIELDS = (
    "dtype",
    "pipeline",
    "epilogue",
    "instance_id",
    "layout",
    "direct_variant",
    "atom",
)
_INT_FIELDS = tuple(
    f for f in DIRECT_SPEC_ORDER + tuple(DIRECT_KNOB_FIELDS) if f not in _STRING_FIELDS
)
#: The UED knobs: every KMD field that tunes a kernel, so the ranking model can tell
#: apart the instances one problem admits. Left out are the fields that only restate
#: the problem the instance was matched to (dtype, layout, the direct filter, stride,
#: padding, dilation and channels per group), the constants (wave_size, grouped) and
#: instance_id, an identity a model would memorise rather than learn from.
ENGINE_KNOBS = (
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
    "unroll_k",
    "async_dma",
    "async_chunk_a",
    "async_chunk_b",
    "block_w",
    "block_waves",
    "block_h",
    "unroll_rows",
    "block_q",
    "block_groups",
    "double_buffer",
    "tile_h",
    "tile_w",
    "tile_k_out",
    "ck",
    "waves_m",
    "waves_n",
    "atom",
    "swizzle_wgm",
    "iglp",
    "waves_per_eu",
    "group_merge",
    "split_c",
    "prefetch_rows",
    "w_fold",
)
#: Direct-sweep algorithm -> the instance's direct_variant, for the algorithms the
#: exact-shape engine's direct launch runs.
DIRECT_ALGORITHMS = {"direct_depthwise": "std", "direct_depthwise_spatial": "spatial"}
#: Every forward direct-sweep algorithm the packaged AOT launch runs, and its
#: direct_variant: AOT_DIRECT_ALGORITHMS of the rocKE adapter.
#: tools/enumerate_conv_aot_instances.py ships every identity of these.
PACKAGED_DIRECT_ALGORITHMS = {
    **DIRECT_ALGORITHMS,
    "direct_depthwise_tiled": "tiled",
    "direct_depthwise_col": "col",
    "direct_grouped": "grouped",
    "direct_nongrouped": "nongrouped",
    "direct_grouped_halo": "grouped_halo",
    "direct_depthwise_win": "win",
}
#: The generator variant (kernel name) of each direct variant. std and spatial keep
#: the name they had when they were the only direct instances.
_DIRECT_VARIANT_NAMES = {
    "std": "gfx950_conv_fwd_aot_dw_{dtype}_{instance_id}",
    "spatial": "gfx950_conv_fwd_aot_dw_{dtype}_{instance_id}",
    "tiled": "gfx950_conv_fwd_aot_dwtiled_{dtype}_{instance_id}",
    "col": "gfx950_conv_fwd_aot_dwcol_{dtype}_{instance_id}",
    "grouped": "gfx950_conv_fwd_aot_grouped_{dtype}_{instance_id}",
    "nongrouped": "gfx950_conv_fwd_aot_nongrouped_{dtype}_{instance_id}",
    "grouped_halo": "gfx950_conv_fwd_aot_groupedhalo_{dtype}_{instance_id}",
    "win": "gfx950_conv_fwd_aot_dwwin_{dtype}_{instance_id}",
}
#: The generator variant of a group-merged implicit-GEMM instance and of a
#: split-C non-grouped one.
_MERGED_VARIANT_NAME = "gfx950_conv_fwd_aot_merged_{dtype}_{instance_id}"
_SPLIT_C_VARIANT_NAME = "gfx950_conv_fwd_aot_nongroupedsplit_{dtype}_{instance_id}"
#: The dispatch a split-C pack names (``hipkernel.gfx950_conv_fwd_aot.split_c_dispatch``):
#: its kernels launch twice, the conv pass and the fold, so its handler declares
#: the launches ``conv`` and ``reduce`` and every other pack keeps the engine's
#: single-launch handler.
SPLIT_C_DISPATCH = "split_c"
#: The launches of a split-C kernel: its conv pass (the instance's own builder)
#: and rocKE's split reduction over the same spec.
SPLIT_C_LAUNCHES = {
    "conv": {},
    "reduce": {"builder": "build_gfx950_conv_fwd_aot_split_reduce"},
}


#: The gfx1250 engine's generator variants: one per direct variant (the adapter's
#: ``GFX1250_DIRECT_ALGORITHMS`` values), each its own kernel name.
_GFX1250_DIRECT_VARIANT_NAMES = {
    "std": "gfx1250_conv_fwd_aot_dw_{dtype}_{instance_id}",
    "spatial": "gfx1250_conv_fwd_aot_dwspatial_{dtype}_{instance_id}",
    "tiled": "gfx1250_conv_fwd_aot_dwtiled_{dtype}_{instance_id}",
    "col": "gfx1250_conv_fwd_aot_dwcol_{dtype}_{instance_id}",
    "dw_wmma": "gfx1250_conv_fwd_aot_dwwmma_{dtype}_{instance_id}",
    "wmma": "gfx1250_conv_fwd_aot_wmma_{dtype}_{instance_id}",
    "smallc_wmma": "gfx1250_conv_fwd_aot_smallcwmma_{dtype}_{instance_id}",
    "smallk_wmma": "gfx1250_conv_fwd_aot_smallkwmma_{dtype}_{instance_id}",
    "gstream_wmma": "gfx1250_conv_fwd_aot_gstreamwmma_{dtype}_{instance_id}",
    "tapouter_wmma": "gfx1250_conv_fwd_aot_tapouterwmma_{dtype}_{instance_id}",
}


@dataclasses.dataclass(frozen=True)
class ArchConfig:
    """What the config of one ``--arch`` engine spells beyond the adapter's tables:
    its generator variant names, the order its direct variants are grouped in (a
    tuple per generator variant) and whether its catalog is ordered (each kernel's
    ``priority`` from its list position, the lists kept in that order)."""

    implicit_name: str
    merged_name: str
    direct_names: dict
    direct_groups: tuple
    ordered: bool


#: Each ``--arch`` engine's :class:`ArchConfig`.
ARCHES = {
    "gfx950": ArchConfig(
        implicit_name="gfx950_conv_fwd_aot_{dtype}_{instance_id}",
        merged_name=_MERGED_VARIANT_NAME,
        direct_names=_DIRECT_VARIANT_NAMES,
        # std and spatial share one variant, as before the other direct variants.
        direct_groups=(
            ("std", "spatial"),
            ("tiled",),
            ("col",),
            ("grouped",),
            ("nongrouped",),
            ("grouped_halo",),
            ("win",),
        ),
        ordered=False,
    ),
    "gfx1250": ArchConfig(
        implicit_name="gfx1250_conv_fwd_aot_{dtype}_{instance_id}",
        merged_name="gfx1250_conv_fwd_aot_merged_{dtype}_{instance_id}",
        direct_names=_GFX1250_DIRECT_VARIANT_NAMES,
        direct_groups=tuple((v,) for v in _GFX1250_DIRECT_VARIANT_NAMES),
        ordered=True,
    ),
}


_bind_rocke = _common.bind_rocke


def _adapter():
    import builders.common.convolution_forward as adapter

    return adapter


def arches() -> tuple:
    """The architectures with a packaged forward engine: the adapter's ``TARGETS``,
    in its order. rocKE must already be bound (:func:`_bind_rocke`)."""
    return tuple(_adapter().TARGETS)


def product_engine(arch: str = DEFAULT_ARCH) -> str:
    """``arch``'s shipped forward engine, its own product engine (ids are uuid5 of
    instance_id): the adapter's ``TARGETS[arch].engine``."""
    targets = _adapter().TARGETS
    if arch not in targets or arch not in ARCHES:
        raise ValueError(f"no packaged forward engine for arch {arch!r}")
    return targets[arch].engine


def product_native_namespace(arch: str = DEFAULT_ARCH) -> str:
    """The native-symbol namespace :func:`product_engine` registers
    (``hipkernel.<arch>_conv_fwd_aot``), which a borrowing engine states verbatim
    as its ``native_symbols_from``."""
    engine = product_engine(arch)
    return f"{engine.split(':', 1)[0]}.{engine_slug(engine)}"


def adapter_builder(arch: str = DEFAULT_ARCH) -> str:
    """The adapter's typed builder ``arch``'s kernels are built by,
    ``build_<arch>_conv_fwd_aot``."""
    product_engine(arch)
    name = f"build_{arch}_conv_fwd_aot"
    if not callable(getattr(_adapter(), name, None)):
        raise ValueError(f"the adapter has no builder {name} for arch {arch!r}")
    return name


def optional_fields(arch: str = DEFAULT_ARCH) -> dict:
    """``name -> builder_default`` of the KMD fields after :data:`SPEC_ORDER`, in
    spec order: :data:`DIRECT_FIELDS` and :data:`DIRECT_KNOB_FIELDS` on gfx950; on
    gfx1250 the same (``unroll_rows`` defaulting to 0, its absent value) and then
    the knobs its spec class adds, each at its dataclass default."""
    fields = {**DIRECT_FIELDS, **DIRECT_KNOB_FIELDS}
    if arch == DEFAULT_ARCH:
        return fields
    adapter = _adapter()
    spec_class = adapter.SPEC_CLASSES[arch]
    base = {f.name for f in dataclasses.fields(adapter.Gfx950ConvFwdAotSpec)}
    defaults = {f.name: f.default for f in dataclasses.fields(spec_class)}
    fields = {name: defaults[name] for name in fields}
    fields.update({n: d for n, d in defaults.items() if n not in base})
    return fields


def string_fields(arch: str = DEFAULT_ARCH) -> tuple:
    """The KMD fields of type string."""
    return _STRING_FIELDS + tuple(
        name
        for name, default in optional_fields(arch).items()
        if isinstance(default, str) and name not in _STRING_FIELDS
    )


def engine_knobs(arch: str = DEFAULT_ARCH) -> tuple:
    """The UED knobs (:data:`ENGINE_KNOBS`), and on gfx1250 every knob its spec
    class adds."""
    if arch == DEFAULT_ARCH:
        return ENGINE_KNOBS
    shared = {**DIRECT_FIELDS, **DIRECT_KNOB_FIELDS}
    return ENGINE_KNOBS + tuple(n for n in optional_fields(arch) if n not in shared)


def direct_variant_fields(spec) -> tuple:
    """The optional fields a direct instance of ``spec``'s variant states, in KMD
    order: :data:`DIRECT_VARIANT_FIELDS` on gfx950; on gfx1250 the fields of the
    variant's knobs (the adapter's ``GFX1250_DIRECT_KNOBS``) past
    :data:`DIRECT_FIELDS`, and ``cpg``/``kpg`` where the variant records them."""
    if spec.TARGET.arch == DEFAULT_ARCH:
        return DIRECT_VARIANT_FIELDS[spec.direct_variant]
    adapter = _adapter()
    used = set(adapter.GFX1250_DIRECT_KNOBS[spec.direct_variant].values())
    if spec.direct_variant in adapter._GFX1250_RECORDED_CAPS:
        used |= {"cpg", "kpg"}
    return tuple(
        n
        for n in optional_fields(spec.TARGET.arch)
        if n in used and n not in DIRECT_FIELDS
    )


def spec_order(spec) -> tuple:
    """The descriptor spec order of ``spec``'s family and direct variant."""
    if not spec.is_direct:
        return MERGED_SPEC_ORDER if spec.is_merged else SPEC_ORDER
    order = DIRECT_SPEC_ORDER + direct_variant_fields(spec)
    return order + ("split_c",) if spec.is_split_c else order


def instance_spec(identity_dict: dict, arch: str = DEFAULT_ARCH):
    """``(spec, "")`` for a supported ``arch`` identity, else ``(None, reason)``.

    The rocKE adapter's ``conv_fwd_aot_spec_for_identity``: the identity must be one
    of ``arch`` that rocKE's forward sweeps build (``job_for_identity`` /
    ``direct_job_for_identity``) and use no feature the packaged launch does not
    model.
    """
    from builders.common.convolution_forward import (
        conv_fwd_aot_spec_for_identity,
        gfx950_conv_fwd_aot_spec_for_identity,
    )
    from kernels.conv.kernel_identity import KernelIdentity

    try:
        identity = KernelIdentity.from_dict(identity_dict)
    except TypeError as exc:
        return None, f"not a KernelIdentity: {exc}"
    if arch == DEFAULT_ARCH:
        return gfx950_conv_fwd_aot_spec_for_identity(identity)
    if identity.arch != arch:
        kind = f"({identity.arch}, {identity.direction}, {identity.algorithm})"
        return None, f"not a {arch} forward implicit-GEMM or direct identity {kind}"
    return conv_fwd_aot_spec_for_identity(identity)


def instance_specs(identities: list, jobs: int = 1, arch: str = DEFAULT_ARCH) -> list:
    """:func:`instance_spec` of each identity, in order, over ``jobs`` forked worker
    processes (serially where ``fork`` is not available)."""
    convert = functools.partial(instance_spec, arch=arch)
    if (
        jobs <= 1
        or len(identities) < 1000
        or "fork" not in multiprocessing.get_all_start_methods()
    ):
        return [convert(identity) for identity in identities]
    chunk = max(1, len(identities) // (jobs * 8))
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        return pool.map(convert, identities, chunksize=chunk)


_variant = _common.variant
engine_slug = _common.engine_slug


#: The shipped engine's dotted native-symbol namespace (``hipkernel.gfx950_conv_fwd_aot``):
#: what another engine states verbatim as its ``native_symbols_from``.
PRODUCT_NATIVE_NAMESPACE = (
    f"{PRODUCT_ENGINE.split(':', 1)[0]}.{engine_slug(PRODUCT_ENGINE)}"
)


#: How :func:`build_config` splits a catalog into packs, one KDP each.
#: ``dtype``: one pack per dtype. ``k-loop``: one pack per dtype, implicit-GEMM
#: pipeline and K loop (``plain``, ``unroll`` or ``async``), and one per dtype for
#: the direct instances, which keeps every KDP of rocKE's full sweep well below
#: the size a source host accepts.
SHARDINGS = ("dtype", "k-loop")


def shard_name(spec, sharding: str = "dtype") -> str:
    """The pack ``spec`` belongs to under ``sharding`` (see :data:`SHARDINGS`)."""
    if sharding == "dtype":
        # A split-C kernel launches through its own dispatch handler.
        return f"{spec.dtype}_split" if spec.is_split_c else spec.dtype
    if sharding != "k-loop":
        raise ValueError(f"unknown sharding {sharding!r}; expected one of {SHARDINGS}")
    if spec.is_direct:
        # A split-C kernel launches twice, through its own dispatch handler,
        # so it cannot share a pack with single-launch kernels.
        return (
            f"{spec.dtype}_direct_split" if spec.is_split_c else f"{spec.dtype}_direct"
        )
    k_loop = "async" if spec.async_dma else ("unroll" if spec.unroll_k else "plain")
    return f"{spec.dtype}_{spec.pipeline}_{k_loop}"


def priorities(specs: list) -> dict:
    """``instance_id -> priority`` of an ordered catalog: within a dtype, falling
    with list position, so the first listed is tried first."""
    by_dtype = collections.defaultdict(list)
    for spec in specs:
        by_dtype[spec.dtype].append(spec.instance_id)
    return {
        instance_id: len(ids) - 1 - index
        for ids in by_dtype.values()
        for index, instance_id in enumerate(ids)
    }


def _ordered_variant(
    name: str, order: tuple, specs: list, priority: dict | None
) -> dict:
    variant = _variant(name, order, specs)
    if priority is not None:
        for shape, spec in zip(variant["shapes"], specs):
            shape["priority"] = priority[spec.instance_id]
    return variant


def build_config(
    specs: list,
    engine_name: str | None = None,
    authored_subpath: str = "",
    sharding: str = "dtype",
    arch: str = DEFAULT_ARCH,
) -> dict:
    """The generator config for ``specs`` (already validated, de-duplicated).

    ``engine_name`` (default: ``arch``'s :func:`product_engine`) other than the
    product is a second engine over the same native code: its descriptors name the
    shipped engine's symbols (``native_symbols_from``). ``authored_subpath``
    defaults to ``rocKE/<engine slug>``. An ordered ``arch`` (:data:`ARCHES`)
    states each kernel's ``priority`` (:func:`priorities`) in ``specs``' order.

    The catalog is split into packs by ``sharding`` (:func:`shard_name`), each
    its own KDP, so no file has to hold the whole catalog. The packs declare no
    discriminator: they are shards of one catalog, and the engine's graph_match
    and the shared kernel matcher admit every one of them. Within a pack,
    implicit-GEMM instances keep their own variant and leave the direct fields to
    their builder_default. Direct instances are one variant per direct kernel
    (std and spatial share one), each stating DIRECT_FIELDS and its own block
    knobs (:func:`direct_variant_fields`) and leaving the others to theirs.
    """
    product = product_engine(arch)
    if engine_name is None:
        engine_name = product
    layout = ARCHES[arch]
    strings = string_fields(arch)
    optional = optional_fields(arch)
    kmd = [
        {"name": name, "type": "string" if name in strings else "int"}
        for name in SPEC_ORDER
    ] + [
        {
            "name": name,
            "type": "string" if name in strings else "int",
            "default_value": default,
            "builder_default": True,
        }
        for name, default in optional.items()
    ]
    metadata = list(SPEC_ORDER + tuple(optional))
    priority = priorities(specs) if layout.ordered else None
    shards = collections.defaultdict(list)
    for spec in specs:
        shards[shard_name(spec, sharding)].append(spec)
    packs = []
    for shard in sorted(shards):
        variants = []
        implicit = [s for s in shards[shard] if not s.is_direct and not s.is_merged]
        if implicit:
            variants.append(
                _ordered_variant(layout.implicit_name, SPEC_ORDER, implicit, priority)
            )
        merged = [s for s in shards[shard] if s.is_merged]
        if merged:
            variants.append(
                _ordered_variant(
                    layout.merged_name, MERGED_SPEC_ORDER, merged, priority
                )
            )
        for group in layout.direct_groups:
            members = [
                s
                for s in shards[shard]
                if s.is_direct and s.direct_variant in group and not s.is_split_c
            ]
            if members:
                variants.append(
                    _ordered_variant(
                        layout.direct_names[group[0]],
                        spec_order(members[0]),
                        members,
                        priority,
                    )
                )
        split = [s for s in shards[shard] if s.is_split_c]
        if split:
            variants.append(
                _variant(_SPLIT_C_VARIANT_NAME, spec_order(split[0]), split)
            )
        defaults = {
            "kind": "rocke",
            "source": "builders/common/convolution_forward.py",
            "builder": adapter_builder(arch),
        }
        pack = {"name": shard, "arch": [arch]}
        if split:
            if len(split) != len(shards[shard]):
                raise ValueError(
                    f"pack {shard!r} mixes split-C kernels with single-launch ones"
                )
            defaults["launches"] = {
                name: dict(override) for name, override in SPLIT_C_LAUNCHES.items()
            }
            pack["dispatch"] = SPLIT_C_DISPATCH
        packs.append({**pack, "kernel_defaults": defaults, "variants": variants})
    engine = {
        "name": engine_name,
        "sdk_version": "1.0.0",
        "knobs": list(engine_knobs(arch)),
        "heuristic": "native",
    }
    if engine_name != product:
        engine["native_symbols_from"] = product_native_namespace(arch)
    return {
        "dialect": "packaged",
        "authored_subpath": authored_subpath or f"rocKE/{engine_slug(engine_name)}",
        "engine": engine,
        "kmd_fields": kmd,
        "kernel_source_kind": "rocke",
        # Only the split-C instances use a workspace (their fp32 partial sums),
        # sized per graph by the native handler.
        "workspace_policy": "derived" if any(s.is_split_c for s in specs) else "none",
        "graph_match": {"shape": "shared_shape", "discriminator": "none"},
        "specialization": {
            "metadata_fields": metadata,
            "matcher_only_fields": [],
            "bindings": {name: {"field": name} for name in metadata},
        },
        "packs": packs,
    }


dump_config = _common.dump_config


def stabilize_ids(
    kdp_path: Path, engine_name: str = PRODUCT_ENGINE, product: str = PRODUCT_ENGINE
) -> int:
    """:func:`conv_aot_common.stabilize_ids` with :data:`PRODUCT_ENGINE` as the
    default for both arguments."""
    return _common.stabilize_ids(kdp_path, engine_name, product)


def kernel_uuid(
    instance_id: str, engine_name: str = PRODUCT_ENGINE, product: str = PRODUCT_ENGINE
) -> str:
    """:func:`conv_aot_common.kernel_uuid` with :data:`PRODUCT_ENGINE` as the
    default for both arguments."""
    return _common.kernel_uuid(instance_id, engine_name, product)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--rocke-root", type=Path, help="rocKE tree (platform/, library/)."
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
        "instance must be one of it (default: %(default)s).",
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
        "derives (default: the --arch engine). Another name borrows that engine's "
        "native symbols.",
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
        default="k-loop",
        help="How the catalog is split into packs, one KDP each (default: "
        "%(default)s; the test-only seed uses dtype).",
    )
    parser.add_argument(
        "--stabilize-ids",
        type=Path,
        nargs="+",
        metavar="KDP",
        help="Rewrite each generated .kdp.json's kernel UUIDs from instance_id, then exit.",
    )
    args = parser.parse_args(argv)
    if args.arch not in ARCHES:
        parser.error(
            f"argument --arch: invalid choice: {args.arch!r} (choose from "
            f"{', '.join(sorted(ARCHES))})"
        )
    # The product engine name is the adapter's; without rocKE (--stabilize-ids
    # alone) it is the engine slug this tool names its variants after.
    product = (
        PRODUCT_ENGINE
        if args.arch == DEFAULT_ARCH
        else f"hipkernel:{args.arch.capitalize()}ConvFwdAot"
    )
    if args.stabilize_ids:
        engine_name = args.engine_name or product
        for kdp in args.stabilize_ids:
            count = stabilize_ids(kdp, engine_name, product)
            print(f"{kdp}: stabilized {count} kernel id(s)")
        return 0
    if not (args.rocke_root and args.instances and args.output):
        parser.error("--rocke-root, --instances and --output are required")
    _bind_rocke(args.rocke_root)
    if product_engine(args.arch) != product:
        parser.error(f"the adapter's {args.arch} engine is {product_engine(args.arch)}")

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
    converted = instance_specs(list(identities.values()), args.jobs, args.arch)
    for index, (spec, reason) in zip(identities, converted):
        if spec is None:
            rejected.append((index, reason))
        elif spec.instance_id not in seen:
            seen.add(spec.instance_id)
            specs.append(spec)
    if not ARCHES[args.arch].ordered:
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
        f"# Generated by tools/make_conv_aot_config.py from {sources}; "
        "do not edit by hand.\n"
    )
    config = build_config(
        specs, args.engine_name, args.authored_subpath, args.sharding, args.arch
    )
    args.output.write_text(header + dump_config(config))
    print(
        f"{len(specs)} instance(s) written to {args.output} "
        f"({len(entries) - len(rejected) - len(specs)} duplicate(s), {len(rejected)} unsupported)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
