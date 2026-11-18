#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Turn rocKE AOT forward-conv instances into a forward-conv AOT engine config.

One tool for every forward engine of the rocKE adapter's ``TARGETS``, chosen by
``--arch`` (:func:`arches`): ``hipkernel:Gfx950ConvFwdAot`` (the default) and
``hipkernel:Gfx942ConvFwdAot``. Each is its own product engine: kernel ids are
``uuid5(INSTANCE_UUID_NAMESPACE, instance_id)`` and the bundle names its own native
symbols.

An instance is one rocKE ``KernelIdentity`` (``kernels/conv/kernel_identity.py``):
a compiled configuration with no problem extents, which serves every problem its
capabilities admit. The input is a JSON list whose entries are either an identity
dict (``KernelIdentity.to_dict()``) or an object carrying one under ``"identity"``;
other keys of such an object are ignored here.

Every entry must be a forward identity of the engine's architecture in fp16 or bf16
that the packaged launch supports: an implicit-GEMM identity of the AOT sweep
(``kernels/conv/aot_jobs.py``), or a forward identity of the direct sweep
(``kernels/conv/direct_aot_jobs.py``) of an algorithm the architecture's packaged
launch runs (the adapter's ``TARGETS[arch].direct_algorithms``). The rocKE adapter
decides (``conv_fwd_aot_spec_for_identity`` in
``builders/common/convolution_forward.py``); it refuses, with a reason, every identity
feature the packaged launch does not model and any other field off its default.
Anything refused is listed with its reason and the tool exits non-zero, unless
``--skip-unsupported`` is given. Entries
are de-duplicated by ``instance_id`` (``KernelIdentity.stable_hash()``), which is the
key that ties a packaged kernel back to the identity it was built from.

An engine's shipped catalog is a curated subset of rocKE's forward sweeps, one list
per dtype (``configs/<arch>_conv_fwd_aot_<dtype>.instances.json``), written by
``tools/enumerate_conv_aot_instances.py``; ``--instances`` takes them all as one
catalog. ``--engine-name`` emits the same instances as another engine over the
shipped engine's native code: the test-only seed is
``configs/gfx950_conv_fwd_aot_seed.instances.json`` as
``hipkernel:Gfx950ConvFwdAotSeed``. The config is an intermediate, regenerated from
the lists on every run; only the seed's small one is kept, as a worked example.

Then run ``generate.py --config <output>`` and ``--stabilize-ids`` on the emitted
bundle: the generator mints random kernel UUIDs, and a recorded autotune winner is
keyed by kernel UUID, so this tool re-derives each kernel's UUID from its
``instance_id`` (and engine) to keep it stable across regenerations. Run under the
catalog's own LLVM flavor (``llvm22`` for both engines), e.g. for gfx942::

    export ARCH=gfx942 ROCKE_LLVM_FLAVOR=llvm22
    python tools/make_conv_aot_config.py --arch "$ARCH" --rocke-root "$ROCKE" \\
        --instances configs/"$ARCH"_conv_fwd_aot_{fp16,bf16}.instances.json \\
        --output "$WORK"/aot.yaml
    python generate.py --config "$WORK"/aot.yaml --output-dir "$GENERATED" --force
    python tools/make_conv_aot_config.py --arch "$ARCH" \\
        --stabilize-ids "$GENERATED"/<...>/*.kdp.json

The config splits the catalog into packs, so KDPs, by ``--sharding``
(:data:`SHARDINGS`): by default one per dtype, pipeline and K loop, plus one per
dtype for the direct instances. Each is a shard of the one catalog, which keeps
every emitted file well below the size a source host accepts.
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

INSTANCE_UUID_NAMESPACE = _common.INSTANCE_UUID_NAMESPACE

#: The default architecture. Its engine, like every other's, is the adapter's
#: ``TARGETS[arch].engine`` (:func:`product_engine`), read once rocKE is bound.
DEFAULT_ARCH = "gfx950"
#: The rocKE adapter every kernel of every forward bundle is built by.
ADAPTER_SOURCE = "builders/common/convolution_forward.py"
#: The in-repo rocKE tree, the default ``--rocke-root``.
DEFAULT_ROCKE_ROOT = (
    launch_surface.find_repo_root(_HERE) / "dnn-providers/hip-kernel-provider/rocke"
)

#: The default architecture's shipped engine, :func:`product_engine` of
#: :data:`DEFAULT_ARCH`, spelled out for the code that names it before rocKE is
#: bound. Another engine name over its catalog (a test-only subset) borrows its
#: native symbols.
PRODUCT_ENGINE = "hipkernel:Gfx950ConvFwdAot"

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
#: Spec field order of an implicit-GEMM instance (``ConvFwdAotSpec``); also its
#: descriptor spec order.
SPEC_ORDER = ("dtype",) + _CARRIED + ("instance_id", "layout")
#: The direct fields ConvFwdAotSpec appends, which every direct instance
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
#: Every forward direct-sweep algorithm a packaged AOT launch runs, and its
#: direct_variant: AOT_DIRECT_ALGORITHMS of the rocKE adapter. An architecture
#: packages those of its ``TARGETS[arch].direct_algorithms``
#: (:func:`packaged_direct_algorithms`).
PACKAGED_DIRECT_ALGORITHMS = {
    **DIRECT_ALGORITHMS,
    "direct_depthwise_tiled": "tiled",
    "direct_depthwise_col": "col",
    "direct_grouped": "grouped",
    "direct_nongrouped": "nongrouped",
    "direct_grouped_halo": "grouped_halo",
    "direct_depthwise_win": "win",
}
#: The generator variant (kernel name) of an implicit-GEMM instance and of each
#: direct variant, after the engine's ``{arch}`` (:func:`_variant_name`). std and
#: spatial keep the name they had when they were the only direct instances.
IMPLICIT_VARIANT_NAME = "{arch}_conv_fwd_aot_{dtype}_{instance_id}"
_DIRECT_VARIANT_NAMES = {
    "std": "{arch}_conv_fwd_aot_dw_{dtype}_{instance_id}",
    "spatial": "{arch}_conv_fwd_aot_dw_{dtype}_{instance_id}",
    "tiled": "{arch}_conv_fwd_aot_dwtiled_{dtype}_{instance_id}",
    "col": "{arch}_conv_fwd_aot_dwcol_{dtype}_{instance_id}",
    "grouped": "{arch}_conv_fwd_aot_grouped_{dtype}_{instance_id}",
    "nongrouped": "{arch}_conv_fwd_aot_nongrouped_{dtype}_{instance_id}",
    "grouped_halo": "{arch}_conv_fwd_aot_groupedhalo_{dtype}_{instance_id}",
    "win": "{arch}_conv_fwd_aot_dwwin_{dtype}_{instance_id}",
}
#: The generator variant of a group-merged implicit-GEMM instance and of a
#: split-C non-grouped one.
_MERGED_VARIANT_NAME = "{arch}_conv_fwd_aot_merged_{dtype}_{instance_id}"
_SPLIT_C_VARIANT_NAME = "{arch}_conv_fwd_aot_nongroupedsplit_{dtype}_{instance_id}"
#: The dispatch a split-C pack names (``hipkernel.<arch>_conv_fwd_aot.split_c_dispatch``):
#: its kernels launch twice, the conv pass and the fold, so its handler declares
#: the launches ``conv`` and ``reduce`` and every other pack keeps the engine's
#: single-launch handler.
SPLIT_C_DISPATCH = "split_c"


def _variant_name(template: str, arch: str) -> str:
    """``template`` for ``arch``, its ``{dtype}`` and ``{instance_id}`` left for the
    generator."""
    return template.replace("{arch}", arch)


_bind_rocke = _common.bind_rocke


def arches() -> tuple:
    """The architectures with a packaged forward engine: the adapter's ``TARGETS``,
    in its order. rocKE must already be bound (:func:`_bind_rocke`)."""
    from builders.common.convolution_forward import TARGETS

    return tuple(TARGETS)


def _target(arch: str):
    """The adapter's ``TARGETS[arch]``; a ValueError for an arch it does not package."""
    from builders.common.convolution_forward import SPEC_CLASSES, TARGETS

    if arch not in TARGETS or arch not in SPEC_CLASSES:
        raise ValueError(f"no packaged forward engine for arch {arch!r}")
    return TARGETS[arch]


def product_engine(arch: str = DEFAULT_ARCH) -> str:
    """``arch``'s shipped forward engine, its own product engine (ids are uuid5 of
    instance_id): the adapter's ``TARGETS[arch].engine``."""
    return _target(arch).engine


def product_native_namespace(arch: str = DEFAULT_ARCH) -> str:
    """The native-symbol namespace :func:`product_engine` registers, which a borrowing
    engine states verbatim as its ``native_symbols_from``."""
    return EngineSpec(name=product_engine(arch)).own_native_symbol_namespace


def adapter_builder(arch: str = DEFAULT_ARCH, launch: str = "") -> str:
    """The adapter's typed builder ``arch``'s kernels are built by,
    ``build_<arch>_conv_fwd_aot``, or that of one of their extra launches
    (``launch="split_reduce"``: ``build_<arch>_conv_fwd_aot_split_reduce``); a
    ValueError when the adapter lacks it."""
    import builders.common.convolution_forward as adapter

    _target(arch)
    name = f"build_{arch}_conv_fwd_aot" + (f"_{launch}" if launch else "")
    if not callable(getattr(adapter, name, None)):
        raise ValueError(f"the adapter has no builder {name} for arch {arch!r}")
    return name


def split_c_launches(arch: str = DEFAULT_ARCH) -> dict:
    """The launches of ``arch``'s split-C kernels: the conv pass (the instance's own
    builder) and rocKE's split reduction over the same spec."""
    return {
        "conv": {},
        "reduce": {"builder": adapter_builder(arch, "split_reduce")},
    }


def packaged_direct_algorithms(arch: str = DEFAULT_ARCH) -> dict:
    """Each forward direct-sweep algorithm with a packaged launch on ``arch``, and its
    ``direct_variant``: the adapter's ``TARGETS[arch].direct_algorithms``, which is
    what refuses any other algorithm (:func:`instance_spec` defers to the adapter)."""
    return {
        algorithm: PACKAGED_DIRECT_ALGORITHMS[algorithm]
        for algorithm in _target(arch).direct_algorithms
    }


def check_arch(parser: argparse.ArgumentParser, arch: str) -> None:
    """Refuse ``--arch`` through ``parser`` unless ``arch`` is one of :func:`arches`;
    rocKE must already be bound. Every forward tool's command line checks with it."""
    if arch not in arches():
        parser.error(
            f"argument --arch: invalid choice: {arch!r} "
            f"(choose from {', '.join(sorted(arches()))})"
        )


def check_engine_name(
    parser: argparse.ArgumentParser, engine_name: str | None, arch: str
) -> None:
    """Refuse ``--engine-name`` through ``parser`` when it is another architecture's
    product engine: that engine's ids and symbols are its own arch's, so a catalog
    of ``arch`` under its name (or its ids derived without its ``--arch``) would be
    neither. rocKE must already be bound."""
    for other in arches():
        if other != arch and engine_name == product_engine(other):
            parser.error(
                f"argument --engine-name: {engine_name} is the {other} engine; "
                f"pass --arch {other}"
            )


def spec_order(spec) -> tuple:
    """The descriptor spec order of ``spec``'s family and direct variant."""
    if not spec.is_direct:
        return MERGED_SPEC_ORDER if spec.is_merged else SPEC_ORDER
    order = DIRECT_SPEC_ORDER + DIRECT_VARIANT_FIELDS[spec.direct_variant]
    return order + ("split_c",) if spec.is_split_c else order


def instance_spec(identity_dict: dict):
    """``(spec, "")`` for a supported identity -- an instance of the spec class of the
    identity's architecture (``Gfx950ConvFwdAotSpec``, ``Gfx942ConvFwdAotSpec``) --
    else ``(None, reason)``.

    The rocKE adapter's ``conv_fwd_aot_spec_for_identity`` for the identity's own
    architecture: the identity must be one rocKE's forward sweeps build
    (``job_for_identity`` / ``direct_job_for_identity``) and use no feature that
    architecture's packaged launch does not model. An architecture without a packaged
    engine is refused by the adapter, with its reason.
    """
    from builders.common.convolution_forward import conv_fwd_aot_spec_for_identity
    from kernels.conv.kernel_identity import KernelIdentity

    try:
        identity = KernelIdentity.from_dict(identity_dict)
    except TypeError as exc:
        return None, f"not a KernelIdentity: {exc}"
    return conv_fwd_aot_spec_for_identity(identity, arch=identity.arch)


def instance_specs(identities: list, jobs: int = 1) -> list:
    """:func:`instance_spec` of each identity, in order, over ``jobs`` forked worker
    processes (serially where ``fork`` is not available)."""
    if (
        jobs <= 1
        or len(identities) < 1000
        or "fork" not in multiprocessing.get_all_start_methods()
    ):
        return [instance_spec(identity) for identity in identities]
    chunk = max(1, len(identities) // (jobs * 8))
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        return pool.map(instance_spec, identities, chunksize=chunk)


_variant = _common.variant
engine_slug = _common.engine_slug


#: :data:`PRODUCT_ENGINE`'s dotted native-symbol namespace
#: (``hipkernel.gfx950_conv_fwd_aot``, :func:`product_native_namespace` of
#: :data:`DEFAULT_ARCH`): what another engine states verbatim as its
#: ``native_symbols_from``.
PRODUCT_NATIVE_NAMESPACE = EngineSpec(name=PRODUCT_ENGINE).own_native_symbol_namespace


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


def build_config(
    specs: list,
    engine_name: str | None = None,
    authored_subpath: str = "",
    sharding: str = "dtype",
) -> dict:
    """The generator config for ``specs`` (already validated, de-duplicated, all of
    one architecture's engine).

    ``engine_name`` (by default the architecture's product engine,
    :func:`product_engine`) other than that product engine is a second engine over
    the same native code: its descriptors name the product's symbols
    (``native_symbols_from`` states :func:`product_native_namespace`).
    ``authored_subpath`` defaults to ``rocKE/<engine slug>``.

    The catalog is split into packs by ``sharding`` (:func:`shard_name`), each
    its own KDP, so no file has to hold the whole catalog. The packs declare no
    discriminator: they are shards of one catalog, and the engine's graph_match
    and the shared kernel matcher admit every one of them. Within a pack,
    implicit-GEMM instances keep their own variant and leave the direct fields to
    their builder_default. Direct instances are one variant per direct kernel
    (std and spatial share one), each stating DIRECT_FIELDS and its own block
    knobs (DIRECT_VARIANT_FIELDS) and leaving the others to theirs.
    """
    spec_arches = {spec.ARCH for spec in specs}
    if len(spec_arches) != 1:
        raise ValueError(
            f"one engine's catalog is one architecture's, got {spec_arches}"
        )
    (arch,) = spec_arches
    product, builder = product_engine(arch), adapter_builder(arch)
    engine_name = engine_name or product
    kmd = [
        {"name": name, "type": "int" if name in _INT_FIELDS else "string"}
        for name in SPEC_ORDER
    ] + [
        {
            "name": name,
            "type": "int" if name in _INT_FIELDS else "string",
            "default_value": default,
            "builder_default": True,
        }
        for name, default in {**DIRECT_FIELDS, **DIRECT_KNOB_FIELDS}.items()
    ]
    metadata = list(DIRECT_SPEC_ORDER + tuple(DIRECT_KNOB_FIELDS))
    shards = collections.defaultdict(list)
    for spec in specs:
        shards[shard_name(spec, sharding)].append(spec)
    packs = []
    for shard in sorted(shards):
        variants = []
        implicit = [s for s in shards[shard] if not s.is_direct and not s.is_merged]
        if implicit:
            variants.append(
                _variant(
                    _variant_name(IMPLICIT_VARIANT_NAME, arch), SPEC_ORDER, implicit
                )
            )
        merged = [s for s in shards[shard] if s.is_merged]
        if merged:
            variants.append(
                _variant(
                    _variant_name(_MERGED_VARIANT_NAME, arch),
                    MERGED_SPEC_ORDER,
                    merged,
                )
            )
        # std and spatial share one variant, as before the other direct variants.
        for group in (
            ("std", "spatial"),
            ("tiled",),
            ("col",),
            ("grouped",),
            ("nongrouped",),
            ("grouped_halo",),
            ("win",),
        ):
            members = [
                s
                for s in shards[shard]
                if s.is_direct and s.direct_variant in group and not s.is_split_c
            ]
            if members:
                variants.append(
                    _variant(
                        _variant_name(_DIRECT_VARIANT_NAMES[group[0]], arch),
                        spec_order(members[0]),
                        members,
                    )
                )
        split = [s for s in shards[shard] if s.is_split_c]
        if split:
            variants.append(
                _variant(
                    _variant_name(_SPLIT_C_VARIANT_NAME, arch),
                    spec_order(split[0]),
                    split,
                )
            )
        defaults = {"kind": "rocke", "source": ADAPTER_SOURCE, "builder": builder}
        pack = {"name": shard, "arch": [arch]}
        if split:
            if len(split) != len(shards[shard]):
                raise ValueError(
                    f"pack {shard!r} mixes split-C kernels with single-launch ones"
                )
            defaults["launches"] = split_c_launches(arch)
            pack["dispatch"] = SPLIT_C_DISPATCH
        packs.append({**pack, "kernel_defaults": defaults, "variants": variants})
    engine = {
        "name": engine_name,
        "sdk_version": "1.0.0",
        "knobs": list(ENGINE_KNOBS),
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
    kdp_path: Path, engine_name: str | None = None, arch: str = DEFAULT_ARCH
) -> int:
    """:func:`conv_aot_common.stabilize_ids` with ``arch``'s engine as the
    product (and, by default, as ``engine_name``)."""
    product = product_engine(arch)
    return _common.stabilize_ids(kdp_path, engine_name or product, product)


def kernel_uuid(
    instance_id: str, engine_name: str | None = None, arch: str = DEFAULT_ARCH
) -> str:
    """:func:`conv_aot_common.kernel_uuid` with ``arch``'s engine as the
    product (and, by default, as ``engine_name``)."""
    product = product_engine(arch)
    return _common.kernel_uuid(instance_id, engine_name or product, product)


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
        "derives (default: the --arch engine). Another name borrows that engine's "
        "native symbols; another architecture's engine is refused (pass its "
        "--arch instead).",
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
    # The engines and arches are the adapter's TARGETS, so rocKE is bound first,
    # --stabilize-ids included.
    _bind_rocke(args.rocke_root)
    check_arch(parser, args.arch)
    check_engine_name(parser, args.engine_name, args.arch)
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
        f"# Generated by tools/make_conv_aot_config.py from {sources}; "
        "do not edit by hand.\n"
    )
    config = build_config(specs, args.engine_name, args.authored_subpath, args.sharding)
    args.output.write_text(header + dump_config(config))
    print(
        f"{len(specs)} instance(s) written to {args.output} "
        f"({len(entries) - len(rejected) - len(specs)} duplicate(s), {len(rejected)} unsupported)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
