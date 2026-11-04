#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Turn rocKE AOT backward-weight conv instances into a hipkernel:Gfx950ConvWgradAot config.

The wgrad sibling of ``make_conv_dgrad_aot_config.py``: same input (JSON lists of
rocKE ``KernelIdentity`` dicts, or objects carrying one under ``"identity"``), same
output (a packaged-dialect generator config), same ``--stabilize-ids`` step. It reuses
that tool's shared base (``make_conv_aot_config.py``)'s YAML dumper, variant layout and
id derivation, with this engine as its own product engine: kernel ids are
``uuid5(INSTANCE_UUID_NAMESPACE, instance_id)`` and the bundle names its own native
symbols (no ``native_symbols_from``).

Every entry must be a gfx950 wgrad identity in fp16 or bf16 that the packaged launch
runs: an implicit-GEMM identity of the wgrad AOT sweep
(``benchmarks/common/kernel_sweep.py``) that is not two-stage (the Stage 2 reduce is
not wired in the adapter), or a wgrad identity of the direct sweep
(``benchmarks/common/direct_kernel_sweep.py``) of a packaged algorithm
(:func:`packaged_direct_algorithms`), including the five partial-slot families' fold
helper (``direct_wgrad_helper``). The helper is packaged as a kernel of its own family
(``kernel_family`` 2) that the kernel matcher never admits; the folded direct dispatch
loads it from the same pack. Anything else is listed with its reason and the tool
exits non-zero, unless ``--skip-unsupported`` is given.

The shipped catalog is ``configs/gfx950_conv_wgrad_aot_<dtype>.instances.json``,
written by ``tools/enumerate_conv_wgrad_aot_instances.py`` (which also drops every
two-stage identity before this tool ever sees one)::

    python tools/make_conv_wgrad_aot_config.py --rocke-root "$ROCKE" \\
        --instances configs/gfx950_conv_wgrad_aot_{fp16,bf16}.instances.json \\
        --output "$WORK"/wgrad_aot.yaml
    python generate.py --config "$WORK"/wgrad_aot.yaml --output-dir "$GENERATED" --force
    python tools/make_conv_wgrad_aot_config.py --stabilize-ids "$GENERATED"/<...>/*.kdp.json

The catalog is split into packs (:data:`SHARDINGS`): by default one per dtype and
kernel family (the implicit-GEMM instances are shape-generic and bake no pipeline
family the way dgrad's tilde kernels do, so a dtype/pipeline split would not shrink
anything; the eight families are the only axis that actually partitions the catalog).
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

import make_conv_aot_config as _fwd  # noqa: E402

#: The shipped wgrad engine: its own product engine (ids are uuid5 of instance_id).
PRODUCT_ENGINE = "hipkernel:Gfx950ConvWgradAot"
#: The rocKE adapter every kernel of the bundle is built by.
ADAPTER_SOURCE = "builders/common/convolution_backward_filter.py"
ADAPTER_BUILDER = "build_gfx950_conv_wgrad_aot"

#: Identity fields the packaged wgrad launch takes from an implicit-GEMM identity. A
#: wgrad binary bakes no filter, stride, padding or channel count at all (unlike
#: dgrad's tilde decomposition), so unlike dgrad there is no bakeable-geometry set:
#: every implicit-GEMM instance runs the problem exactly as the graph states it.
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
    "grouped",
    "async_dma",
    "unroll_k",
    "lds_k_pad",
    "lds_layout",
    "group_merge",
    "ws_replicas",
    "prefetch_addr",
    # The catalog's own placeholder (never the value actually launched --
    # gfx950_conv_wgrad_aot_split_k always re-resolves it from the real problem), but
    # still carried: it is read back by to_identity() for an implicit-GEMM instance, so
    # it must round-trip to instance_id like any other carried field.
    "split_k",
    # A packageable, but not plannable, gap (gfx950_conv_wgrad_aot_plan refuses to plan
    # one): a two-stage identity is still a valid spec this converter accepts, carried
    # the same way. tools/enumerate_conv_wgrad_aot_instances.py drops every two-stage
    # identity before it reaches this catalog, so every shipped instance states 0 here.
    "two_stage",
)
_BOOL_CARRIED = (
    "lds_k_outer",
    "grouped",
    "async_dma",
    "unroll_k",
    "prefetch_addr",
    "two_stage",
)
#: Identity fields an implicit-GEMM identity must hold at their KernelIdentity default
#: (they select emission the packaged launch or matcher does not model, or are a
#: capability a wgrad binary never bakes -- the filter, stride, padding, dilation and
#: channel counts, all 0/runtime for implicit GEMM here, unlike dgrad's dY-halo/patch
#: fields which simply do not exist for wgrad).
_CONSTANT = (
    "chiplet_swizzle",
    "waves_per_eu",
    "acc_epilogue",
    "split_k_cluster",
    "streamk",
    "streamk_reduction",
    "uniform_k_decode",
    "patch",
    "tap_outer_k",
    "dy_halo",
    "dy_halo_setprio",
    "dy_halo_kouter_pad",
    "num_load_waves",
    "cshuffle_no_alias",
    "is_3d",
    "is_pointwise",
    "filter_h",
    "filter_w",
    "filter_d",
    "stride_h",
    "stride_w",
    "dilation_h",
    "dilation_w",
    "pad_h",
    "pad_w",
    "cpg",
    "kpg",
    "block_h",
    "max_sub_gemms",
    "halo_max_wo",
    "async_chunk_a",
    "async_chunk_b",
    "knobs",
)
#: Spec field order of an implicit-GEMM Gfx950ConvWgradAotSpec; also its descriptor
#: spec order (the KMD's required fields).
SPEC_ORDER = ("dtype",) + _CARRIED + ("instance_id", "layout")
#: The direct placeholders every direct or helper instance's implicit-GEMM fields hold
#: (a wgrad implicit-GEMM instance bakes no capability at all, so unlike dgrad there is
#: no distinct "direct geometry" field set to add after these -- the capability fields
#: below are it). Mirrors the adapter's own ``_DIRECT_PLACEHOLDERS``, not re-derived
#: from it: ``to_identity()`` ignores these for a direct spec regardless of what is
#: written here, so they must spell exactly what the adapter's validator requires.
DIRECT_PLACEHOLDERS = {
    "tile_m": 0,
    "tile_n": 0,
    "tile_k": 0,
    "warp_m": 0,
    "warp_n": 0,
    "warp_tile_m": 0,
    "warp_tile_n": 0,
    "warp_tile_k": 0,
    "pipeline": "direct",
    "epilogue": "direct",
    "vector_size_a": 0,
    "vector_size_b": 0,
    "vector_size_c": 0,
    "lds_k_outer": 0,
    "grouped": 0,
    "async_dma": 0,
    "unroll_k": 0,
    "lds_k_pad": 0,
    "lds_layout": "default",
    "group_merge": 1,
    "ws_replicas": 0,
    "prefetch_addr": 0,
    "split_k": -1,
    "two_stage": 0,
}
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
#: The capability fields a direct instance bakes; a fold-helper instance (unlike the
#: pipeline that feeds it) bakes none of them -- see _direct_instance_spec.
_HELPER_CAPABILITY_FIELDS = tuple(
    name for name in DIRECT_FIELDS if name not in ("kernel_family", "direct_variant")
)


def _direct_knob_fields() -> dict:
    """Every direct/helper knob with its KMD default: ``_KNOB_DEFAULTS`` of the
    adapter, not a second list that could drift from it."""
    from builders.common.convolution_backward_filter import _KNOB_DEFAULTS

    return dict(_KNOB_DEFAULTS)


def _direct_variant_fields() -> dict:
    """Per variant, the knobs it states, in KMD order (``AOT_WGRAD_DIRECT_KNOBS``
    of the adapter). Every wgrad direct/helper variant states only its own knobs,
    unlike dgrad's transpose/reorganize helpers which also carry ``fold_k32``."""
    from builders.common.convolution_backward_filter import AOT_WGRAD_DIRECT_KNOBS

    order = list(_direct_knob_fields())
    stated = {
        variant: set(mapping.values()) for variant, mapping in AOT_WGRAD_DIRECT_KNOBS.items()
    }
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
    "lds_layout",
)


def _int_fields() -> tuple:
    """Every KMD field that is an int. Resolved on first use: the knob set comes from
    the adapter, which is bound at run time."""
    return tuple(
        f
        for f in DIRECT_SPEC_ORDER + tuple(_direct_knob_fields())
        if f not in _STRING_FIELDS
    )


#: The UED knobs: every KMD field that tunes a kernel, so the ranking model can tell
#: apart the instances one problem admits; together they make every per-graph
#: candidate tuple unique. Left out are the fields that only restate the problem the
#: instance was matched to (dtype, layout, the direct filter, stride, padding,
#: dilation and channel counts), the constants (wave_size) and instance_id.
def engine_knobs() -> tuple:
    """The UED knobs: every KMD field that tunes a kernel, so the ranking model can
    tell apart the instances one problem admits. The per-variant knobs come from the
    adapter, so a pipeline this engine gains is rankable without a change here."""
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
        "async_dma",
        "unroll_k",
        "lds_k_pad",
        "lds_layout",
        "group_merge",
        "ws_replicas",
        "prefetch_addr",
    )
    return fixed + tuple(_direct_knob_fields())


#: Every wgrad direct-sweep algorithm the packaged launch runs (or loads, for the fold
#: helper), and its direct_variant: AOT_WGRAD_DIRECT_ALGORITHMS of the adapter. Derived,
#: not restated: the adapter decides what has a packaged launch, and a second list here
#: could only drift from it. Resolved on first use, since rocKE is bound at run time
#: (_bind_rocke), not at import.
_PACKAGED_DIRECT_CACHE = None


def packaged_direct_algorithms() -> dict:
    """Each wgrad direct-sweep algorithm with a packaged launch, and its
    ``direct_variant`` (AOT_WGRAD_DIRECT_ALGORITHMS of the adapter)."""
    return _packaged_direct()[0]


def direct_directions() -> dict:
    """The ``KernelIdentity.direction`` each packaged direct algorithm carries."""
    return _packaged_direct()[1]


def _packaged_direct():
    global _PACKAGED_DIRECT_CACHE
    if _PACKAGED_DIRECT_CACHE is None:
        from builders.common.convolution_backward_filter import (
            AOT_WGRAD_DIRECT_ALGORITHMS,
            KERNEL_FAMILY_WGRAD_HELPER,
        )

        algorithms, directions = {}, {}
        for algorithm, (family, variant) in AOT_WGRAD_DIRECT_ALGORITHMS.items():
            algorithms[algorithm] = variant
            directions[algorithm] = (
                "direct_wgrad_helper"
                if family == KERNEL_FAMILY_WGRAD_HELPER
                else "direct_wgrad"
            )
        _PACKAGED_DIRECT_CACHE = (algorithms, directions)
    return _PACKAGED_DIRECT_CACHE


#: The generator variant (kernel name) of the implicit-GEMM instances and of each
#: direct and helper variant.
IMPLICIT_VARIANT_NAME = "gfx950_conv_wgrad_aot_{dtype}_{instance_id}"
_DIRECT_VARIANT_NAMES = {
    "direct_wgrad": "gfx950_conv_wgrad_aot_dw_{dtype}_{instance_id}",
    "direct_wgrad_tiled": "gfx950_conv_wgrad_aot_tiled_{dtype}_{instance_id}",
    "direct_wgrad_dw": "gfx950_conv_wgrad_aot_dwdw_{dtype}_{instance_id}",
    "direct_wgrad_dw_cols": "gfx950_conv_wgrad_aot_dwcols_{dtype}_{instance_id}",
    "direct_wgrad_dw_mfma": "gfx950_conv_wgrad_aot_dwmfma_{dtype}_{instance_id}",
    "direct_wgrad_gather": "gfx950_conv_wgrad_aot_gather_{dtype}_{instance_id}",
    "direct_wgrad_grouped_mfma": "gfx950_conv_wgrad_aot_groupedmfma_{dtype}_{instance_id}",
    "direct_wgrad_fold": "gfx950_conv_wgrad_aot_fold_{dtype}_{instance_id}",
}

_bind_rocke = _fwd._bind_rocke
engine_slug = _fwd.engine_slug
dump_config = _fwd.dump_config
INSTANCE_UUID_NAMESPACE = _fwd.INSTANCE_UUID_NAMESPACE


def kernel_uuid(instance_id: str, engine_name: str = PRODUCT_ENGINE) -> str:
    """:func:`make_conv_aot_config.kernel_uuid` with this engine as the product."""
    return _fwd.kernel_uuid(instance_id, engine_name, PRODUCT_ENGINE)


def stabilize_ids(kdp_path: Path, engine_name: str = PRODUCT_ENGINE) -> int:
    """:func:`make_conv_aot_config.stabilize_ids` with this engine as the product."""
    return _fwd.stabilize_ids(kdp_path, engine_name, PRODUCT_ENGINE)


def spec_order(spec) -> tuple:
    """The descriptor spec order of ``spec``'s family and variant."""
    if not spec.is_direct:
        return SPEC_ORDER
    return DIRECT_SPEC_ORDER + _direct_variant_fields()[spec.direct_variant]


def instance_spec(identity_dict: dict):
    """``(Gfx950ConvWgradAotSpec, "")`` for a supported identity, else ``(None, reason)``."""
    from benchmarks.common.kernel_cache import KernelIdentity

    identity = KernelIdentity.from_dict(identity_dict)
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
    from builders.common.convolution_backward_filter import (
        supports_gfx950_conv_wgrad_aot,
    )

    ok, reason = supports_gfx950_conv_wgrad_aot(spec)
    return (spec, "") if ok else (None, reason)


def _implicit_instance_spec(identity):
    from benchmarks.common.kernel_cache import KernelIdentity
    from builders.common.convolution_backward_filter import Gfx950ConvWgradAotSpec

    if (identity.arch, identity.direction, identity.algorithm) != (
        "gfx950",
        "wgrad",
        "implicit_gemm",
    ):
        return None, (
            f"not a gfx950 wgrad implicit-GEMM identity "
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
    # Every packaged identity must be one rocKE's wgrad generator produces.
    from benchmarks.common.kernel_sweep import job_for_identity

    try:
        job_for_identity(identity)
    except LookupError as exc:
        return None, str(exc)
    values = {name: getattr(identity, name) for name in _CARRIED}
    for name in _BOOL_CARRIED:
        values[name] = int(values[name])
    spec = Gfx950ConvWgradAotSpec(
        dtype=identity.dtype_a, instance_id=identity.stable_hash(), **values
    )
    return _checked(spec)


def _direct_instance_spec(identity):
    """A wgrad direct or fold-helper identity: the direct sweep must be able to build
    it, and the packaged launch must run (or load) its algorithm."""
    from benchmarks.common.direct_kernel_sweep import direct_job_for_identity
    from builders.common.convolution_backward_filter import (
        Gfx950ConvWgradAotSpec,
        aot_wgrad_direct_instance_fields,
    )

    if identity.arch != "gfx950":
        return None, f"not a gfx950 wgrad direct identity ({identity.arch})"
    if identity.algorithm not in packaged_direct_algorithms():
        return None, (
            f"direct algorithm {identity.algorithm!r} has no packaged wgrad launch "
            f"(supported: {sorted(packaged_direct_algorithms())})"
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
        fields = aot_wgrad_direct_instance_fields(identity)
    except (LookupError, TypeError, ValueError) as exc:
        return None, str(exc)
    if identity.direction == "direct_wgrad_helper":
        # kernels.conv.direct_aot_jobs._helper_identity keeps the feeding pipeline's
        # filter and dilation on rocKE's own helper identity (its build-cache key only
        # zeros pad/stride/cpg/kpg), but the fold reads a flat taps count alone
        # (direct_wgrad_fold_knobs) and bakes no filter, stride, padding, dilation or
        # channel count -- supports_gfx950_conv_wgrad_aot requires a helper instance to
        # record every one at 0 (mirrors the adapter's own
        # gfx950_conv_wgrad_aot_fold_fields, which builds a helper's fields from a
        # folded spec rather than from this raw identity).
        fields.update(dict.fromkeys(_HELPER_CAPABILITY_FIELDS, 0))
    spec = Gfx950ConvWgradAotSpec(
        dtype=identity.dtype_a,
        instance_id=identity.stable_hash(),
        wave_size=identity.wave_size,
        **DIRECT_PLACEHOLDERS,
        **fields,
    )
    return _checked(spec)


#: How :func:`build_config` splits a catalog into packs, one KDP each.
#: ``dtype``: one pack per dtype. ``family``: one per dtype and kernel family (the
#: eight pipelines, implicit GEMM included) -- wgrad's eight families are far more
#: heterogeneous than dgrad's single implicit-GEMM pipeline plus a handful of direct
#: ones, so this keeps each shard to one pipeline's instances.
SHARDINGS = ("dtype", "family")


def shard_name(spec, sharding: str = "family") -> str:
    """The pack ``spec`` belongs to under ``sharding`` (see :data:`SHARDINGS`)."""
    if sharding == "dtype":
        return spec.dtype
    if sharding != "family":
        raise ValueError(f"unknown sharding {sharding!r}; expected one of {SHARDINGS}")
    if not spec.is_direct:
        return f"{spec.dtype}_implicit"
    return f"{spec.dtype}_{spec.direct_variant}"


def kmd_fields() -> list:
    """The KMD: SPEC_ORDER required, then the direct fields and knobs with defaults."""
    return [
        {"name": name, "type": "int" if name in _int_fields() else "string"}
        for name in SPEC_ORDER
    ] + [
        {
            "name": name,
            "type": "int" if name in _int_fields() else "string",
            "default_value": default,
            "builder_default": True,
        }
        for name, default in {**DIRECT_FIELDS, **_direct_knob_fields()}.items()
    ]


def build_config(
    specs: list,
    engine_name: str = PRODUCT_ENGINE,
    authored_subpath: str = "",
    sharding: str = "family",
) -> dict:
    """The generator config for ``specs`` (already validated, de-duplicated).

    The packs are shards of one catalog with no discriminator. Within a pack the
    implicit-GEMM instances are one variant, and each direct or helper variant its
    own, stating DIRECT_FIELDS and its knobs and leaving the others to their
    builder_default. ``engine_name`` other than :data:`PRODUCT_ENGINE` borrows this
    engine's native symbols.
    """
    metadata = list(DIRECT_SPEC_ORDER + tuple(_direct_knob_fields()))
    shards = collections.defaultdict(list)
    for spec in specs:
        shards[shard_name(spec, sharding)].append(spec)
    packs = []
    for shard in sorted(shards):
        variants = []
        implicit = [s for s in shards[shard] if not s.is_direct]
        if implicit:
            variants.append(
                _fwd._variant(IMPLICIT_VARIANT_NAME, spec_order(implicit[0]), implicit)
            )
        for variant_name in _DIRECT_VARIANT_NAMES:
            members = [
                s
                for s in shards[shard]
                if s.is_direct and s.direct_variant == variant_name
            ]
            if members:
                variants.append(
                    _fwd._variant(
                        _DIRECT_VARIANT_NAMES[variant_name],
                        spec_order(members[0]),
                        members,
                    )
                )
        packs.append(
            {
                "name": shard,
                "arch": ["gfx950"],
                "kernel_defaults": {
                    "kind": "rocke",
                    "source": ADAPTER_SOURCE,
                    "builder": ADAPTER_BUILDER,
                },
                "variants": variants,
            }
        )
    engine = {
        "name": engine_name,
        "sdk_version": "1.0.0",
        "knobs": list(engine_knobs()),
        "heuristic": "native",
    }
    if engine_name != PRODUCT_ENGINE:
        engine["native_symbols_from"] = PRODUCT_ENGINE
    return {
        "dialect": "packaged",
        "authored_subpath": authored_subpath or f"rocKE/{engine_slug(engine_name)}",
        "engine": engine,
        "kmd_fields": kmd_fields(),
        "kernel_source_kind": "rocke",
        # The wgrad launch bakes no sub-GEMM record table or MFMA weight scratch; the
        # native handler needs no per-problem workspace sizing.
        "workspace_policy": "none",
        "graph_match": {"shape": "shared_shape", "discriminator": "none"},
        "specialization": {
            "metadata_fields": metadata,
            "matcher_only_fields": [],
            "bindings": {name: {"field": name} for name in metadata},
        },
        "packs": packs,
    }


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
        "--skip-unsupported",
        action="store_true",
        help="Drop unsupported identities (each is still reported) instead of failing.",
    )
    parser.add_argument(
        "--engine-name",
        default=PRODUCT_ENGINE,
        help="Engine the bundle declares, and whose kernel ids --stabilize-ids "
        "derives (default: %(default)s).",
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
        default="family",
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

    if args.stabilize_ids:
        for kdp in args.stabilize_ids:
            count = stabilize_ids(kdp, args.engine_name)
            print(f"{kdp}: stabilized {count} kernel id(s)")
        return 0
    if not (args.rocke_root and args.instances and args.output):
        parser.error("--rocke-root, --instances and --output are required")
    _bind_rocke(args.rocke_root)

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
        f"# Generated by tools/make_conv_wgrad_aot_config.py from {sources}; "
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
